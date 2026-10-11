"""Re-IoU: IoU of the best label-compatible detection after an entity returns into view following an absence."""
from __future__ import annotations

import argparse
import json

import numpy as np

from .common import GeneratedSample, iou_xyxy, read_json, write_json
from .metric_entity import DEFAULT_LABEL_MAP, _label_ok, _pred_index, evaluate as evaluate_entity

BUCKETS = (("le2", 0.0, 2.0), ("2to5", 2.0, 5.0), ("5to10", 5.0, 10.0), ("gt10", 10.0, float("inf")))


def bucket_of(absence: float) -> str:
    for name, lo, hi in BUCKETS:
        if lo < absence <= hi or (name == "le2" and absence <= hi):
            return name
    return "gt10"


def absence_events(rows: list[dict], min_absence: float = 1.0) -> list[dict]:
    """rows = metric_entity per-entity `instants` (dict t, gt, ...). An event is a maximal run of instants with
    gt None that is bounded by visible instants on both sides and spans >= min_absence seconds (return - leave)."""
    vis = [r.get("gt") is not None for r in rows]
    events, k, n = [], 0, len(rows)
    while k < n:
        if vis[k]:
            k += 1; continue
        s = k
        while k < n and not vis[k]:
            k += 1
        e = k
        if s == 0 or e >= n:
            continue
        leave_t, return_t = float(rows[s - 1]["t"]), float(rows[e]["t"])
        absence = return_t - leave_t
        if absence + 1e-9 < min_absence:
            continue
        events.append(dict(leave_index=s - 1, return_index=e, leave_t=leave_t, return_t=return_t,
                           absence=absence, bucket=bucket_of(absence)))
    return events


def best_match_rows(rows: list[dict], label: str, pred_idx: dict, out_frames: list[int], label_map=None) -> list[dict]:
    """Per instant: the label-compatible detection (any track) with the highest IoU against the reference box.
    Returns rows aligned with `rows`: dict(t, gt, best_box, best_iou, best_track, center_err); gt None -> invisible."""
    label_map = label_map or DEFAULT_LABEL_MAP
    out = []
    for k, r in enumerate(rows):
        g = r.get("gt")
        if g is None:
            out.append(dict(t=r["t"], gt=None)); continue
        best, bb, bt = 0.0, None, None
        gc = np.array([(g[0] + g[2]) / 2, (g[1] + g[3]) / 2]); side = max(g[2] - g[0], g[3] - g[1], 1e-6)
        near, nd = None, None
        for tid, lab, b in pred_idx.get(out_frames[k], []):
            if not _label_ok(label, lab, label_map):
                continue
            v = iou_xyxy(g, b)
            if v > best:
                best, bb, bt = v, b, tid
            pc = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]); d = float(np.linalg.norm(gc - pc)) / side
            if nd is None or d < nd:
                near, nd = b, d
        ce = None
        if bb is not None:
            pc = np.array([(bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2])
            ce = float(np.linalg.norm(gc - pc) / np.sqrt(2.0))
        out.append(dict(t=r["t"], gt=g, best_box=bb, best_iou=float(best), best_track=bt, center_err=ce,
                        near_box=near, near_dist=nd))
    return out


def _window_stats(best: list[dict], rows: list[dict], idx: list[int], lenient_dist: float = 1.0) -> dict:
    found = [i for i in idx if best[i].get("best_box") is not None]
    def lenient_box(i):
        """strict hit (overlap) or, failing that, the nearest detection if its centre is within lenient_dist sides"""
        if best[i].get("best_box") is not None:
            return best[i]["best_box"]
        if best[i].get("near_dist") is not None and best[i]["near_dist"] <= lenient_dist:
            return best[i]["near_box"]
        return None
    near_boxes = [lenient_box(i) for i in idx]
    near = [i for i, b in zip(idx, near_boxes) if b is not None]
    ious = [float(best[i]["best_iou"]) for i in idx]
    tracked = [i for i in idx if rows[i].get("pred") is not None]
    return dict(instants=idx, n=len(idx), n_found=len(found), n_near=len(near),
                rate=(len(found) / len(idx)) if idx else None,
                rate_lenient=(len(near) / len(idx)) if idx else None,
                near_boxes=near_boxes,
                iou=float(np.mean(ious)) if ious else None,
                iou_present=float(np.mean([best[i]["best_iou"] for i in found])) if found else None,
                center_err=float(np.mean([best[i]["center_err"] for i in found])) if found else None,
                boxes=[best[i].get("best_box") for i in idx],
                tracked_rate=(len(tracked) / len(idx)) if idx else None,
                tracked_iou=float(np.mean([rows[i]["iou"] for i in idx])) if idx else None)


def event_windows(rows: list[dict], ev: dict, window: float = 1.0) -> tuple[list[int], list[int]]:
    lt, rt = ev["leave_t"], ev["return_t"]
    pre = [i for i, r in enumerate(rows) if r.get("gt") is not None and lt - window < r["t"] <= lt + 1e-9]
    post = [i for i, r in enumerate(rows) if r.get("gt") is not None and rt - 1e-9 <= r["t"] < rt + window]
    return pre, post


def _mean(vals):
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else None


def weak_reasons(rows: list[dict], ev: dict, grid_hz: float, out_h: int, min_post_visible: float = 0.5,
                 min_gt_height_px: float = 16.0, edge_eps: float = 0.005) -> list[str]:
    """Why an event is a weak test of re-appearance (empty list = strong)."""
    k = ev["return_index"]; n = 0
    while k < len(rows) and rows[k].get("gt") is not None:
        n += 1; k += 1
    reasons = []
    if n / grid_hz < min_post_visible - 1e-9:
        reasons.append("short_return")
    g = rows[ev["return_index"]]["gt"]
    if g[0] <= edge_eps or g[1] <= edge_eps or g[2] >= 1 - edge_eps or g[3] >= 1 - edge_eps:
        reasons.append("edge_reentry")
    if (g[3] - g[1]) * out_h < min_gt_height_px:
        reasons.append("small_box")
    return reasons


def evaluate(gt: dict, prediction: dict, sample: GeneratedSample, grid_hz: float = 8.0, entities: str = "dynamic",
             min_absence: float = 1.0, window: float = 1.0, entity_metric: dict | None = None, lenient_dist: float = 1.0,
             min_post_visible: float = 0.5, min_gt_height_px: float = 16.0) -> dict:
    ent = entity_metric or evaluate_entity(gt, prediction, sample, grid_hz=grid_hz, entities=entities)
    pred_idx = _pred_index(prediction) if prediction.get("frames") is not None else {}
    out_frames = ent["alignment"]["output_frames"]
    events, per_entity = [], {}
    for e, pe in ent["per_entity"].items():
        rows = pe["instants"]
        best = best_match_rows(rows, pe["label"], pred_idx, out_frames)
        vis = [b for b in best if b["gt"] is not None]
        per_entity[e] = dict(label=pe["label"], n_visible_instants=len(vis),
                             best_iou=float(np.mean([b["best_iou"] for b in vis])) if vis else None,
                             found_rate=float(np.mean([b["best_box"] is not None for b in vis])) if vis else None,
                             best_center_err=_mean([b["center_err"] for b in vis]),
                             tracked_iou=pe["mean_iou"], tracked_miss_rate=pe["miss_rate"], instants=best)
        for ev in absence_events(rows, min_absence):
            pre, post = event_windows(rows, ev, window)
            weak = weak_reasons(rows, ev, grid_hz, sample.height, min_post_visible, min_gt_height_px)
            events.append(dict(entity=e, label=pe["label"], track_id=pe["track_id"], **ev, weak=weak,
                               pre=_window_stats(best, rows, pre, lenient_dist), post=_window_stats(best, rows, post, lenient_dist)))

    def agg(evs):
        return dict(n_events=len(evs),
                    reappear_rate=_mean([x["post"]["rate"] for x in evs]),
                    reappear_rate_lenient=_mean([x["post"]["rate_lenient"] for x in evs]),
                    pre_rate_lenient=_mean([x["pre"]["rate_lenient"] for x in evs]),
                    reappear_iou=_mean([x["post"]["iou"] for x in evs]),
                    reappear_iou_present=_mean([x["post"]["iou_present"] for x in evs]),
                    reappear_center_err=_mean([x["post"]["center_err"] for x in evs]),
                    pre_rate=_mean([x["pre"]["rate"] for x in evs]),
                    pre_iou=_mean([x["pre"]["iou"] for x in evs]),
                    pre_center_err=_mean([x["pre"]["center_err"] for x in evs]),
                    tracked_reappear_rate=_mean([x["post"]["tracked_rate"] for x in evs]),
                    tracked_reappear_iou=_mean([x["post"]["tracked_iou"] for x in evs]))

    by_bucket = {name: agg([x for x in events if x["bucket"] == name]) for name, _, _ in BUCKETS}
    strong = [x for x in events if not x["weak"]]
    valid = [v for v in per_entity.values() if v["n_visible_instants"] > 0]
    summary = dict(
        **agg(events), n_entities_with_events=len({x["entity"] for x in events}),
        strong=dict(**agg(strong), by_bucket={name: agg([x for x in strong if x["bucket"] == name]) for name, _, _ in BUCKETS}),
        n_weak=len(events) - len(strong),
        weak_reasons={r: sum(r in x["weak"] for x in events) for r in ("short_return", "edge_reentry", "small_box")},
        weak_criteria=dict(min_post_visible=min_post_visible, min_gt_height_px=min_gt_height_px, edge_eps=0.005),
        lenient_dist=lenient_dist,
        absence_mean=_mean([x["absence"] for x in events]),
        absence_max=max([x["absence"] for x in events]) if events else None,
        min_absence=min_absence, window=window, grid_hz=grid_hz, entities=entities, matching="best_match_per_instant",
        clip=dict(best_iou=_mean([v["best_iou"] for v in valid]), found_rate=_mean([v["found_rate"] for v in valid]),
                  best_center_err=_mean([v["best_center_err"] for v in valid]),
                  tracked_iou=ent["summary"]["mean_iou"], tracked_center_err=ent["summary"]["mean_center_err"],
                  tracked_miss_rate=ent["summary"]["miss_rate"], n_instants=ent["summary"]["n_instants"],
                  n_entities=ent["summary"]["n_entities"], tracker=ent["summary"]["tracker"]),
    )
    return dict(schema="prowbench.metric_memory.v3", protocol="memory_30s_v1", sample_id=sample.sample_id,
                method=sample.method, run_id=sample.run_id, summary=summary, by_bucket=by_bucket, events=events,
                per_entity=per_entity, assignment=ent["assignment"], alignment=ent["alignment"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--prediction", required=True)
    ap.add_argument("--sample", required=True, help="GeneratedSample json string or path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--entity-metric", default=None, help="existing metric_entity json to reuse the assignment")
    ap.add_argument("--grid-hz", type=float, default=8.0)
    ap.add_argument("--entities", default="dynamic")
    ap.add_argument("--min-absence", type=float, default=1.0)
    ap.add_argument("--window", type=float, default=1.0)
    a = ap.parse_args(argv)
    s = json.loads(a.sample) if a.sample.strip().startswith("{") else read_json(a.sample)
    res = evaluate(read_json(a.gt), read_json(a.prediction), GeneratedSample.from_dict(s), a.grid_hz, a.entities,
                   a.min_absence, a.window, read_json(a.entity_metric) if a.entity_metric else None)
    write_json(a.out, res)
    sm = res["summary"]
    print(f"{s['sample_id']}: {sm['n_events']} events, reappear rate {sm['reappear_rate']}, iou {sm['reappear_iou']} -> {a.out}")


if __name__ == "__main__":
    main()
