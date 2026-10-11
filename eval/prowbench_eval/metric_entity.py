"""Entity control: IoU, centre error (CErr) and miss rate of the tracked entities against the projected ground-truth
boxes on the evaluation grid."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from .common import (GeneratedSample, eval_grid, evaluated_entities, hungarian, iou_xyxy, output_frame_for_time,
                     read_json, source_frame_for_time, write_json)

DEFAULT_LABEL_MAP = {
    "person": {"person", "man", "woman", "human", "people", "character"},
    "crate": {"crate", "box", "wooden box", "parcel", "package"},
    "parcel": {"parcel", "package", "box", "crate", "wooden box"},
    "ball": {"ball"},
    "bench": {"bench", "seat", "chair"},
    "chair": {"chair", "seat", "bench"},
    "platform": {"platform", "lift", "elevator", "deck"},
    "cable": {"cable", "zipline", "rope", "wire"},
    "handle": {"handle", "trolley", "pulley", "grip"},
    "tree": {"tree"},
    "building": {"building", "house", "hut"},
    "car": {"car", "vehicle"},
    "horse": {"horse"},
    "boat": {"boat"},
    "zipline": {"zipline", "cable", "rope"},
    "jetpack": {"jetpack"},
    "bow": {"bow", "recurve bow"},
    "arrow": {"arrow"},
    "target": {"target", "archery target", "bullseye"},
    "door": {"door", "wooden door"},
}


def _label_ok(gt_label: str, pred_label: str, label_map) -> bool:
    if gt_label is None or pred_label is None:
        return True
    acc = label_map.get(gt_label, {gt_label})
    p = pred_label.lower().strip()
    return p in acc or any(a in p for a in acc)


def _pred_index(prediction: dict):
    """frame index -> list of (track_id, label, box_norm)"""
    W, H = prediction["width"], prediction["height"]
    if not np.isfinite([W, H]).all() or min(W, H) <= 0:
        raise ValueError("Invalid tracker image size")
    idx = defaultdict(list)
    frame_ids = set()
    for fr in prediction["frames"]:
        if fr['frame'] in frame_ids:
            raise ValueError("Duplicate tracker frame record")
        frame_ids.add(fr['frame'])
        tids = set()
        for d in fr["detections"]:
            if d['track_id'] in tids:
                raise ValueError("Duplicate track identity within one frame")
            tids.add(d['track_id'])
            if np.asarray(d['box']).shape != (4,) or not np.isfinite(d['box']).all():
                raise ValueError("Tracker box must contain four finite coordinates")
            x0, y0, x1, y1 = d["box"]
            if x1 <= x0 or y1 <= y0:
                raise ValueError("Tracker box must have positive area")
            idx[fr["frame"]].append((d["track_id"], d.get("label"), [x0 / W, y0 / H, x1 / W, y1 / H]))
    return idx


def select_entities(gt: dict, which: str = "dynamic_or_candidate") -> list[str]:
    """Evaluated entities: dynamic or interactionCandidate, minus EVAL_EXCLUDED_LABELS (terrain/effect/geometry)."""
    return evaluated_entities(gt, which)


def evaluate(gt: dict, prediction: dict, sample: GeneratedSample, grid_hz: float = 8.0,
             entities: str = "dynamic_or_candidate", label_map=None, min_visible_area: float = 1e-4,
             min_track_iou: float = 0.05) -> dict:
    label_map = label_map or DEFAULT_LABEL_MAP
    sp = sample.spatial
    instants = eval_grid(grid_hz)
    if not np.isfinite(sample.fps) or sample.fps <= 0 or sample.frame_count <= 0:
        raise ValueError("Invalid generated-video FPS or frame count")
    if max(instants) > (sample.frame_count - .5) / sample.fps + 1e-9:
        raise ValueError("Generated video does not cover the evaluation time grid")
    for key in ("width", "height", "frame_count"):
        if key in prediction and prediction[key] != getattr(sample, key):
            raise ValueError(f"Tracker {key} disagrees with generated sample")
    if 'fps' in prediction and not np.isclose(prediction['fps'], sample.fps, rtol=1e-5):
        raise ValueError("Tracker FPS disagrees with generated sample")
    pred_idx = _pred_index(prediction)
    ent_ids = select_entities(gt, entities)

    gt_boxes = {e: [] for e in ent_ids}
    src_frames, out_frames = [], []
    for t in instants:
        sf = source_frame_for_time(t)
        of = min(output_frame_for_time(t, sample.fps), sample.frame_count - 1)
        src_frames.append(sf); out_frames.append(of)
        fb = gt["frames"][sf]["boxes"]
        for e in ent_ids:
            b = fb.get(e)
            if b and b["visible"]:
                nb = sp.box_src_to_out_norm(b["box"])
                area = max(0.0, nb[2] - nb[0]) * max(0.0, nb[3] - nb[1])
                gt_boxes[e].append(nb.tolist() if area >= min_visible_area else None)
            else:
                gt_boxes[e].append(None)

    track_ids = sorted({tid for fr in pred_idx.values() for tid, _, _ in fr})
    track_label = {}
    for fr in pred_idx.values():
        for tid, lab, _ in fr:
            track_label.setdefault(tid, lab)

    n_e, n_t = len(ent_ids), len(track_ids)
    score = np.zeros((n_e, n_t))
    for i, e in enumerate(ent_ids):
        for j, tid in enumerate(track_ids):
            if not _label_ok(gt["entities"][e]["label"], track_label[tid], label_map):
                score[i, j] = -1.0
                continue
            vals = []
            for k, of in enumerate(out_frames):
                g = gt_boxes[e][k]
                if g is None:
                    continue
                pb = next((b for t2, _, b in pred_idx.get(of, []) if t2 == tid), None)
                vals.append(iou_xyxy(g, pb) if pb is not None else 0.0)
            score[i, j] = float(np.mean(vals)) if vals else 0.0

    assignment = {}
    if n_e and n_t:
        rows, cols = hungarian(1.0 - np.where(score < 0, -1.0, score))
        for r, c in zip(rows, cols):
            if score[r, c] >= min_track_iou:
                assignment[ent_ids[r]] = track_ids[c]

    per_entity = {}
    for e in ent_ids:
        tid = assignment.get(e)
        ious, cerr, present, missing = [], [], 0, 0
        rows_out = []
        for k, (t, of) in enumerate(zip(instants, out_frames)):
            g = gt_boxes[e][k]
            if g is None:
                rows_out.append(dict(t=t, gt=None)); continue
            present += 1
            pb = None
            if tid is not None:
                pb = next((b for t2, _, b in pred_idx.get(of, []) if t2 == tid), None)
            if pb is None:
                missing += 1; ious.append(0.0)
                rows_out.append(dict(t=t, gt=g, pred=None, iou=0.0, center_err=None)); continue
            iou = iou_xyxy(g, pb)
            gc = np.array([(g[0] + g[2]) / 2, (g[1] + g[3]) / 2]); pc = np.array([(pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2])
            ce = float(np.linalg.norm(gc - pc) / np.sqrt(2.0))
            ious.append(iou); cerr.append(ce)
            rows_out.append(dict(t=t, gt=g, pred=pb, iou=iou, center_err=ce))
        per_entity[e] = dict(
            label=gt["entities"][e]["label"], track_id=tid, n_visible_instants=present,
            mean_iou=float(np.mean(ious)) if ious else None,
            mean_center_err=float(np.mean(cerr)) if cerr else None,
            miss_rate=(missing / present) if present else None,
            instants=rows_out)

    valid = [v for v in per_entity.values() if v["n_visible_instants"] > 0]
    summary = dict(
        mean_iou=float(np.mean([v["mean_iou"] for v in valid])) if valid else None,
        mean_center_err=float(np.mean([v["mean_center_err"] for v in valid if v["mean_center_err"] is not None])) if any(v["mean_center_err"] is not None for v in valid) else None,
        miss_rate=float(np.mean([v["miss_rate"] for v in valid])) if valid else None,
        n_entities=len(valid), n_instants=len(instants), grid_hz=grid_hz,
        n_tracks=n_t, n_assigned=len(assignment),
        unassigned_tracks=[t for t in track_ids if t not in assignment.values()],
        tracker=(prediction.get("tracker") or {}).get("name", "dino_iou"),
    )
    return dict(schema="prowbench.metric_entity.v1", sample_id=sample.sample_id, method=sample.method, run_id=sample.run_id,
                summary=summary, per_entity=per_entity, assignment=assignment,
                alignment=dict(instants=instants, source_frames=src_frames, output_frames=out_frames,
                               crop_xywh=list(sample.crop_xywh), out_size=[sample.width, sample.height]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--prediction", required=True)
    ap.add_argument("--sample", required=True, help="json string or path of a samples.jsonl row")
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid-hz", type=float, default=8.0)
    ap.add_argument("--entities", default="dynamic_or_candidate", choices=["dynamic_or_candidate", "dynamic", "all"],
                    help="evaluated entities (common.evaluated_entities): dynamic_or_candidate (five-second datasets), "
                         "dynamic (long-horizon dataset) or all")
    args = ap.parse_args(argv)
    s = args.sample
    sample = GeneratedSample.from_dict(json.loads(s) if s.strip().startswith("{") else read_json(s))
    res = evaluate(read_json(args.gt), read_json(args.prediction), sample, args.grid_hz, args.entities)
    write_json(args.out, res)
    print(json.dumps(res["summary"]))


if __name__ == "__main__":
    main()
