"""Identity of returning entities: DINOv2 embedding similarity between entity crops before and after an absence."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from .common import GeneratedSample, iter_video_frames, read_json, write_json

DEFAULT_MODEL = "facebook/dinov2-base"


def _load(model: str, device: str):
    import torch
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained(model)
    net = AutoModel.from_pretrained(model).to(device).eval()
    return proc, net, torch


def crop_box(frame: np.ndarray, box_norm, margin: float = 0.25, min_side: int = 32) -> np.ndarray:
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = box_norm
    bw, bh = (x1 - x0) * w, (y1 - y0) * h
    cx, cy = (x0 + x1) / 2 * w, (y0 + y1) / 2 * h
    bw, bh = max(bw * (1 + 2 * margin), min_side), max(bh * (1 + 2 * margin), min_side)
    xa, xb = int(max(0, cx - bw / 2)), int(min(w, cx + bw / 2))
    ya, yb = int(max(0, cy - bh / 2)), int(min(h, cy + bh / 2))
    return frame[ya:max(ya + 1, yb), xa:max(xa + 1, xb)]


def embed(crops: list[np.ndarray], loaded, batch: int = 32) -> np.ndarray:
    proc, net, torch = loaded
    from PIL import Image
    feats = []
    for i in range(0, len(crops), batch):
        ims = [Image.fromarray(c) for c in crops[i:i + batch]]
        inputs = proc(images=ims, return_tensors="pt").to(net.device)
        with torch.no_grad():
            out = net(**inputs)
        f = out.last_hidden_state[:, 0]
        feats.append(torch.nn.functional.normalize(f, dim=-1).float().cpu().numpy())
    return np.concatenate(feats) if feats else np.zeros((0, 1))


def _needed(memory: dict, short_gap: float, grid_hz: float):
    """(entity, instant index) pairs that need a crop: event windows + short-range pairs; returns
    rows {entity: best-match rows}, {(entity, k): box_norm}, event crop keys, short pairs [(entity, k, k2)]."""
    rows = {e: pe["instants"] for e, pe in memory["per_entity"].items()}
    need, ev_keys, short_pairs = {}, [], []
    for ev in memory["events"]:
        e = ev["entity"]
        pre = [k for k in ev["pre"]["instants"] if rows[e][k].get("best_box") is not None]
        post = [k for k in ev["post"]["instants"] if rows[e][k].get("best_box") is not None]
        for k in pre + post:
            need[(e, k)] = rows[e][k]["best_box"]
        ev_keys.append((pre, post))
    step = int(round(short_gap * grid_hz))
    for e, r in rows.items():
        for k in range(len(r) - step):
            seg = r[k:k + step + 1]
            if all(x.get("gt") is not None for x in seg) and r[k].get("best_box") is not None and r[k + step].get("best_box") is not None:
                need[(e, k)] = r[k]["best_box"]; need[(e, k + step)] = r[k + step]["best_box"]
                short_pairs.append((e, k, k + step))
    return rows, need, ev_keys, short_pairs


def evaluate(video: str, memory: dict, sample: GeneratedSample, loaded, short_gap: float = 1.0,
             margin: float = 0.25, min_side: int = 32) -> dict:
    grid_hz = memory["summary"]["grid_hz"]
    out_frames = memory["alignment"]["output_frames"]
    rows, need, ev_keys, short_pairs = _needed(memory, short_gap, grid_hz)
    for ev in memory["events"]:
        for k in ev["post"]["instants"]:
            for e2, r2 in rows.items():
                if e2 != ev["entity"] and r2[k].get("best_box") is not None:
                    need[(e2, k)] = r2[k]["best_box"]
    frame_ids = sorted({out_frames[k] for (_, k) in need})
    frames = {i: im[:, :, ::-1].copy() for i, im in iter_video_frames(video, set(frame_ids))}
    keys = sorted(need)
    crops = [crop_box(frames[out_frames[k]], need[(e, k)], margin, min_side) for (e, k) in keys]
    feats = embed(crops, loaded) if crops else np.zeros((0, 1))
    F = {key: feats[i] for i, key in enumerate(keys)}

    def cos(a, b):
        return float(np.dot(F[a], F[b]))

    events_out = []
    for ev, (pre, post) in zip(memory["events"], ev_keys):
        e = ev["entity"]
        pairs = [cos((e, a), (e, b)) for a in pre for b in post]
        cross = [cos((e, k), (e2, k)) for k in post for e2 in rows if e2 != e and (e2, k) in F]
        events_out.append(dict(entity=e, leave_t=ev["leave_t"], return_t=ev["return_t"], absence=ev["absence"],
                               bucket=ev["bucket"], n_pre=len(pre), n_post=len(post),
                               identity_revisit=float(np.mean(pairs)) if pairs else None,
                               identity_cross=float(np.mean(cross)) if cross else None))
    short = [cos((e, a), (e, b)) for e, a, b in short_pairs]
    scored = [x for x in events_out if x["identity_revisit"] is not None]
    by_bucket = defaultdict(list)
    for x in scored:
        by_bucket[x["bucket"]].append(x["identity_revisit"])
    summary = dict(
        identity_revisit=float(np.mean([x["identity_revisit"] for x in scored])) if scored else None,
        identity_cross=float(np.mean([x["identity_cross"] for x in scored if x["identity_cross"] is not None]))
        if any(x["identity_cross"] is not None for x in scored) else None,
        identity_short=float(np.mean(short)) if short else None,
        n_events=len(events_out), n_events_scored=len(scored), n_short_pairs=len(short_pairs),
        by_bucket={b: dict(n=len(v), identity_revisit=float(np.mean(v))) for b, v in by_bucket.items()},
        model=loaded[1].name_or_path if hasattr(loaded[1], "name_or_path") else None,
        short_gap=short_gap, margin=margin, min_side=min_side, n_crops=len(crops))
    return dict(schema="prowbench.memory_identity.v1", protocol="memory_30s_v1", sample_id=sample.sample_id,
                method=sample.method, run_id=sample.run_id, summary=summary, events=events_out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--memory", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--short-gap", type=float, default=1.0)
    a = ap.parse_args(argv)
    s = json.loads(a.sample) if a.sample.strip().startswith("{") else read_json(a.sample)
    sample = GeneratedSample.from_dict(s)
    res = evaluate(sample.video, read_json(a.memory), sample, _load(a.model, a.device), a.short_gap)
    write_json(a.out, res)
    print(f"{sample.sample_id}: identity revisit {res['summary']['identity_revisit']} "
          f"(short {res['summary']['identity_short']}, cross {res['summary']['identity_cross']}) -> {a.out}")


if __name__ == "__main__":
    main()
