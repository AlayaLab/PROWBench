"""Re-State: VLM judgment of whether a returning person keeps the recorded walking or standing state."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from .common import read_json, write_json
from .interaction_windowed import composite
from .vlm_judge import Judge, sample_frames

PROMPT = """You see {n} frames, in time order, from a {t0:.2f}-{t1:.2f} s segment of a video (frame index and time are
given below). Each image shows the full frame on the LEFT and a zoom on ONE person of interest on the RIGHT (the
same frame; the zoom box follows where that person should be). Some videos are realistic, some are low-poly or
block-figure renders; treat block figures as people. The camera moves, so judge the person's motion relative to
the ground and the background, not relative to the image border.

Step 1 - describe the person in the zoom: clothing / colours, posture, and whether their position relative to
the ground and surrounding fixed objects changes across the frames.

Step 2 - answer with exactly one of "walking" (the person moves along the ground between the first and last
frame), "standing" (the person stays in place; small sway or turning counts as standing), or "unknown" (no
person visible in the zoom, or the frames do not show enough to decide).

Return only JSON:
{{"description": "...", "motion": "walking|standing|unknown", "evidence": [frame indices]}}
Frames (index, seconds): {frames}
"""

VALID = ("walking", "standing", "unknown")


def entity_speed(episode_dir: str, entity: str, fps: int) -> np.ndarray:
    """Per-frame speed (m/s) from the engine state; falls back to finite differences of the position."""
    ep = Path(episode_dir)
    man = read_json(ep / "manifest.json")
    init = man["initial"]["entities"].get(entity, {})
    speeds, pos = [], []
    with gzip.open(ep / man["states"], "rt") as f:
        for line in f:
            fr = json.loads(line)
            st = fr["entities"].get(entity, init)
            sp = (st.get("state") or {}).get("speed")
            speeds.append(float(sp) if sp is not None else np.nan)
            pos.append(st.get("position", [np.nan] * 3))
    speeds = np.asarray(speeds, dtype=float)
    if np.isnan(speeds).any():
        p = np.asarray(pos, dtype=float)
        fd = np.linalg.norm(np.diff(p, axis=0), axis=1) * fps
        fd = np.concatenate([fd[:1], fd])
        speeds = np.where(np.isnan(speeds), fd, speeds)
    return speeds


def gt_state(speeds: np.ndarray, t0: float, t1: float, fps: int, speed_thresh: float) -> tuple[str, float]:
    a, b = int(round(t0 * fps)), int(round(t1 * fps))
    seg = speeds[max(0, a):max(a + 1, min(len(speeds), b + 1))]
    v = float(np.nanmean(seg)) if len(seg) else float("nan")
    return ("walking" if v > speed_thresh else "standing"), v


def detection_zoom_box(boxes_norm: list, W: int, H: int, margin: float = 0.35, min_frac: float = 0.25) -> tuple:
    """Union of matched detection boxes (normalised generated-frame coords) -> expanded pixel box in the generated frame."""
    bs = [b for b in boxes_norm if b]
    x0, y0 = min(b[0] for b in bs) * W, min(b[1] for b in bs) * H
    x1, y1 = max(b[2] for b in bs) * W, max(b[3] for b in bs) * H
    mw, mh = (x1 - x0) * margin + 0.02 * W, (y1 - y0) * margin + 0.02 * H
    x0, x1, y0, y1 = x0 - mw, x1 + mw, y0 - mh, y1 + mh
    if x1 - x0 < min_frac * W:
        c = (x0 + x1) / 2; x0, x1 = c - min_frac * W / 2, c + min_frac * W / 2
    if y1 - y0 < min_frac * H:
        c = (y0 + y1) / 2; y0, y1 = c - min_frac * H / 2, c + min_frac * H / 2
    return (int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1)))


def judge_event(sample: dict, gt: dict, ev: dict, judge: Judge, n_frames: int, window: float, crop_margin: float,
                tile_h: int = 360) -> dict:
    t0 = ev["return_t"]
    t1 = min(t0 + window, (sample["frame_count"] - 1) / sample["fps"])
    raw = sample_frames(sample["video"], t0, t1, n_frames, sample["fps"], max_side=4096)
    W, H = sample["width"], sample["height"]
    box = detection_zoom_box(ev["post"]["near_boxes"], W, H, crop_margin)
    frames = [(i, composite(im, box, (W, H), tile_h)) for i, im in raw]
    frame_times = [(int(i), round(i / sample["fps"], 3)) for i, _ in frames]
    prompt = PROMPT.format(n=len(frames), t0=t0, t1=t1, frames=json.dumps(frame_times))
    resp = judge.ask(frames, prompt)
    p = resp.get("parsed") or {}
    ans = p.get("motion") if p.get("motion") in VALID else "invalid"
    return dict(window=[t0, t1], zoom_box=list(box), frames=frame_times, answer=ans, description=p.get("description"),
                evidence=p.get("evidence"), prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                judge_cache=str(resp.get("_cache_path", "")))


def evaluate(sample: dict, gt: dict, memory: dict, judge: Judge, n_frames: int = 6, window: float = 1.5,
             crop_margin: float = 0.35, speed_thresh: float = 0.2, pre_window: float = 1.0) -> dict:
    fps = int(gt["fps"])
    episode_dir = gt["source"]["episode_dir"]
    speeds = {}
    events, excluded = [], set()
    for ev in memory["events"]:
        e = ev["entity"]
        if (gt["entities"].get(e) or {}).get("label") != "person":     # the question is about a person's motion
            excluded.add(e)
            continue
        if e not in speeds:
            speeds[e] = entity_speed(episode_dir, e, fps)
        post_state, post_v = gt_state(speeds[e], ev["return_t"], ev["return_t"] + window, fps, speed_thresh)
        pre_state, pre_v = gt_state(speeds[e], ev["leave_t"] - pre_window, ev["leave_t"], fps, speed_thresh)
        base = dict(entity=e, leave_t=ev["leave_t"], return_t=ev["return_t"], absence=ev["absence"], bucket=ev["bucket"],
                    weak=ev.get("weak", []), gt_pre=pre_state, gt_pre_speed=pre_v, gt_post=post_state, gt_post_speed=post_v,
                    kind="persist" if pre_state == post_state else "evolve", n_near=ev["post"].get("n_near", 0))
        if not any(ev["post"].get("near_boxes") or []):
            events.append(dict(**base, judgeable=False, answer="not_judgeable", correct=False, invalid=False)); continue
        j = judge_event(sample, gt, ev, judge, n_frames, window, crop_margin)
        events.append(dict(**base, judgeable=True, correct=(j["answer"] == post_state), invalid=(j["answer"] == "invalid"), **j))
    judgeable = [x for x in events if x["judgeable"]]
    valid = [x for x in judgeable if not x["invalid"]]
    persist = [x for x in valid if x["kind"] == "persist"]
    evolve = [x for x in valid if x["kind"] == "evolve"]
    known = [x for x in valid if x["answer"] != "unknown"]
    strong = [x for x in valid if not x["weak"]]
    summary = dict(
        n_events=len(events), excluded_entities=sorted(excluded), n_judgeable=len(judgeable), n_not_judgeable=len(events) - len(judgeable),
        n_invalid=len(judgeable) - len(valid), n_unknown=sum(x["answer"] == "unknown" for x in valid),
        state_acc_strong=float(np.mean([x["correct"] for x in strong])) if strong else None, n_strong=len(strong),
        state_acc=float(np.mean([x["correct"] for x in valid])) if valid else None,
        state_acc_known=float(np.mean([x["correct"] for x in known])) if known else None,
        state_persist=float(np.mean([x["correct"] for x in persist])) if persist else None, n_persist=len(persist),
        state_evolve=float(np.mean([x["correct"] for x in evolve])) if evolve else None, n_evolve=len(evolve),
        gt_post_states={s: sum(x["gt_post"] == s for x in events) for s in ("walking", "standing")},
        answers={s: sum(x["answer"] == s for x in events) for s in VALID + ("invalid", "not_judgeable")},
        gate="lenient match (centre within one reference-box side) at >= 1 post instant; zoom on matched detections",
        by_bucket={b: dict(n=len(v), state_acc=float(np.mean([x["correct"] for x in v])))
                   for b in {x["bucket"] for x in valid} for v in [[x for x in valid if x["bucket"] == b]]},
        judge=judge.model_name, n_frames=n_frames, window=window, crop_margin=crop_margin, speed_thresh=speed_thresh)
    return dict(schema="prowbench.memory_state.v2", protocol="memory_30s_v1", sample_id=sample["sample_id"],
                method=sample["method"], run_id=sample["run_id"], summary=summary, events=events)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--memory", required=True)
    ap.add_argument("--sample", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cache-dir", default=".judge_cache")
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--window", type=float, default=1.5)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    a = ap.parse_args(argv)
    s = json.loads(a.sample) if a.sample.strip().startswith("{") else read_json(a.sample)
    judge = Judge(backend="hf", model=a.model, device=a.device, cache_dir=a.cache_dir, max_new_tokens=a.max_new_tokens,
                  enable_thinking=False)
    res = evaluate(s, read_json(a.gt), read_json(a.memory), judge, a.frames, a.window)
    write_json(a.out, res)
    print(f"{s['sample_id']}: state acc {res['summary']['state_acc']} on {res['summary']['n_events']} events -> {a.out}")


if __name__ == "__main__":
    main()
