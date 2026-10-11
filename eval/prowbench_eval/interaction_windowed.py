"""Interaction Success Rate (ISR): judgment of every event group of a view on frames of its window, shown as the full
frame next to a zoom on the participants."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from .common import read_json, sha256_file
from .vlm_judge import JudgeResponseError, sample_frames

PROMPT = """You see {n} frames, in time order, from a {t0:.2f}-{t1:.2f} s segment of a short video (frame index and
time are given below). Each image shows the full frame on the LEFT and a zoom on the main actors on the RIGHT
(same frame). Some videos are realistic, some are low-poly or block-figure renders; treat block figures and
boxes as people and objects. The camera may move or follow the actors, so judge motion of a person or object
relative to the ground and background, not relative to the image border.

Step 1 - describe, from the frames only: who or what is present, what happens between the first and the last
frame, and the state at the end. Do not assume anything not visible.

Step 2 - compare with the EXPECTED description below and answer each question with exactly "yes", "no" or
"unknown" ("unknown" only when the frames do not show enough to decide). Cite evidence frame indices.
EXPECTED action in this segment: {action}
EXPECTED state at the end of the segment: {outcome}
EXPECTED participants: {participants}

Return only JSON:
{{"description": "...", "end_state": "...",
  "action": "yes|no|unknown", "action_evidence": [frame indices],
  "outcome": "yes|no|unknown", "outcome_evidence": [frame indices],
  "participants": "yes|no|unknown", "participants_evidence": "..."}}
Frames (index, seconds): {frames}
"""

VALID = ("yes", "no", "unknown")


def load_visual_events(path: str, episode: str, registry: dict) -> list[dict]:
    """Merge registered phase instances into the visual events of `episode`; times come from the registry.
    A phase name refers to its last instance in the registry; `name#k` refers to its k-th instance in time order."""
    table = read_json(path)
    groups = table["episodes"][episode]
    instances = [e for e in registry["instances"] if e["episode"] == episode and e["kind"] == "phase"]
    phases = {e["name"]: e for e in instances}
    seen = {}
    for e in sorted(instances, key=lambda e: (e["start_time"], e["end_time"])):
        seen[e["name"]] = seen.get(e["name"], 0) + 1
        phases[f"{e['name']}#{seen[e['name']]}"] = e
    pad = float(table.get("window_pad_seconds", 0.15))
    events = []
    for g in groups:
        missing = [p for p in g["phases"] if p not in phases]
        if missing:
            raise ValueError(f"{episode}: visual event {g['name']} references unregistered phases {missing}")
        members = [phases[p] for p in g["phases"]]
        start = min(m["start_time"] for m in members); end = max(m["end_time"] for m in members)
        events.append(dict(id=f"{episode}/{g['name']}", name=g["name"], phases=g["phases"], start=start, end=end,
                           window=(max(0.0, start - pad), end + pad), participants=g["participants"],
                           action=g["action"], outcome=g["outcome"],
                           phase_ids=[m["event_id"] for m in members]))
    return events


def zoom_box(gt: dict, entity_ids: list[str], t0: float, t1: float, margin: float = 0.35, min_frac: float = 0.25):
    """Union of the projected GT boxes of `entity_ids` over frames in [t0, t1], in source pixels, expanded."""
    W, H = gt["width"], gt["height"]
    xs, ys = [], []
    for fr in gt["frames"]:
        if not (t0 <= fr["time"] <= t1):
            continue
        for eid in entity_ids:
            b = (fr.get("boxes") or {}).get(eid)
            if b and b.get("visible") and b.get("box"):
                x0, y0, x1, y1 = b["box"]; xs += [x0, x1]; ys += [y0, y1]
    if not xs:
        return (0, 0, W, H)
    x0, x1, y0, y1 = max(0, min(xs)), min(W, max(xs)), max(0, min(ys)), min(H, max(ys))
    mw, mh = (x1 - x0) * margin + 0.02 * W, (y1 - y0) * margin + 0.02 * H
    x0, x1, y0, y1 = x0 - mw, x1 + mw, y0 - mh, y1 + mh
    if x1 - x0 < min_frac * W:
        c = (x0 + x1) / 2; x0, x1 = c - min_frac * W / 2, c + min_frac * W / 2
    if y1 - y0 < min_frac * H:
        c = (y0 + y1) / 2; y0, y1 = c - min_frac * H / 2, c + min_frac * H / 2
    return (int(max(0, x0)), int(max(0, y0)), int(min(W, x1)), int(min(H, y1)))


def composite(frame_rgb: np.ndarray, box_src, src_wh, tile_h: int = 360):
    """full frame | zoom, both `tile_h` high. `box_src` is in source pixels; frame may have another size."""
    import cv2
    h, w = frame_rgb.shape[:2]
    sx, sy = w / src_wh[0], h / src_wh[1]
    x0, y0, x1, y1 = box_src
    crop = frame_rgb[int(y0 * sy):max(int(y0 * sy) + 1, int(y1 * sy)), int(x0 * sx):max(int(x0 * sx) + 1, int(x1 * sx))]
    full = cv2.resize(frame_rgb, (int(w * tile_h / h), tile_h), interpolation=cv2.INTER_AREA)
    ch, cw = crop.shape[:2]
    zoom = cv2.resize(crop, (max(1, int(cw * tile_h / ch)), tile_h), interpolation=cv2.INTER_CUBIC)
    sep = np.full((tile_h, 6, 3), 255, dtype=np.uint8)
    return np.concatenate([full, sep, zoom], axis=1)


def judge_event(video: str, fps: float, frame_count: int, event: dict, gt: dict, judge, n_frames: int = 12,
                crop_margin: float = 0.35, tile_h: int = 360, sample_meta: dict | None = None) -> dict:
    t0, t1 = event["window"]
    t1 = min(t1, (frame_count - 1) / fps)
    raw = sample_frames(video, t0, t1, n_frames, fps, max_side=4096)
    src_wh = (gt["width"], gt["height"])
    box = zoom_box(gt, event["participants"], t0, t1, crop_margin)
    if sample_meta and sample_meta.get("crop_xywh"):
        cx, cy, cw, chh = sample_meta["crop_xywh"]
        box = (box[0] - cx, box[1] - cy, box[2] - cx, box[3] - cy); src_wh = (cw, chh)
        box = (max(0, box[0]), max(0, box[1]), min(cw, box[2]), min(chh, box[3]))
    frames = [(i, composite(im, box, src_wh, tile_h)) for i, im in raw]
    frame_times = [(int(i), round(i / fps, 3)) for i, _ in frames]
    prompt = PROMPT.format(n=len(frames), t0=t0, t1=t1, action=event["action"], outcome=event["outcome"],
                           participants=", ".join(event["participants"]), frames=json.dumps(frame_times))
    resp = judge.ask(frames, prompt)
    p = resp.get("parsed") or {}
    ans = {k: (p.get(k) if p.get(k) in VALID else "invalid") for k in ("action", "outcome", "participants")}
    shown = {i for i, _ in frame_times}
    ev = {k: [i for i in (p.get(k + "_evidence") or []) if isinstance(i, int) and i in shown] for k in ("action", "outcome")}
    return dict(event_id=event["id"], name=event["name"], phases=event["phases"], window=[t0, t1], zoom_box=list(box),
                frames=frame_times, answers=ans, evidence=ev, description=p.get("description"), end_state=p.get("end_state"),
                correct=(ans["action"] == "yes" and ans["outcome"] == "yes"), invalid=any(v == "invalid" for v in ans.values()),
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), judge_cache=str(resp.get("_cache_path", "")))


def score_sample(results: list[dict]) -> dict:
    n = len(results); valid = [r for r in results if not r["invalid"]]
    def rate(k, v="yes"):
        return sum(r["answers"][k] == v for r in valid) / n if n else None
    return dict(interaction_success=sum(r["correct"] for r in valid) / n if n else None, n_events=n,
                n_correct=sum(r["correct"] for r in valid), n_invalid=n - len(valid),
                action_rate=rate("action"), outcome_rate=rate("outcome"), participants_rate=rate("participants"),
                unknown_action_rate=rate("action", "unknown"), unknown_outcome_rate=rate("outcome", "unknown"))


def evaluate_sample(sample: dict, events: list[dict], gt: dict, judge, n_frames: int = 12, crop_margin: float = 0.35) -> dict:
    per_event, failures = [], []
    for ev in events:
        try:
            per_event.append(judge_event(sample["video"], sample["fps"], sample["frame_count"], ev, gt, judge, n_frames, crop_margin,
                                         sample_meta=sample))
        except JudgeResponseError as exc:      # an invalid answer; any other error fails the sample
            failures.append(dict(event_id=ev["id"], error=repr(exc)))
            per_event.append(dict(event_id=ev["id"], name=ev["name"], phases=ev["phases"], window=list(ev["window"]), frames=[],
                                  answers=dict(action="invalid", outcome="invalid", participants="invalid"), evidence={},
                                  description=None, end_state=None, correct=False, invalid=True, error=repr(exc)))
    return dict(schema="prowbench.interaction.windowed.v2", sample_id=sample["sample_id"], method=sample["method"], run_id=sample["run_id"],
                summary=score_sample(per_event), events=per_event, failures=failures,
                validation_status="uncalibrated_judge; calibrate on white source renders", video_sha256=sha256_file(sample["video"]))
