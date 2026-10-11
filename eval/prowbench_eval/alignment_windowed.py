"""Logic-Render Alignment (LRA): timeline-segment, end-state, static-fact and negative-control (foil) judgments of a
generated video against the description record of its view."""
from __future__ import annotations

import hashlib
import json
import re


from .common import EVAL_DURATION, SOURCE_FPS, evaluated_entities, sha256_file
from .interaction_windowed import composite, zoom_box
from .vlm_judge import JudgeResponseError, sample_frames

VALID = ("yes", "no", "unknown")

DYNAMIC_PROMPT = """You see {n} frames, in time order, from the {t0:.2f}-{t1:.2f} s window of a short video (frame index and
time are given below). Each image shows the full frame on the LEFT and a zoom on the moving actors and objects on the
RIGHT (same frame). Some videos are realistic, some are low-poly or block-figure renders; treat block figures and
boxes as people and objects. The camera may move, orbit or follow the actors, so a figure that stays near the image
centre may still be moving: judge motion of a person or object relative to the ground, the props (platform, chair,
crate, cable) and the background, not relative to the image border. The window is slightly wider than the segment,
so the previous or next segment may show at its edges.

Step 1 - describe, from the frames only: who or what is present, what happens between the first and the last frame,
and the state at the end. Do not assume anything not visible.

Step 2 - the video was generated from a written timeline.
Segment before this window: "{before}"
EXPECTED in this window: "{text}"
Segment after this window: "{after}"
Answer two questions with exactly "yes", "no" or "unknown" ("unknown" only when the frames do not show enough to
decide), citing evidence frame indices:
  action:  does the movement or pose change described in EXPECTED happen in this window (the same action, in the
           same direction, by the same kind of actor), even if fine details cannot be verified on a block figure?
  details: are all the specifics stated in EXPECTED visibly satisfied (e.g. feet near the ground, clears the edge,
           hands near the thighs)?

Return only JSON:
{{"description": "...", "action": "yes|no|unknown", "action_evidence": [frame indices],
  "details": "yes|no|unknown", "details_evidence": [frame indices], "reason": "..."}}
Frames (index, seconds): {frames}
"""

END_STATE_PROMPT = """You see {n} frames, in time order, from the last {dur:.2f} s of a short video (frame index and time are given
below). Each image shows the full frame on the LEFT and a zoom on the moving actors and objects on the RIGHT (same
frame). Some videos are realistic, some are low-poly or block-figure renders; treat block figures and boxes as people
and objects.

The video was generated from a written description that specifies how the clip must END. Check each expected
end-state fact against these frames and answer, in order, exactly "yes", "no" or "unknown" ("unknown" only when the
frames do not show enough to decide). Facts:
{facts}

Return only JSON:
{{"description": "...", "answers": ["yes|no|unknown", ...], "reason": "..."}}
Frames (index, seconds): {frames}
"""

STATIC_PROMPT = """You see {n} frames sampled uniformly across a whole short video, in time order (frame index and time are given
below). Some videos are realistic, some are low-poly or block-figure renders. Judge what is ACTUALLY shown: a white
block figure is not a person in an olive shirt, a plain grey box is not a wooden chair.

The video was generated from a written scene description. Check each fact below against the frames and answer, in
order, exactly "yes" (the fact is visibly true), "no" (visibly false or absent) or "unknown" (cannot be decided).
Facts:
{facts}

Return only JSON, one answer and one short reason (at most 15 words) per fact, in order:
{{"answers": ["yes|no|unknown", ...], "reasons": ["...", ...]}}
Frames (index, seconds): {frames}
"""

INSTRUCTION_PREFIXES = ("preserve", "keep", "maintain", "follow", "reveal", "introduce", "realize", "realise", "retain", "leave")
STATIC_SECTIONS = ("World", "Protagonist", "Companion", "Companions", "Objects", "Object")
_PARA = re.compile(r"^\s*([A-Z][A-Za-z ]{2,30}):\s*(.*\S)\s*$")


def paragraphs(desc_json: dict) -> dict:
    out = {}
    for line in desc_json.get("description", "").split("\n"):
        m = _PARA.match(line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]


def load_timeline(desc_json: dict, duration: float = EVAL_DURATION, min_seconds: float = 0.2) -> list[dict]:
    fps = float(desc_json.get("fps") or SOURCE_FPS)
    segs = []
    for s in desc_json.get("timeline", []):
        t0 = float(s["startSeconds"] if "startSeconds" in s else s["startFrame"] / fps)
        t1 = float(s["endSeconds"] if "endSeconds" in s else s["endFrameExclusive"] / fps)
        text = (s.get("description") or s.get("text") or "").strip()
        t1 = min(t1, duration)
        if not text or t1 - t0 < min_seconds:
            continue
        segs.append(dict(phase=s.get("phase"), text=text, t0=t0, t1=t1))
    if not segs:
        raise ValueError("description json has no usable timeline segment")
    return segs


def scene_facts(desc_json: dict, max_facts: int = 8, sections=STATIC_SECTIONS) -> list[str]:
    """Factual sentences of the static paragraphs; instructions about the reference are not facts."""
    paras = paragraphs(desc_json)
    facts = []
    for name in sections:
        for sent in split_sentences(paras.get(name, "")):
            if 20 < len(sent) < 220 and not sent.lower().startswith(INSTRUCTION_PREFIXES):
                facts.append(sent)
    if not facts:
        raise ValueError("description json has no static scene fact")
    return facts[:max_facts]


def end_state_facts(desc_json: dict, segs: list[dict]) -> list[str]:
    """Sentences of the 'Action timeline' paragraph that are not timed segments = required end state."""
    para = paragraphs(desc_json).get("Action timeline", "")
    facts = [s for s in split_sentences(para) if not s.startswith("From ") and not s.lower().startswith(INSTRUCTION_PREFIXES)]
    return facts or [segs[-1]["text"]]


def foil_episode(episodes: list[str], episode: str, offset: int | None = None) -> str:
    """Deterministic negative control: the episode `offset` places later in sorted order (cyclic).
    Default offset = half the list, i.e. the episode farthest away in sorted order; offset 1 (the neighbour) paired
    jump-landing with moving-platform, two 'get onto a platform' stories that share half their timeline."""
    eps = sorted(episodes)
    if episode not in eps or len(eps) < 2:
        raise ValueError(f"cannot pick a foil for {episode!r} among {eps}")
    offset = len(eps) // 2 if offset is None else int(offset)
    if offset % len(eps) == 0:
        raise ValueError("foil offset must not map an episode onto itself")
    return eps[(eps.index(episode) + offset) % len(eps)]


def _answer(value):
    return value if value in VALID else "invalid"


def _answers(parsed: dict, n: int) -> list[str]:
    ans = parsed.get("answers")
    if not isinstance(ans, list) or len(ans) != n:
        return ["invalid"] * n
    return [_answer(a) for a in ans]


def _frames_with_zoom(video, fps, frame_count, t0, t1, n, gt, sample, crop_margin, tile_h=360):
    t1 = min(t1, (frame_count - 1) / fps)
    raw = sample_frames(video, t0, t1, n, fps, max_side=4096)
    src_wh = (gt["width"], gt["height"])
    box = zoom_box(gt, evaluated_entities(gt), t0, t1, crop_margin)
    if sample.get("crop_xywh"):
        cx, cy, cw, chh = sample["crop_xywh"]
        box = (max(0, box[0] - cx), max(0, box[1] - cy), min(cw, box[2] - cx), min(chh, box[3] - cy)); src_wh = (cw, chh)
    frames = [(i, composite(im, box, src_wh, tile_h)) for i, im in raw]
    return frames, [(int(i), round(i / fps, 3)) for i, _ in frames], list(box), t1


def judge_segment(sample, seg, gt, judge, n_frames, pad, crop_margin, before: str = "", after: str = "") -> dict:
    t0 = max(0.0, seg["t0"] - pad); t1 = seg["t1"] + pad
    frames, times, box, t1 = _frames_with_zoom(sample["video"], sample["fps"], sample["frame_count"], t0, t1, n_frames, gt, sample, crop_margin)
    prompt = DYNAMIC_PROMPT.format(n=len(frames), t0=t0, t1=t1, text=seg["text"], before=before or "(clip start)", after=after or "(clip end)",
                                   frames=json.dumps(times))
    p = judge.ask(frames, prompt).get("parsed") or {}
    shown = {i for i, _ in times}
    def ev(k):
        return [i for i in (p.get(k) or []) if isinstance(i, int) and i in shown]
    return dict(phase=seg["phase"], text=seg["text"], window=[t0, t1], zoom_box=box, frames=times,
                depicted=_answer(p.get("action")), details=_answer(p.get("details")),
                evidence=ev("action_evidence"), details_evidence=ev("details_evidence"),
                description=p.get("description"), reason=p.get("reason"), prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())


def judge_end_state(sample, facts, gt, judge, n_frames, seconds, crop_margin) -> dict:
    dur = min(sample["frame_count"] / sample["fps"], EVAL_DURATION)
    t0 = max(0.0, dur - seconds)
    frames, times, box, t1 = _frames_with_zoom(sample["video"], sample["fps"], sample["frame_count"], t0, dur - 1e-3, n_frames, gt, sample, crop_margin)
    prompt = END_STATE_PROMPT.format(n=len(frames), dur=t1 - t0, facts="\n".join(f"{i + 1}. {f}" for i, f in enumerate(facts)), frames=json.dumps(times))
    p = judge.ask(frames, prompt).get("parsed") or {}
    return dict(window=[t0, t1], zoom_box=box, frames=times, facts=facts, answers=_answers(p, len(facts)),
                description=p.get("description"), reason=p.get("reason"), prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())


def judge_static(sample, facts, judge, n_frames) -> dict:
    dur = min(sample["frame_count"] / sample["fps"], EVAL_DURATION)
    frames = sample_frames(sample["video"], 0.0, dur - 1e-3, n_frames, sample["fps"], max_side=768)
    times = [(int(i), round(i / sample["fps"], 3)) for i, _ in frames]
    prompt = STATIC_PROMPT.format(n=len(frames), facts="\n".join(f"{i + 1}. {f}" for i, f in enumerate(facts)), frames=json.dumps(times))
    p = judge.ask(frames, prompt).get("parsed") or {}
    reasons = p.get("reasons") if isinstance(p.get("reasons"), list) else None
    return dict(frames=times, facts=facts, answers=_answers(p, len(facts)), reasons=reasons,
                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())


def _rates(answers: list[str]) -> dict:
    n = len(answers)
    if n == 0:
        return dict(score=None, n=0, n_yes=0, n_unknown=0, n_invalid=0)
    return dict(score=sum(a == "yes" for a in answers) / n, n=n, n_yes=sum(a == "yes" for a in answers),
                n_unknown=sum(a == "unknown" for a in answers), n_invalid=sum(a == "invalid" for a in answers))


def score_sample(segments, end_state, static, foil_segments=None, foil_static=None, judge_name=None) -> dict:
    d = _rates([s["depicted"] for s in segments]); e = _rates(end_state["answers"]); st = _rates(static["answers"])
    strict = _rates([(s.get("details") or "invalid") if s["depicted"] == "yes" else s["depicted"] for s in segments])
    out = dict(dynamic_checklist=d["score"], dynamic_strict=strict["score"], n_segments=d["n"],
               dynamic_unknown_rate=d["n_unknown"] / d["n"] if d["n"] else None,
               end_state=e["score"], n_end_facts=e["n"],
               static_facts=st["score"], n_static_facts=st["n"], static_unknown_rate=st["n_unknown"] / st["n"] if st["n"] else None,
               n_invalid=d["n_invalid"] + e["n_invalid"] + st["n_invalid"], judge=judge_name)
    if foil_segments is not None:
        fd = _rates([s["depicted"] for s in foil_segments])
        out.update(dynamic_foil=fd["score"], dynamic_margin=(d["score"] - fd["score"]) if None not in (d["score"], fd["score"]) else None)
        out["n_invalid"] += fd["n_invalid"]
    if foil_static is not None:
        fs = _rates(foil_static["answers"])
        out.update(static_foil=fs["score"], static_margin=(st["score"] - fs["score"]) if None not in (st["score"], fs["score"]) else None)
        out["n_invalid"] += fs["n_invalid"]
    return out


def _invalid_segment(seg, exc):
    return dict(phase=seg["phase"], text=seg["text"], window=[seg["t0"], seg["t1"]], zoom_box=None, frames=[], depicted="invalid",
                details="invalid", evidence=[], details_evidence=[], description=None, reason=None, error=repr(exc))


def evaluate_sample(sample: dict, desc: dict, gt: dict, judge, cfg: dict, foil_desc: dict | None = None, foil_name: str | None = None) -> dict:
    """One sample -> per-question records + summary. A judge response that is not valid JSON marks that question invalid
    (it stays in the denominator); any other error fails the sample."""
    n_frames = int(cfg.get("frames", 8)); pad = float(cfg.get("window_pad", 0.15)); margin = float(cfg.get("crop_margin", 0.35))
    segs = load_timeline(desc); facts = scene_facts(desc, int(cfg.get("max_static_facts", 8))); ends = end_state_facts(desc, segs)
    failures = []

    def run_segments(seg_list):
        rows = []
        for k, seg in enumerate(seg_list):
            before = seg_list[k - 1]["text"] if k > 0 else ""; after = seg_list[k + 1]["text"] if k + 1 < len(seg_list) else ""
            try:
                rows.append(judge_segment(sample, seg, gt, judge, n_frames, pad, margin, before, after))
            except JudgeResponseError as exc:
                failures.append(dict(question=f"segment:{seg['phase']}", error=repr(exc))); rows.append(_invalid_segment(seg, exc))
        return rows

    segments = run_segments(segs)
    try:
        end_state = judge_end_state(sample, ends, gt, judge, int(cfg.get("end_state_frames", 4)), float(cfg.get("end_state_seconds", 0.5)), margin)
    except JudgeResponseError as exc:
        failures.append(dict(question="end_state", error=repr(exc)))
        end_state = dict(facts=ends, answers=["invalid"] * len(ends), error=repr(exc))
    try:
        static = judge_static(sample, facts, judge, int(cfg.get("static_frames", 12)))
    except JudgeResponseError as exc:
        failures.append(dict(question="static", error=repr(exc)))
        static = dict(facts=facts, answers=["invalid"] * len(facts), error=repr(exc))
    foil_segments = foil_static = None
    if foil_desc is not None:
        foil_segments = run_segments(load_timeline(foil_desc))
        ffacts = scene_facts(foil_desc, int(cfg.get("max_static_facts", 8)))
        try:
            foil_static = judge_static(sample, ffacts, judge, int(cfg.get("static_frames", 12)))
        except JudgeResponseError as exc:
            failures.append(dict(question="foil_static", error=repr(exc)))
            foil_static = dict(facts=ffacts, answers=["invalid"] * len(ffacts), error=repr(exc))
    summary = score_sample(segments, end_state, static, foil_segments, foil_static, getattr(judge, "model_name", None))
    return dict(schema="prowbench.alignment.windowed.v2", protocol="alignment_v2", sample_id=sample["sample_id"], method=sample["method"],
                run_id=sample["run_id"], summary=summary, segments=segments, end_state=end_state, static=static,
                foil=dict(episode=foil_name, segments=foil_segments, static=foil_static) if foil_desc is not None else None,
                failures=failures, description_sha256=hashlib.sha256(json.dumps(desc, sort_keys=True).encode()).hexdigest(),
                validation_status="uncalibrated_judge; calibrate on white source renders", video_sha256=sha256_file(sample["video"]))
