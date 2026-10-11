"""Table-level aggregation for the long-horizon (30-second) setting.

Inputs are the per-recording outputs of the released modules for one method (one generated video per recording):
``metric_entity`` (``summary.mean_iou``, ``summary.mean_center_err``), ``metric_memory`` (``summary.reappear_iou``),
``memory_state`` (``summary.state_persist``, ``summary.n_persist``), ``metric_camera`` with the ``cami2v`` protocol
(``summary.RotErr_mean`` in radians, ``summary.TransErr_rel_mean``, ``summary.CamMC_rel_mean``) and per-recording
ISR/LRA scores (``long_horizon_logic`` when adjudicated, otherwise the windowed judge summaries).

Rules:
- IoU, CErr, Rot, Trans and CamMC: equal mean over recordings.
- Re-IoU and Re-State: equal mean over the recordings where the value is defined (a recording without a returning
  entity, or without a judgeable persistence event, has no value and is skipped).
- State events: the number of persistence events entering Re-State, summed over recordings.
- Rot is reported in degrees: the mean of the per-recording radian errors converted with ``math.degrees``.
- gISR / gLRA: per-recording IoU times ISR / LRA, then the equal mean over recordings (``joint_control.score_samples``).
- Window curve: for every window ``[k*w, (k+1)*w)``, the mean IoU over all (recording, entity, instant) triples with a
  reference box in that window, pooled over recordings; a missed detection counts as IoU 0.
"""
from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean

from .joint_control import score_samples
from .report_tables import CENTER_ERROR_MISS


def mean_defined(values):
    """Mean of the values that are not None; None if no value is defined."""
    vals = [v for v in values if v is not None]
    return mean(vals) if vals else None


def scene_scores(rows):
    """Per-recording mean of ``correct`` for rows with ``episode`` and ``correct`` (one method, one metric)."""
    rows = list(rows)
    if len({(r.get("method"), r.get("kind")) for r in rows}) > 1:
        raise ValueError("Rows must belong to one method and one metric")
    scenes = defaultdict(list)
    for r in rows:
        scenes[r["episode"]].append(bool(r["correct"]))
    return {ep: mean(v) for ep, v in scenes.items()}


def method_row(recordings):
    """Aggregate one method. ``recordings`` is a list of dicts, one per recording, with keys
    ``entity``, ``memory``, ``state`` and ``camera`` (the metric outputs; ``state`` and ``camera`` may be None) and
    optional ``isr`` / ``lra`` (per-recording scores)."""
    recs = list(recordings)
    if not recs:
        raise ValueError("No recordings")
    for key in ("entity", "memory"):
        if any(r.get(key) is None for r in recs):
            raise ValueError(f"{key} results missing for {sum(r.get(key) is None for r in recs)} recordings")
    if 0 < sum(r.get("state") is None for r in recs) < len(recs):
        raise ValueError("state results missing for some recordings; run memory_state for all of them")
    ent = [r["entity"]["summary"] for r in recs]
    mem = [r["memory"]["summary"] for r in recs]
    sta = [(r.get("state") or {}).get("summary") or {} for r in recs]
    # as in report_tables: a recording with evaluated entities that are never matched has CErr 1.0; one without
    # evaluated entities (mean_iou None) has no entity scores
    cerr = [CENTER_ERROR_MISS if e["mean_center_err"] is None and e["mean_iou"] is not None else e["mean_center_err"]
            for e in ent]
    cam = [(r.get("camera") or {}).get("summary") for r in recs]
    out = dict(
        n_recordings=len(recs),
        iou=mean_defined(e["mean_iou"] for e in ent),
        cerr=mean_defined(cerr),
        reappear_iou=mean_defined(m["reappear_iou"] for m in mem),
        n_reappear_recordings=sum(m["reappear_iou"] is not None for m in mem),
        state_persist=mean_defined(s.get("state_persist") for s in sta),
        n_state_recordings=sum(s.get("state_persist") is not None for s in sta),
        n_state_events=sum(int(s.get("n_persist") or 0) for s in sta),
    )
    if all(c is not None for c in cam):
        out.update(rot_deg=math.degrees(mean(c["RotErr_mean"] for c in cam)),
                   trans=mean(c["TransErr_rel_mean"] for c in cam),
                   cammc=mean(c["CamMC_rel_mean"] for c in cam))
    if all(r.get("isr") is not None and r.get("lra") is not None for r in recs):
        out.update(isr=mean(r["isr"] for r in recs), lra=mean(r["lra"] for r in recs))
        g = score_samples(dict(iou=e["mean_iou"], interaction=r["isr"], align=r["lra"]) for e, r in zip(ent, recs))
        out.update(gisr=g["gisr"], glra=g["glra"])
    return out


def window_iou_curve(entity_results, window: float = 5.0, duration: float = 30.0):
    """Pooled mean IoU per time window over a list of ``metric_entity`` outputs (one method).

    Every instant of every evaluated entity with a reference box contributes once; an instant without a matched
    detection contributes IoU 0. Returns a list with one value (or None for an empty window) per window."""
    n = int(math.ceil(duration / window - 1e-9))
    pooled = [[] for _ in range(n)]
    for res in entity_results:
        for ent in res["per_entity"].values():
            for x in ent["instants"]:
                if x.get("gt") is None:
                    continue
                k = min(int(x["t"] // window), n - 1)
                pooled[k].append(float(x.get("iou") or 0.0))
    return [mean(v) if v else None for v in pooled]


def count_below(values, threshold: float):
    """Number of values strictly below ``threshold`` (e.g. per-recording IoU below 0.3)."""
    return sum(v < threshold for v in values)
