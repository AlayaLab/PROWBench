"""Spatially grounded interaction and alignment scores (gISR / gLRA)."""


def _grounded(rows, key):
    return [r["iou"] * r[key] for r in rows if r.get("iou") is not None and r.get(key) is not None]


def score_samples(rows):
    """Multiply each sample's IoU by its ISR (`interaction`) / LRA (`align`), then average the samples equally.
    Samples where either factor is missing (None, e.g. a view without evaluated entities) are left out of that mean;
    the value is None if no sample has both factors."""
    rows = list(rows)
    out = {}
    for name, key in (("gisr", "interaction"), ("glra", "align")):
        vals = _grounded(rows, key)
        out[name] = sum(vals) / len(vals) if vals else None
    return out
