"""Multi-view table aggregation: per-view rows, four-view scene means, equally weighted scenes.

Each synchronized view is scored independently with the single-view metrics. A scene value is the
mean of its four views and the table value is the mean of the scene values. A view without a value
makes its scene undefined, and an undefined scene makes the table value undefined; views are never
dropped silently.
"""
import math
import statistics

VIEWS = ("follow-a", "follow-b", "follow-c", "follow-d")

# column -> (metric file kind, summary field, factor)
CONTROL_COLUMNS = {
    "IoU": ("entity", "mean_iou", 1.0),
    "CenterErr": ("entity", "mean_center_err", 1.0),
    "MissRate": ("entity", "miss_rate", 1.0),
    "RotErr_deg": ("camera", "RotErr_mean", 180 / math.pi),
    "TransErr": ("camera", "TransErr_rel_mean", 1.0),
    "CamMC": ("camera", "CamMC_rel_mean", 1.0),
}


def control_row(entity, camera):
    """Table columns of one view from `metric_entity.evaluate` and `metric_camera.evaluate` (CamI2V) outputs."""
    if camera.get("protocol") != "cami2v_all_motion_v1":
        raise ValueError("camera scores must use the cami2v protocol")
    summaries = dict(entity=entity["summary"], camera=camera["summary"])
    row = {}
    for column, (kind, field, factor) in CONTROL_COLUMNS.items():
        value = summaries[kind][field]
        if value is not None and not math.isfinite(value):
            raise ValueError(f"non-finite {field}")
        row[column] = None if value is None else value * factor
    return row


def logic_row(interaction, alignment):
    """ISR and LRA of one view from windowed interaction and alignment outputs (all judgments valid)."""
    for result in (interaction, alignment):
        if not result.get("summary") or result["summary"].get("n_invalid", 1) > 0 or result.get("failures"):
            raise ValueError("invalid or failed judgments must be resolved before aggregation")
    return dict(ISR=interaction["summary"]["interaction_success"], LRA=alignment["summary"]["dynamic_checklist"])


def grounded_row(row):
    """Per-view products IoU x ISR and IoU x LRA (gISR, gLRA)."""
    if row.get("IoU") is None:
        return dict(gISR=None, gLRA=None)
    return dict(gISR=row["IoU"] * row["ISR"], gLRA=row["IoU"] * row["LRA"])


def scene_macro(rows, columns, views=VIEWS, n_scenes=None):
    """Mean of the views within each scene, then the equal mean of the scenes.

    `rows` carry `episode`, `view` and the columns, and must belong to one method. Every scene must contain
    exactly the given views. Returns per-scene values and the overall values.
    """
    rows = list(rows)
    if len({r.get("method") for r in rows}) > 1:
        raise ValueError("rows must belong to one method")
    by_scene = {}
    for r in rows:
        by_scene.setdefault(r["episode"], []).append(r)
    if n_scenes is not None and len(by_scene) != n_scenes:
        raise ValueError(f"expected {n_scenes} scenes, got {len(by_scene)}")
    scenes = {}
    for episode in sorted(by_scene):
        group = by_scene[episode]
        if sorted(r["view"] for r in group) != sorted(views):
            raise ValueError(f"{episode}: views must be exactly {list(views)}")
        scenes[episode] = {c: statistics.mean(r[c] for r in group) if all(r.get(c) is not None for r in group) else None
                           for c in columns}
    overall = {c: statistics.mean(s[c] for s in scenes.values()) if all(s[c] is not None for s in scenes.values()) else None
               for c in columns}
    return dict(per_scene=scenes, overall=overall)


def summarize_method(views, n_scenes=10):
    """Full multi-view row of one method.

    `views` is a list of dicts with `method`, `episode`, `view`, `entity`, `camera`, `interaction` and `alignment`
    (the metric outputs of that view). Returns the scene-macro averages of the control, logic and grounded columns.
    """
    rows = []
    for v in views:
        row = dict(method=v["method"], episode=v["episode"], view=v["view"])
        row.update(control_row(v["entity"], v["camera"]))
        row.update(logic_row(v["interaction"], v["alignment"]))
        row.update(grounded_row(row))
        rows.append(row)
    columns = list(CONTROL_COLUMNS) + ["ISR", "LRA", "gISR", "gLRA"]
    return dict(per_view=rows, **scene_macro(rows, columns, n_scenes=n_scenes))
