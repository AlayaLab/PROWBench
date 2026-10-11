import math

from prowbench_eval import report_tables as rt


def _summaries(iou=0.5, center=0.1, rot=math.pi / 180, isr=0.5, lra=0.75, n_entities=2):
    return dict(entity=dict(mean_iou=iou, mean_center_err=center, n_entities=n_entities),
                camera=dict(RotErr_mean=rot, TransErr_rel_mean=0.2, CamMC_rel_mean=0.3),
                temporal=dict(warp_error_x1000=0.4), clipscore=dict(clip_score=25.0),
                vbench={"vbench.imaging_quality": 70.0, "vbench.aesthetic_quality": 0.6},
                vbench_motion={"vbench.motion_smoothness": 0.99},
                interaction=dict(interaction_success=isr, n_invalid=0), alignment=dict(dynamic_checklist=lra, dynamic_foil=0.25))


def test_sample_row_columns_and_units():
    row = rt.sample_row(**_summaries())
    assert abs(row["rot"] - 1.0) < 1e-12
    assert (row["iou"], row["cerr"], row["trans"], row["cammc"]) == (0.5, 0.1, 0.2, 0.3)
    assert (row["imaging"], row["aesthetic"], row["msmooth"], row["warp"], row["clip"]) == (70.0, 0.6, 0.99, 0.4, 25.0)
    assert (row["interaction"], row["align"], row["center_error_penalty"]) == (0.5, 0.75, False)


def test_center_error_penalty_only_for_unmatched_targets():
    unmatched = rt.sample_row(**_summaries(iou=0.0, center=None))
    assert unmatched["cerr"] == rt.CENTER_ERROR_MISS and unmatched["center_error_penalty"]
    empty = rt.sample_row(**_summaries(iou=None, center=None, n_entities=0))
    assert empty["cerr"] is None and not empty["center_error_penalty"]


def test_missing_inputs_give_none():
    row = rt.sample_row({}, {}, {}, {}, {}, {}, {}, {})
    assert all(row[c] is None for c in rt.SAMPLE_COLUMNS)


def test_mean_skips_none_and_nan():
    assert rt.mean([1.0, None, float("nan"), 3.0]) == (2.0, 2)
    assert rt.mean([None]) == (None, 0)


def test_grounded_scores_are_per_sample_products():
    rows = [dict(iou=0.5, interaction=1.0, align=0.5), dict(iou=1.0, interaction=0.0, align=1.0), dict(iou=None, interaction=1.0, align=1.0)]
    assert rt.cell(rows, "gisr") == (0.25, 2)
    assert rt.cell(rows, "glra") == ((0.25 + 1.0) / 2, 2)
    assert rt.cell(rows, "iou") == (0.75, 2)


def test_table_excludes_pairs_without_entities_from_entity_columns():
    def r(ep, iou, n):
        return dict(episode=ep, camera="follow", n_entities=n, iou=iou, cerr=None if iou is None else 0.1, interaction=1.0, align=1.0)
    rows = {"a": [r("e1", 0.4, 2), r("e2", None, 0)], "b": [r("e1", 0.6, 2), r("e2", None, 0)]}
    tab = rt.table(rows, columns=("iou", "cerr", "interaction", "gisr"))
    assert tab["a"]["iou"] == dict(mean=0.4, n=1, expected=1)
    assert tab["a"]["gisr"]["expected"] == 1 and tab["a"]["interaction"] == dict(mean=1.0, n=2, expected=2)
    rows["b"][1] = r("e2", 0.0, 3)
    assert rt.table(rows, columns=("iou",))["a"]["iou"]["expected"] == 2


def _gt(boxes):
    return dict(width=100, height=100, entities={k: {} for k in boxes},
                frames=[dict(time=t, boxes={k: dict(box=b, visible=True) for k, b in boxes.items()}) for t in range(10)])


def test_joint_control_skips_missing_factors():
    from prowbench_eval.joint_control import score_samples
    rows = [dict(iou=0.5, interaction=1.0, align=0.5), dict(iou=None, interaction=1.0, align=1.0), dict(iou=1.0, interaction=None, align=1.0)]
    assert score_samples(rows) == dict(gisr=0.5, glra=(0.25 + 1.0) / 2)
    assert score_samples([dict(iou=None, interaction=1.0, align=1.0)]) == dict(gisr=None, glra=None)
