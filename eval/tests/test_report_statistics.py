"""Synthetic checks for the derived method-comparison statistics."""
import math

from prowbench_eval import report_statistics as rs


def _scores(table):
    """{scene: {method: {metric: value}}} -> flat "<scene>|<method>" mapping."""
    return {f"{s}|{m}": v for s, row in table.items() for m, v in row.items()}


def test_scene_scores_and_duplicates():
    rows = [dict(ep="a", m="x", iou=0.5), dict(ep="a", m="y", iou=None)]
    out = rs.scene_scores(rows, ["iou"], lambda r: r["ep"], lambda r: r["m"])
    assert out == {"a|x": {"iou": 0.5}, "a|y": {"iou": None}}
    try:
        rs.scene_scores(rows + rows[:1], ["iou"], lambda r: r["ep"], lambda r: r["m"])
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate rows must be rejected")


def test_class_mean_requires_all_methods():
    row = {"a": {"k": 1.0}, "b": {"k": None}, "c": {"k": 3.0}}
    assert rs.class_mean(row, ["a", "c"], "k") == 2.0
    assert rs.class_mean(row, ["a", "b"], "k") is None


def test_paired_comparison_ties_direction_and_p():
    table = {f"s{i}": {"a": {"k": 1.0}, "b": {"k": 0.0}} for i in range(6)}
    table["t"] = {"a": {"k": 0.5}, "b": {"k": 0.5}}          # tie
    table["u"] = {"a": {"k": 0.0}, "b": {"k": 1.0}}          # loss
    table["v"] = {"a": {"k": None}, "b": {"k": 1.0}}         # skipped
    r = rs.paired_comparison(_scores(table), ["a"], ["b"], "k")
    assert (r["win"], r["loss"], r["tie"], r["n"]) == (6, 1, 1, 8)
    # exact two-sided binomial: P(X<=1 or X>=6 | n=7, p=.5) = 16/128
    assert abs(r["p"] - 16 / 128) < 1e-12
    assert r["median_diff"] == 1.0
    low = rs.paired_comparison(_scores(table), ["a"], ["b"], "k", better="lower")
    assert (low["win"], low["loss"], low["tie"]) == (1, 6, 1)
    assert low["median_diff"] == 1.0                          # raw A - B, independent of direction


def test_paired_comparison_all_tied_gives_nan():
    table = {"s": {"a": {"k": 1.0}, "b": {"k": 1.0 + 1e-13}}}
    r = rs.paired_comparison(_scores(table), ["a"], ["b"], "k")
    assert r["tie"] == 1 and math.isnan(r["p"])


def test_count_scenes_all_better():
    table = {"s1": {"a": {"k": 2.0}, "b": {"k": 3.0}, "c": {"k": 1.0}},
             "s2": {"a": {"k": 2.0}, "b": {"k": 0.5}, "c": {"k": 1.0}}}
    assert rs.count_scenes_all_better(_scores(table), [["a"], ["b"]], ["c"], "k") == 1


def test_method_means_and_spread():
    table = {"s1": {"a": {"k": 1.0}, "b": {"k": None}}, "s2": {"a": {"k": 3.0}, "b": {"k": 4.0}}}
    m = rs.method_means(_scores(table), ["a", "b"])
    assert m == {("a", "k"): 2.0, ("b", "k"): 4.0}
    assert rs.class_mean_of_means(m, ["a", "b"], "k") == 3.0
    assert abs(rs.relative_range([2.0, 4.0]) - 2 / 3) < 1e-12
    assert abs(rs.coefficient_of_variation([2.0, 4.0]) - 1 / 3) < 1e-12
    assert rs.best_method(m, ["a", "b"], "k") == "b"
    m2 = {("a", "warp"): 1.0, ("b", "warp"): 2.0}
    assert rs.best_method(m2, ["a", "b"], "warp") == "a"


def test_spearman_helpers():
    a = {("x", "k"): 1.0, ("y", "k"): 2.0, ("z", "k"): 3.0}
    b = {("x", "k"): 10.0, ("y", "k"): 30.0, ("z", "k"): 20.0}
    assert abs(rs.spearman_between_settings(a, b, ["x", "y", "z"], "k") - 0.5) < 1e-12
    table = {"s1": {"x": {"i": 1, "j": 1}, "y": {"i": 2, "j": 2}, "z": {"i": 3, "j": 3}},
             "s2": {"x": {"i": 1, "j": 3}, "y": {"i": 2, "j": 2}, "z": {"i": 3, "j": 1}},
             "s3": {"x": {"i": 1, "j": 1}, "y": {"i": 1, "j": 2}, "z": {"i": 1, "j": 3}}}   # constant x: skipped
    mean, n = rs.within_scene_spearman(_scores(table), ["x", "y", "z"], "i", "j")
    assert n == 2 and abs(mean) < 1e-12


def test_protagonist_and_others():
    pe = {"player": dict(n_visible_instants=5, mean_iou=0.6), "car": dict(n_visible_instants=3, mean_iou=None),
          "dog": dict(n_visible_instants=2, mean_iou=0.4), "hidden": dict(n_visible_instants=0, mean_iou=0.9)}
    assert rs.protagonist_and_others(pe) == (0.6, 0.2)
    assert rs.protagonist_and_others({"player": pe["player"], "hidden": pe["hidden"]}) is None
    p, o, n = rs.role_means({"m": [(0.6, 0.2), (0.4, 0.4)]})["m"]
    assert abs(p - 0.5) < 1e-12 and abs(o - 0.3) < 1e-12 and n == 2


def test_matched_vs_foil_and_grouped_means():
    rows = [dict(align=1.0, _align_foil=0.0), dict(align=0.5, _align_foil=0.5), dict(align=0.2, _align_foil=None)]
    assert rs.matched_vs_foil(rows) == (0.75, 0.25, 0.5, 2)
    g = rs.grouped_means([dict(c="a", v=1.0), dict(c="a", v=3.0), dict(c="b", v=5.0)], lambda r: r["c"], ["v"])
    assert g == {"a": dict(n=2, v=2.0), "b": dict(n=1, v=5.0)}
    cam = rs.camera_summary_degrees(dict(RotErr_mean=math.pi / 2, TransErr_rel_mean=0.1, CamMC_rel_mean=0.2))
    assert abs(cam["rot"] - 90.0) < 1e-12 and cam["trans"] == 0.1 and cam["cammc"] == 0.2
