"""Derived statistics over per-sample metric values (method comparisons in the analysis section).

Input contract
--------------
``scores`` maps a key ``"<scene>|<method>"`` to a dict of per-sample metric values (for example ``iou``, ``cerr``,
``interaction``, ``align``, ``rot``, ``trans``, ``cammc``, ``imaging``, ``aesthetic``, ``warp``, ``msmooth``, ``clip``).
A scene is one scene-camera pair; a missing metric value is ``None``.  :func:`scene_scores` builds this mapping from
per-pair table rows.

Conventions (all functions below follow them)
---------------------------------------------
* Class mean of one scene: plain mean over the methods of the class; it is undefined (``None``) unless every method of
  the class has a value for that scene.
* Paired comparison of two classes (or two single methods) over scenes: a scene counts as a win for A when A is better
  than B by more than ``1e-12`` in the stated direction, as a loss when B is better, and as a tie otherwise.  Scenes where
  either class mean is undefined are skipped.  The p-value is the exact two-sided binomial (sign) test with success
  probability 0.5 over the non-tied scenes (``scipy.stats.binomtest``); it is NaN when every scene is tied.  The reported
  median is the median of the raw differences A - B over all compared scenes (ties included), independent of direction.
* Method mean: mean of the defined per-sample values of that method.
* Rank correlation: Spearman's rho (``scipy.stats.spearmanr``, average ranks for ties).
* Spread of method means: ``relative_range`` = (max - min) / mean; ``coefficient_of_variation`` = population standard
  deviation / mean.
"""
from __future__ import annotations

import math
import statistics

from scipy.stats import binomtest, spearmanr

TIE_TOLERANCE = 1e-12
LOWER_IS_BETTER = frozenset({"cerr", "rot", "trans", "cammc", "warp"})


# ----------------------------------------------------------------------------------------------- per-scene scores
def scene_scores(rows, metrics, scene_of, method_of):
    """Map per-pair rows to ``{"<scene>|<method>": {metric: value}}``.

    ``scene_of(row)`` and ``method_of(row)`` give the scene key and the method name of a row.  Every (scene, method)
    combination must occur at most once.
    """
    out = {}
    for row in rows:
        key = f"{scene_of(row)}|{method_of(row)}"
        if key in out:
            raise ValueError(f"duplicate scene/method row: {key}")
        out[key] = {m: row.get(m) for m in metrics}
    return out


def by_scene(scores):
    """``{"<scene>|<method>": values}`` -> ``{scene: {method: values}}`` (insertion order kept)."""
    out = {}
    for key, values in scores.items():
        scene, method = key.split("|")
        out.setdefault(scene, {})[method] = values
    return out


# ----------------------------------------------------------------------------------------------- class comparisons
def class_mean(row, methods, metric):
    """Mean of ``metric`` over ``methods`` for one scene row ``{method: values}``; ``None`` unless all are defined."""
    xs = [row[m][metric] for m in methods if row[m].get(metric) is not None]
    return sum(xs) / len(xs) if len(xs) == len(methods) else None


def paired_comparison(scores, group_a, group_b, metric, better="higher"):
    """Scene-paired comparison of the class means of ``group_a`` and ``group_b`` (see module conventions).

    Returns ``dict(win, loss, tie, n, p, median_diff)``.
    """
    if better not in ("higher", "lower"):
        raise ValueError("better must be 'higher' or 'lower'")
    win = loss = tie = 0
    diffs = []
    for row in by_scene(scores).values():
        a, b = class_mean(row, group_a, metric), class_mean(row, group_b, metric)
        if a is None or b is None:
            continue
        d = a - b if better == "higher" else b - a
        diffs.append(a - b)
        if abs(d) < TIE_TOLERANCE:
            tie += 1
        elif d > 0:
            win += 1
        else:
            loss += 1
    p = binomtest(win, win + loss).pvalue if win + loss else float("nan")
    return dict(win=win, loss=loss, tie=tie, n=win + loss + tie, p=p, median_diff=statistics.median(diffs))


def count_scenes_all_better(scores, groups, reference, metric):
    """Number of scenes where every class in ``groups`` has a strictly higher class mean than ``reference``.

    Scenes where the first class mean is undefined are skipped.
    """
    n = 0
    for row in by_scene(scores).values():
        ref = class_mean(row, reference, metric)
        vals = [class_mean(row, g, metric) for g in groups]
        if vals[0] is None:
            continue
        if all(v > ref for v in vals):
            n += 1
    return n


# ----------------------------------------------------------------------------------------------- method means
def method_means(scores, methods, metrics=None):
    """``{(method, metric): mean}`` over the defined per-sample values of each method.

    ``metrics`` defaults to the metric keys of the first entry of ``scores``.
    """
    if metrics is None:
        metrics = list(scores[next(iter(scores))].keys())
    out = {}
    for m in methods:
        for k in metrics:
            xs = [v[k] for key, v in scores.items() if key.endswith("|" + m) and v.get(k) is not None]
            out[(m, k)] = sum(xs) / len(xs)
    return out


def class_mean_of_means(means, methods, metric):
    """Unweighted mean of the method means of a class."""
    return sum(means[(m, metric)] for m in methods) / len(methods)


def spearman_between_settings(means_a, means_b, methods, metric):
    """Spearman's rho between the method means of two settings for one metric."""
    return spearmanr([means_a[(m, metric)] for m in methods], [means_b[(m, metric)] for m in methods]).statistic


def within_scene_spearman(scores, methods, metric_x, metric_y):
    """Mean over scenes of Spearman's rho between two metrics across methods.

    A scene is skipped when any method lacks either value or when either metric is constant across methods.
    Returns ``(mean_rho, number_of_scenes)``.
    """
    rs = []
    for row in by_scene(scores).values():
        x = [row[m][metric_x] for m in methods]
        y = [row[m][metric_y] for m in methods]
        if None in x or None in y or len(set(x)) < 2 or len(set(y)) < 2:
            continue
        rs.append(spearmanr(x, y).statistic)
    return sum(rs) / len(rs), len(rs)


def relative_range(values):
    """(max - min) / mean of a list of method means."""
    values = list(values)
    return (max(values) - min(values)) / (sum(values) / len(values))


def coefficient_of_variation(values):
    """Population standard deviation / mean of a list of method means."""
    values = list(values)
    return statistics.pstdev(values) / (sum(values) / len(values))


def best_method(means, methods, metric):
    """Method with the best mean (lowest for metrics in ``LOWER_IS_BETTER``, else highest; first one on ties)."""
    pick = min if metric in LOWER_IS_BETTER else max
    return pick(methods, key=lambda m: means[(m, metric)])


# ----------------------------------------------------------------------------------------------- entity roles
def protagonist_and_others(per_entity, protagonist="player"):
    """Per-sample protagonist IoU and mean IoU of the other entities.

    ``per_entity`` is the ``per_entity`` block of an entity metric file.  Only entities with at least one visible
    reference instant are used; a missing ``mean_iou`` counts as 0.  Returns ``None`` when the protagonist is not
    evaluated or no other entity is.
    """
    pe = {k: e for k, e in per_entity.items() if (e.get("n_visible_instants") or 0) > 0}
    if protagonist not in pe or len(pe) < 2:
        return None
    others = statistics.mean((e.get("mean_iou") or 0.0) for k, e in pe.items() if k != protagonist)
    return (pe[protagonist].get("mean_iou") or 0.0), others


def role_means(pairs_by_method):
    """``{method: [(protagonist, others), ...]}`` -> ``{method: (mean protagonist, mean others, n)}``."""
    return {m: (statistics.mean(x for x, _ in v), statistics.mean(y for _, y in v), len(v))
            for m, v in pairs_by_method.items()}


# ----------------------------------------------------------------------------------------------- mismatched timeline
def matched_vs_foil(rows, matched="align", foil="_align_foil"):
    """Mean matched score, mean foil score and share of samples with matched > foil.

    Only rows where both values are defined are used.  Returns ``(matched_mean, foil_mean, share_higher, n)``.
    """
    rr = [r for r in rows if r.get(foil) is not None and r.get(matched) is not None]
    return (statistics.mean(r[matched] for r in rr), statistics.mean(r[foil] for r in rr),
            sum(r[matched] > r[foil] for r in rr) / len(rr), len(rr))


# ----------------------------------------------------------------------------------------------- grouped means
def grouped_means(rows, group_of, metrics):
    """``{group: dict(n=..., <metric>=mean)}`` over rows, grouped by ``group_of(row)`` (plain means)."""
    groups = {}
    for r in rows:
        groups.setdefault(group_of(r), []).append(r)
    return {g: dict(n=len(rr), **{k: sum(r[k] for r in rr) / len(rr) for k in metrics}) for g, rr in groups.items()}


def camera_summary_degrees(summary):
    """Camera metric summary -> ``dict(rot=<degrees>, trans=..., cammc=...)`` (rotation stored in radians)."""
    return dict(rot=summary["RotErr_mean"] * 180 / math.pi, trans=summary["TransErr_rel_mean"],
                cammc=summary["CamMC_rel_mean"])
