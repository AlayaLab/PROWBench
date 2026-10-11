"""Per-sample rows and table cells of the main-track tables.

A table row is built from the `summary` of each per-sample metric file (entity, camera, temporal, CLIP) and the per-sample
VBench rows (imaging and aesthetic quality, motion smoothness), plus the `summary` of the interaction and alignment judgments.
A table cell is the plain mean of the per-sample values of one method over the evaluated pairs.

Rules:
- `cerr`: `metric_entity` averages the center error over matched instants only. A sample whose evaluated entities are never
  matched has a BBox IoU of 0 and no center error; it receives `CENTER_ERROR_MISS` (the maximum, since errors are normalized by
  the unit-square diagonal) so that the failure counts against the method. A sample without evaluated entities has neither value.
- `rot`: `RotErr_mean` (radians per instant) converted to degrees. `trans` and `cammc` are `TransErr_rel_mean` and `CamMC_rel_mean`.
- `warp` is `warp_error_x1000`, `clip` is `clip_score` (cosine x 100), `imaging`, `aesthetic` and `msmooth` are the VBench scores.
- `interaction` (ISR) is `interaction_success`; `align` (LRA) is `dynamic_checklist`.
- Missing values (None) and NaN are skipped by `mean`; every other sample counts with equal weight.
- A pair for which no method has an evaluated entity (`n_entities == 0` in every row) has no entity score for any method and is
  excluded from the expected count of the entity columns (`no_entity_pairs`, `expected_count`).
- gISR / gLRA: the per-sample product of BBox IoU and ISR / LRA, averaged over the samples where both factors are defined.

Ground truth: entity scores of first-person views use the record filtered by `gt_export.filter_first_person`
(`gt_export --first-person-filter auto`); follow views use the exported record unchanged.

Command line:
  rows   one row per sample of a samples.jsonl from the metric files of that run:
         <metrics-dir>/<key>.{entity,camera,temporal,clipscore}.json, <judge-dir>/<key>.{interaction,alignment}.json
         (windowed_judging), and the VBench per_sample.jsonl files (metric_vbench summarise) -> rows.jsonl.
         <key> is the sample id with "/" replaced by "__"; a missing file leaves its columns empty (None).
  table  cells of one or more methods from rows.jsonl files (one method per file; pairs without evaluated entities are
         determined across all given files) -> table.json, printed as one line per method. With --dataset, every released
         view of the dataset is an evaluated pair, so missing samples show as n < expected.
"""
import argparse
import json
import math
from pathlib import Path

from .common import write_text

CENTER_ERROR_MISS = 1.0
COLUMNS = ("iou", "cerr", "interaction", "align", "gisr", "glra", "rot", "trans", "cammc", "imaging", "aesthetic", "warp",
           "msmooth", "clip")
SAMPLE_COLUMNS = ("iou", "cerr", "rot", "trans", "cammc", "imaging", "aesthetic", "warp", "msmooth", "clip", "interaction", "align")
ENTITY_COLUMNS = ("iou", "cerr")
GROUNDED = {"gisr": "interaction", "glra": "align"}


def sample_row(entity: dict, camera: dict, temporal: dict, clipscore: dict, vbench: dict, vbench_motion: dict,
               interaction: dict, alignment: dict) -> dict:
    """One table row from the per-sample metric summaries. `vbench` / `vbench_motion` are the per-sample VBench rows with the
    keys `vbench.imaging_quality`, `vbench.aesthetic_quality` and `vbench.motion_smoothness`; any argument may be an empty dict."""
    iou, center = entity.get("mean_iou"), entity.get("mean_center_err")
    penalty = center is None and iou is not None
    rot = camera.get("RotErr_mean")
    return dict(iou=iou, cerr=CENTER_ERROR_MISS if penalty else center, rot=None if rot is None else math.degrees(rot),
                trans=camera.get("TransErr_rel_mean"), cammc=camera.get("CamMC_rel_mean"),
                imaging=vbench.get("vbench.imaging_quality"), aesthetic=vbench.get("vbench.aesthetic_quality"),
                warp=temporal.get("warp_error_x1000"), msmooth=vbench_motion.get("vbench.motion_smoothness"),
                clip=clipscore.get("clip_score"), interaction=interaction.get("interaction_success"),
                align=alignment.get("dynamic_checklist"), n_entities=entity.get("n_entities"), center_error_penalty=penalty,
                align_foil=alignment.get("dynamic_foil"), interaction_invalid=interaction.get("n_invalid"))


def _defined(v) -> bool:
    return v is not None and not (isinstance(v, float) and math.isnan(v))


def mean(values):
    """(mean, n) over the defined values; (None, 0) if there is none."""
    vals = [v for v in values if _defined(v)]
    return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)


def grounded_values(rows, column: str) -> list:
    """Per-sample IoU x ISR (`gisr`) or IoU x LRA (`glra`) for the rows where both factors are defined."""
    key = GROUNDED[column]
    return [r[key] * r["iou"] for r in rows if r.get(key) is not None and r.get("iou") is not None]


def cell(rows, column: str):
    """(mean, n) of one column over the rows of one method."""
    if column in GROUNDED:
        return mean(grounded_values(rows, column))
    return mean(r.get(column) for r in rows)


def method_cells(rows, columns=COLUMNS) -> dict:
    """{column: {"mean": ..., "n": ...}} for the rows of one method (one row per evaluated pair)."""
    out = {}
    for c in columns:
        m, n = cell(rows, c)
        out[c] = dict(mean=m, n=n)
    return out


def no_entity_pairs(rows_by_method: dict) -> set:
    """Pairs (episode, camera) for which every method row has zero evaluated entities."""
    pairs = {}
    for rows in rows_by_method.values():
        for r in rows:
            pairs.setdefault((r["episode"], r["camera"]), []).append(r.get("n_entities"))
    return {p for p, ns in pairs.items() if ns and all(n == 0 for n in ns)}


def expected_count(pairs, column: str, excluded=()) -> int:
    """Number of pairs a complete cell averages: entity columns leave out the pairs without evaluated entities."""
    return len([p for p in pairs if not (column in ENTITY_COLUMNS and p in excluded)])


def table(rows_by_method: dict, columns=COLUMNS, pairs=None) -> dict:
    """{method: {column: {"mean", "n", "expected"}}}; `rows_by_method` maps a method to its rows (keys episode, camera, n_entities
    and the sample columns), one row per pair. `expected` follows `expected_count` over `pairs` (the evaluated (episode, camera)
    pairs, e.g. every released view of the dataset; default: the pairs present in the method's rows), so a cell with
    n < expected has missing samples. Rows of pairs outside `pairs` are an error."""
    excluded = no_entity_pairs(rows_by_method)
    out = {}
    for method, rows in rows_by_method.items():
        present = [(r["episode"], r["camera"]) for r in rows]
        if len(set(present)) != len(present):
            raise ValueError(f"{method}: more than one row for a pair")
        if pairs is not None and not set(present) <= set(pairs):
            raise ValueError(f"{method}: rows of pairs outside the evaluated pairs: {sorted(set(present) - set(pairs))[:5]}")
        pairs_m = sorted(set(pairs) if pairs is not None else set(present))
        cells = method_cells(rows, columns)
        for c, v in cells.items():
            src = GROUNDED.get(c)
            v["expected"] = expected_count(pairs_m, "iou" if src else c, excluded)
        out[method] = cells
    return out


def _summary(path):
    p = Path(path)
    return json.loads(p.read_text()).get("summary") or {} if p.exists() else {}


def _vbench(path):
    if not path:
        return {}
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return {r["sample_id"]: r for r in rows}


def rows_from_files(samples, metrics_dir, judge_dir=None, vbench=None, vbench_motion=None):
    """Table rows of the samples (samples.jsonl rows) from the metric files of one run (see the module docstring)."""
    vb, vbm = _vbench(vbench), _vbench(vbench_motion)
    out = []
    for s in samples:
        key = s["sample_id"].replace("/", "__")
        m = Path(metrics_dir)
        j = Path(judge_dir) if judge_dir else None
        row = sample_row(_summary(m / f"{key}.entity.json"), _summary(m / f"{key}.camera.json"),
                         _summary(m / f"{key}.temporal.json"), _summary(m / f"{key}.clipscore.json"),
                         vb.get(s["sample_id"], {}), vbm.get(s["sample_id"], {}),
                         _summary(j / f"{key}.interaction.json") if j else {},
                         _summary(j / f"{key}.alignment.json") if j else {})
        out.append(dict(row, sample_id=s["sample_id"], episode=s["episode"], camera=s["camera"],
                        representation=s["representation"], method=s["method"], run_id=s["run_id"]))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("rows")
    r.add_argument("--samples", required=True)
    r.add_argument("--metrics-dir", required=True, help="<key>.entity/.camera/.temporal/.clipscore.json")
    r.add_argument("--judge-dir", default=None, help="<key>.interaction/.alignment.json")
    r.add_argument("--vbench", default=None, help="per_sample.jsonl of the imaging/aesthetic VBench run")
    r.add_argument("--vbench-motion", default=None, help="per_sample.jsonl of the motion-smoothness VBench run")
    r.add_argument("--out", required=True)
    t = sub.add_parser("table")
    t.add_argument("--rows", nargs="+", required=True)
    t.add_argument("--dataset", default=None,
                   help="released dataset directory; its views are the evaluated pairs (expected counts, missing samples)")
    t.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.command == "rows":
        samples = [json.loads(line) for line in Path(a.samples).read_text().splitlines() if line.strip()]
        rows = rows_from_files(samples, a.metrics_dir, a.judge_dir, a.vbench, a.vbench_motion)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        write_text(a.out, "".join(json.dumps(x) + "\n" for x in rows))
        print(f"{len(rows)} rows -> {a.out}")
        return
    by_method = {}
    for path in a.rows:
        rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        methods = {x["method"] for x in rows}
        if len(methods) != 1:
            raise SystemExit(f"{path}: rows of one method expected, got {sorted(methods)}")
        method = methods.pop()
        if method in by_method:
            raise SystemExit(f"method {method} given twice")
        by_method[method] = rows
    pairs = None
    if a.dataset:
        views = [json.loads(line) for line in Path(a.dataset, "samples.jsonl").read_text().splitlines() if line.strip()]
        pairs = sorted({(v["episode"], v["camera"]) for v in views})
    result = table(by_method, pairs=pairs)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    write_text(a.out, json.dumps(result, indent=1))
    for method, cells in result.items():
        print(method, json.dumps({c: (None if v["mean"] is None else round(v["mean"], 4)) for c, v in cells.items()}))
        short = {c: f"{v['n']}/{v['expected']}" for c, v in cells.items() if v["n"] < v["expected"]}
        if short:
            print(f"  {method}: incomplete cells (n/expected): {short}")


if __name__ == "__main__":
    main()
