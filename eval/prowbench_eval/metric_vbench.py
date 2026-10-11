"""VBench bridge: lossless trimming of the generated videos for VBench, per-sample rows from its results, summaries and
rank correlations."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import EVAL_DURATION, read_json, read_jsonl, video_meta, write_json, write_text

VBENCH_DIMS = ["subject_consistency", "background_consistency", "motion_smoothness", "dynamic_degree",
               "aesthetic_quality", "imaging_quality", "temporal_flickering"]


_X264 = {"yuv420p", "yuv422p", "yuv444p", "yuvj420p", "yuvj422p", "yuvj444p", "yuv420p10le", "yuv422p10le", "yuv444p10le",
         "gray", "gray10le"}
_X264RGB = {"rgb24", "bgr24", "bgr0"}


def _lossless_codec(video: str, meta: dict) -> list[str]:
    """Encoder arguments that keep the decoded frames of `video` exactly: its own pixel format (chroma layout and bit
    depth); 4:2:0 / 4:2:2 frames whose size does not allow subsampling are stored as 4:4:4 of the same bit depth."""
    r = subprocess.run([os.environ.get("FFMPEG", "ffmpeg"), "-hide_banner", "-i", video], capture_output=True, text=True)
    m = re.search(r"Video: [^,]+(?:\([^)]*\))?, ([a-z0-9_]+)", r.stderr)
    pix = m.group(1) if m else ""        # yuvj*: full-range YUV, kept as such
    odd_w, odd_h = meta["width"] % 2, meta["height"] % 2
    if pix.startswith(("yuv420p", "yuvj420p")) and (odd_w or odd_h) or pix.startswith(("yuv422p", "yuvj422p")) and odd_w:
        pix = pix.replace("420", "444").replace("422", "444")
    if pix in _X264:
        return ["-c:v", "libx264", "-qp", "0", "-preset", "veryfast", "-pix_fmt", pix]
    if pix in _X264RGB or pix == "gbrp":          # gbrp: planar 8-bit RGB, stored as rgb24 without loss
        return ["-c:v", "libx264rgb", "-qp", "0", "-preset", "veryfast", "-pix_fmt", "rgb24" if pix == "gbrp" else pix]
    raise ValueError(f"{video}: pixel format {pix or 'unknown'} cannot be trimmed losslessly; provide a video of exactly "
                     f"{int(EVAL_DURATION * meta['fps'] + 1e-6)} frames")


def prepare(samples_jsonl: str, out_dir: str, prompt_lookup=None):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    mapping = read_json(out / "mapping.json") if (out / "mapping.json").exists() else []
    seen = {m["file"] for m in mapping}
    for s in read_jsonl(samples_jsonl):
        name = f"{s['method']}__{s['run_id']}__{s['sample_id'].replace('/', '__')}.mp4"
        dst = out / name
        if not dst.exists():
            meta = video_meta(s["video"]); keep = int(EVAL_DURATION * meta["fps"] + 1e-6)
            if meta["frame_count"] <= keep:
                try:
                    os.symlink(os.path.abspath(s["video"]), dst)
                except OSError:
                    tmp = dst.with_name(f".{dst.stem}.{uuid.uuid4().hex}.part.mp4")
                    try:
                        shutil.copy2(s["video"], tmp)
                        os.replace(tmp, dst)
                    finally:
                        tmp.unlink(missing_ok=True)
            else:
                tmp = dst.with_name(f".{dst.stem}.{uuid.uuid4().hex}.part.mp4")
                try:
                    r = subprocess.run([os.environ.get("FFMPEG", "ffmpeg"), "-v", "error", "-y", "-i", s["video"], "-frames:v",
                                        str(keep), "-an", *_lossless_codec(s["video"], meta), str(tmp)], capture_output=True, text=True)
                    if r.returncode or video_meta(tmp)["frame_count"] != keep:
                        raise RuntimeError(f"trim to {keep} frames failed for {s['video']}: {r.stderr[-300:]}")
                    os.replace(tmp, dst)
                finally:
                    tmp.unlink(missing_ok=True)         # nothing is left behind that VBench would read
        if name not in seen:
            mapping.append(dict(file=name, **{k: s[k] for k in ("method", "run_id", "sample_id", "representation")}))
            seen.add(name)
    write_json(out / "mapping.json", mapping)
    print(f"{len(mapping)} videos in {out} (mapping merged)")


def _rank(a):
    a = np.asarray(a, float); order = a.argsort(); r = np.empty(len(a)); r[order] = np.arange(len(a))
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, r)
    return sums[inv] / cnt[inv]


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y)); x, y = x[m], y[m]
    if len(x) < 3:
        return None, len(x)
    rx, ry = _rank(x), _rank(y)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return (float((rx * ry).sum() / d) if d > 0 else None), int(len(x))


def kendall(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = ~(np.isnan(x) | np.isnan(y)); x, y = x[m], y[m]
    n = len(x)
    if n < 3:
        return None
    c = d = 0
    for i in range(n):
        for j in range(i + 1, n):
            s = np.sign(x[i] - x[j]) * np.sign(y[i] - y[j])
            c += s > 0; d += s < 0
    return float((c - d) / (n * (n - 1) / 2))


def load_vbench_results(path: str, mapping: list[dict]) -> dict:
    """VBench per-video output: {dim: [score, [{"video_path":..., "video_results":...}]]}. Returns file->{dim: score}."""
    raw = read_json(path)
    per = {}
    for dim, val in raw.items():
        entries = val[1] if isinstance(val, list) and len(val) > 1 else []
        for e in entries:
            f = os.path.basename(e.get("video_path", ""))
            per.setdefault(f, {})[dim] = e.get("video_results")
    return per


def correlate(vbench_json: str, mapping_json: str, table_jsonl: str, out: str, task_metrics=None):
    mapping = read_json(mapping_json)
    per_file = load_vbench_results(vbench_json, mapping)
    rows = {(r["method"], r["run_id"], r["sample_id"]): r for r in read_jsonl(table_jsonl)}
    task_metrics = task_metrics or ["entity.mean_iou", "entity.mean_center_err", "entity.miss_rate",
                                    "interaction.interaction_success", "alignment.timeline_checklist",
                                    "alignment.caption_consistency", "camera.ate_over_path",
                                    "camera.RotErr_mean", "camera.TransErr_rel_mean", "camera.CamMC_rel_mean"]
    joined = []
    for m in mapping:
        key = (m["method"], m["run_id"], m["sample_id"])
        if key in rows and m["file"] in per_file:
            joined.append(dict(**rows[key], vbench=per_file[m["file"]]))
    res = {}
    for tm in task_metrics:
        x = [r.get(tm, np.nan) if r.get(tm) is not None else np.nan for r in joined]
        for dim in VBENCH_DIMS:
            y = [r["vbench"].get(dim, np.nan) if r["vbench"].get(dim) is not None else np.nan for r in joined]
            rho, n = spearman(x, y)
            res[f"{tm} ~ {dim}"] = dict(spearman=rho, kendall=kendall(x, y), n=n)
    write_json(out, dict(schema="prowbench.vbench_correlation.v1", n_joined=len(joined), correlations=res))
    print(json.dumps({k: v for k, v in res.items() if v["spearman"] is not None}, indent=1)[:3000])



VBENCH_FPS_INDEPENDENT = ["imaging_quality", "aesthetic_quality", "dynamic_degree"]


def vbench_rows(vbench_json: str, mapping_json: str) -> list[dict]:
    """One row per prepared video: sample fields + `vbench.<dim>` floats (dynamic_degree bool -> 0/1).
    Every mapped video must appear in every dimension of the VBench json; missing entries raise."""
    mapping = read_json(mapping_json)
    per_file = load_vbench_results(vbench_json, mapping)
    dims = sorted({d for v in per_file.values() for d in v})
    rows = []
    for m in mapping:
        if m["file"] not in per_file:
            raise KeyError(f"{m['file']} missing from VBench results")
        row = dict(method=m["method"], run_id=m["run_id"], sample_id=m["sample_id"], representation=m["representation"])
        for d in dims:
            if d not in per_file[m["file"]]:
                raise KeyError(f"{m['file']} has no {d} result")
            v = per_file[m["file"]][d]
            row[f"vbench.{d}"] = float(v) if isinstance(v, (bool, int, float)) else float(v[0])
        rows.append(row)
    return rows


_HIGHER_BETTER = {"mean_iou", "interaction_success", "event_precision", "participants_rate", "timing_rate", "outcome_rate", "action_rate", "caption_consistency",
                  "timeline_checklist", "scene_facts", "dynamic_checklist", "dynamic_strict", "end_state", "static_facts", "dynamic_margin", "static_margin", "onset_hit_rate", "onset_precision",
                  "imaging_quality", "aesthetic_quality", "dynamic_degree", "subject_consistency", "background_consistency",
                  "motion_smoothness", "temporal_flickering"}


def _boot_ci(v, n_boot=2000, seed=0):
    v = np.asarray(v, float)
    if len(v) < 2:
        return (None, None)
    rng = np.random.default_rng(seed)
    means = [rng.choice(v, len(v), replace=True).mean() for _ in range(n_boot)]
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def _ci_text(ci, fmt=".3f"):
    """bootstrap interval for the summary table; n/a when fewer than two values give no interval"""
    lo, hi = ci
    return "[n/a]" if lo is None or hi is None else f"[{lo:{fmt}}, {hi:{fmt}}]"


def _summarise_rows(rows, group_keys=("method", "run_id", "representation")):
    """Mean, std, n and bootstrap 95% CI of every metric column per (method, run, representation)."""
    for r in rows:
        for k, v in r.items():
            if '.' in k and isinstance(v, (int, float)) and not np.isfinite(v):
                raise ValueError(f"Non-finite metric {k} in sample {r.get('sample_id')}")
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in group_keys)].append(r)
    out = []
    metric_cols = [c for c in rows[0] if "." in c] if rows else []
    for g, rs in sorted(groups.items()):
        row = dict(zip(group_keys, g)); row["n"] = len(rs)
        for c in metric_cols:
            v = [r[c] for r in rs if isinstance(r.get(c), (int, float)) and not isinstance(r.get(c), bool)]
            if v:
                lo, hi = _boot_ci(v)
                row[c] = dict(mean=float(np.mean(v)), std=float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, n=len(v), ci95=[lo, hi])
            else:
                row[c] = None
        out.append(row)
    return out


def _paired(rows, a: tuple, b: tuple, key=("method", "run_id")):
    """Paired difference b - a on the intersection of sample ids (same representation enforced by sample_id)."""
    for group in (a, b):
        selected = [r['sample_id'] for r in rows if tuple(r[k] for k in key) == group]
        if len(selected) != len(set(selected)):
            raise ValueError(f"Duplicate paired sample IDs in {group}; seeds need explicit pairing")
    A = {r["sample_id"]: r for r in rows if tuple(r[k] for k in key) == a}
    B = {r["sample_id"]: r for r in rows if tuple(r[k] for k in key) == b}
    common = sorted(set(A) & set(B))
    res = dict(a=a, b=b, n_common=len(common), samples=common, metrics={})
    if not common:
        return res
    for c in [c for c in rows[0] if "." in c]:
        d = [B[s][c] - A[s][c] for s in common if isinstance(A[s].get(c), (int, float)) and isinstance(B[s].get(c), (int, float))
             and not isinstance(A[s].get(c), bool)]
        if d:
            lo, hi = _boot_ci(d)
            wins = sum(1 for x in d if (x > 0) == (c.split(".")[1] in _HIGHER_BETTER) and x != 0)
            res["metrics"][c] = dict(mean_diff=float(np.mean(d)), ci95=[lo, hi], n=len(d), b_wins=wins, b_win_rate=wins / len(d))
    return res


def summarise(vbench_json: str, mapping_json: str, out_dir: str, pairs=(), expected=None):
    from .common import write_jsonl
    rows = vbench_rows(vbench_json, mapping_json)
    if expected is not None and len(rows) != expected:
        raise ValueError(f"expected {expected} samples, got {len(rows)}")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "per_sample.jsonl", rows)
    summary = _summarise_rows(rows)
    write_json(out / "summary.json", summary)
    cols = [c for c in rows[0] if c.startswith("vbench.")]
    paired = {}
    for p in pairs:
        a, b = [tuple(x.split(":")) for x in p.split(",")]
        paired[p] = _paired(rows, a, b)
    write_json(out / "paired.json", paired)
    lines = ["| method | run | rep | n | " + " | ".join(c.split(".", 1)[1] for c in cols) + " |",
             "|" + "---|" * (4 + len(cols))]
    for r in summary:
        cells = []
        for c in cols:
            v = r[c]
            cells.append("--" if v is None else f"{v['mean']:.3f} ± {v['std']:.3f} {_ci_text(v['ci95'])}")
        lines.append(f"| {r['method']} | {r['run_id']} | {r['representation']} | {r['n']} | " + " | ".join(cells) + " |")
    lines += ["", "paired (b − a, mean diff [95% CI], b wins / n):"]
    for p, res in paired.items():
        for c in cols:
            m = res["metrics"].get(c)
            if m:
                lines.append(f"- {p} {c}: {m['mean_diff']:+.3f} {_ci_text(m['ci95'], '+.3f')} {m['b_wins']}/{m['n']}")
    write_text(out / "summary.md", "\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--samples", required=True); p.add_argument("--out-dir", required=True)
    c = sub.add_parser("correlate"); c.add_argument("--vbench-json", required=True); c.add_argument("--mapping", required=True)
    c.add_argument("--table", required=True, help="per-sample jsonl with the task metric columns"); c.add_argument("--out", required=True)
    s = sub.add_parser("summarise"); s.add_argument("--vbench-json", required=True); s.add_argument("--mapping", required=True)
    s.add_argument("--out-dir", required=True); s.add_argument("--pairs", nargs="*", default=[]); s.add_argument("--expected", type=int)
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        prepare(a.samples, a.out_dir)
    elif a.cmd == "summarise":
        summarise(a.vbench_json, a.mapping, a.out_dir, a.pairs, a.expected)
    else:
        correlate(a.vbench_json, a.mapping, a.table, a.out)


if __name__ == "__main__":
    main()
