"""Command-line driver for the multi-view appearance (Comp. / Cons.) and MEt3R metrics of one method.

Inputs:
  --dataset   released multi-view dataset (data/multi4)
  --samples   samples.jsonl of the method (make_samples); one row per episode and view, all with the same crop_xywh
  --out       output directory

  appearance  builds the judge jobs (multiview_runner.appearance_jobs), asks the VLM judge (greedy, thinking off, at most
              2400 new tokens), parses every response strictly and writes <out>/appearance/<job id>.json. A failed
              judgment is written as <job id>.error.json, is not counted as an absent or unobservable observation and
              makes the command exit non-zero; successful results are kept and skipped on the next run.
  met3r       scores all six view pairs at the MEt3R instants (multiview_met3r.measure) into <out>/met3r/.
  summarize   Comp. and Cons. (multiview_appearance.aggregate_observations, in percent) and MEt3R
              (multiview_met3r.aggregate_scenes) from the stored results -> <out>/summary.json. A score is reported only
              when every expected judgment (job lists written by `appearance`) or every view pair and instant is present
              exactly once (no result outside that set, no second result for the same item) and no judgment failed;
              otherwise it is null and the missing, unexpected or duplicate items are listed.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .common import read_json, read_jsonl, write_json
from .multiview_appearance import aggregate_observations
from .multiview_geometry import Clip
from .multiview_runner import JUDGE_SETTINGS, VIEWS, appearance_jobs, met3r_pairs, observation_rows, parse_response
from .vlm_judge import DEFAULT_JUDGE_MODEL, Judge


def load_samples(path):
    """(method, media, crop, episodes): media maps the method to {"<episode>/<view>": video path}."""
    rows = list(read_jsonl(path))
    methods = {r["method"] for r in rows}
    crops = {tuple(r.get("crop_xywh") or (0, 0, 1280, 720)) for r in rows}
    if len(methods) != 1 or len(crops) != 1:
        raise SystemExit("samples must belong to one method and share one crop_xywh")
    method = methods.pop()
    keys = [f"{r['episode']}/{r['camera']}" for r in rows]
    if len(set(keys)) != len(keys):
        raise SystemExit("samples contain more than one video for a view (select one representation with make_samples --representations)")
    media = dict(zip(keys, (r["video"] for r in rows)))
    episodes = sorted({r["episode"] for r in rows})
    for ep in episodes:
        missing = [v for v in VIEWS if f"{ep}/{v}" not in media]
        if missing:
            raise SystemExit(f"{ep}: no video for views {missing}")
    return method, {method: media}, list(crops.pop()), episodes


def job_id(job):
    views = "-".join(v.replace("follow-", "") for v in job["views"])
    return re.sub(r"[^A-Za-z0-9_.-]", "_", f"{job['kind']}__{job['episode']}__{job['t']}__{job['participant']}__{views}")


def run_appearance(args, judge=None):
    method, media, crop, episodes = load_samples(args.samples)
    ns = argparse.Namespace(release=args.dataset, min_pix=2500, exclusions=[], crops={method: crop})
    out = Path(args.out) / "appearance"
    out.mkdir(parents=True, exist_ok=True)
    if judge is None:
        import torch
        torch.manual_seed(0)
        protocol = read_json(args.protocol) if args.protocol else {}
        model = protocol.get("judge", {}).get("model") or DEFAULT_JUDGE_MODEL
        judge = Judge(model=model, device=args.device, cache_dir=str(out / ".judge_cache"), **JUDGE_SETTINGS)
    import cv2
    done = failed = 0
    for ep in episodes[args.worker::args.workers]:
        cache = {}
        try:
            jobs = appearance_jobs(ns, media, method, ep, cache)
        finally:
            for key, value in cache.items():
                if key[0] == "clip":
                    value.close()
        names = [job_id(j) for j in jobs]
        if len(set(names)) != len(names):
            raise SystemExit(f"{ep}: duplicate appearance jobs")
        write_json(out / f"jobs__{ep}.json", dict(episode=ep, method=method, jobs=names))
        for job in jobs:
            name = job_id(job)
            dest = out / f"{name}.json"
            if dest.exists():
                continue
            meta = {k: job[k] for k in ("kind", "method", "episode", "t", "participant", "views", "prompt")}
            try:
                frames = [(i, cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) for i, im in enumerate(job["images"])]
                rec = judge.ask(frames, job["prompt"])
                parsed = parse_response(job, rec["raw"])
                if parsed != rec["parsed"]:
                    raise ValueError("judge response is not strict JSON")
                write_json(dest, dict(job=meta, parsed=parsed, raw=rec["raw"], rows=observation_rows(job, parsed),
                                      judge=dict(model=getattr(judge, "model_name", None), **JUDGE_SETTINGS)))
                (out / f"{name}.error.json").unlink(missing_ok=True)
                done += 1
            except Exception as exc:
                write_json(out / f"{name}.error.json", dict(job=meta, error=repr(exc)))
                failed += 1
    print(f"appearance: {done} judged, {failed} failed")
    if failed:
        raise SystemExit(f"{failed} failed judgments (not counted as absent or unobservable); re-run to retry")
    return done


def run_met3r(args, model=None):
    from .multiview_met3r import CONFIG, load_model, measure
    method, media, _crop, episodes = load_samples(args.samples)
    out = Path(args.out) / "met3r"
    out.mkdir(parents=True, exist_ok=True)
    model = model or load_model()
    n = 0
    for ep in episodes[args.worker::args.workers]:
        clips = {v: Clip(media[method][f"{ep}/{v}"]) for v in VIEWS}
        try:
            for t, a, b in met3r_pairs():
                dest = out / f"{ep}__{t}__{a}__{b}.json"
                if dest.exists():
                    continue
                res = measure(model, clips[a].at(t), clips[b].at(t), dest.with_suffix(".masks.npz"))
                write_json(dest, dict(method=method, episode=ep, t=t, views=[a, b], config=CONFIG, **res))
                n += 1
        finally:
            for c in clips.values():
                c.close()
    print(f"met3r: {n} pairs scored")
    return n


def summarize(args):
    from .multiview_met3r import aggregate_scenes
    method, _media, _crop, episodes = load_samples(args.samples)
    missing_scenes = sorted(set(read_json(Path(args.dataset) / "index.json")["episodes"]) - set(episodes))
    out = Path(args.out)
    app = out / "appearance"
    rows = {"prompt": [], "consistency": []}
    errors = sorted(p.name for p in app.glob("*.error.json"))
    missing, unexpected, expected_names = [], [], set()
    for ep in episodes:
        manifest = app / f"jobs__{ep}.json"
        if not manifest.exists():
            missing.append(f"{ep}: no job list (appearance not run)")
            continue
        names = read_json(manifest)["jobs"]
        expected_names |= set(names)
        missing += [n for n in names if not (app / f"{n}.json").exists()]
    for p in sorted(app.glob("*.json")):
        if p.name.endswith(".error.json") or p.name.startswith("jobs__"):
            continue
        if p.stem not in expected_names:
            unexpected.append(p.name)
            continue
        rec = read_json(p)
        if rec["job"]["method"] != method:
            unexpected.append(p.name)
            continue
        for r in rec["rows"]:
            rows[r["kind"]].append(r)
    complete = not (errors or missing or unexpected or missing_scenes)
    summary = dict(method=method, episodes=episodes, missing_scenes=missing_scenes, appearance_complete=complete,
                   failed_judgments=errors,
                   missing_judgments=missing[:50], n_missing=len(missing), unexpected_results=unexpected[:50])
    for name, kind in (("compliance", "prompt"), ("consistency", "consistency")):
        agg = aggregate_observations(rows[kind], episodes) if rows[kind] else dict(score=None, scene_scores={})
        summary[name] = dict(percent=None if agg["score"] is None or not complete else 100 * agg["score"],
                             scene_scores=agg["scene_scores"], observations=len(rows[kind]),
                             scored=sum(r["score"] is not None for r in rows[kind]))
    want = {(ep, t, a, b) for ep in episodes for t, a, b in met3r_pairs()}
    pairs, seen, extra, duplicate = [], set(), [], []
    for p in sorted((out / "met3r").glob("*.json")):
        rec = dict(read_json(p), kind="met3r")
        key = (rec["episode"], rec["t"], *rec["views"])
        if key not in want or rec.get("method") != method:
            extra.append(p.name)
        elif key in seen:
            duplicate.append(p.name)
        else:
            seen.add(key)
            pairs.append(rec)
    met = aggregate_scenes(pairs, episodes) if pairs else dict(score=None, per_scene={})
    ok = seen == want and not extra and not duplicate and not missing_scenes
    summary["met3r"] = dict(score=met["score"] if ok else None, per_scene=met["per_scene"], pairs=len(pairs),
                            expected_pairs=len(want), missing_pairs=len(want - seen), unexpected_results=extra[:50],
                            duplicate_results=duplicate[:50])
    write_json(out / "summary.json", summary)
    print(json.dumps({k: (v["percent"] if k in ("compliance", "consistency") else v["score"])
                      for k, v in summary.items() if k in ("compliance", "consistency", "met3r")}))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["appearance", "met3r", "summarize"])
    ap.add_argument("--dataset", required=True, help="released multi-view dataset directory (data/multi4)")
    ap.add_argument("--samples", required=True, help="samples.jsonl of one method (all episodes and views)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--protocol", default=None, help="protocol json; its judge.model is used (default Qwen/Qwen3.6-27B)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args(argv)
    if not 0 <= args.worker < args.workers:
        ap.error("--worker must be in [0, --workers)")
    {"appearance": run_appearance, "met3r": run_met3r, "summarize": summarize}[args.command](args)


if __name__ == "__main__":
    main()
