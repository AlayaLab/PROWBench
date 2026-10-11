"""Command-line driver for the windowed VLM-judge metrics: Interaction Success Rate (interaction_windowed) and
Logic-Render Alignment (alignment_windowed), one JSON result per generated sample.

Inputs (all supplied by the caller):
  --samples       JSONL of generated samples (common.GeneratedSample fields)
  --gt-dir        exported ground truth, <gt-dir>/<episode>/<camera>.json (gt_export)
  --protocol      JSON with `judge` (model, max_new_tokens, enable_thinking), `interaction` (frames, crop_margin) and
                  `alignment` (frames, window_pad, crop_margin, end_state_frames, end_state_seconds, static_frames,
                  max_static_facts, seed, optional foil and foil_offset) sections, as in protocol/<dataset>/settings.json
  ISR:  --registry (event_registry output) and --visual-events (event groups per episode: protocol/<dataset>/events.json)
  LRA:  --descriptions, the description records: protocol/<dataset>/descriptions.json, or a directory with
        episodes/<episode>/<camera>/<representation>/video.description.json (falls back to the `white` representation);
        the negative-control foil is chosen among its episodes.

A sample whose judging raises is written with validation_status="failed" and is re-run on the next invocation;
existing successful results are kept. Individual failing judge calls inside a sample are recorded as invalid
answers and stay in the denominator of the sample score.
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from .alignment_windowed import evaluate_sample as alignment_sample, foil_episode, load_timeline
from .common import read_json, read_jsonl, write_json
from .interaction_windowed import evaluate_sample as interaction_sample, load_visual_events
from .vlm_judge import DEFAULT_JUDGE_MODEL, Judge


_TABLES = {}


def _description_table(root):
    st = Path(root).stat()
    key = (str(Path(root).resolve()), st.st_mtime_ns, st.st_size)
    if key not in _TABLES:
        _TABLES.clear()
        _TABLES[key] = read_json(root)["episodes"]
    return _TABLES[key]


def read_description(root, sample, episode=None):
    """Description record of `episode` (default: the sample's) seen from the sample's camera, from a descriptions file
    ({"episodes": {episode: {camera: record}}}) or a description directory."""
    ep = episode or sample["episode"]
    if Path(root).is_file():
        views = _description_table(root)[ep]
        if sample["camera"] not in views:
            raise KeyError(f"no description of {ep}/{sample['camera']} in {root}")
        return views[sample["camera"]]
    return read_json(description_path(root, sample, episode))


def description_path(root, sample, episode=None):
    """Description of `episode` (default: the sample's) seen from the sample's camera; descriptions are identical
    across representations, so a missing representation falls back to `white`."""
    ep = episode or sample["episode"]
    p = Path(root) / "episodes" / ep / sample["camera"] / sample["representation"] / "video.description.json"
    if not p.exists():
        p = Path(root) / "episodes" / ep / sample["camera"] / "white" / "video.description.json"
    return p


def description_episodes(root):
    if Path(root).is_file():
        return sorted(_description_table(root))
    return sorted(d for d in os.listdir(Path(root) / "episodes") if (Path(root) / "episodes" / d).is_dir())


_FP_NOTE = re.compile(r"\s*\(First-person view:.*?\)\s*$")


def _scene_type(ep):
    return re.sub(r"^[a-z0-9]+-\d+-", "", ep)


def _timeline_sentences(desc):
    return {_FP_NOTE.sub("", s["text"]).strip().lower() for s in load_timeline(desc)}


def foil_for(root, sample, mode="next_episode", offset=None):
    """Negative-control episode and its description.
    next_episode: the episode `offset` places later in sorted order (default half the list).
    distinct_timeline: the same start, then the first episode (cyclic, sorted order) of another scene type none of
    whose timeline segment texts appears in the sample's own timeline."""
    if mode == "none":
        return None, None
    eps = description_episodes(root)
    ep = foil_episode(eps, sample["episode"], offset)
    if mode == "distinct_timeline":
        own = _timeline_sentences(read_description(root, sample))
        k = eps.index(ep)
        for j in range(len(eps)):
            cand = eps[(k + j) % len(eps)]
            if cand != sample["episode"] and _scene_type(cand) != _scene_type(sample["episode"]) \
                    and not own & _timeline_sentences(read_description(root, sample, cand)):
                ep = cand
                break
        else:
            raise ValueError(f"no distinct-timeline foil for {sample['episode']}")
    elif mode != "next_episode":
        raise ValueError(f"unknown foil mode {mode!r}")
    return ep, read_description(root, sample, ep)


def make_judge(protocol, device, cache_dir):
    cfg = protocol.get("judge", {})
    return Judge(model=cfg.get("model") or DEFAULT_JUDGE_MODEL, device=device, cache_dir=cache_dir,
                 max_new_tokens=cfg.get("max_new_tokens", 2048), enable_thinking=cfg.get("enable_thinking", False))


def run(args, judge=None):
    protocol = read_json(args.protocol)
    samples = list(read_jsonl(args.samples))[args.worker::args.workers]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    registry = read_json(args.registry) if args.metric == "interaction" else None
    if judge is None:
        import torch
        torch.manual_seed(protocol.get("alignment", {}).get("seed", 0))
        judge = make_judge(protocol, args.device, args.judge_cache or str(out / ".judge_cache"))
    done = failed = 0
    for s in samples:
        dest = out / f"{s['sample_id'].replace('/', '__')}.{args.metric}.json"
        if dest.exists() and read_json(dest).get("validation_status") != "failed":
            continue
        gt = read_json(Path(args.gt_dir) / s["episode"] / f"{s['camera']}.json")
        try:
            if args.metric == "interaction":
                icfg = protocol.get("interaction", {})
                events = load_visual_events(args.visual_events, s["episode"], registry)
                result = interaction_sample(s, events, gt, judge, int(icfg.get("frames", 8)), float(icfg.get("crop_margin", 0.35)))
            else:
                acfg = protocol.get("alignment", {})
                desc = read_description(args.descriptions, s)
                foil_name, foil_desc = foil_for(args.descriptions, s, acfg.get("foil", "next_episode"), acfg.get("foil_offset"))
                result = alignment_sample(s, desc, gt, judge, acfg, foil_desc, foil_name)
            done += 1
        except Exception as exc:
            result = dict(sample_id=s["sample_id"], method=s["method"], run_id=s["run_id"], validation_status="failed",
                          error=repr(exc))
            failed += 1
        write_json(dest, result)
        summary = result.get("summary") or {}
        key = "interaction_success" if args.metric == "interaction" else "dynamic_checklist"
        state = "failed" if result.get("validation_status") == "failed" else "ok"
        print(f"{s['sample_id']}: {state} {key}={summary.get(key)}", flush=True)
    print(f"{args.metric}: {done} judged, {failed} failed, worker {args.worker}/{args.workers}")
    return done, failed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("metric", choices=["interaction", "alignment"])
    ap.add_argument("--samples", required=True)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--out", required=True, help="output dir; writes <sample key>.<metric>.json")
    ap.add_argument("--registry", help="event registry json (interaction)")
    ap.add_argument("--visual-events", help="event groups json, protocol/<dataset>/events.json (interaction)")
    ap.add_argument("--descriptions", help="description records: protocol/<dataset>/descriptions.json or a directory with "
                                           "episodes/<episode>/<camera>/<representation>/ (alignment)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--judge-cache", default=None, help="judge response cache (default <out>/.judge_cache)")
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args(argv)
    if args.metric == "interaction" and not (args.registry and args.visual_events):
        ap.error("interaction needs --registry and --visual-events")
    if args.metric == "alignment" and not args.descriptions:
        ap.error("alignment needs --descriptions")
    if not 0 <= args.worker < args.workers:
        ap.error("--worker must be in [0, --workers)")
    _, failed = run(args)
    if failed:
        raise SystemExit(f"{failed} sample(s) failed; their results are written with validation_status=\"failed\" and are "
                         "re-run on the next call")


if __name__ == "__main__":
    main()
