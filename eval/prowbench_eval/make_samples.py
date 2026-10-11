"""Build the samples.jsonl that the metric commands read (one common.GeneratedSample per generated video).

Generated videos are expected at <videos>/<episode>/<camera>/<representation>/video.mp4, with <videos> =
model_output_video/<dataset>/<method> in the repository, and an optional output.json next to each video
(fps, width, height, frame_count, seed, crop_xywh). Every released view of the dataset that has a video there becomes
one row; views without a video are counted as missing. Video metadata (fps, size, frame count) is read from the file;
values given in output.json must agree with it. `crop_xywh` is the rectangle of the source frame (dataset width x
height, in source pixels) that the generated frame shows:
  auto    (default) crop_xywh from output.json when it has one, otherwise full;
  full    the whole source frame, for outputs resized from the full proxy frame (any aspect ratio);
  center  the largest centered source rectangle with the output's aspect ratio, for methods that center-cropped
          their input to their own aspect ratio;
  x,y,w,h an explicit rectangle.
`extra.reference` is set to the released proxy video of the same view and representation (used by metric_temporal
to bound the number of frame pairs).

With --gt-dir and --work, the job lists of the GPU stages are written as well: <work>/detect_jobs.json for
`detect_track --jobs` (tracks to <work>/preds/<sample key>.json) and <work>/camera_jobs.json for
`estimate_camera --jobs` (trajectories to <work>/camera_traj/<sample key>.json); <sample key> is the sample id with
"/" replaced by "__"."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


from .common import EVAL_DURATION, video_meta, write_text


def crop_for(mode: str, src_w: int, src_h: int, out_w: int, out_h: int) -> list:
    if mode == "full":
        return [0, 0, src_w, src_h]
    if mode == "center":
        aspect = out_w / out_h
        if src_w / src_h > aspect:
            w = src_h * aspect
            return [round((src_w - w) / 2, 1), 0, round(w, 1), src_h]
        h = src_w / aspect
        return [0, round((src_h - h) / 2, 1), src_w, round(h, 1)]
    vals = [float(v) for v in mode.split(",")]
    if len(vals) != 4 or vals[2] <= 0 or vals[3] <= 0:
        raise ValueError(f"crop must be full, center or x,y,w,h (got {mode!r})")
    return [int(v) if v.is_integer() else v for v in vals]


def output_record(video, meta) -> dict:
    """output.json next to a video (empty if absent); its fps / width / height / frame_count must match the file."""
    p = Path(video).with_name("output.json")
    if not p.exists():
        return {}
    rec = json.loads(p.read_text())
    for key in ("width", "height", "frame_count"):
        if key in rec and int(rec[key]) != meta[key]:
            raise ValueError(f"{p}: {key}={rec[key]} but the video has {meta[key]}")
    if "fps" in rec and abs(float(rec["fps"]) - meta["fps"]) > 1e-3:
        raise ValueError(f"{p}: fps={rec['fps']} but the video has {meta['fps']}")
    return rec


def build(dataset, videos, method, run_id, crop="auto", representations=None):
    """(rows, missing): rows in samples.jsonl order of the dataset, missing = sample ids without a video. Raises if a video
    is shorter than the evaluated duration (EVAL_DURATION)."""
    dataset, videos = Path(dataset), Path(videos)
    rows, missing, short = [], [], []
    for line in (dataset / "samples.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rep = r["representation"]
        if representations and rep not in representations:
            continue
        sid = f"{r['episode']}/{r['camera']}/{rep}"
        video = videos / r["episode"] / r["camera"] / rep / "video.mp4"
        if not video.exists():
            missing.append(sid)
            continue
        m = video_meta(video)
        if m["frame_count"] < int(EVAL_DURATION * m["fps"] + 1e-6):
            short.append(f"{sid} ({m['frame_count']} frames at {m['fps']:g} fps)")
            continue
        rec = output_record(video, m)
        if crop == "auto":
            box = list(rec["crop_xywh"]) if rec.get("crop_xywh") else crop_for("full", r["width"], r["height"], m["width"], m["height"])
        else:
            box = crop_for(crop, r["width"], r["height"], m["width"], m["height"])
        row = dict(sample_id=sid, episode=r["episode"], camera=r["camera"], representation=rep, method=method,
                   run_id=run_id, video=str(video.resolve()), fps=m["fps"], width=m["width"], height=m["height"],
                   frame_count=m["frame_count"], crop_xywh=box, extra=dict(reference=str((dataset / r["video"]).resolve())))
        if rec.get("seed") is not None:
            row["seed"] = rec["seed"]
        rows.append(row)
    if short:
        raise ValueError(f"{len(short)} videos are shorter than {EVAL_DURATION:g} s: " + ", ".join(short[:5])
                         + (" ..." if len(short) > 5 else ""))
    return rows, missing


def sample_key(row) -> str:
    return row["sample_id"].replace("/", "__")


def stage_jobs(rows, gt_dir, work, detect_entities=None):
    """(detect jobs, camera jobs) in the job format of detect_track --jobs / estimate_camera --jobs. `detect_entities`
    sets the tracked entity selection of every detect job (default of detect_track: dynamic_or_candidate)."""
    gt_dir, work = Path(gt_dir), Path(work)
    detect, camera = [], []
    for r in rows:
        gt = gt_dir / r["episode"] / f"{r['camera']}.json"
        if not gt.exists():
            raise FileNotFoundError(f"no exported ground truth {gt} (run gt_export first)")
        job = dict(video=r["video"], gt=str(gt), out=str(work / "preds" / f"{sample_key(r)}.json"), sample=r)
        if detect_entities:
            job["entities"] = detect_entities
        detect.append(job)
        camera.append(dict(video=r["video"], gt=str(gt), out=str(work / "camera_traj" / f"{sample_key(r)}.json"), sample=r))
    return detect, camera


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="released dataset directory (data/<dataset>)")
    ap.add_argument("--videos", required=True, help="directory with <episode>/<camera>/<representation>/video.mp4")
    ap.add_argument("--method", required=True)
    ap.add_argument("--run-id", default="default", help="label stored in every result; distinguishes several runs of one method")
    ap.add_argument("--crop", default="auto", help="auto (default: output.json, else full), full, center, or x,y,w,h in source pixels")
    ap.add_argument("--representations", nargs="*", default=None,
                    help="representations the method was run on (default: every representation that has a video)")
    ap.add_argument("--out", required=True, help="samples.jsonl to write")
    ap.add_argument("--gt-dir", default=None, help="exported ground truth <gt-dir>/<episode>/<camera>.json (with --work)")
    ap.add_argument("--work", default=None, help="directory for detect_jobs.json / camera_jobs.json (with --gt-dir)")
    ap.add_argument("--detect-entities", choices=["dynamic_or_candidate", "dynamic", "all"], default=None,
                    help="entity selection of the detect jobs (dynamic for the long-horizon dataset)")
    args = ap.parse_args(argv)
    if bool(args.gt_dir) != bool(args.work):
        ap.error("--gt-dir and --work go together")
    rows, missing = build(args.dataset, args.videos, args.method, args.run_id, args.crop, args.representations)
    if not rows:
        raise SystemExit(f"no generated videos found under {args.videos}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_text(args.out, "".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} samples -> {args.out}; {len(missing)} dataset rows without a video")
    if args.work:
        detect, camera = stage_jobs(rows, args.gt_dir, args.work, args.detect_entities)
        Path(args.work).mkdir(parents=True, exist_ok=True)
        write_text(Path(args.work, "detect_jobs.json"), json.dumps(detect, indent=1))
        write_text(Path(args.work, "camera_jobs.json"), json.dumps(camera, indent=1))
        print(f"{len(detect)} jobs -> {Path(args.work, 'detect_jobs.json')}, {Path(args.work, 'camera_jobs.json')}")


if __name__ == "__main__":
    main()
