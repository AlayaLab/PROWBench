"""Camera-only rectification of views whose recorded intrinsics change over time (zoom).

ViPE pose estimation assumes one pinhole camera per clip. For a view whose focal length changes, every
generated frame is resampled into a constant pinhole with the largest recorded focal length (a center crop
for the zoomed-out frames), with bilinear spatial sampling and lossless RGB encoding. Resolution, frame rate,
frame count and camera poses are unchanged; nothing is padded or interpolated in time. The rectified video and
the matching ground truth (constant intrinsics) are used for camera estimation and camera scoring only; all
other metrics use the native video.

Command line: for every sample of a samples.jsonl, writes <work>/camera_gt/<key>.json (the rectified ground truth for a
zoom view, otherwise a copy of the exported one), the rectified video <work>/rectified/<key>.mp4 for zoom views, and
<work>/camera_jobs.json for `estimate_camera --jobs` (trajectories to <work>/camera_traj/<key>.json). Score the
trajectories with `metric_camera --gt <work>/camera_gt/<key>.json`. <key> is the sample id with "/" replaced by "__".
"""
import argparse
import copy
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from .common import write_text


def per_frame_intrinsics(sample, gt):
    """Intrinsics of every generated frame in output pixels, from the nearest ground-truth frame.

    Returns `(ks, target)`: an `(n, 4)` array of `fx, fy, cx, cy` and the constant target `fx, fy, cx, cy`
    (largest focal lengths, principal point of frame 0).
    """
    sx, sy = sample["width"] / sample["crop_xywh"][2], sample["height"] / sample["crop_xywh"][3]
    x, y = sample["crop_xywh"][:2]
    ks = []
    for n in range(sample["frame_count"]):
        f = min(round(n / sample["fps"] * gt["fps"]), gt["frame_count"] - 1)
        k = gt["camera_trajectory"][f]["intrinsics"]
        ks.append([k["fx"] * sx, k["fy"] * sy, (k["cx"] - x) * sx, (k["cy"] - y) * sy])
    ks = np.array(ks)
    if not (np.ptp(ks[:, 2:], axis=0).max() < 1e-8):
        raise ValueError('camera_rectify: check failed: np.ptp(ks[:, 2:], axis=0).max() < 1e-8')
    target = ks[0].copy()
    target[:2] = ks[:, :2].max(axis=0)
    return ks, target


def needs_rectification(ks):
    """True when the intrinsics change within the clip."""
    return bool(np.max(np.ptp(ks, axis=0)) >= 1e-7)


def rectified_ground_truth(sample, gt, target):
    """Copy of the ground truth with every frame's intrinsics set to the target (in source pixels)."""
    new = copy.deepcopy(gt)
    x, y, cw, ch = sample["crop_xywh"]
    sx, sy = sample["width"] / cw, sample["height"] / ch
    tk = dict(fx=target[0] / sx, fy=target[1] / sy, cx=target[2] / sx + x, cy=target[3] / sy + y)
    for frame in new["camera_trajectory"]:
        frame["intrinsics"] = tk
    return new


def rectify_frame(image, k, target):
    """Resample one BGR frame with intrinsics `k` into the target pinhole."""
    import cv2
    height, width = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    mx = ((xx - target[2]) * k[0] / target[0] + k[2]).astype("float32")
    my = ((yy - target[3]) * k[1] / target[1] + k[3]).astype("float32")
    if not (mx.min() >= -1e-4 and mx.max() <= width - 1 + 1e-4):
        raise ValueError('camera_rectify: check failed: mx.min() >= -1e-4 and mx.max() <= width - 1 + 1e-4')
    if not (my.min() >= -1e-4 and my.max() <= height - 1 + 1e-4):
        raise ValueError('camera_rectify: check failed: my.min() >= -1e-4 and my.max() <= height - 1 + 1e-4')
    return cv2.remap(image, mx, my, cv2.INTER_LINEAR)


def rectify_video(video, sample, gt, out_path, ffmpeg="ffmpeg"):
    """Write the rectified video to `out_path`; return the rectified ground truth, or None if no zoom.

    The video is encoded losslessly (libx264rgb, crf 0) at the native size and frame rate.
    """
    import cv2
    ks, target = per_frame_intrinsics(sample, gt)
    if not needs_rectification(ks):
        return None
    width, height = sample["width"], sample["height"]
    cap = cv2.VideoCapture(video)
    proc = subprocess.Popen([ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
                             "-r", str(sample["fps"]), "-i", "-", "-an", "-c:v", "libx264rgb", "-crf", "0", "-preset", "veryfast",
                             "-threads", "2", str(out_path)], stdin=subprocess.PIPE)
    try:
        try:
            for i, k in enumerate(ks):
                ok, image = cap.read()
                if not (ok):
                    raise ValueError((video, i))
                proc.stdin.write(rectify_frame(image, k, target).tobytes())
            if cap.read()[0]:
                raise ValueError("video has more frames than frame_count")
        finally:
            cap.release()
            proc.stdin.close()
    finally:
        rc = proc.wait()                 # also when writing failed; the original error is raised after it
    if rc != 0:
        raise RuntimeError(f"ffmpeg exited with status {rc} while writing {out_path}")
    return rectified_ground_truth(sample, gt, target)


def prepare_camera_inputs(samples, gt_dir, work, ffmpeg="ffmpeg"):
    """Camera-estimation jobs of the samples (see the module docstring); returns (jobs, number of rectified views)."""
    work = Path(work)
    jobs, n_rect = [], 0
    for s in samples:
        key = s["sample_id"].replace("/", "__")
        gt = json.loads((Path(gt_dir) / s["episode"] / f"{s['camera']}.json").read_text())
        video = work / "rectified" / f"{key}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        rect = rectify_video(s["video"], s, gt, video, ffmpeg=ffmpeg)
        cam_gt = work / "camera_gt" / f"{key}.json"
        cam_gt.parent.mkdir(parents=True, exist_ok=True)
        write_text(cam_gt, json.dumps(rect if rect is not None else gt))
        sample = dict(s, video=str(video)) if rect is not None else s
        n_rect += rect is not None
        jobs.append(dict(video=sample["video"], gt=str(cam_gt), out=str(work / "camera_traj" / f"{key}.json"), sample=sample))
    return jobs, n_rect


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--gt-dir", required=True, help="exported ground truth <gt-dir>/<episode>/<camera>.json")
    ap.add_argument("--work", required=True)
    ap.add_argument("--ffmpeg", default=os.environ.get("FFMPEG", "ffmpeg"))
    a = ap.parse_args(argv)
    samples = [json.loads(line) for line in Path(a.samples).read_text().splitlines() if line.strip()]
    jobs, n_rect = prepare_camera_inputs(samples, a.gt_dir, a.work, a.ffmpeg)
    write_text(Path(a.work, "camera_jobs.json"), json.dumps(jobs, indent=1))
    print(f"{len(jobs)} camera jobs -> {Path(a.work, 'camera_jobs.json')}; {n_rect} zoom views rectified")


if __name__ == "__main__":
    main()
