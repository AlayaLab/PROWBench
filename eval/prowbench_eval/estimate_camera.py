"""Camera trajectories of generated videos with ViPE: intrinsics fixed to the recorded camera (adjusted for crop and
resize), dynamic entities masked. Writes prowbench.traj.v1 files."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from .common import EVAL_EXCLUDED_LABELS, GeneratedSample, read_json, write_json
from .detect_track import LABEL_QUERY

TRAJ_SCHEMA = "prowbench.traj.v1"
CV_TO_GL = np.diag([1.0, -1.0, -1.0, 1.0])
# `pose_only`: the default ViPE pipeline without post-hoc depth alignment, set by overrides so that ViPE v1.2.0 needs no
# extra configuration file. Any other name selects configs/pipeline/<name>.yaml of the ViPE checkout.
VIPE_PIPELINES = {"pose_only": ["pipeline=default", "pipeline.post.depth_align_model=null"]}


def positive_depth_summary(depths, min_points=32):
    values = np.asarray(depths, float).reshape(-1)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) < min_points:
        raise ValueError(f"Scene scale unavailable: {len(values)} valid background points, need {min_points}")
    return dict(depth=float(np.median(values)), n_points=len(values),
                p10=float(np.percentile(values, 10)), p90=float(np.percentile(values, 90)))


def slam_scene_scale(xyz_world, c2w_cv):
    """Same-map camera-Z depth; world frame and global reconstruction scale cancel."""
    xyz = np.asarray(xyz_world, float)
    pose = np.asarray(c2w_cv, float)
    camera_points = (xyz - pose[:3, 3]) @ pose[:3, :3]
    return dict(positive_depth_summary(camera_points[:, 2]), frame=0,
                source="vipe_first_keyframe_filtered_static_slam_map",
                shares_trajectory_scale=True)


def intrinsics_for_sample(gt: dict, sample: GeneratedSample) -> list[float]:
    """[fx, fy, cx, cy] of the generated video, from the recorded source intrinsics + crop + resize.
    crop_xywh is in source pixels of the recording; the crop is then resized to (width, height)."""
    K = gt["camera_trajectory"][0]["intrinsics"]
    x, y, w, h = sample.crop_xywh
    sx, sy = sample.width / w, sample.height / h
    return [float(K["fx"] * sx), float(K["fy"] * sy), float((K["cx"] - x) * sx), float((K["cy"] - y) * sy)]


def dynamic_phrases(gt: dict) -> list[str]:
    """Detector phrases for the engine's *dynamic* entities (things that move independently of the camera).
    Interaction candidates that are static (decks, cable, landing platform) are deliberately NOT masked:
    they are exactly the rigid structure SLAM should track. Labels without a detector phrase (`effect`) are
    skipped; text phrases are label-level, so e.g. `platform` also masks a static platform in the same scene."""
    out = ["person"]
    for d in gt["entities"].values():
        lab = d.get("label")
        if not d.get("dynamic") or lab in EVAL_EXCLUDED_LABELS or lab not in LABEL_QUERY:
            continue
        q = LABEL_QUERY[lab]
        if q not in out:
            out.append(q)
    return out


def rotmat_to_quat_xyzw(R: np.ndarray) -> list[float]:
    """Shepperd's method; returns [x, y, z, w] with w >= 0."""
    m = np.asarray(R, float)
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w], float)
    if w < 0:
        q = -q
    return (q / np.linalg.norm(q)).tolist()


def c2w_cv_to_traj(c2w_cv: np.ndarray, fps: float, estimator: dict) -> dict:
    """(N,4,4) OpenCV camera-to-world -> prowbench.traj.v1 (OpenGL convention, per-frame)."""
    frames = []
    for i, M in enumerate(np.asarray(c2w_cv, float)):
        G = M @ CV_TO_GL
        frames.append(dict(frame=i, position=G[:3, 3].tolist(), quaternion=rotmat_to_quat_xyzw(G[:3, :3])))
    return dict(schema=TRAJ_SCHEMA, fps=float(fps), convention="camera_to_world, OpenGL (y up, looks -z), xyzw",
                frames=frames, estimator=estimator)


class VipeEstimator:
    """ViPE camera estimation with fixed intrinsics and masks on the dynamic entities."""

    def __init__(self, vipe_root: str, device: str = "cuda:0", vipe_pipeline: str = "pose_only"):
        self.vipe_root = str(Path(vipe_root).resolve())
        if self.vipe_root not in sys.path:
            sys.path.insert(0, self.vipe_root)
        import torch
        self.torch = torch
        torch.cuda.set_device(torch.device(device))
        from vipe.config.parse import parse_typed_config
        from vipe.pipeline import _as_dictconfig
        from vipe.pipeline.default import DefaultAnnotationPipeline
        from vipe.pipeline.processors import TrackAnythingProcessor
        from vipe.streams.base import FrameAttribute, ProcessedVideoStream, StreamProcessor
        from vipe.utils.cameras import CameraType
        import copy

        typed = parse_typed_config(
            config_name="default", config_dir=str(Path(self.vipe_root) / "configs"),
            hydra_args=VIPE_PIPELINES.get(vipe_pipeline, [f"pipeline={vipe_pipeline}"]) + [
                        "streams=raw_mp4_stream", "streams.base_path=unused",
                        "pipeline.init.intrinsics=gt", "pipeline.output.save_viz=false",
                        "pipeline.output.path=" + str(Path(os.environ.get("TMPDIR", "/tmp")) / "prowbench_vipe_unused")])
        pcfg = copy.deepcopy(_as_dictconfig(typed.pipeline))
        del pcfg.instance
        self.version = _vipe_version(self.vipe_root)
        self.pipeline_name = vipe_pipeline

        class FixedIntrinsicsProcessor(StreamProcessor):
            def __init__(self, K):
                super().__init__()
                self.K = torch.as_tensor(K).float()

            def update_attributes(self, previous_attributes):
                return previous_attributes | {FrameAttribute.INTRINSICS}

            def __call__(self, frame_idx, frame):
                frame.intrinsics = self.K.clone()
                frame.camera_type = CameraType.PINHOLE
                return frame

        outer = self

        class ProwBenchPipeline(DefaultAnnotationPipeline):
            def _add_init_processors(self, video_stream):
                procs = [FixedIntrinsicsProcessor(outer._K)]
                if self.init_cfg.instance is not None:
                    procs.append(TrackAnythingProcessor(
                        outer._phrases, add_sky=self.init_cfg.instance.add_sky,
                        sam_run_gap=int(video_stream.fps() * self.init_cfg.instance.kf_gap_sec),
                        model_cache=self.model_cache))
                return ProcessedVideoStream(video_stream, procs)

        self.pipeline = ProwBenchPipeline(**pcfg)
        self.pipeline.return_payload = True
        self._K, self._phrases = None, None

    def estimate(self, video: str, K: list[float], phrases: list[str], name: str) -> tuple[np.ndarray, dict]:
        from vipe.streams.raw_mp4_stream import RawMp4Stream
        self._K, self._phrases = K, phrases
        stream = RawMp4Stream(Path(video), name=name)
        n = len(stream)
        t0 = time.time()
        out = self.pipeline.run(stream)
        slam = out.payload
        traj = slam.get_view_trajectory(0)
        c2w = traj.matrix().detach().cpu().numpy()
        if c2w.shape[0] != n:
            raise RuntimeError(f"ViPE returned {c2w.shape[0]} poses for {n} frames ({video})")
        intr = slam.intrinsics[0].detach().cpu().numpy().tolist()
        info = dict(name="vipe", version=self.version, pipeline=self.pipeline_name, intrinsics_fixed=[float(k) for k in K],
                    intrinsics_slam=intr, phrases=list(phrases), ba_residual=float(slam.ba_residual),
                    frames=int(n), fps=float(stream.fps()), seconds=round(time.time() - t0, 1))
        try:
            if slam.slam_map is None or 0 not in slam.slam_map.dense_disp_frame_inds:
                raise ValueError("ViPE has no frame-zero SLAM map")
            index = slam.slam_map.dense_disp_frame_inds.index(0)
            xyz, _ = slam.slam_map.get_dense_disp_pcd(index, 0)
            info["scene_scale"] = slam_scene_scale(xyz.detach().cpu().numpy(), c2w[0])
        except ValueError as exc:
            info["scene_scale_error"] = str(exc)
        self.torch.cuda.empty_cache()
        return c2w, info


def _vipe_version(root: str) -> str:
    try:
        import subprocess
        rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True,
                                      stderr=subprocess.DEVNULL).strip()
    except Exception:
        rev = "unknown"
    try:
        import vipe
        v = getattr(vipe, "__version__", None) or "?"
    except Exception:
        v = "?"
    return f"{v}@{rev}"


def make_estimator(backend: str, **kw):
    if backend == "vipe":
        return VipeEstimator(kw["vipe_root"], kw.get("device", "cuda:0"), kw.get("vipe_pipeline", "pose_only"))
    raise ValueError(f"unknown backend {backend!r}")


def run_job(est, job: dict, backend: str) -> str:
    out = Path(job["out"])
    if out.exists() and not job.get("force"):
        return "skip"
    gt = read_json(job["gt"])
    s = job["sample"]
    sample = GeneratedSample.from_dict(s) if isinstance(s, dict) else GeneratedSample.from_dict(read_json(s))
    K = intrinsics_for_sample(gt, sample)
    phrases = dynamic_phrases(gt)
    c2w, info = est.estimate(job["video"], K, phrases, name=sample.sample_id.replace("/", "__"))
    traj = c2w_cv_to_traj(c2w, sample.fps, info)
    traj.update(sample_id=sample.sample_id, method=sample.method, run_id=sample.run_id, video=job["video"])
    write_json(out, traj)
    return "ok"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", help="json list of jobs (see module doc)")
    ap.add_argument("--video"); ap.add_argument("--gt"); ap.add_argument("--sample"); ap.add_argument("--out")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--backend", default="vipe", choices=["vipe"])
    ap.add_argument("--vipe-root", default=os.environ.get("VIPE_ROOT", ""))
    ap.add_argument("--vipe-pipeline", default="pose_only")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if args.jobs:
        jobs = read_json(args.jobs)
    else:
        if not (args.video and args.gt and args.sample and args.out):
            ap.error("--video/--gt/--sample/--out are required without --jobs")
        s = args.sample
        jobs = [dict(video=args.video, gt=args.gt, out=args.out, force=args.force,
                     sample=json.loads(s) if s.strip().startswith("{") else s)]
    pending = [j for j in jobs if args.force or j.get("force") or not Path(j["out"]).exists()]
    print(f"{len(jobs)} jobs, {len(pending)} pending ({args.backend})", flush=True)
    if not pending:
        return
    est = make_estimator(args.backend, vipe_root=args.vipe_root, device=args.device, vipe_pipeline=args.vipe_pipeline)
    n_ok = 0
    for j in pending:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)
        est.torch.manual_seed(args.seed)
        if args.force:
            j = dict(j, force=True)
        st = run_job(est, j, args.backend)
        n_ok += st == "ok"
        print(f"[{st}] {j['out']}", flush=True)
    print(f"done: {n_ok} estimated")


if __name__ == "__main__":
    main()
