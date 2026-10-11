import json
import tempfile
from pathlib import Path

import numpy as np

from prowbench_eval import camera_rectify


def _gt(fx_values):
    return dict(fps=30, frame_count=len(fx_values),
                camera_trajectory=[dict(frame=i, intrinsics=dict(fx=f, fy=f, cx=640, cy=360)) for i, f in enumerate(fx_values)])


def test_constant_intrinsics_need_no_rectification():
    sample = dict(sample_id="ep/follow-a/x", episode="ep", camera="follow-a", video="v.mp4", fps=30, width=1280, height=720,
                  frame_count=3, crop_xywh=[0, 0, 1280, 720])
    ks, target = camera_rectify.per_frame_intrinsics(sample, _gt([700.0] * 3))
    assert not camera_rectify.needs_rectification(ks) and np.allclose(target, [700, 700, 640, 360])
    ks, target = camera_rectify.per_frame_intrinsics(sample, _gt([700.0, 750.0, 800.0]))
    assert camera_rectify.needs_rectification(ks) and target[0] == 800.0
    with tempfile.TemporaryDirectory() as tmp:
        gt_dir = Path(tmp) / "gt" / "ep"
        gt_dir.mkdir(parents=True)
        (gt_dir / "follow-a.json").write_text(json.dumps(_gt([700.0] * 3)))
        jobs, n = camera_rectify.prepare_camera_inputs([sample], Path(tmp) / "gt", Path(tmp) / "work")
        assert n == 0 and jobs[0]["video"] == "v.mp4" and jobs[0]["out"].endswith("camera_traj/ep__follow-a__x.json")
        assert json.loads(Path(jobs[0]["gt"]).read_text())["camera_trajectory"][0]["intrinsics"]["fx"] == 700.0


def test_rectified_ground_truth_uses_target():
    sample = dict(width=1280, height=720, crop_xywh=[0, 0, 1280, 720])
    new = camera_rectify.rectified_ground_truth(sample, _gt([700.0, 800.0]), np.array([800.0, 800.0, 640.0, 360.0]))
    assert all(f["intrinsics"]["fx"] == 800.0 for f in new["camera_trajectory"])


def test_ffmpeg_is_awaited_when_writing_fails():
    from unittest.mock import patch
    import pytest

    class Proc:
        waited = 0
        def __init__(self, *a, **k):
            self.stdin = self
        def write(self, data): pass
        def close(self): pass
        def wait(self):
            Proc.waited += 1
            return 0

    sample = dict(sample_id="ep/follow-a/x", episode="ep", camera="follow-a", fps=30, width=1280, height=720,
                  frame_count=3, crop_xywh=[0, 0, 1280, 720])
    with tempfile.TemporaryDirectory() as tmp, patch.object(camera_rectify.subprocess, "Popen", Proc):
        with pytest.raises(ValueError):            # the video cannot be read
            camera_rectify.rectify_video(str(Path(tmp) / "missing.mp4"), sample, _gt([700.0, 750.0, 800.0]), Path(tmp) / "o.mp4")
    assert Proc.waited == 1
