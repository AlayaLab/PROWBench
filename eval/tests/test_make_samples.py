import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from prowbench_eval import make_samples


@pytest.fixture(autouse=True)
def half_second_clips(monkeypatch):
    """the synthetic videos of these tests are 0.5 s long"""
    monkeypatch.setattr(make_samples, "EVAL_DURATION", 0.5)


def _video(path, w, h, n, fps):
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(n):
        out.write(np.full((h, w, 3), i * 10 % 255, np.uint8))
    out.release()


def test_crop_modes():
    assert make_samples.crop_for("full", 1280, 720, 1344, 768) == [0, 0, 1280, 720]
    assert make_samples.crop_for("center", 1280, 720, 1280, 704) == [0, 8.0, 1280, 704.0]
    assert make_samples.crop_for("center", 1280, 720, 960, 544) == [4.7, 0, 1270.6, 720]
    assert make_samples.crop_for("16,0,1248,720", 1280, 720, 1344, 768) == [16, 0, 1248, 720]


def test_build_rows_and_missing():
    with tempfile.TemporaryDirectory() as tmp:
        ds, vids = Path(tmp) / "ds", Path(tmp) / "out"
        ds.mkdir()
        rows = [dict(episode="ep", camera="follow", representation=r, width=1280, height=720,
                     video=f"episodes/ep/follow/{r}/video.mp4") for r in ("white", "colored-obb")]
        (ds / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        _video(vids / "ep" / "follow" / "colored-obb" / "video.mp4", 64, 36, 12, 24)
        out, missing = make_samples.build(ds, vids, "m", "r1")
        assert missing == ["ep/follow/white"]
        assert len(out) == 1 and out[0]["sample_id"] == "ep/follow/colored-obb"
        assert out[0]["fps"] == 24.0 and out[0]["frame_count"] == 12 and out[0]["crop_xywh"] == [0, 0, 1280, 720]
        assert out[0]["extra"]["reference"].endswith("episodes/ep/follow/colored-obb/video.mp4")
        from prowbench_eval.common import GeneratedSample
        assert GeneratedSample.from_dict(out[0]).frame_count == 12


def test_stage_jobs():
    with tempfile.TemporaryDirectory() as tmp:
        gt = Path(tmp) / "gt" / "ep"
        gt.mkdir(parents=True)
        (gt / "follow.json").write_text("{}")
        row = dict(sample_id="ep/follow/white", episode="ep", camera="follow", video="v.mp4")
        detect, camera = make_samples.stage_jobs([row], Path(tmp) / "gt", Path(tmp) / "work")
        assert detect[0]["out"].endswith("work/preds/ep__follow__white.json") and detect[0]["sample"] == row
        assert camera[0]["out"].endswith("work/camera_traj/ep__follow__white.json")
        try:
            make_samples.stage_jobs([dict(row, camera="first-person")], Path(tmp) / "gt", Path(tmp) / "work")
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("missing ground truth accepted")


def test_output_json_crop_and_consistency():
    with tempfile.TemporaryDirectory() as tmp:
        ds, vids = Path(tmp) / "ds", Path(tmp) / "out"
        ds.mkdir()
        (ds / "samples.jsonl").write_text(json.dumps(dict(episode="ep", camera="follow", representation="white", width=832,
                                                          height=480, video="episodes/ep/follow/white/video.mp4")) + "\n")
        v = vids / "ep" / "follow" / "white" / "video.mp4"
        _video(v, 64, 36, 12, 16)
        rows, _ = make_samples.build(ds, vids, "m", "r1")
        assert rows[0]["crop_xywh"] == [0, 0, 832, 480] and "seed" not in rows[0]
        v.with_name("output.json").write_text(json.dumps(dict(fps=16, width=64, height=36, frame_count=12, seed=7,
                                                              crop_xywh=[16, 0, 800, 480])))
        rows, _ = make_samples.build(ds, vids, "m", "r1")
        assert rows[0]["crop_xywh"] == [16, 0, 800, 480] and rows[0]["seed"] == 7
        assert make_samples.build(ds, vids, "m", "r1", crop="full")[0][0]["crop_xywh"] == [0, 0, 832, 480]
        v.with_name("output.json").write_text(json.dumps(dict(fps=24, width=64, height=36, frame_count=12)))
        try:
            make_samples.build(ds, vids, "m", "r1")
        except ValueError:
            pass
        else:
            raise AssertionError("inconsistent output.json accepted")


def test_rejects_videos_shorter_than_the_clip():
    with tempfile.TemporaryDirectory() as tmp:
        ds, vids = Path(tmp) / "ds", Path(tmp) / "out"
        ds.mkdir()
        row = dict(episode="ep", camera="follow", representation="white", width=1280, height=720,
                   video="episodes/ep/follow/white/video.mp4")
        (ds / "samples.jsonl").write_text(json.dumps(row) + "\n")
        _video(vids / "ep" / "follow" / "white" / "video.mp4", 64, 36, 11, 24)
        with pytest.raises(ValueError, match="shorter than 0.5 s"):
            make_samples.build(ds, vids, "m", "r1")
