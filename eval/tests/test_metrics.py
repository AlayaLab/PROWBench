"""Tests of the entity, camera and VBench metric modules and of the ground-truth export."""
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prowbench_eval import common, metric_entity, metric_camera, metric_vbench, gt_export


def test_iou_and_hungarian():
    assert abs(common.iou_xyxy([0, 0, 1, 1], [0, 0, 1, 1]) - 1) < 1e-9
    assert abs(common.iou_xyxy([0, 0, 1, 1], [0.5, 0, 1.5, 1]) - 1 / 3) < 1e-9
    cost = np.array([[1, 0.2], [0.1, 1]])
    r, c = common.hungarian(cost)
    assert list(zip(r.tolist(), c.tolist())) == [(0, 1), (1, 0)]
    r2, c2 = common._hungarian_fallback(cost)
    assert list(zip(r2.tolist(), c2.tolist())) == [(0, 1), (1, 0)]


def test_spatial_map_crop_resize():
    sp = common.SpatialMap(832, 480, (16, 0, 1248, 720))
    b = sp.box_src_to_out_norm([16, 0, 1264, 720])
    assert np.allclose(b, [0, 0, 1, 1])
    sp2 = common.SpatialMap(1024, 576)
    assert np.allclose(sp2.box_src_to_out_norm([640, 360, 1280, 720]), [0.5, 0.5, 1, 1])


def test_eval_grid_and_frame_mapping():
    g = common.eval_grid(8.0)
    assert len(g) == 40 and g[0] == 0 and g[-1] < 5
    assert common.source_frame_for_time(4.999) == 149
    assert common.output_frame_for_time(2.5, 16) == 40 and common.output_frame_for_time(2.5, 24) == 60


def test_projection_matches_dataset_reader():
    cam = dict(worldToCamera=[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -5, 1],
               intrinsics=dict(fx=1000, fy=1000, cx=640, cy=360), near=0.1, far=100)
    pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    pix, depth = gt_export.project_points(cam, pts)
    assert np.allclose(depth, 5)
    assert np.allclose(pix[0], [640, 360]) and np.allclose(pix[1], [840, 360]) and np.allclose(pix[2], [640, 160])
    box, n = gt_export.box_from_corners(cam, np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float))
    assert n == 8 and np.allclose(box, [390, 110, 890, 610], atol=1e-6)


def _fake_gt(n_frames=150, W=1280, H=720):
    frames = []
    for i in range(n_frames):
        x = 100 + i * 4
        frames.append(dict(frame=i, time=i / 30, phase="push", boxes={
            "player": dict(box=[x, 200, x + 100, 500], visible=True),
            "crate": dict(box=[x + 150, 300, x + 250, 400], visible=i < 120),
            "tree": dict(box=[10, 10, 200, 300], visible=True)}))
    return dict(schema="prowbench.gt.v1", episode="fake", camera="follow", fps=30, frame_count=n_frames, width=W, height=H,
                entities={"player": dict(label="person", dynamic=True, interaction_candidate=True),
                          "crate": dict(label="crate", dynamic=True, interaction_candidate=True),
                          "tree": dict(label="tree", dynamic=False, interaction_candidate=False)},
                phases=[dict(phase="push", start_frame=0, end_frame_exclusive=n_frames, start_time=0, end_time=5)],
                events=[], frames=frames, camera_trajectory=[])


def _fake_pred_from_gt(gt, sample, jitter=0.0, swap_at=None, drop_crate=False):
    sp = sample.spatial
    frames = []
    for j in range(sample.frame_count):
        t = j / sample.fps
        sf = common.source_frame_for_time(t)
        dets = []
        for e, tid, lab in (("player", 1, "person"), ("crate", 2, "wooden crate")):
            b = gt["frames"][sf]["boxes"][e]
            if not b["visible"] or (drop_crate and e == "crate"):
                continue
            nb = sp.box_src_to_out_norm(b["box"])
            box = [nb[0] * sample.width, nb[1] * sample.height, nb[2] * sample.width, nb[3] * sample.height]
            box = [v + jitter for v in box]
            if swap_at is not None and t >= swap_at:
                tid = 3 - tid
            dets.append(dict(track_id=tid, label=lab, score=0.9, box=box))
        frames.append(dict(frame=j, detections=dets))
    return dict(schema="prowbench.tracks.v1", width=sample.width, height=sample.height, fps=sample.fps,
                frame_count=sample.frame_count, frames=frames)


def _sample(fps=24, w=1024, h=576, crop=(0, 0, 1280, 720)):
    return common.GeneratedSample("fake/follow/mixed", "fake", "follow", "mixed", "m", "r", "/dev/null", fps, w, h,
                                  int(fps * 5), crop)


def test_entity_excluded_labels_and_query_table():
    from prowbench_eval import detect_track
    gt = _fake_gt()
    gt["entities"]["exhaust"] = dict(label="effect", dynamic=True, interaction_candidate=False)
    gt["entities"]["deco"] = dict(label="geometry", dynamic=False, interaction_candidate=False)
    gt["entities"]["trolley"] = dict(label="handle", dynamic=True, interaction_candidate=True)
    assert metric_entity.select_entities(gt) == ["player", "crate", "trolley"]
    assert set(metric_entity.select_entities(gt, "all")) == {"player", "crate", "tree", "trolley"}
    assert detect_track.queries_from_gt(gt) == ["person", "wooden crate", "zipline trolley handle"]
    gt["entities"]["x"] = dict(label="dragon", dynamic=True)
    try:
        detect_track.queries_from_gt(gt); raise AssertionError("unmapped label must raise")
    except KeyError:
        pass
    m = np.zeros((10, 10), bool); m[2:5, 3:7] = True
    assert detect_track.mask_to_box(m) == [3.0, 2.0, 7.0, 5.0] and detect_track.mask_to_box(np.zeros((4, 4), bool)) is None


def test_entity_perfect_prediction_gives_iou_one():
    gt = _fake_gt()
    for s in (_sample(), _sample(16, 832, 480, (16, 0, 1248, 720)), _sample(24, 1344, 768)):
        res = metric_entity.evaluate(gt, _fake_pred_from_gt(gt, s), s)
        assert res["summary"]["n_entities"] == 2
        assert res["summary"]["mean_iou"] > 0.999, res["summary"]
        assert res["summary"]["miss_rate"] == 0
        assert res["summary"]["mean_center_err"] < 1e-9


def test_entity_missing_counts_zero_and_miss_rate():
    gt = _fake_gt(); s = _sample()
    res = metric_entity.evaluate(gt, _fake_pred_from_gt(gt, s, drop_crate=True), s)
    pe = res["per_entity"]
    assert pe["crate"]["track_id"] is None and pe["crate"]["mean_iou"] == 0 and pe["crate"]["miss_rate"] == 1
    assert pe["player"]["mean_iou"] > 0.999
    assert abs(res["summary"]["mean_iou"] - 0.5) < 1e-6


def test_entity_identity_swap_is_penalised():
    gt = _fake_gt(); s = _sample()
    res = metric_entity.evaluate(gt, _fake_pred_from_gt(gt, s, swap_at=2.5), s)
    assert 0.3 < res["summary"]["mean_iou"] < 0.75, res["summary"]


def test_entity_jitter_reduces_iou_monotonically():
    gt = _fake_gt(); s = _sample()
    ious = [metric_entity.evaluate(gt, _fake_pred_from_gt(gt, s, jitter=j), s)["summary"]["mean_iou"] for j in (0, 10, 40)]
    assert ious[0] > ious[1] > ious[2]


def _circle_traj(n, r=3.0, scale=1.0, rot=None, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    fr = []
    for i in range(n):
        a = 2 * math.pi * i / n * 0.5
        p = np.array([r * math.cos(a), 1.5, r * math.sin(a)]) * scale
        if rot is not None:
            p = rot @ p
        p = p + rng.normal(0, noise, 3)
        yaw = math.atan2(-p[0], -p[2]) if abs(p[0]) + abs(p[2]) > 0 else 0
        q = [0, math.sin(yaw / 2), 0, math.cos(yaw / 2)]
        fr.append(dict(frame=i, position=p.tolist(), quaternion=q))
    return fr


def test_camera_sim3_invariance():
    gt = dict(camera_trajectory=_circle_traj(150))
    R = metric_camera.quat_to_R([0, math.sin(0.7), 0, math.cos(0.7)])
    pred = dict(fps=24, frames=_circle_traj(120, scale=0.37, rot=R))
    for f in pred["frames"]:
        Rq = R @ metric_camera.quat_to_R(f["quaternion"])
        yaw = math.atan2(Rq[0, 2], Rq[2, 2]); f["quaternion"] = [0, math.sin(yaw / 2), 0, math.cos(yaw / 2)]
    res = metric_camera.evaluate(gt, pred, 24, protocol="ate")["summary"]
    assert res["ate_m"] < 0.03 and abs(res["sim3_scale"] * 0.37 - 1) < 0.01, res
    assert res["rot_err_offset_free_deg_mean"] < 1.0
    noisy = dict(fps=24, frames=_circle_traj(120, noise=0.05))
    assert metric_camera.evaluate(gt, noisy, 24, protocol="ate")["summary"]["ate_m"] > 0.02


def test_camera_cami2v_metrics_scale_and_frame_invariance():
    gt = dict(camera_trajectory=_circle_traj(150))
    instants = common.eval_grid(8)
    Pg = np.array([gt["camera_trajectory"][common.source_frame_for_time(t)]["position"] for t in instants])
    Qg = np.array([gt["camera_trajectory"][common.source_frame_for_time(t)]["quaternion"] for t in instants])
    perfect = metric_camera.cami2v_metrics(Pg, Qg, Pg, Qg, static=False)
    assert perfect["RotErr"] < 1e-6 and perfect["TransErr_rel"] < 1e-6 and perfect["CamMC_abs"] < 1e-6, perfect
    R = metric_camera.quat_to_R([0.3, 0.2, 0.1, 0.9])
    Pp = (0.37 * (R @ Pg.T)).T + np.array([5, -2, 1])
    from prowbench_eval import estimate_camera
    Qp = np.array([estimate_camera.rotmat_to_quat_xyzw(R @ metric_camera.quat_to_R(q)) for q in Qg])
    r = metric_camera.cami2v_metrics(Pg, Qg, Pp, Qp, static=False)
    assert r["RotErr"] < 1e-5 and r["TransErr_rel"] < 1e-5 and r["CamMC_rel"] < 1e-5, r
    assert r["TransErr_abs"] > 0.1, r
    assert metric_camera.cami2v_metrics(Pg, Qg, Pg, Qg, static=True)["TransErr_rel"] < 1e-6


def test_estimate_camera_intrinsics_quaternion_and_convention():
    from prowbench_eval import estimate_camera
    K = dict(fx=691.55, fy=691.55, cx=640, cy=360)
    gt = dict(camera_trajectory=[dict(intrinsics=K)])
    s = common.GeneratedSample(sample_id="e/c/r", episode="e", camera="c", representation="r", method="m", run_id="x",
                               video="v.mp4", fps=16, width=832, height=480, frame_count=81, crop_xywh=(16, 0, 1248, 720))
    fx, fy, cx, cy = estimate_camera.intrinsics_for_sample(gt, s)
    assert abs(fx - 691.55 * 832 / 1248) < 1e-6 and abs(fy - 691.55 * 480 / 720) < 1e-6
    assert abs(cx - (640 - 16) * 832 / 1248) < 1e-6 and abs(cy - 240) < 1e-6
    for q in ([0, 0, 0, 1], [0.3, 0.2, 0.1, 0.9], [-0.7, 0.1, 0.5, 0.2], [0.9, 0.05, -0.3, 0.1]):
        Rq = metric_camera.quat_to_R(q)
        assert np.allclose(metric_camera.quat_to_R(estimate_camera.rotmat_to_quat_xyzw(Rq)), Rq, atol=1e-9)
    gt = dict(camera_trajectory=_circle_traj(150))
    c2w_gl = np.array([np.block([[metric_camera.quat_to_R(f["quaternion"]), np.array(f["position"])[:, None]], [np.zeros((1, 3)), 1]])
                       for f in gt["camera_trajectory"]])
    c2w_cv = c2w_gl @ np.linalg.inv(estimate_camera.CV_TO_GL)
    traj = estimate_camera.c2w_cv_to_traj(c2w_cv, 30, dict(name="test"))
    assert traj["schema"] == "prowbench.traj.v1" and len(traj["frames"]) == 150
    res = metric_camera.evaluate(gt, traj, 30, protocol="ate")["summary"]
    assert res["ate_m"] < 1e-6 and res["rot_err_deg_mean"] < 1e-6 and res["RotErr"] < 1e-6, res
    assert abs(res["sim3_scale"] - 1) < 1e-6


def test_spearman_and_kendall():
    x = [1, 2, 3, 4, 5]; y = [2, 3, 4, 5, 7]
    assert abs(metric_vbench.spearman(x, y)[0] - 1) < 1e-9 and abs(metric_vbench.kendall(x, y) - 1) < 1e-9
    assert abs(metric_vbench.spearman(x, y[::-1])[0] + 1) < 1e-9


def test_vbench_paired_uses_intersection_only():
    rows = [dict(method=meth, run_id="r", sample_id=sid, representation=sid.split("/")[2], **{"vbench.imaging_quality": val})
            for meth, ids, val in (("a", ["e1/follow/white", "e2/follow/white"], 0.5), ("b", ["e1/follow/white", "e3/follow/mixed"], 0.7))
            for sid in ids]
    pr = metric_vbench._paired(rows, ("a", "r"), ("b", "r"))
    assert pr["n_common"] == 1 and abs(pr["metrics"]["vbench.imaging_quality"]["mean_diff"] - 0.2) < 1e-9
    assert pr["metrics"]["vbench.imaging_quality"]["b_wins"] == 1
    summ = metric_vbench._summarise_rows(rows)
    reps = {(r["method"], r["representation"]) for r in summ}
    assert ("b", "mixed") in reps and ("b", "white") in reps


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))


def test_vbench_summarise_rows_and_pairs():
    with tempfile.TemporaryDirectory() as d:
        t = Path(d)
        mapping = [dict(file="a__r1__ep__cam__white.mp4", method="a", run_id="r1", sample_id="ep/cam/white", representation="white"),
                   dict(file="b__r2__ep__cam__white.mp4", method="b", run_id="r2", sample_id="ep/cam/white", representation="white"),
                   dict(file="a__r1__ep2__cam__white.mp4", method="a", run_id="r1", sample_id="ep2/cam/white", representation="white"),
                   dict(file="b__r2__ep2__cam__white.mp4", method="b", run_id="r2", sample_id="ep2/cam/white", representation="white")]
        vb = {"imaging_quality": [0.5, [dict(video_path=f"/x/{m['file']}", video_results=0.4 + 0.1 * i) for i, m in enumerate(mapping)]],
              "dynamic_degree": [0.5, [dict(video_path=f"/x/{m['file']}", video_results=(i % 2 == 0)) for i, m in enumerate(mapping)]]}
        common.write_json(t / "vb.json", vb); common.write_json(t / "map.json", mapping)
        rows = metric_vbench.vbench_rows(str(t / "vb.json"), str(t / "map.json"))
        assert [r["vbench.dynamic_degree"] for r in rows] == [1.0, 0.0, 1.0, 0.0]
        metric_vbench.summarise(str(t / "vb.json"), str(t / "map.json"), str(t / "t"), pairs=["a:r1,b:r2"], expected=4)
        summ = common.read_json(t / "t" / "summary.json")
        a = [s for s in summ if s["method"] == "a"][0]
        assert abs(a["vbench.imaging_quality"]["mean"] - 0.5) < 1e-9 and a["n"] == 2
        paired = common.read_json(t / "t" / "paired.json")["a:r1,b:r2"]
        assert paired["n_common"] == 2 and abs(paired["metrics"]["vbench.imaging_quality"]["mean_diff"] - 0.1) < 1e-9
        assert paired["metrics"]["vbench.imaging_quality"]["b_wins"] == 2
        vb["imaging_quality"][1].pop()
        common.write_json(t / "vb2.json", vb)
        try:
            metric_vbench.vbench_rows(str(t / "vb2.json"), str(t / "map.json"))
        except KeyError:
            pass
        else:
            raise AssertionError("missing VBench entry must raise")



def _ffmpeg():
    import shutil
    import pytest
    ff = os.environ.get("FFMPEG", "ffmpeg")
    if not shutil.which(ff):
        pytest.skip("ffmpeg not available")
    return ff


def _raw(ff, path, pix, frames):
    import subprocess
    return subprocess.run([ff, "-v", "error", "-i", str(path), "-frames:v", str(frames), "-f", "rawvideo", "-pix_fmt", pix, "-"],
                          check=True, capture_output=True).stdout


def test_vbench_prepare_trims_losslessly_in_the_input_format():
    import subprocess, tempfile
    import pytest
    ff = _ffmpeg()
    cases = [("odd444.mp4", "853x481", ["-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p"], "yuv444p"),
             ("full444.mp4", "64x36", ["-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuvj444p"], "rgb24"),
             ("ten.mp4", "640x360", ["-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p10le"], "yuv420p10le"),
             ("rgb.mp4", "640x360", ["-c:v", "libx264rgb", "-qp", "0", "-pix_fmt", "rgb24"], "rgb24"),
             ("even420.mp4", "640x360", ["-c:v", "libx264", "-pix_fmt", "yuv420p"], "yuv420p")]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp); rows = []
        for name, size, enc, _ in cases:
            subprocess.run([ff, "-v", "error", "-f", "lavfi", "-i", f"testsrc=size={size}:rate=24", "-frames:v", "144", *enc,
                            str(tmp / name)], check=True)
            assert "x".join(str(common.video_meta(tmp / name)[k]) for k in ("width", "height")) == size
            rows.append(dict(sample_id=f"ep/{name[:-4]}/white", method="m", run_id="r", representation="white", video=str(tmp / name)))
        (tmp / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        metric_vbench.prepare(str(tmp / "samples.jsonl"), str(tmp / "out"))
        for name, _, _, pix in cases:
            out = tmp / "out" / f"m__r__ep__{name[:-4]}__white.mp4"
            meta = common.video_meta(out)
            assert meta["frame_count"] == 120 and (meta["width"], meta["height"]) == tuple(common.video_meta(tmp / name)[k] for k in ("width", "height"))
            assert _raw(ff, out, pix, 120) == _raw(ff, tmp / name, pix, 120), name
        subprocess.run([ff, "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=64x36:rate=24", "-frames:v", "144", "-c:v", "ffv1",
                        "-pix_fmt", "yuv420p12le", str(tmp / "twelve.mkv")], check=True)
        (tmp / "s12.jsonl").write_text(json.dumps(dict(sample_id="ep/twelve/white", method="m", run_id="r", representation="white",
                                                      video=str(tmp / "twelve.mkv"))) + "\n")
        with pytest.raises(ValueError, match="cannot be trimmed losslessly"):
            metric_vbench.prepare(str(tmp / "s12.jsonl"), str(tmp / "out12"))
