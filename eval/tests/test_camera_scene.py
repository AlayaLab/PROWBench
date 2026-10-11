"""Tests of the scene-scale camera metrics and the trajectory conversion of `estimate_camera`."""
import numpy as np

from prowbench_eval.metric_camera import scene_depth_metrics, evaluate, quat_to_R
from prowbench_eval.estimate_camera import rotmat_to_quat_xyzw, slam_scene_scale, positive_depth_summary


def test_static_perfect_drift_and_pure_rotation():
    p = np.zeros((40, 3)); q = np.tile([0., 0, 0, 1], (40, 1))
    m, _ = scene_depth_metrics(p, q, p, q, 10, 10)
    assert all(abs(v) < 1e-7 for v in m.values())
    drift = p.copy(); drift[:, 0] = np.linspace(0, 1, 40)
    a, _ = scene_depth_metrics(p, q, drift, q, 10, 10)
    b, _ = scene_depth_metrics(p, q, 2 * drift, q, 10, 10)
    assert abs(a['TransErr_scene_mean'] - .05) < 1e-10
    assert abs(b['TransErr_scene_mean'] - 2*a['TransErr_scene_mean']) < 1e-10
    assert abs(a['CamMC_scene_mean'] - .05) < 1e-10
    yaw = np.linspace(0, np.pi/2, 40)
    qr = np.array([[0, np.sin(t/2), 0, np.cos(t/2)] for t in yaw])
    r, _ = scene_depth_metrics(p, q, p, qr, 10, 10)
    assert abs(r['RotErr_deg_mean'] - 45) < 1e-6 and r['TransErr_scene_mean'] == 0
    perfect_rotation, _ = scene_depth_metrics(p, qr, p, qr, 10, 10)
    assert perfect_rotation['CamMC_scene_mean'] < 1e-12


def test_joint_scene_scale_invariance_but_motion_amplitude_penalized():
    p = np.zeros((40, 3)); p[:, 0] = np.linspace(0, 2, 40)
    q = np.tile([0., 0, 0, 1], (40, 1))
    R = quat_to_R([.3, .2, .1, .9])
    pp = .37 * p @ R.T + [5, -2, 1]
    qp = np.tile(rotmat_to_quat_xyzw(R), (40, 1))
    same, _ = scene_depth_metrics(p, q, pp, qp, 10, 3.7)
    assert same['TransErr_scene_mean'] < 1e-10 and same['CamMC_scene_mean'] < 1e-10
    short, _ = scene_depth_metrics(p, q, .1*p, q, 10, 10)
    stopped, _ = scene_depth_metrics(p, q, np.zeros_like(p), q, 10, 10)
    assert abs(short['TransErr_scene_mean']-.09) < 1e-10
    assert stopped['TransErr_scene_mean'] > short['TransErr_scene_mean']


def test_scene_scale_uses_same_world_frame_and_rejects_invalid_depth():
    xyz = np.tile([[0., 0, 10]], (40, 1))
    pose = np.eye(4)
    assert slam_scene_scale(xyz, pose)['depth'] == 10
    R = quat_to_R([.3, .2, .1, .9]); pose[:3, :3] = R; pose[:3, 3] = [4, 5, 6]
    transformed = 2*xyz @ R.T + pose[:3, 3]
    assert abs(slam_scene_scale(transformed, pose)['depth']-20) < 1e-10
    for bad in [[], [0]*40, [float('nan')]*40]:
        try:
            positive_depth_summary(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid scene depth accepted')


def test_scene_protocol_requires_scale_and_keeps_every_static_instant():
    frames = [dict(frame=i, position=[0, 0, 0], quaternion=[0, 0, 0, 1]) for i in range(150)]
    gt = dict(camera_trajectory=frames)
    pred = dict(fps=30, frames=frames)
    try:
        evaluate(gt, pred, 30, protocol="scene_depth")
    except ValueError:
        pass
    else:
        raise AssertionError('Missing metadata silently accepted')
    scale = dict(depth=10, shares_trajectory_scale=True)
    pred['estimator'] = dict(scene_scale=scale)
    result = evaluate(gt, pred, 30, protocol="scene_depth", gt_scene_scale=scale)
    assert result['schema'] == 'prowbench.metric_camera.scene_depth.v1'
    assert len(result['per_instant']) == 40
    assert result['summary']['TransErr_scene_mean'] == 0
