"""Tests of the CamI2V camera metrics of `metric_camera`."""
import numpy as np
from prowbench_eval.metric_camera import cami2v_metrics, evaluate, _c2w


def test_cami2v_static_and_pure_rotation_have_finite_scores():
    p = np.zeros((40, 3)); q = np.tile([0., 0, 0, 1], (40, 1))
    perfect = cami2v_metrics(p, q, p, q, static=True)
    assert all(v == 0 for v in perfect.values())
    drift = p.copy(); drift[:, 0] = np.linspace(0, .01, 40)
    score = cami2v_metrics(p, q, drift, q, static=True)
    assert all(np.isfinite(v) for v in score.values())
    assert abs(score['TransErr_rel_mean'] - .5 * .01/(.01+1e-9)) < 1e-12
    assert score['CamMC_rel_mean'] == score['TransErr_rel_mean']
    yaw = np.linspace(0, np.pi/2, 40)
    qr = np.array([[0, np.sin(t/2), 0, np.cos(t/2)] for t in yaw])
    rotation = cami2v_metrics(p, q, p, qr, static=True)
    assert abs(rotation['RotErr_mean'] - np.pi/4) < 1e-8
    assert rotation['TransErr_rel'] == 0 and rotation['CamMC_rel'] > 0


def test_cami2v_matches_official_sum_and_normalize_t_formulas():
    rng = np.random.default_rng(13)
    pg, pp = rng.normal(size=(40, 3)), rng.normal(size=(40, 3))
    qg, qp = rng.normal(size=(40, 4)), rng.normal(size=(40, 4))
    g, p = _c2w(pg, qg), _c2w(pp, qp)
    g, p = np.linalg.inv(g[0]) @ g, np.linalg.inv(p[0]) @ p
    r = np.arccos(np.clip((np.trace(g[:, :3, :3].transpose(0, 2, 1) @ p[:, :3, :3], axis1=1, axis2=2)-1)/2, -1, 1))
    result, rows = cami2v_metrics(pg, qg, pp, qp, return_per_instant=True)
    assert np.isclose(result['RotErr'], r.sum())
    for tag in ['rel', 'abs']:
        gn, pn = g.copy(), p.copy()
        sg = np.linalg.norm(g[:, :3, 3], axis=1).max()+1e-9
        sp = np.linalg.norm(p[:, :3, 3], axis=1).max()+1e-9 if tag == 'rel' else sg
        gn[:, :3, 3] /= sg; pn[:, :3, 3] /= sp
        expected_t = np.linalg.norm(gn[:, :3, 3]-pn[:, :3, 3], axis=1)
        expected_c = np.linalg.norm((gn[:, :3, :4]-pn[:, :3, :4]).reshape(-1, 12), axis=1)
        for key, values in [('TransErr_'+tag, expected_t), ('CamMC_'+tag, expected_c)]:
            assert np.isclose(result[key], values.sum())
            assert np.isclose(result[key+'_mean'], values.mean())
            assert np.allclose([row[key] for row in rows], values)


def test_default_camera_protocol_needs_no_depth_and_covers_static():
    frames = [dict(frame=i, position=[0, 0, 0], quaternion=[0, 0, 0, 1]) for i in range(150)]
    result = evaluate(dict(camera_trajectory=frames), dict(fps=30, frames=frames), 30)
    assert result['protocol'] == 'cami2v_all_motion_v1'
    assert result['summary']['TransErr_rel_mean'] == 0
    assert result['summary']['CamMC_rel_mean'] == 0
    assert len(result['per_instant']) == 40
