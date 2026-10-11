"""Camera control: rotation error, translation error and CamMC of an estimated trajectory against the recorded camera
on the evaluation grid (CamI2V protocol)."""
from __future__ import annotations

import argparse
import json

import numpy as np

from .common import eval_grid, output_frame_for_time, read_json, source_frame_for_time, write_json


def quat_to_R(q):
    if np.asarray(q).shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError("Invalid quaternion: expected four finite components and nonzero norm")
    x, y, z, w = q
    n = np.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale=True):
    """Find s, R, t minimising || dst - (s R src + t) ||."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    cov = D.T @ S / len(src)
    U, sig, Vt = np.linalg.svd(cov)
    d = np.ones(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        d[-1] = -1
    R = U @ np.diag(d) @ Vt
    var_s = (S ** 2).sum() / len(src)
    s = (sig * d).sum() / var_s if with_scale and var_s > 0 else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def load_traj(path) -> dict:
    p = str(path)
    if p.endswith(".json"):
        return read_json(p)
    frames = []
    with open(p) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            v = [float(x) for x in line.split()]
            frames.append(dict(time=v[0], position=v[1:4], quaternion=v[4:8]))
    return dict(frames=frames)


def _sample_traj(traj: dict, instants, fps=None, frame_count=None):
    frames = traj["frames"]
    if not frames:
        raise ValueError("Empty camera trajectory")
    if "time" in frames[0]:
        times = np.asarray([fr["time"] for fr in frames], float)
        if not np.isfinite(times).all() or (np.diff(times) <= 0).any():
            raise ValueError("Camera timestamps must be finite and strictly increasing")
        tolerance = 1e-9
    else:
        rate = fps or traj.get("fps")
        if rate is None or not np.isfinite(rate) or rate <= 0:
            raise ValueError("Camera trajectory needs a positive finite FPS")
        if fps is not None and traj.get("fps") is not None and not np.isclose(fps, traj['fps'], rtol=1e-5):
            raise ValueError("Prediction FPS disagrees with trajectory metadata")
        if [fr.get("frame", i) for i, fr in enumerate(frames)] != list(range(len(frames))):
            raise ValueError("Frame-index trajectory must be contiguous from frame zero")
        times = np.arange(len(frames)) / rate
        tolerance = .5 / rate + 1e-9
    if min(instants) < times[0] - tolerance or max(instants) > times[-1] + tolerance:
        raise ValueError("Camera trajectory does not cover the evaluation time grid")
    P, Q = [], []
    for t in instants:
        if "time" in frames[0]:
            i = int(np.argmin(abs(times - t)))
        else:
            f = output_frame_for_time(t, rate)
            f = min(f, (frame_count or len(frames)) - 1)
            i = min(f, len(frames) - 1)
        P.append(frames[i]["position"]); Q.append(frames[i]["quaternion"])
    return np.asarray(P, float), np.asarray(Q, float)


def _c2w(P, Q):
    M = np.tile(np.eye(4), (len(P), 1, 1))
    for i in range(len(P)):
        M[i, :3, :3] = quat_to_R(Q[i]); M[i, :3, 3] = P[i]
    return M


def cami2v_metrics(Pg, Qg, Pp, Qp, static: bool = False, eps: float = 1e-9,
                   return_per_instant: bool = False):
    """RotErr / TransErr / CamMC exactly as CamI2V's `metric()`, on relative-to-first-instant c2w poses."""
    G, E = _c2w(Pg, Qg), _c2w(Pp, Qp)
    G = np.linalg.inv(G[0])[None] @ G
    E = np.linalg.inv(E[0])[None] @ E
    n = len(G)
    Rg, Re = G[:, :3, :3], E[:, :3, :3]
    tr = np.einsum("nij,nij->n", Rg, Re)
    rot = np.arccos(np.clip((tr - 1) / 2, -1, 1))
    out = dict(RotErr=float(rot.sum()), RotErr_mean=float(rot.mean()))
    rows = [dict(RotErr=float(r)) for r in rot]
    sg = np.linalg.norm(G[:, :3, 3], axis=1).max() + eps
    se = np.linalg.norm(E[:, :3, 3], axis=1).max() + eps
    for tag, s_e in (("rel", se), ("abs", sg)):
        tg, te = G[:, :3, 3] / sg, E[:, :3, 3] / s_e
        terr = np.linalg.norm(tg - te, axis=1)
        RTg = np.concatenate([Rg, tg[:, :, None]], axis=2).reshape(n, 12)
        RTe = np.concatenate([Re, te[:, :, None]], axis=2).reshape(n, 12)
        mc = np.linalg.norm(RTg - RTe, axis=1)
        out[f"TransErr_{tag}"] = float(terr.sum()); out[f"TransErr_{tag}_mean"] = float(terr.mean())
        out[f"CamMC_{tag}"] = float(mc.sum()); out[f"CamMC_{tag}_mean"] = float(mc.mean())
        for row, t, c in zip(rows, terr, mc):
            row[f"TransErr_{tag}"] = float(t)
            row[f"CamMC_{tag}"] = float(c)
    return (out, rows) if return_per_instant else out


def scene_depth_metrics(Pg, Qg, Pp, Qp, gt_scale, pred_scale):
    """All motions, including zero translation and pure rotation; no Sim(3) fit.

    Each trajectory uses one first-frame static scene depth. This removes a joint
    map/pose scale ambiguity, but preserves motion amplitude relative to the scene.
    """
    scales = np.asarray([gt_scale, pred_scale], float)
    if not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("Scene depths must be finite and strictly positive")
    G, E = _c2w(Pg, Qg), _c2w(Pp, Qp)
    if not np.isfinite(G).all() or not np.isfinite(E).all():
        raise ValueError("Non-finite camera trajectory")
    G = np.linalg.inv(G[0])[None] @ G
    E = np.linalg.inv(E[0])[None] @ E
    G[:, :3, 3] /= gt_scale
    E[:, :3, 3] /= pred_scale
    tr = np.einsum("nij,nij->n", G[:, :3, :3], E[:, :3, :3])
    rot = np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))
    trans = np.linalg.norm(G[:, :3, 3] - E[:, :3, 3], axis=1)
    pose = np.linalg.norm((G[:, :3, :4] - E[:, :3, :4]).reshape(len(G), 12), axis=1)
    summary = dict(RotErr_deg_mean=float(rot.mean()), TransErr_scene_mean=float(trans.mean()),
                   CamMC_scene_mean=float(pose.mean()))
    rows = [dict(rot_err_deg=float(r), trans_err_scene=float(t), cammc_scene=float(c))
            for r, t, c in zip(rot, trans, pose)]
    return summary, rows


def evaluate(gt: dict, pred_traj: dict, pred_fps: float | None, grid_hz: float = 8.0,
             protocol: str = "cami2v", gt_scene_scale: dict | None = None) -> dict:
    instants = eval_grid(grid_hz)
    gt_frames = {f["frame"]: f for f in gt["camera_trajectory"]}
    Pg = np.array([gt_frames[source_frame_for_time(t)]["position"] for t in instants], float)
    Qg = np.array([gt_frames[source_frame_for_time(t)]["quaternion"] for t in instants], float)
    Pp, Qp = _sample_traj(pred_traj, instants, pred_fps)

    if protocol == "cami2v":
        if not all(np.isfinite(x).all() for x in (Pg, Qg, Pp, Qp)):
            raise ValueError("Non-finite camera trajectory")
        summary, rows = cami2v_metrics(Pg, Qg, Pp, Qp, return_per_instant=True)
        summary.update(n_instants=len(instants), grid_hz=grid_hz)
        return dict(schema="prowbench.metric_camera.cami2v.v1", protocol="cami2v_all_motion_v1",
                    summary=summary, estimator=pred_traj.get("estimator"),
                    normalization=dict(relative_to_first=True, translation="own_max_norm_plus_eps",
                                       eps=1e-9, rotation_units="radians", static_exclusion=False),
                    per_instant=[dict(t=float(t), **row) for t, row in zip(instants, rows)])

    if protocol == "scene_depth":
        pred_scale = (pred_traj.get("estimator") or {}).get("scene_scale")
        for label, value in (("GT", gt_scene_scale), ("prediction", pred_scale)):
            if not value or value.get("shares_trajectory_scale") is not True:
                raise ValueError(f"{label} scene scale missing or not verified to share trajectory scale")
        summary, rows = scene_depth_metrics(Pg, Qg, Pp, Qp, gt_scene_scale["depth"], pred_scale["depth"])
        summary.update(n_instants=len(instants), grid_hz=grid_hz)
        return dict(schema="prowbench.metric_camera.scene_depth.v1", protocol="scene_depth_v1",
                    summary=summary, scene_scale=dict(gt=gt_scene_scale, prediction=pred_scale),
                    estimator=pred_traj.get("estimator"),
                    per_instant=[dict(t=float(t), **row) for t, row in zip(instants, rows)])
    if protocol != "ate":
        raise ValueError(f"Unknown camera protocol: {protocol}")

    path_len = float(np.linalg.norm(np.diff(Pg, axis=0), axis=1).sum())
    static = path_len < 0.05
    s, R, t = umeyama(Pp, Pg, with_scale=not static)
    Pa = (s * (R @ Pp.T)).T + t
    ate = float(np.sqrt(((Pa - Pg) ** 2).sum(1).mean()))

    rot_err = []
    for qg, qp in zip(Qg, Qp):
        Rg, Rp = quat_to_R(qg), R @ quat_to_R(qp)
        c = np.clip((np.trace(Rg.T @ Rp) - 1) / 2, -1, 1)
        rot_err.append(float(np.degrees(np.arccos(c))))
    rot_err = np.asarray(rot_err)
    rel_rots = []
    for i in range(len(Qg)):
        rel_rots.append(quat_to_R(Qg[i]).T @ (R @ quat_to_R(Qp[i])))
    M = sum(rel_rots) / len(rel_rots); U, _, Vt = np.linalg.svd(M); R0 = U @ Vt
    rot_err_offset_free = []
    for Rr in rel_rots:
        c = np.clip((np.trace(R0.T @ Rr) - 1) / 2, -1, 1); rot_err_offset_free.append(float(np.degrees(np.arccos(c))))

    rpe_t, rpe_r = [], []
    for i in range(1, len(instants)):
        dg = Pg[i] - Pg[i - 1]; dp = Pa[i] - Pa[i - 1]
        if not static and np.linalg.norm(dg) > 1e-6:
            rpe_t.append(float(np.linalg.norm(dp - dg) / np.linalg.norm(dg)))
        Rg = quat_to_R(Qg[i - 1]).T @ quat_to_R(Qg[i]); Rp = (R @ quat_to_R(Qp[i - 1])).T @ (R @ quat_to_R(Qp[i]))
        c = np.clip((np.trace(Rg.T @ Rp) - 1) / 2, -1, 1); rpe_r.append(float(np.degrees(np.arccos(c))))

    summary = dict(
        ate_m=ate, ate_over_path=(ate / path_len) if not static else None, gt_path_length_m=path_len,
        sim3_scale=float(s), static_camera=bool(static),
        rot_err_deg_mean=float(rot_err.mean()), rot_err_offset_free_deg_mean=float(np.mean(rot_err_offset_free)),
        rpe_trans_rel_mean=float(np.mean(rpe_t)) if rpe_t else None, rpe_rot_deg_mean=float(np.mean(rpe_r)) if rpe_r else None,
        n_instants=len(instants), grid_hz=grid_hz)
    summary.update(cami2v_metrics(Pg, Qg, Pp, Qp, static))
    return dict(schema="prowbench.metric_camera.v1", summary=summary, estimator=pred_traj.get("estimator"),
                per_instant=[dict(t=t, gt=Pg[i].tolist(), pred_aligned=Pa[i].tolist(), rot_err_deg=float(rot_err[i]))
                             for i, t in enumerate(instants)])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred-traj", required=True, help="prowbench.traj.v1 json or TUM txt")
    ap.add_argument("--pred-fps", type=float, default=None, help="fps of the generated video if traj has frame indices")
    ap.add_argument("--out", required=True)
    ap.add_argument("--grid-hz", type=float, default=8.0)
    ap.add_argument("--protocol", choices=["cami2v", "scene_depth", "ate"], default="cami2v")
    ap.add_argument("--gt-scene-scale", help="JSON metadata for GT first-frame static background depth")
    args = ap.parse_args(argv)
    res = evaluate(read_json(args.gt), load_traj(args.pred_traj), args.pred_fps, args.grid_hz,
                   protocol=args.protocol,
                   gt_scene_scale=read_json(args.gt_scene_scale) if args.gt_scene_scale else None)
    write_json(args.out, res)
    print(json.dumps(res["summary"]))


if __name__ == "__main__":
    main()
