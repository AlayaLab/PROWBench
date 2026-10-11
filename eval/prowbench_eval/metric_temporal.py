"""Warp error: photometric error (x1000) between adjacent generated frames after RAFT flow warping, with
forward-backward occlusion masks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .common import EVAL_DURATION, video_meta, write_text


def read_frames(path, size, max_frames):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < max_frames:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        frames.append(rgb)
    cap.release()
    return np.stack(frames)


def load_raft(weights, device):
    from vbench.third_party.RAFT.core.raft import RAFT
    args = argparse.Namespace(small=False, mixed_precision=False, alternate_corr=False, dropout=0)
    model = RAFT(args)
    state = torch.load(weights, map_location="cpu")
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def flow(model, a, b, iters=20):
    """a, b: 1x3xHxW float in [0,255] (RAFT convention). Returns 1x2xHxW flow a->b."""
    _, fl = model(a, b, iters=iters, test_mode=True)
    return fl


def warp(img, fl):
    """Backward-warp img (1xCxHxW) with flow (1x2xHxW): out(x) = img(x + flow(x))."""
    _, _, h, w = img.shape
    ys, xs = torch.meshgrid(torch.arange(h, device=img.device), torch.arange(w, device=img.device), indexing="ij")
    gx = (xs[None].float() + fl[:, 0]) / (w - 1) * 2 - 1
    gy = (ys[None].float() + fl[:, 1]) / (h - 1) * 2 - 1
    grid = torch.stack([gx, gy], dim=-1)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)


def occlusion_mask(fw, bw):
    """Forward-backward consistency (Sundaram et al. 2010 thresholds as used by Lai et al.): True = occluded / unreliable."""
    bw_w = warp(bw, fw)
    diff = (fw + bw_w).pow(2).sum(1)
    mag = fw.pow(2).sum(1) + bw_w.pow(2).sum(1)
    return diff > 0.01 * mag + 0.5


def to_tensor(frame, device):
    return torch.from_numpy(frame).permute(2, 0, 1)[None].float().to(device)


def evaluate(model, gen, ref, device, iters, gen_fps=None, ref_fps=None):
    """Warp error over every adjacent pair of the generated frames. The flow comparison with the reference (tof_epe)
    pairs frames by index, so it is computed only when both videos have the same frame rate, over their common frames."""
    n = len(gen) - 1
    same_rate = gen_fps is None or ref_fps is None or abs(gen_fps - ref_fps) < 1e-6
    n_tof = min(len(gen), len(ref)) - 1 if same_rate else 0
    rows, warp_errs, tofs, occ = [], [], [], []
    for t in range(n):
        g0, g1 = to_tensor(gen[t], device), to_tensor(gen[t + 1], device)
        fw = flow(model, g0, g1, iters)
        bw = flow(model, g1, g0, iters)
        mask = ~occlusion_mask(fw, bw)
        g1w = warp(g1, fw)
        err = ((g0 - g1w) / 255.0).pow(2).mean(1)
        we = float(err[mask].mean()) if mask.any() else float("nan")
        epe = None
        if t < n_tof:
            r0, r1 = to_tensor(ref[t], device), to_tensor(ref[t + 1], device)
            rf = flow(model, r0, r1, iters)
            epe = float((fw - rf).pow(2).sum(1).sqrt().mean())
            tofs.append(epe)
        rows.append({"t": t, "warp_error": we, "tof_epe": epe, "occluded_frac": float((~mask).float().mean())})
        warp_errs.append(we); occ.append(rows[-1]["occluded_frac"])
    return {"warp_error_x1000": 1000 * float(np.nanmean(warp_errs)), "tof_epe": float(np.mean(tofs)) if tofs else None,
            "occluded_frac": float(np.mean(occ)), "n_pairs": n, "tof_pairs": len(tofs)}, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--raft", required=True, help="raft-things.pth")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--size", default="672x384", help="analysis resolution WxH")
    ap.add_argument("--frames", type=int, default=10 ** 6, help="hard cap on the frames of each video to use")
    ap.add_argument("--frames-by-time", type=int, default=1, help="1: keep the first floor(EVAL_DURATION * fps) frames of each video (no resampling)")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    w, h = (int(x) for x in a.size.split("x"))
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model = load_raft(a.raft, a.device)
    for line in open(a.samples):
        s = json.loads(line)
        key = s["sample_id"].replace("/", "__")
        out = out_dir / f"{key}.temporal.json"
        if out.exists() and not a.force:
            continue
        ref_path = s.get("extra", {}).get("reference")
        if not ref_path:
            raise SystemExit(f"{key}: samples.jsonl has no extra.reference (proxy video)")
        def keep(p):
            return min(a.frames, int(EVAL_DURATION * video_meta(p)["fps"] + 1e-6)) if a.frames_by_time else a.frames
        gen = read_frames(s["video"], (w, h), keep(s["video"]))
        ref = read_frames(ref_path, (w, h), keep(ref_path))
        summary, rows = evaluate(model, gen, ref, a.device, a.iters, video_meta(s["video"])["fps"], video_meta(ref_path)["fps"])
        summary.update(video=s["video"], reference=ref_path, size=a.size, raft=str(a.raft), iters=a.iters,
                       gen_frames=int(len(gen)), ref_frames=int(len(ref)), schema="prowbench.metric_temporal.v1")
        write_text(out, json.dumps({"sample_id": s["sample_id"], "summary": summary, "per_pair": rows}, indent=1))
        tof = "n/a" if summary["tof_epe"] is None else f"{summary['tof_epe']:.3f}"
        print(f"{key}: warp_error_x1000={summary['warp_error_x1000']:.2f} tof_epe={tof} "
              f"occ={summary['occluded_frac']:.3f} pairs={summary['n_pairs']}", flush=True)


if __name__ == "__main__":
    main()
