'Ground-truth export: projected entity boxes, phases, events and the camera trajectory of engine recordings.'
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .common import SOURCE_FPS, SOURCE_H, SOURCE_W, read_json, read_jsonl, write_json

STATIC_LABELS_SKIP = {"terrain"}


def _mat(m):
    """Column-major 16 -> (4,4) row-major numpy."""
    return np.asarray(m, dtype=np.float64).reshape(4, 4).T


def obb_corners_world(bbox: dict) -> np.ndarray:
    mn, mx = np.asarray(bbox["min"]), np.asarray(bbox["max"])
    corners = np.array([[x, y, z] for x in (mn[0], mx[0]) for y in (mn[1], mx[1]) for z in (mn[2], mx[2])])
    M = _mat(bbox["worldMatrix"])
    h = np.concatenate([corners, np.ones((8, 1))], axis=1)
    return (M @ h.T).T[:, :3]


def project_points(cam: dict, pts: np.ndarray):
    """Returns (pixels (N,2), depth (N,)) using the recording's intrinsics; camera looks down -Z."""
    W2C = _mat(cam["worldToCamera"])
    k = cam["intrinsics"]
    h = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    c = (W2C @ h.T).T[:, :3]
    depth = -c[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = k["fx"] * c[:, 0] / depth + k["cx"]
        v = k["cy"] - k["fy"] * c[:, 1] / depth
    return np.stack([u, v], axis=1), depth


def box_from_corners(cam: dict, corners: np.ndarray, near_eps: float = 1e-3):
    """Pixel AABB of the projected corners. Corners behind the camera are clipped
    against the near plane along the edges to the visible corners (cheap and good
    enough for a bbox)."""
    pix, depth = project_points(cam, corners)
    front = depth > near_eps
    if not front.any():
        return None, 0
    if not front.all():
        W2C = _mat(cam["worldToCamera"]); k = cam["intrinsics"]
        cc = (W2C @ np.concatenate([corners, np.ones((8, 1))], 1).T).T[:, :3]
        edges = [(i, j) for i in range(8) for j in range(i + 1, 8) if bin(i ^ j).count("1") == 1]
        extra = []
        for i, j in edges:
            if front[i] != front[j]:
                a, b = cc[i], cc[j]
                za, zb = -a[2], -b[2]
                t = (near_eps - za) / (zb - za)
                p = a + t * (b - a)
                d = -p[2]
                extra.append([k["fx"] * p[0] / d + k["cx"], k["cy"] - k["fy"] * p[1] / d])
        pix = np.concatenate([pix[front], np.asarray(extra).reshape(-1, 2)], axis=0)
    else:
        pix = pix[front]
    x0, y0 = pix.min(0); x1, y1 = pix.max(0)
    return [float(x0), float(y0), float(x1), float(y1)], int(front.sum())


def clip_box(b, w=SOURCE_W, h=SOURCE_H):
    x0, y0, x1, y1 = b
    x0, x1 = max(0.0, x0), min(float(w), x1)
    y0, y1 = max(0.0, y0), min(float(h), y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def load_episode(ep_dir: Path):
    manifest = read_json(ep_dir / "manifest.json")
    frames = list(read_jsonl(ep_dir / manifest["states"]))
    frames.sort(key=lambda f: f["index"])
    return manifest, frames


def load_camera(ep_dir: Path, track: str):
    cam = read_json(ep_dir / track / "camera.json")
    return cam["frames"]


def entity_defs(manifest):
    return {e["id"]: e for e in manifest["entities"]}


def frame_entity_states(manifest, frame):
    init = manifest["initial"]["entities"]
    return {eid: frame["entities"].get(eid, init[eid]) for eid in entity_defs(manifest)}


def _diff_events(frames, manifest):
    """Derive discrete events from consecutive frames."""
    events = []
    prev = None
    ent_ids = list(entity_defs(manifest).keys())
    for f in frames:
        t = f["index"] / SOURCE_FPS
        inter = f.get("interaction", {}) or {}
        cur = {
            "phase": inter.get("phase"),
            "relations": _norm_relations(inter.get("relations")),
            "supports": _supports(inter.get("contacts") or []),
            "state": {eid: (f["entities"].get(eid) or {}).get("state") for eid in ent_ids},
        }
        if prev is not None:
            if cur["phase"] != prev["phase"]:
                events.append(dict(type="phase", frame=f["index"], time=t, from_=prev["phase"], to=cur["phase"],
                                   participants=inter.get("participants") or []))
            for key in sorted(cur["relations"] - prev["relations"]):
                events.append(dict(type="relation_start", frame=f["index"], time=t, relation=key))
            for key in sorted(prev["relations"] - cur["relations"]):
                events.append(dict(type="relation_end", frame=f["index"], time=t, relation=key))
            for actor, tgt in cur["supports"].items():
                if prev["supports"].get(actor) != tgt:
                    events.append(dict(type="support_change", frame=f["index"], time=t, actor=actor,
                                       from_=prev["supports"].get(actor), to=tgt))
            for eid in ent_ids:
                a, b = prev["state"].get(eid), cur["state"].get(eid)
                if a != b and b is not None:
                    changed = {k: v for k, v in (b or {}).items() if (a or {}).get(k) != v and k != "phase"}
                    if changed:
                        events.append(dict(type="state_change", frame=f["index"], time=t, entity=eid, changed=changed))
        prev = cur
    return events


def _norm_relations(rel):
    out = set()
    if not rel:
        return out
    if isinstance(rel, dict):
        for k, v in rel.items():
            out.add(f"{k}:{_stable(v)}")
    elif isinstance(rel, list):
        for r in rel:
            out.add(_stable(r))
    return out


def _stable(v):
    import json
    return json.dumps(v, sort_keys=True) if not isinstance(v, str) else v


def _supports(contacts):
    sup = {}
    for c in contacts:
        if c.get("type") == "support":
            sup[c.get("actorId")] = c.get("targetEntityId")
    return sup


def phase_intervals(frames):
    """Contiguous phase runs as [start_frame, end_frame_exclusive)."""
    out = []
    cur, start = None, 0
    for f in frames:
        p = (f.get("interaction") or {}).get("phase")
        if p != cur:
            if cur is not None:
                out.append(dict(phase=cur, start_frame=start, end_frame_exclusive=f["index"]))
            cur, start = p, f["index"]
    if cur is not None:
        out.append(dict(phase=cur, start_frame=start, end_frame_exclusive=frames[-1]["index"] + 1))
    for iv in out:
        iv["start_time"] = iv["start_frame"] / SOURCE_FPS
        iv["end_time"] = iv["end_frame_exclusive"] / SOURCE_FPS
    return out


def export_episode(ep_dir: Path, track: str, min_box_px: float = 2.0) -> dict:
    manifest, frames = load_episode(ep_dir)
    cams = load_camera(ep_dir, track)
    if not (len(cams) == len(frames) == manifest["frameCount"]):
        raise ValueError((len(cams), len(frames), manifest["frameCount"]))
    defs = entity_defs(manifest)
    dyn = set(manifest.get("dynamicEntityIds", []))

    entities_out = {}
    for eid, d in defs.items():
        entities_out[eid] = dict(id=eid, label=d.get("semanticLabel"), description=d.get("description"),
                                 dynamic=eid in dyn, interaction_candidate=bool(d.get("interactionCandidate")),
                                 surface=bool(d.get("surface")))

    per_frame = []
    for f, cam in zip(frames, cams):
        states = frame_entity_states(manifest, f)
        boxes = {}
        for eid, st in states.items():
            if defs[eid].get("semanticLabel") in STATIC_LABELS_SKIP:
                continue
            if not st.get("visible", True) or "bbox" not in st:
                boxes[eid] = dict(box=None, visible=False, reason="hidden")
                continue
            raw, n_front = box_from_corners(cam, obb_corners_world(st["bbox"]))
            clipped = clip_box(raw, manifest["width"], manifest["height"]) if raw else None
            vis = clipped is not None and (clipped[2] - clipped[0]) >= min_box_px and (clipped[3] - clipped[1]) >= min_box_px
            boxes[eid] = dict(box=clipped, box_unclipped=raw, visible=bool(vis), corners_in_front=n_front,
                              world_position=st.get("position"))
        per_frame.append(dict(frame=f["index"], time=f["index"] / SOURCE_FPS, boxes=boxes,
                              phase=(f.get("interaction") or {}).get("phase")))

    cam_traj = [dict(frame=i, position=c["position"], quaternion=c["quaternion"], fov=c.get("fov"),
                     intrinsics=c["intrinsics"], near=c.get("near"), far=c.get("far"))
                for i, c in enumerate(cams)]

    return dict(
        schema="prowbench.gt.v1",
        episode=manifest.get("caseId") or ep_dir.name,
        camera=track,
        fps=manifest["fps"], frame_count=manifest["frameCount"],
        width=manifest["width"], height=manifest["height"],
        entities=entities_out,
        phases=phase_intervals(frames),
        events=_diff_events(frames, manifest),
        frames=per_frame,
        camera_trajectory=cam_traj,
        source=dict(episode_dir=str(ep_dir), states=manifest["states"], camera_file=f"{track}/camera.json"),
    )


FIRST_PERSON_CAMERA = "first-person"
PROTAGONIST_ID = "player"


def first_person_excluded(gt: dict, flagged=(), coverage: float = 0.9) -> list[str]:
    """Entities removed from a first-person view before scoring: the protagonist (the camera wearer), entities
    flagged by the caller (e.g. the wearer's own vehicle), and every entity whose visible box covers at least
    `coverage` of the image in at least `coverage` of its visible frames (a carrier that fills the view)."""
    own = {PROTAGONIST_ID} | set(flagged)
    W, H = gt["width"], gt["height"]
    for e in gt["entities"]:
        if e in own:
            continue
        vis = [f["boxes"][e]["box"] for f in gt["frames"] if f["boxes"].get(e, {}).get("visible")]
        if vis and sum(((b[2] - b[0]) * (b[3] - b[1])) >= coverage * W * H for b in vis) >= coverage * len(vis):
            own.add(e)
    return sorted(e for e in own if e in gt["entities"])


def filter_first_person(gt: dict, flagged=()) -> dict:
    """Copy of `gt` without the entities of `first_person_excluded`; their ids are listed in `first_person_excluded`."""
    drop = first_person_excluded(gt, flagged)
    out = dict(gt)
    out["entities"] = {k: v for k, v in gt["entities"].items() if k not in drop}
    out["frames"] = [dict(fr, boxes={k: v for k, v in fr["boxes"].items() if k not in drop}) for fr in gt["frames"]]
    out["first_person_excluded"] = drop
    return out


def episode_cameras(ep_dir: Path) -> list[str]:
    """Camera tracks of an episode: subdirectories that contain camera.json."""
    return sorted(p.parent.name for p in Path(ep_dir).glob("*/camera.json"))


def main(argv=None):
    ap = argparse.ArgumentParser(description="Export per-frame ground truth (entity boxes, phases, events, camera "
                                             "trajectory) of engine recordings to <out>/<episode>/<camera>.json.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="dataset root containing episodes/<episode>/manifest.json")
    ap.add_argument("--out", required=True, help="output dir; writes <out>/<episode>/<camera>.json")
    ap.add_argument("--episodes", nargs="*", default=None, help="episode ids (default: every episode of the dataset)")
    ap.add_argument("--cameras", nargs="*", default=None,
                    help="camera tracks (default: every <episode>/<camera>/camera.json of each episode)")
    ap.add_argument("--first-person-filter", choices=["auto", "off"], default="off",
                    help="auto: drop the camera wearer and view-filling carriers from the 'first-person' camera "
                         "(first_person_excluded); off (default): keep every entity. The report used auto for the "
                         "five-second datasets and for the long-horizon ISR/LRA ground truth, and off for the "
                         "long-horizon entity, memory, state and camera ground truth (see USAGE.md)")
    args = ap.parse_args(argv)
    root = Path(args.dataset)
    eps = args.episodes or sorted(p.name for p in (root / "episodes").iterdir() if (p / "manifest.json").exists())
    for ep in eps:
        available = episode_cameras(root / "episodes" / ep)
        for cam in args.cameras or available:
            if cam not in available:
                if args.cameras and len(args.cameras) == 1:
                    raise SystemExit(f"{ep}: no camera track {cam!r} (available: {available})")
                continue
            gt = export_episode(root / "episodes" / ep, cam)
            if cam == FIRST_PERSON_CAMERA and args.first_person_filter == "auto":
                gt = filter_first_person(gt)
            out = Path(args.out) / ep / f"{cam}.json"
            write_json(out, gt)
            n_ev = len(gt["events"]); n_ph = len(gt["phases"])
            excl = f", excluded {gt['first_person_excluded']}" if "first_person_excluded" in gt else ""
            print(f"{ep}/{cam}: {gt['frame_count']} frames, {len(gt['entities'])} entities{excl}, {n_ph} phases, "
                  f"{n_ev} events -> {out}")


if __name__ == "__main__":
    main()
