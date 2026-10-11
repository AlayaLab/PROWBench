"""Multi-view geometry helpers: projection of oriented boxes, authored participant appearance, frame access by time and
participant crops for the appearance judge."""
import collections, gzip, json, re
from pathlib import Path
import numpy as np

def col_major(m): return np.array(m, dtype=float).reshape(4, 4).T


def project(e, w2c, K, near=0.05):
    """OBB -> axis-aligned box in SOURCE 1280x720 pixel coordinates."""
    c = np.array(e["center"], float); hs = np.array(e["halfSize"], float); ax = np.array(e["axes"], float)
    signs = [(sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    cam = {s: (w2c @ np.array([*(c + s[0] * hs[0] * ax[0] + s[1] * hs[1] * ax[1] + s[2] * hs[2] * ax[2]), 1.0]))[:3] for s in signs}
    edges = [(a, b) for a in signs for b in signs if a < b and sum(x != y for x, y in zip(a, b)) == 1]
    pts = []
    def add(q): z = -q[2]; pts.append((K["cx"] + K["fx"] * q[0] / z, K["cy"] - K["fy"] * q[1] / z))
    for s in signs:
        if -cam[s][2] >= near: add(cam[s])
    for a, b in edges:
        qa, qb = cam[a], cam[b]; za, zb = -qa[2], -qb[2]
        if (za >= near) != (zb >= near): add(qa + (near - za) / (zb - za) * (qb - qa))
    if not pts: return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return [min(xs), min(ys), max(xs), max(ys)]


def map_box(b, crop, W, H):
    """Map a box from source pixels into a generated frame of size W x H that shows the region `crop` (x, y, w, h)."""
    x0, y0, wc, hc = crop
    sx, sy = W / wc, H / hc
    return [(b[0] - x0) * sx, (b[1] - y0) * sy, (b[2] - x0) * sx, (b[3] - y0) * sy]


_PROTAGONIST_LOOK = re.compile(r"The protagonist is participant ([A-Z]), (.+?)\. ")
_SUBJECT_LOOK = re.compile(r"\(participant ([A-Z])\) is (.+?), and keeps exactly this appearance\.")
_LATE_ENTRY = re.compile(r"^not in frame at .*?; from then on it is ")


def prompt_outfits(rel, ep):
    """Authored appearance of every participant label of a multi-view episode, read from the released benchmark
    prompts (<view>/<representation>/video.h3-ref2va.txt of all views): the sentence "The protagonist is participant X,
    <appearance>." and "(participant X) is <appearance>, and keeps exactly this appearance."; a late-entry prefix
    ("not in frame at ...; from then on it is ") is removed. All prompts of an episode must agree."""
    look = {}
    for path in sorted(Path(rel, "episodes", ep).glob("*/*/video.h3-ref2va.txt")):
        text = path.read_text()
        for rx in (_PROTAGONIST_LOOK, _SUBJECT_LOOK):
            for label, desc in rx.findall(text):
                desc = _LATE_ENTRY.sub("", desc)
                if look.setdefault(label, desc) != desc:
                    raise ValueError(f"{ep}: inconsistent appearance of participant {label} in {path}")
    return look


def participant_labels(manifest):
    """entity id -> participant label (A, B, ...). A manifest with `actors` lists them (`entityId`, `label`); otherwise the
    participants are the entity ids of `multiview.actors`, labelled A, B, C, ... in that order."""
    if "actors" in manifest:
        return {x["entityId"]: x["label"] for x in manifest["actors"]}
    ids = manifest["multiview"]["actors"]
    if not ids or len(ids) > 26 or len(set(ids)) != len(ids):
        raise ValueError(f"{manifest.get('name')}: unexpected multiview.actors {ids}")
    return {e: chr(ord("A") + i) for i, e in enumerate(ids)}


def outfits(rel, ep, view):
    """(entity id -> label, label -> authored appearance). The appearance comes from the `People:` section of the view's
    engine.description.txt when present, otherwise from the released prompts (`prompt_outfits`)."""
    lab = participant_labels(json.load(open(f"{rel}/episodes/{ep}/manifest.json")))
    txt = open(f"{rel}/episodes/{ep}/{view}/engine.description.txt").read()
    if "People:" not in txt:
        allowed = set(lab.values())
        return lab, {k: v for k, v in prompt_outfits(rel, ep).items() if k in allowed}
    look = {}
    for line in txt.split("People:", 1)[1].strip().splitlines():
        m = re.match(r"([\w-]+): (.+)$", line.strip())
        if m and m.group(1) in lab: look[lab[m.group(1)]] = m.group(2)
    return lab, look


class Clip:
    """one generated video: frame access by evaluation time, using its own fps."""
    def __init__(self, path):
        import cv2
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 24.0
        self.W = int(self.cap.get(3)); self.H = int(self.cap.get(4))
        self.n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.cache = {}
    def at(self, t):
        idx = int(round(t * self.fps))
        if not (0 <= idx < self.n):
            raise ValueError((t, self.fps, self.n))
        if idx not in self.cache:
            self.cap.set(1, idx); ok, fr = self.cap.read()
            if not (ok and fr is not None):
                raise ValueError((t,idx))
            self.cache[idx] = fr
            if len(self.cache) > 8: self.cache.pop(next(iter(self.cache)))
        return self.cache[idx]
    def close(self): self.cap.release()


def crops_for(a, media, method, ep, cams, t, rel, cache):
    """-> {participant: {cam: (crop_bgr, full_frame, box_in_output)}} for every view where the engine says it is visible."""
    out = collections.defaultdict(dict)
    for cam in cams:
        key = (ep, cam)
        if key not in cache:
            base = f"{a.release}/episodes/{ep}/{cam}"
            camj = json.load(open(f"{base}/camera.json"))
            obb = [json.loads(line) for line in gzip.open(f"{base}/colored-obb/obb-frames.jsonl.gz", "rt")]
            vis = json.load(open(f"{base}/visibility.json")); lab, look = outfits(a.release, ep, cam)
            cache[key] = (camj["frames"], obb, vis, lab, look, camj.get("subjectEntityId") or vis.get("subjectEntityId"))
        frames, obb, vis, lab, look, subj = cache[key]
        vp = media[method].get(f"{ep}/{cam}")
        if not (vp):
            raise ValueError((method,ep,cam))
        ck = ("clip", method, ep, cam)
        if ck not in cache: cache[ck] = Clip(vp)
        clip = cache[ck]
        crop_xywh = a.crops[method]
        fi = int(round(t * 30))
        w2c = col_major(frames[fi]["worldToCamera"]); K = frames[fi]["intrinsics"]
        frame = clip.at(t)
        if not (frame is not None):
            raise ValueError('multiview_geometry: check failed: frame is not None')
        if not (fi < len(obb) and fi < len(frames)):
            raise ValueError('multiview_geometry: check failed: fi < len(obb) and fi < len(frames)')
        for e in obb[fi]["entities"]:
            eid = e["entityId"]
            if eid not in lab: continue
            px = vis["actors"].get(eid, {}).get("visiblePixelCount", [])
            if not (fi < len(px)):
                raise ValueError((ep,cam,eid,fi))
            if px[fi] < a.min_pix:
                a.exclusions.append(dict(method=method,episode=ep,view=cam,t=t,entity=eid,reason="gt_visibility_below_threshold",pixels=px[fi])); continue
            b = project(e, w2c, K)
            if not b:
                a.exclusions.append(dict(method=method,episode=ep,view=cam,t=t,entity=eid,reason="no_front_projection")); continue
            b = map_box(b, crop_xywh, clip.W, clip.H)
            x0, y0 = max(0, b[0]), max(0, b[1]); x1, y1 = min(clip.W - 1, b[2]), min(clip.H - 1, b[3])
            if x1 - x0 < 12 or y1 - y0 < 24:
                a.exclusions.append(dict(method=method,episode=ep,view=cam,t=t,entity=eid,reason="small_or_outside_native_crop")); continue
            mx, my = 0.15 * (x1 - x0), 0.08 * (y1 - y0)
            X0, Y0 = int(max(0, x0 - mx)), int(max(0, y0 - my))
            X1, Y1 = int(min(clip.W, x1 + mx)), int(min(clip.H, y1 + my))
            c = frame[Y0:Y1, X0:X1]
            if not (c.size > 0):
                raise ValueError('multiview_geometry: check failed: c.size > 0')
            out[lab[eid]][cam] = (c, frame, (X0, Y0, X1, Y1), eid == subj, look)
    return out

