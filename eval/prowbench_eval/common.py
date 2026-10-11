"""Shared data types and helpers: generated-sample records, JSON I/O, the evaluation time grid, the mapping of source
boxes into the generated frame, IoU and assignment."""
from __future__ import annotations

import contextlib
import gzip
import json
import math
import os
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

SOURCE_FPS = 30
SOURCE_W, SOURCE_H = 1280, 720
# PROWBENCH_EVAL_DURATION; the former name GAMEBENCH_EVAL_DURATION is still read when the new one is unset
EVAL_DURATION = float(os.environ.get("PROWBENCH_EVAL_DURATION", os.environ.get("GAMEBENCH_EVAL_DURATION", "5.0")))
if not (EVAL_DURATION > 0):
    raise ValueError("PROWBENCH_EVAL_DURATION must be a positive number of seconds")

EVAL_EXCLUDED_LABELS = frozenset({"terrain", "effect", "geometry"})


def evaluated_entities(gt: dict, which: str = "dynamic_or_candidate") -> list[str]:
    """Entity ids that the entity metrics evaluate. `all` = everything except excluded labels;
    `dynamic` = only the engine's dynamic entities (long street scenes list dozens of static interaction
    candidates - parked cars, doors, ladders - whose labels have no tracker prompt and which never move)."""
    if which not in ("all", "dynamic", "dynamic_or_candidate"):
        raise ValueError(f"Unknown entity selection {which!r}")
    out = []
    for e, d in gt["entities"].items():
        if d.get("label") in EVAL_EXCLUDED_LABELS:
            continue
        if which == "all" or d.get("dynamic") or (which == "dynamic_or_candidate" and d.get("interaction_candidate")):
            out.append(e)
    return out


def read_json(path):
    with open(path) as f:
        return json.load(f)


@contextlib.contextmanager
def atomic_open(path, mode="w", **kw):
    """Write through a temporary file next to `path` that replaces `path` only when the block completes, so an
    interrupted write never leaves a partial file under the final name (resumed stages skip existing outputs)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)   # permissions from the umask, as with open()
    try:
        with os.fdopen(fd, mode, **kw) as f:
            yield f
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def write_text(path, text: str):
    with atomic_open(path) as f:
        f.write(text)


def write_json(path, obj, indent=2):
    with atomic_open(path) as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def read_jsonl(path) -> Iterator[dict]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path, rows: Iterable[dict]):
    with atomic_open(path) as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256_file(path, chunk=1 << 20) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


@dataclass
class SpatialMap:
    """Maps pixel coordinates of the 1280x720 source render into the generated
    frame. Every generator we know of does (optional centre crop) -> resize, so
    this is an affine map. `crop_xywh` is in source pixels.
    """
    out_w: int
    out_h: int
    crop_xywh: tuple = (0, 0, SOURCE_W, SOURCE_H)

    def src_to_out(self, xy: np.ndarray) -> np.ndarray:
        cx, cy, cw, ch = self.crop_xywh
        xy = np.asarray(xy, dtype=np.float64)
        out = np.empty_like(xy)
        out[..., 0] = (xy[..., 0] - cx) * self.out_w / cw
        out[..., 1] = (xy[..., 1] - cy) * self.out_h / ch
        return out

    def box_src_to_out_norm(self, box_xyxy) -> np.ndarray:
        """Source-pixel box -> generated-frame box, normalised to [0,1]."""
        p = self.src_to_out(np.asarray(box_xyxy, dtype=np.float64).reshape(2, 2))
        p[:, 0] /= self.out_w
        p[:, 1] /= self.out_h
        return np.clip(p.reshape(4), 0.0, 1.0)


def output_frame_for_time(t: float, fps: float) -> int:
    return int(round(t * fps))


def source_frame_for_time(t: float) -> int:
    return int(np.clip(round(t * SOURCE_FPS), 0, int(EVAL_DURATION * SOURCE_FPS) - 1))


def eval_grid(grid_hz: float = 8.0, duration: float = EVAL_DURATION) -> list[float]:
    """Common evaluation instants for every generator: k / grid_hz, k>=0, t<duration.
    8 Hz -> 40 instants on a 5 s clip. Lower than every generator's fps, so we
    never interpolate: each instant picks the nearest generated frame and the
    nearest source frame."""
    if not np.isfinite([grid_hz, duration]).all() or grid_hz <= 0 or duration <= 0:
        raise ValueError("Evaluation grid frequency and duration must be positive and finite")
    n = int(math.floor(duration * grid_hz - 1e-9)) + 1
    return [k / grid_hz for k in range(n) if k / grid_hz < duration - 1e-9]


def iou_xyxy(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    ua = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0) + max(0.0, bx1 - bx0) * max(0.0, by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def iou_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64).reshape(-1, 4)
    B = np.asarray(B, dtype=np.float64).reshape(-1, 4)
    if len(A) == 0 or len(B) == 0:
        return np.zeros((len(A), len(B)))
    x0 = np.maximum(A[:, None, 0], B[None, :, 0])
    y0 = np.maximum(A[:, None, 1], B[None, :, 1])
    x1 = np.minimum(A[:, None, 2], B[None, :, 2])
    y1 = np.minimum(A[:, None, 3], B[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    def area(X):
        return np.clip(X[:, 2] - X[:, 0], 0, None) * np.clip(X[:, 3] - X[:, 1], 0, None)
    union = area(A)[:, None] + area(B)[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def center(box):
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])


def hungarian(cost: np.ndarray):
    """Minimal-cost assignment. Uses scipy if present, else a small O(n^3) fallback."""
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2 or not np.isfinite(cost).all():
        raise ValueError("Assignment cost must be a finite matrix")
    if cost.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(cost)
        return np.asarray(r), np.asarray(c)
    except Exception:
        return _hungarian_fallback(cost)


def _hungarian_fallback(cost):
    if np.asarray(cost).ndim != 2 or not np.isfinite(cost).all():
        raise ValueError("Assignment cost must be a finite matrix")
    if cost.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    n, m = cost.shape
    big = np.full((max(n, m), max(n, m)), cost.max() + 1.0)
    big[:n, :m] = cost
    N = big.shape[0]
    u = np.zeros(N + 1); v = np.zeros(N + 1)
    p = np.zeros(N + 1, dtype=int); way = np.zeros(N + 1, dtype=int)
    for i in range(1, N + 1):
        p[0] = i; j0 = 0
        minv = np.full(N + 1, np.inf); used = np.zeros(N + 1, dtype=bool)
        while True:
            used[j0] = True; i0 = p[j0]; delta = np.inf; j1 = 0
            for j in range(1, N + 1):
                if not used[j]:
                    cur = big[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur; way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]; j1 = j
            for j in range(N + 1):
                if used[j]:
                    u[p[j]] += delta; v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]; p[j0] = p[j1]; j0 = j1
            if j0 == 0:
                break
    rows, cols = [], []
    for j in range(1, N + 1):
        if p[j] and p[j] - 1 < n and j - 1 < m:
            rows.append(p[j] - 1); cols.append(j - 1)
    order = np.argsort(rows)
    return np.asarray(rows)[order], np.asarray(cols)[order]


def video_meta(path) -> dict:
    """Width/height/fps/frame count via cv2 (falls back to ffprobe)."""
    try:
        import cv2
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError("cv2 cannot open")
        meta = dict(width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    fps=float(cap.get(cv2.CAP_PROP_FPS)), frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        cap.release()
        if not all(np.isfinite(v) and v > 0 for v in meta.values()):
            raise ValueError("Invalid video metadata")
        return meta
    except Exception:
        import subprocess
        out = subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                                       "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames",
                                       "-of", "json", str(path)])
        s = json.loads(out)["streams"][0]
        num, den = s["r_frame_rate"].split("/")
        return dict(width=int(s["width"]), height=int(s["height"]), fps=float(num) / float(den),
                    frame_count=int(s["nb_read_frames"]))


def iter_video_frames(path, frame_indices: set[int] | None = None):
    """Yield (index, BGR ndarray). Decodes sequentially so indices are exact."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Cannot open video: {path}")
    if frame_indices is not None and (not frame_indices or min(frame_indices) < 0):
        cap.release()
        raise ValueError("Requested video frame indices must be nonempty and nonnegative")
    i = 0
    seen = set()
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if frame_indices is None or i in frame_indices:
            seen.add(i)
            yield i, fr
        i += 1
        if frame_indices is not None and i > max(frame_indices):
            break
    cap.release()
    if frame_indices is not None and seen != frame_indices:
        raise ValueError(f"Missing requested video frames in {path}: {sorted(frame_indices - seen)}")


@dataclass
class GeneratedSample:
    """One generated video plus everything needed to align it with the recording."""
    sample_id: str
    episode: str
    camera: str
    representation: str
    method: str
    run_id: str
    video: str
    fps: float
    width: int
    height: int
    frame_count: int
    crop_xywh: tuple = (0, 0, SOURCE_W, SOURCE_H)
    seed: int | None = None
    conditions: list = field(default_factory=list)
    prompt_sha256: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def spatial(self) -> SpatialMap:
        return SpatialMap(self.width, self.height, tuple(self.crop_xywh))

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        d = dict(d)
        d["crop_xywh"] = tuple(d.get("crop_xywh", (0, 0, SOURCE_W, SOURCE_H)))
        return GeneratedSample(**d)
