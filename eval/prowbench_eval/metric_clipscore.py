'CLIP-Score: mean cosine similarity (x100) between CLIP image embeddings of grid frames and the episode title.'
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .common import EVAL_DURATION, write_text


def sample_frames(path, per_second):
    """Frames of the video nearest to the instants k / per_second within the evaluation duration."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    want = sorted({min(n - 1, int(round(k / per_second * fps))) for k in range(int(EVAL_DURATION * per_second))})
    frames, i = [], 0
    while want:
        ok, bgr = cap.read()
        if not ok:
            break
        while want and want[0] == i:
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)); want.pop(0)
        i += 1
    cap.release()
    return frames


def load_titles(path) -> dict:
    """Episode id -> text label scored by CLIP. Accepts a title index {"episodes": [{"episodeId", "title"}, ...]}, or a
    released dataset (its directory or its index.json, whose `episodes` lists episode ids); for a released dataset the
    title of each episode is the `title` field of episodes/<episode>/scene-context.json."""
    p = Path(path)
    index = p / "index.json" if p.is_dir() else p
    episodes = json.load(open(index))["episodes"]
    if all(isinstance(e, dict) for e in episodes):
        return {e["episodeId"]: e["title"] for e in episodes}
    if not all(isinstance(e, str) for e in episodes):
        raise ValueError(f"{index}: `episodes` must be title records or episode ids")
    titles = {}
    for ep in episodes:
        ctx = index.parent / "episodes" / ep / "scene-context.json"
        title = json.load(open(ctx)).get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"{ctx}: no title")
        titles[ep] = title
    return titles


def main():
    import torch  # model dependency: imported when scoring, so the module loads without it
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--index", required=True, help="episode titles: a title index {\"episodes\": [{\"episodeId\", \"title\"}]}, or a released dataset directory / its index.json (titles read from episodes/<episode>/scene-context.json)")
    ap.add_argument("--clip", required=True, help="ViT-B-32.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--frames-per-second", type=float, default=8.0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    import clip
    from PIL import Image
    titles = load_titles(a.index)
    model, preprocess = clip.load(a.clip, device=a.device)
    model.eval()
    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for line in open(a.samples):
        s = json.loads(line)
        key = s["sample_id"].replace("/", "__")
        out = out_dir / f"{key}.clipscore.json"
        if out.exists() and not a.force:
            continue
        caption = titles[s["episode"]]
        frames = sample_frames(s["video"], a.frames_per_second)
        with torch.no_grad():
            text = model.encode_text(clip.tokenize([caption]).to(a.device)).float()
            text = text / text.norm(dim=-1, keepdim=True)
            sims = []
            for i in range(0, len(frames), 16):
                batch = torch.stack([preprocess(Image.fromarray(f)) for f in frames[i:i + 16]]).to(a.device)
                img = model.encode_image(batch).float()
                img = img / img.norm(dim=-1, keepdim=True)
                sims.extend((img @ text.T).squeeze(1).tolist())
        score = 100.0 * float(np.mean(sims))
        write_text(out, json.dumps({"sample_id": s["sample_id"], "summary": {"clip_score": score, "caption": caption, "n_frames": len(sims),
                                   "model": "ViT-B/32", "schema": "prowbench.metric_clipscore.v1"}, "per_frame": sims}, indent=1))
        print(f"{key}: clip_score={score:.2f} ({len(sims)} frames) '{caption}'", flush=True)


if __name__ == "__main__":
    main()
