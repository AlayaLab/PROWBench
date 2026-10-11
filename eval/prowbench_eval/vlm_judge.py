"""VLM judge: frame sampling, JSON answer parsing and validation, and a response cache keyed by frames, prompt and
settings."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
from pathlib import Path

import numpy as np

from .common import iter_video_frames, video_meta, atomic_open

DEFAULT_JUDGE_MODEL = "Qwen/Qwen3.6-27B"


def require_bool(obj, key):
    value = obj.get(key)
    if type(value) is not bool:
        raise ValueError(f"Judge field {key!r} must be a JSON boolean, got {value!r}")
    return value


def require_score(obj, key="score"):
    value = obj.get(key)
    if type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Judge field {key!r} must be a finite score in [0, 1]")
    return float(value)


def sample_frames(video: str, t0: float, t1: float, n: int, fps: float | None = None, max_side: int = 768):
    """n frames uniformly in [t0, t1] (inclusive), resized so the long side <= max_side. Returns list of RGB uint8."""
    import cv2
    meta = video_meta(video); fps = fps or meta["fps"]
    last = meta["frame_count"] - 1
    if n <= 0 or not np.isfinite([t0, t1, fps]).all() or fps <= 0 or t0 < 0 or t1 < t0:
        raise ValueError("Invalid judge frame sampling window")
    idx = sorted({min(last, int(round(t * fps))) for t in np.linspace(t0, t1, n)})
    out = []
    for i, im in iter_video_frames(video, set(idx)):
        h, w = im.shape[:2]; s = min(1.0, max_side / max(h, w))
        if s < 1.0:
            im = cv2.resize(im, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        out.append((i, im[:, :, ::-1].copy()))
    if [i for i, _ in out] != idx:
        raise ValueError(f"Video decode did not supply requested judge frames: {video}")
    return out


def _png_b64(rgb: np.ndarray) -> str:
    from PIL import Image
    buf = io.BytesIO(); Image.fromarray(rgb).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _cache_key(frames, prompt, model) -> str:
    h = hashlib.sha256()
    for _, im in frames:
        h.update(np.ascontiguousarray(im).tobytes()); h.update(str(im.shape).encode())
    h.update(prompt.encode()); h.update(model.encode())
    return h.hexdigest()


def _close_brackets(frag: str) -> str:
    """Strip the (possibly wrong or missing) trailing closers of `frag` and re-close every open [ / { in order,
    skipping brackets inside strings. Used only when the model's output is not valid JSON as written."""
    frag = frag.rstrip()
    while frag and frag[-1] in "]} \n\t":
        frag = frag[:-1]
    stack, in_str, esc = [], False, False
    for ch in frag:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "[{":
            stack.append("]" if ch == "[" else "}")
        elif ch in "]}":
            if stack and stack[-1] == ch:
                stack.pop()
    if in_str:
        raise ValueError("last string value is cut off")
    return frag + "".join(reversed(stack))


def extract_json(text: str):
    """The JSON object in a judge response; missing or wrong closing brackets are repaired where possible. Raises
    ValueError when no JSON object can be parsed."""
    text = text.strip()
    start = text.find("{")
    if start < 0:
        raise ValueError(f"no JSON in judge output: {text[:200]!r}")
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    frag = re.sub(r"```\s*$", "", text[start:]).rstrip()
    if frag and frag[-1] in "]}":
        try:
            return json.loads(_close_brackets(frag))
        except (json.JSONDecodeError, ValueError):
            pass
    raise ValueError(f"no JSON in judge output: {text[:200]!r}")


class JudgeResponseError(ValueError):
    """The judge answered, but its response is not valid JSON (an invalid answer, not an execution error)."""


class Judge:
    def __init__(self, backend="hf", model=None, device="cuda:0", cache_dir=".judge_cache", base_url=None,
                 api_key=None, max_new_tokens=512, dtype="bfloat16", enable_thinking=None):
        self.backend, self.model_name, self.device = backend, model, device
        self.cache_dir = Path(cache_dir); self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self._hf = None; self.base_url, self.api_key, self.dtype = base_url, api_key, dtype

    def _hf_load(self):
        import torch
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as M
        except ImportError:
            from transformers import Qwen2_5_VLForConditionalGeneration as M
        proc = AutoProcessor.from_pretrained(self.model_name)
        net = M.from_pretrained(self.model_name, torch_dtype=getattr(torch, self.dtype)).to(self.device).eval()
        self._hf = (proc, net, torch)

    def _ask_hf(self, frames, prompt) -> str:
        if self._hf is None:
            self._hf_load()
        proc, net, torch = self._hf
        from PIL import Image
        content = [{"type": "image", "image": Image.fromarray(im)} for _, im in frames]
        content.append({"type": "text", "text": prompt})
        msgs = [{"role": "user", "content": content}]
        template_kwargs = {} if self.enable_thinking is None else {"enable_thinking": self.enable_thinking}
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **template_kwargs)
        images = [Image.fromarray(im) for _, im in frames] or None
        inputs = proc(text=[text], images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            gen = net.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return proc.batch_decode(gen[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

    def _ask_openai(self, frames, prompt) -> str:
        import urllib.request
        content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + _png_b64(im)}} for _, im in frames]
        content.append({"type": "text", "text": prompt})
        payload = dict(model=self.model_name, temperature=0, max_tokens=self.max_new_tokens,
                       messages=[{"role": "user", "content": content}])
        if self.enable_thinking is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        body = json.dumps(payload).encode()
        req = urllib.request.Request(self.base_url.rstrip("/") + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json",
                                              "Authorization": f"Bearer {self.api_key or os.environ.get('OPENAI_API_KEY', 'x')}"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)["choices"][0]["message"]["content"]

    def ask(self, frames, prompt: str) -> dict:
        settings = json.dumps(dict(backend=self.backend, model=self.model_name, dtype=self.dtype,
                                   max_new_tokens=self.max_new_tokens, base_url=self.base_url,
                                   frame_indices=[int(i) for i, _ in frames],
                                   enable_thinking=self.enable_thinking), sort_keys=True)
        key = _cache_key(frames, prompt, settings)
        cp = self.cache_dir / f"{key}.json"
        if cp.exists():
            rec = json.load(open(cp))
            if rec.get("parsed", {}).get("_parse_error"):
                raise JudgeResponseError(f"Cached judge JSON parsing failed: {cp}")
            return rec
        raw = self._ask_hf(frames, prompt) if self.backend == "hf" else self._ask_openai(frames, prompt)
        try:
            parsed = extract_json(raw)
        except ValueError as e:          # not valid JSON (json.JSONDecodeError is a ValueError); other errors propagate
            parsed = {"_parse_error": str(e)}
        rec = dict(raw=raw, parsed=parsed, model=self.model_name, backend=self.backend, prompt=prompt,
                   frames=[int(i) for i, _ in frames], settings=json.loads(settings))
        with atomic_open(cp) as f:
            json.dump(rec, f, ensure_ascii=False, indent=1)
        if "_parse_error" in parsed:
            raise JudgeResponseError(f"Judge JSON parsing failed; raw response saved at {cp}")
        return rec


def add_judge_args(ap):
    ap.add_argument("--judge-backend", default="hf", choices=["hf", "openai"])
    ap.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    ap.add_argument("--judge-thinking", choices=["default", "on", "off"], default="off")
    ap.add_argument("--judge-device", default="cuda:0")
    ap.add_argument("--judge-base-url", default=None)
    ap.add_argument("--judge-cache", default=".judge_cache")
    ap.add_argument("--judge-frames", type=int, default=8, help="frames per window shown to the judge")


def judge_from_args(args) -> Judge:
    thinking = {"default": None, "on": True, "off": False}[args.judge_thinking]
    return Judge(args.judge_backend, args.judge_model, args.judge_device, args.judge_cache,
                 args.judge_base_url, enable_thinking=thinking)
