"""Entity tracking on generated videos: SAM 3 text-prompted detection and tracking (optionally completed by SAM 2.1),
SAM 2.1 seeded with ground-truth boxes, or Grounding DINO with an IoU tracker. Writes prowbench.tracks.v1 files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .common import (EVAL_DURATION, GeneratedSample, eval_grid, evaluated_entities, iou_matrix,
                     output_frame_for_time, read_json, source_frame_for_time, video_meta, write_json, iter_video_frames)

LABEL_QUERY = {
    "person": "person", "crate": "wooden crate", "parcel": "wooden box", "ball": "ball", "bench": "bench",
    "chair": "chair", "platform": "platform", "tree": "tree", "building": "building", "car": "car", "horse": "horse",
    "boat": "boat", "zipline": "zipline cable", "cable": "zipline cable", "handle": "zipline trolley handle",
    "jetpack": "jetpack",
    "bow": "bow", "arrow": "arrow", "target": "archery target", "door": "door",
    "vehicle": "vehicle", "cart": "handcart", "obstacle": "fallen log", "control-panel": "control panel switch",
    "checkpoint": "start gate arch", "barrier": "guard rail", "road-marking": "road marking",
    "ladder": "ladder", "weapon": "weapon", "fishing-rod": "fishing rod", "fishing-line": "fishing line", "fish": "fish",
    "machine": "arcade claw machine", "claw": "claw", "prize": "plush toy", "tool": "comb", "dart": "dart", "cup": "cup", "table": "table", "sculpture": "statue",
    "dog": "dog", "cat": "cat", "rabbit": "rabbit", "bird": "bird", "duck": "duck", "animal": "animal",
}


def queries_from_gt(gt: dict, include_static: bool = False) -> list[str]:
    """Detector phrases for the evaluated entities (never terrain / effect / geometry). Raises on an unmapped
    label so a new episode cannot silently query the raw label."""
    labs = []
    for e in evaluated_entities(gt, "all" if include_static else "dynamic_or_candidate"):
        lab = gt["entities"][e]["label"]
        if lab not in LABEL_QUERY:
            raise KeyError(f"GT label {lab!r} (entity {e}) has no detector phrase in LABEL_QUERY")
        q = LABEL_QUERY[lab]
        if q not in labs:
            labs.append(q)
    return labs


class IoUTracker:
    def __init__(self, iou_thresh=0.3, max_age=8):
        self.iou_thresh, self.max_age = iou_thresh, max_age
        self.tracks = []
        self.next_id = 1

    def update(self, boxes: np.ndarray, labels: list[str], scores: np.ndarray):
        out = []
        alive = [t for t in self.tracks if t["age"] <= self.max_age]
        for t in alive:
            t["age"] += 1
        if len(boxes) and alive:
            M = iou_matrix([t["box"] for t in alive], boxes)
            for i, t in enumerate(alive):
                for j in range(len(boxes)):
                    if t["label"] != labels[j]:
                        M[i, j] = 0.0
            used_t, used_d = set(), set()
            for flat in np.argsort(-M, axis=None):
                i, j = divmod(int(flat), M.shape[1])
                if M[i, j] < self.iou_thresh:
                    break
                if i in used_t or j in used_d:
                    continue
                used_t.add(i); used_d.add(j)
                alive[i].update(box=boxes[j], age=0)
                out.append((alive[i]["id"], labels[j], float(scores[j]), boxes[j]))
            new = [j for j in range(len(boxes)) if j not in used_d]
        else:
            new = list(range(len(boxes)))
        for j in new:
            t = dict(id=self.next_id, box=boxes[j], label=labels[j], age=0); self.next_id += 1
            alive.append(t); out.append((t["id"], labels[j], float(scores[j]), boxes[j]))
        self.tracks = alive
        return out


def _load_detector(model: str, device: str):
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    proc = AutoProcessor.from_pretrained(model)
    net = AutoModelForZeroShotObjectDetection.from_pretrained(model).to(device).eval()
    return proc, net, torch


def detect_batch(proc, net, torch, images, text: str, box_thresh: float, text_thresh: float, device: str):
    from PIL import Image
    pil = [Image.fromarray(im[:, :, ::-1]) for im in images]
    inputs = proc(images=pil, text=[text] * len(pil), return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = net(**inputs)
    sizes = torch.tensor([[im.shape[0], im.shape[1]] for im in images], device=device)
    try:
        res = proc.post_process_grounded_object_detection(outputs, inputs.input_ids, threshold=box_thresh,
                                                          text_threshold=text_thresh, target_sizes=sizes)
    except TypeError:
        res = proc.post_process_grounded_object_detection(outputs, inputs.input_ids, box_threshold=box_thresh,
                                                          text_threshold=text_thresh, target_sizes=sizes)
    out = []
    for r in res:
        labels = r.get("text_labels") or r.get("labels")
        out.append((r["boxes"].cpu().numpy(), [str(name) for name in labels], r["scores"].cpu().numpy()))
    return out


def run(video: str, queries: list[str], out: str, model: str, device: str = "cuda:0", batch: int = 8,
        box_thresh: float = 0.3, text_thresh: float = 0.25, frames: str = "grid", grid_hz: float = 8.0,
        max_per_label: int | None = None, detector=None):
    """`detector` = (proc, net, torch) from _load_detector; pass it to reuse one loaded model across videos."""
    meta = video_meta(video)
    if frames == "all":
        wanted = None
    else:
        sel = set()
        for t in eval_grid(grid_hz):
            f = min(output_frame_for_time(t, meta["fps"]), meta["frame_count"] - 1)
            sel.update({max(0, f - 1), f, min(meta["frame_count"] - 1, f + 1)})
        wanted = sel
    proc, net, torch = detector or _load_detector(model, device)
    text = ". ".join(queries) + "."
    tracker = IoUTracker()
    results = []
    buf = []

    def flush():
        if not buf:
            return
        dets = detect_batch(proc, net, torch, [b for _, b in buf], text, box_thresh, text_thresh, device)
        for (idx, _), (boxes, labels, scores) in zip(buf, dets):
            labels = [_canon(name, queries) for name in labels]
            keep = [k for k, name in enumerate(labels) if name is not None]
            boxes, labels, scores = boxes[keep], [labels[k] for k in keep], scores[keep]
            if max_per_label:
                keep = []
                for lab in set(labels):
                    idxs = [k for k, name in enumerate(labels) if name == lab]
                    idxs.sort(key=lambda k: -scores[k]); keep += idxs[:max_per_label]
                keep.sort(); boxes, labels, scores = boxes[keep], [labels[k] for k in keep], scores[keep]
            tr = tracker.update(np.asarray(boxes, dtype=np.float64).reshape(-1, 4), labels, np.asarray(scores))
            results.append(dict(frame=idx, detections=[dict(track_id=tid, label=lab, score=sc, box=[float(x) for x in bx])
                                                       for tid, lab, sc, bx in tr]))
        buf.clear()

    for idx, im in iter_video_frames(video, wanted):
        buf.append((idx, im))
        if len(buf) >= batch:
            flush()
    flush()
    results.sort(key=lambda r: r["frame"])
    payload = dict(schema="prowbench.tracks.v1", video=str(Path(video).resolve()), **meta, queries=queries,
                   detector=dict(model=model, box_thresh=box_thresh, text_thresh=text_thresh, frames=frames, grid_hz=grid_hz),
                   frames=results)
    write_json(out, payload)
    n = sum(len(r["detections"]) for r in results)
    print(f"{video}: {len(results)} frames, {n} detections, {tracker.next_id - 1} tracks -> {out}")
    return payload


def _load_sam2(model: str, device: str):
    import torch
    from transformers import Sam2VideoModel, Sam2VideoProcessor
    net = Sam2VideoModel.from_pretrained(model, dtype=torch.bfloat16).to(device).eval()
    proc = Sam2VideoProcessor.from_pretrained(model)
    return proc, net, torch


def mask_to_box(mask, min_pixels: int = 4):
    """Tight pixel AABB [x0,y0,x1,y1] of a binary mask, or None if fewer than min_pixels are set."""
    import numpy as np
    ys, xs = np.nonzero(mask)
    if len(xs) < min_pixels:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def _sam2_propagate(proc, net, torch, frames, gt, ent_ids, seeds, n_frames, meta, device, min_pixels, non_overlapping,
                    reseed_every, reseed_min_iou, detector, init_model, entities):
    """SAM2 propagation for the entities in `seeds` (entity -> dict(frame, box, ...)); returns (dets per frame,
    reseed log). track_id = 1-based index of the entity in ent_ids. Used by run_sam2 and by run_sam3's fallback."""
    dets = {i: [] for i in range(n_frames)}
    groups = {}
    for e, v in seeds.items():
        groups.setdefault(v["frame"], []).append((e, v["box"]))
    reseed_frames = []
    if reseed_every and reseed_every > 0:
        step = max(1, int(round(reseed_every * meta["fps"])))
        reseed_frames = list(range(step, n_frames, step))
    reseed_log = []
    if reseed_frames and detector is None:
        detector = _load_detector(init_model, device)
    if reseed_frames:
        dproc, dnet, _ = detector
        queries = queries_from_gt(gt, include_static=(entities == "all"))
        text = ". ".join(queries) + "."
        from .metric_entity import DEFAULT_LABEL_MAP, _label_ok

    def _collect(o, sess, box_now):
        m = proc.post_process_masks([o.pred_masks], original_sizes=[[sess.video_height, sess.video_width]],
                                    binarize=True, apply_non_overlapping_constraints=non_overlapping)[0]
        for i, oid in enumerate(sess.obj_ids):
            mk = m[i, 0].cpu().numpy().astype(bool)
            box = mask_to_box(mk, min_pixels)
            if box is None:
                continue
            e = ent_ids[oid - 1]
            box_now[oid] = box
            dets[o.frame_idx].append(dict(track_id=int(oid), label=gt["entities"][e]["label"], entity=e,
                                          score=float(mk.mean()), box=box))

    for f0, members in sorted(groups.items()):
        sess = proc.init_video_session(video=frames, inference_device=device, dtype=torch.bfloat16)
        obj_ids = [ent_ids.index(e) + 1 for e, _ in members]
        proc.add_inputs_to_inference_session(inference_session=sess, frame_idx=f0, obj_ids=obj_ids,
                                             input_boxes=[[box for _, box in members]])
        box_now = {}
        boundaries = [f for f in reseed_frames if f > f0] + [n_frames]
        with torch.inference_mode():
            _collect(net(inference_session=sess, frame_idx=f0), sess, box_now)
            cur = f0
            for nxt in boundaries:
                for o in net.propagate_in_video_iterator(sess, start_frame_idx=cur, max_frame_num_to_track=nxt - cur):
                    if o.frame_idx == cur:
                        continue
                    _collect(o, sess, box_now)
                if nxt >= n_frames:
                    break
                boxes, labels, scores = detect_batch(dproc, dnet, torch, [frames[nxt][:, :, ::-1]], text, 0.3, 0.25, device)[0]
                labels = [_canon(name, queries) for name in labels]
                new_ids, new_boxes = [], []
                for oid in sess.obj_ids:
                    e = ent_ids[oid - 1]; lab = gt["entities"][e]["label"]; B = box_now.get(oid)
                    if B is None:
                        continue
                    best, best_iou = None, 0.0
                    for bx, dl in zip(boxes, labels):
                        if dl is None or not _label_ok(lab, dl, DEFAULT_LABEL_MAP):
                            continue
                        i = iou_matrix([B], [bx])[0, 0]
                        if i > best_iou:
                            best, best_iou = [float(x) for x in bx], float(i)
                    if best is not None and best_iou >= reseed_min_iou:
                        new_ids.append(int(oid)); new_boxes.append(best)
                        reseed_log.append(dict(frame=nxt, entity=e, iou_with_track=round(best_iou, 3)))
                if new_ids:
                    dets[nxt] = [d for d in dets[nxt] if d["track_id"] not in new_ids]
                    proc.add_inputs_to_inference_session(inference_session=sess, frame_idx=nxt, obj_ids=new_ids,
                                                         input_boxes=[new_boxes])
                    _collect(net(inference_session=sess, frame_idx=nxt), sess, box_now)
                cur = nxt
        del sess
        torch.cuda.empty_cache()
    return dets, reseed_log


def run_sam2(video: str, gt: dict, sample: GeneratedSample, out: str, model: str, device: str = "cuda:0",
             grid_hz: float = 8.0, entities: str = "dynamic_or_candidate", tracker=None, min_pixels: int = 4,
             min_init_area: float = 1e-4, init: str = "gt", init_model: str = "IDEA-Research/grounding-dino-tiny",
             init_min_iou: float = 0.4, non_overlapping: bool = False, detector=None, reseed_every: float = 0.0,
             reseed_min_iou: float = 0.3):
    """Track the entities of one video with SAM 2.1, seeded by ground-truth boxes (default) or by detector boxes matched
    to them (`init="dino"`); writes prowbench.tracks.v1 to `out`."""
    proc, net, torch = tracker or _load_sam2(model, device)
    meta = video_meta(video)
    n_frames = min(meta["frame_count"], int(round(EVAL_DURATION * meta["fps"])))
    W, H = meta["width"], meta["height"]
    sp = sample.spatial
    frames = [im[:, :, ::-1].copy() for _, im in iter_video_frames(video, set(range(n_frames)))]

    ent_ids = evaluated_entities(gt, entities)
    seeds = {}
    for e in ent_ids:
        for t in eval_grid(grid_hz):
            b = gt["frames"][source_frame_for_time(t)]["boxes"].get(e)
            if not (b and b["visible"]):
                continue
            nb = sp.box_src_to_out_norm(b["box"])
            if (nb[2] - nb[0]) * (nb[3] - nb[1]) < min_init_area:
                continue
            of = min(output_frame_for_time(t, meta["fps"]), n_frames - 1)
            gb = [float(nb[0] * W), float(nb[1] * H), float(nb[2] * W), float(nb[3] * H)]
            seeds[e] = dict(frame=of, gt_box=gb, box=gb, source="gt", init_iou=None)
            break

    if init == "dino" and seeds:
        from .metric_entity import DEFAULT_LABEL_MAP, _label_ok
        dproc, dnet, _ = detector or _load_detector(init_model, device)
        queries = queries_from_gt(gt, include_static=(entities == "all"))
        text = ". ".join(queries) + "."
        for f0 in sorted({v["frame"] for v in seeds.values()}):
            boxes, labels, scores = detect_batch(dproc, dnet, torch, [frames[f0][:, :, ::-1]], text, 0.3, 0.25, device)[0]
            labels = [_canon(name, queries) for name in labels]
            for e, v in seeds.items():
                if v["frame"] != f0:
                    continue
                lab = gt["entities"][e]["label"]
                best, best_iou = None, 0.0
                for bx, dl in zip(boxes, labels):
                    if dl is None or not _label_ok(lab, dl, DEFAULT_LABEL_MAP):
                        continue
                    i = iou_matrix([v["gt_box"]], [bx])[0, 0]
                    if i > best_iou:
                        best, best_iou = [float(x) for x in bx], float(i)
                v["init_iou"] = best_iou
                if best is not None and best_iou >= init_min_iou:
                    v["box"], v["source"] = best, "dino"

    dets, reseed_log = _sam2_propagate(proc, net, torch, frames, gt, ent_ids, seeds, n_frames, meta, device, min_pixels,
                                       non_overlapping, reseed_every, reseed_min_iou, detector, init_model, entities)
    reseed_frames = bool(reseed_every and reseed_every > 0)
    results = [dict(frame=i, detections=dets[i]) for i in range(n_frames)]
    payload = dict(schema="prowbench.tracks.v1", video=str(Path(video).resolve()), **meta,
                   queries=[gt["entities"][e]["label"] for e in ent_ids],
                   tracker=dict(name=f"sam2_{init}init" + ("_nonoverlap" if non_overlapping else "") + (f"_reseed{reseed_every:g}s" if reseed_frames else ""), model=model,
                                init_mode=init, init_model=init_model if init == "dino" else None, init_min_iou=init_min_iou,
                                non_overlapping=non_overlapping, grid_hz=grid_hz, min_pixels=min_pixels,
                                reseed_every_s=reseed_every, reseed_min_iou=reseed_min_iou, reseeds=reseed_log,
                                init=seeds, not_initialised=[e for e in ent_ids if e not in seeds]),
                   frames=results)
    write_json(out, payload)
    n = sum(len(r["detections"]) for r in results)
    src = {k: sum(1 for v in seeds.values() if v["source"] == k) for k in ("dino", "gt")}
    print(f"{video}: sam2 {len(results)} frames, {n} boxes, {len(seeds)}/{len(ent_ids)} entities seeded (dino {src['dino']}, gt {src['gt']}) -> {out}")
    return payload


def _load_sam3(model: str, device: str, dtype: str = "bfloat16"):
    import torch
    from transformers import Sam3VideoModel, Sam3VideoProcessor
    net = Sam3VideoModel.from_pretrained(model, dtype=getattr(torch, dtype)).to(device).eval()
    proc = Sam3VideoProcessor.from_pretrained(model)
    return proc, net, torch


SAM3_PROMPT = {}


def run_sam3(video: str, gt: dict, sample: GeneratedSample, out: str, model: str, device: str = "cuda:0",
             grid_hz: float = 8.0, entities: str = "dynamic_or_candidate", tracker=None, min_pixels: int = 4,
             dtype: str = "bfloat16", score_thresh: float | None = None, prompt_overrides: dict | None = None,
             fallback: str = "none", sam2_model: str | None = None, sam2=None, frames_mode: str = "all"):
    """Detect and track the entities of one video with SAM 3 text prompts (optional SAM 2.1 fallback); writes
    prowbench.tracks.v1 to `out`."""
    import numpy as np
    proc, net, torch = tracker or _load_sam3(model, device, dtype)
    meta = video_meta(video)
    n_frames = min(meta["frame_count"], int(round(EVAL_DURATION * meta["fps"])))
    if frames_mode == "all":
        frame_ids = list(range(n_frames))
    elif frames_mode == "grid":
        if fallback != "none":
            raise NotImplementedError("sam2 fallback is not supported with frames_mode='grid'")
        frame_ids = sorted({min(output_frame_for_time(t, meta["fps"]), n_frames - 1) for t in eval_grid(grid_hz)})
    else:
        raise ValueError(f"Unknown frames_mode {frames_mode!r}")
    frames = [im[:, :, ::-1].copy() for _, im in iter_video_frames(video, set(frame_ids))]
    if len(frames) != len(frame_ids):
        raise ValueError(f"Video decode returned {len(frames)} of {len(frame_ids)} requested frames: {video}")
    ent_ids = evaluated_entities(gt, entities)
    labels = []
    for e in ent_ids:
        lab = gt["entities"][e]["label"]
        if lab not in LABEL_QUERY:
            raise KeyError(f"GT label {lab!r} (entity {e}) has no prompt phrase in LABEL_QUERY")
        if lab not in labels:
            labels.append(lab)
    pov = dict(SAM3_PROMPT); pov.update(prompt_overrides or {})
    prompts = {pov.get(name, LABEL_QUERY[name]): name for name in labels}
    if score_thresh is not None:
        net.score_threshold_detection = float(score_thresh)
    sess = proc.init_video_session(video=frames, inference_device=device, processing_device="cpu",
                                   video_storage_device="cpu", dtype=getattr(torch, dtype))
    sess = proc.add_text_prompt(inference_session=sess, text=list(prompts))
    n_fed = len(frame_ids)
    dets = {i: [] for i in range(n_fed)}
    obj_label = {}
    with torch.inference_mode():
        for o in net.propagate_in_video_iterator(inference_session=sess, start_frame_idx=0, max_frame_num_to_track=n_fed):
            po = proc.postprocess_outputs(sess, o)
            if o.frame_idx >= n_fed:
                continue
            p2o = po.get("prompt_to_obj_ids") or {}
            for phrase, ids in p2o.items():
                for oid in ids:
                    obj_label.setdefault(int(oid), prompts.get(phrase, phrase))
            ids = po["object_ids"].tolist() if hasattr(po["object_ids"], "tolist") else list(po["object_ids"])
            for i, oid in enumerate(ids):
                mk = po["masks"][i]
                mk = mk.cpu().numpy().astype(bool) if hasattr(mk, "cpu") else np.asarray(mk, bool)
                box = mask_to_box(mk, min_pixels)
                if box is None:
                    continue
                sc = po["scores"][i]; sc = float(sc.item() if hasattr(sc, "item") else sc)
                dets[o.frame_idx].append(dict(track_id=int(oid) + 1, label=obj_label.get(int(oid), "?"), score=sc, box=box))
    del sess
    torch.cuda.empty_cache()
    fb_info = {}
    if fallback not in ("none", "sam2", "sam2_unmatched"):
        raise ValueError(f"Unknown sam3 fallback {fallback!r}")
    unmatched = []
    if fallback in ("sam2", "sam2_unmatched"):
        found_labels = set(obj_label.values())
        missing_labels = [name for name in labels if name not in found_labels]
        fb_ents = [e for e in ent_ids if gt["entities"][e]["label"] in missing_labels]
        if fallback == "sam2_unmatched":
            from .metric_entity import evaluate as _entity_eval
            pre = dict(meta, frames=[dict(frame=int(frame_ids[i]), detections=dets[i]) for i in range(n_fed)])
            asg = _entity_eval(gt, pre, sample, grid_hz=grid_hz, entities=entities)["assignment"]
            unmatched = [e for e in ent_ids if e not in asg and e not in fb_ents]
            fb_ents = fb_ents + unmatched
        if fb_ents:
            sp = sample.spatial; W, H = meta["width"], meta["height"]
            seeds = {}
            for e in fb_ents:
                for t in eval_grid(grid_hz):
                    b = gt["frames"][source_frame_for_time(t)]["boxes"].get(e)
                    if not (b and b["visible"]):
                        continue
                    nb = sp.box_src_to_out_norm(b["box"])
                    if (nb[2] - nb[0]) * (nb[3] - nb[1]) < 1e-4:
                        continue
                    of = min(output_frame_for_time(t, meta["fps"]), n_frames - 1)
                    gb = [float(nb[0] * W), float(nb[1] * H), float(nb[2] * W), float(nb[3] * H)]
                    seeds[e] = dict(frame=of, gt_box=gb, box=gb, source="gt", init_iou=None)
                    break
            s2proc, s2net, _ = sam2 or _load_sam2(sam2_model, device)
            dets2, _ = _sam2_propagate(s2proc, s2net, torch, frames, gt, ent_ids, seeds, n_frames, meta, device, min_pixels,
                                       False, 0.0, 0.3, None, "IDEA-Research/grounding-dino-tiny", entities)
            for i in range(n_frames):
                for d in dets2[i]:
                    dets[i].append(dict(track_id=d["track_id"] + 1000, label=d["label"], entity=d["entity"], score=d["score"],
                                        box=d["box"], source="sam2_gtbox"))
            fb_info = dict(labels=missing_labels, entities=fb_ents, unmatched=unmatched, seeds=seeds, sam2_model=sam2_model)
    results = [dict(frame=int(frame_ids[i]), detections=dets[i]) for i in range(n_fed)]
    cfg = getattr(net, "config", None)
    cfg_note = {k: getattr(cfg, k) for k in ("score_threshold_detection", "det_nms_thresh", "hotstart_delay",
                                              "hotstart_unmatch_thresh", "max_num_objects", "high_conf_thresh") if hasattr(cfg, k)}
    payload = dict(schema="prowbench.tracks.v1", video=str(Path(video).resolve()), **meta, queries=list(prompts),
                   tracker=dict(name="sam3_text", model=model, dtype=dtype, prompts=prompts, grid_hz=grid_hz,
                                frames_mode=frames_mode, n_frames_fed=n_fed,
                                min_pixels=min_pixels, config=cfg_note, score_thresh=net.score_threshold_detection,
                                n_objects=len(obj_label), fallback=fallback, fallback_used=fb_info),
                   frames=results)
    write_json(out, payload)
    n = sum(len(r["detections"]) for r in results)
    print(f"{video}: sam3 {len(results)} frames, {n} boxes, {len(obj_label)} objects for prompts {list(prompts)}"
          + (f", sam2 fallback for {fb_info['entities']}" if fb_info else "") + f" -> {out}")
    return payload


def run_jobs(jobs: list[dict], model: str, device: str = "cuda:0", tracker: str = "dino", **kw):
    """Process many videos with ONE loaded model. Each job: {"video", "gt" | "queries", "out", ["include_static"],
    ["sample"] (GeneratedSample dict, required for tracker=sam2)}. Jobs whose `out` exists are skipped unless
    job["force"]. `tracker` = "dino" (Grounding-DINO + IoU tracker) or "sam2" (GT-box-initialised SAM 2.1)."""
    loaded = None; det = None
    done = 0
    sam2_kw = dict(init=kw.pop("sam2_init", "gt"), init_model=kw.pop("sam2_init_model", "IDEA-Research/grounding-dino-tiny"),
                   non_overlapping=kw.pop("sam2_nonoverlap", False), reseed_every=kw.pop("sam2_reseed", 0.0))
    for j in jobs:
        if Path(j["out"]).exists() and not j.get("force"):
            continue
        if tracker == "sam3":
            if loaded is None:
                loaded = _load_sam3(model, device)
            if kw.get("sam3_fallback") in ("sam2", "sam2_unmatched") and det is None:
                det = _load_sam2(kw["sam2_model"], device)
            run_sam3(j["video"], read_json(j["gt"]), GeneratedSample.from_dict(j["sample"]), j["out"], model, device,
                     grid_hz=kw.get("grid_hz", 8.0),
                     entities=j.get("entities") or ("all" if j.get("include_static") else "dynamic_or_candidate"),
                     tracker=loaded, score_thresh=kw.get("sam3_score_thresh"), prompt_overrides=kw.get("sam3_prompts"),
                     fallback=kw.get("sam3_fallback", "none"), sam2_model=kw.get("sam2_model"), sam2=det,
                     frames_mode=j.get("frames_mode", "all"))
        elif tracker == "sam2":
            if loaded is None:
                loaded = _load_sam2(model, device)
            if det is None and (sam2_kw["init"] == "dino" or sam2_kw["reseed_every"]):
                det = _load_detector(sam2_kw["init_model"], device)
            run_sam2(j["video"], read_json(j["gt"]), GeneratedSample.from_dict(j["sample"]), j["out"], model, device,
                     grid_hz=kw.get("grid_hz", 8.0), entities="all" if j.get("include_static") else "dynamic_or_candidate",
                     tracker=loaded, detector=det, **sam2_kw)
        else:
            if loaded is None:
                loaded = _load_detector(model, device)
            q = list(j.get("queries") or queries_from_gt(read_json(j["gt"]), j.get("include_static", False)))
            q += [x for x in j.get("extra_queries", []) if x not in q]
            run(j["video"], q, j["out"], model, device, detector=loaded, **kw)
        done += 1
    print(f"run_jobs[{tracker}]: {done} processed, {len(jobs) - done} skipped")


def _canon(label: str, queries: list[str]):
    """Map detector phrase back to one of our queries (GroundingDINO may merge/split words)."""
    phrase = label.lower().strip()
    if not phrase:
        return None
    for q in queries:
        if phrase == q or q in phrase or phrase in q:
            return q
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", help="single video (or use --jobs)")
    ap.add_argument("--jobs", help="json list of {video, gt|queries, out} processed with one loaded model")
    ap.add_argument("--gt", help="GT json; queries are derived from its entity labels")
    ap.add_argument("--queries", nargs="*", help="explicit query phrases (override --gt)")
    ap.add_argument("--extra-queries", nargs="*", default=[])
    ap.add_argument("--include-static", action="store_true", help="also query trees/buildings")
    ap.add_argument("--out", help="output json (single-video mode)")
    ap.add_argument("--model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--box-thresh", type=float, default=0.3)
    ap.add_argument("--text-thresh", type=float, default=0.25)
    ap.add_argument("--frames", default="grid", choices=["grid", "all"])
    ap.add_argument("--grid-hz", type=float, default=8.0)
    ap.add_argument("--max-per-label", type=int, default=None)
    ap.add_argument("--tracker", default="dino", choices=["dino", "sam2", "sam3"],
                    help="dino: Grounding-DINO + IoU tracker; sam2: SAM 2.1 video tracker initialised by GT boxes; "
                         "sam3: SAM 3 text-prompted instance detection + tracking (needs transformers>=5)")
    ap.add_argument("--sample", default=None, help="samples.jsonl row (json string or path); required for --tracker sam2")
    ap.add_argument("--sam2-init", default="gt", choices=["gt", "dino"], help="seed box: GT box (default) or detector box matched to GT")
    ap.add_argument("--sam2-init-model", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--sam2-nonoverlap", action="store_true", help="enable SAM2 non-overlapping mask post-processing")
    ap.add_argument("--sam2-reseed", type=float, default=0.0, help="seconds between detector re-anchoring prompts (0 = off)")
    ap.add_argument("--sam3-score-thresh", type=float, default=None, help="SAM 3 detection score threshold (model default 0.5)")
    ap.add_argument("--sam3-prompts", default=None, help="json dict GT label -> prompt phrase overrides")
    ap.add_argument("--sam3-fallback", default="none", choices=["none", "sam2", "sam2_unmatched"], help="track labels SAM 3 never detects with SAM2 GT-box seeding (sam2_unmatched: also entities left unassigned; engine renders only)")
    ap.add_argument("--sam2-model", default=None, help="SAM 2.1 checkpoint for --sam3-fallback sam2")
    ap.add_argument("--force", action="store_true", help="with --jobs: recompute jobs whose output exists")
    args = ap.parse_args(argv)
    if args.tracker == "sam3" and args.sam3_fallback != "none" and not args.sam2_model:
        ap.error("--sam3-fallback " + args.sam3_fallback + " needs --sam2-model (a SAM 2.1 checkpoint)")
    kw = dict(batch=args.batch, box_thresh=args.box_thresh, text_thresh=args.text_thresh, frames=args.frames,
              grid_hz=args.grid_hz, max_per_label=args.max_per_label)
    if args.tracker == "sam2":
        kw.update(sam2_init=args.sam2_init, sam2_init_model=args.sam2_init_model, sam2_nonoverlap=args.sam2_nonoverlap, sam2_reseed=args.sam2_reseed)
    if args.tracker == "sam3":
        kw.update(sam3_score_thresh=args.sam3_score_thresh, sam3_prompts=json.loads(args.sam3_prompts) if args.sam3_prompts else None,
                  sam3_fallback=args.sam3_fallback, sam2_model=args.sam2_model)
    if args.jobs:
        jobs = read_json(args.jobs)
        if args.force:
            jobs = [dict(j, force=True) for j in jobs]
        run_jobs(jobs, args.model, args.device, tracker=args.tracker, **kw)
        return
    if args.tracker in ("sam2", "sam3"):
        if not (args.video and args.out and args.gt and args.sample):
            raise SystemExit(f"--tracker {args.tracker} needs --video, --gt, --sample and --out")
        s = args.sample
        sample = GeneratedSample.from_dict(json.loads(s) if s.strip().startswith("{") else read_json(s))
        if args.tracker == "sam3":
            run_sam3(args.video, read_json(args.gt), sample, args.out, args.model, args.device, grid_hz=args.grid_hz,
                     entities="all" if args.include_static else "dynamic_or_candidate", score_thresh=args.sam3_score_thresh,
                     prompt_overrides=json.loads(args.sam3_prompts) if args.sam3_prompts else None,
                     fallback=args.sam3_fallback, sam2_model=args.sam2_model)
            return
        run_sam2(args.video, read_json(args.gt), sample, args.out, args.model, args.device, grid_hz=args.grid_hz,
                 entities="all" if args.include_static else "dynamic_or_candidate", init=args.sam2_init,
                 init_model=args.sam2_init_model, non_overlapping=args.sam2_nonoverlap, reseed_every=args.sam2_reseed)
        return
    if not args.video or not args.out:
        raise SystemExit("need --video and --out, or --jobs")
    if args.queries:
        q = list(args.queries)
    elif args.gt:
        q = queries_from_gt(read_json(args.gt), args.include_static)
    else:
        raise SystemExit("need --gt or --queries")
    q += [x for x in args.extra_queries if x not in q]
    run(args.video, q, args.out, args.model, args.device, args.batch, args.box_thresh, args.text_thresh,
        args.frames, args.grid_hz, args.max_per_label)


if __name__ == "__main__":
    main()
