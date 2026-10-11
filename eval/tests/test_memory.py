"""Tests of the memory metrics (`metric_memory`, `memory_state`)."""
import os
import subprocess
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prowbench_eval.common import GeneratedSample, evaluated_entities
from prowbench_eval.metric_memory import BUCKETS, absence_events, bucket_of, evaluate, event_windows

ROOT = Path(__file__).resolve().parents[1]


def _rows(vis_pattern: str, hz: float = 8.0, pred=True):
    """'1' visible with a matching box, '0' invisible, 'm' visible but missing prediction."""
    rows = []
    for k, c in enumerate(vis_pattern):
        t = k / hz
        if c == "0":
            rows.append(dict(t=t, gt=None))
        elif c == "m":
            rows.append(dict(t=t, gt=[0.1, 0.1, 0.2, 0.2], pred=None, iou=0.0, center_err=None))
        else:
            rows.append(dict(t=t, gt=[0.1, 0.1, 0.2, 0.2], pred=[0.1, 0.1, 0.2, 0.2], iou=1.0 if pred else 0.5, center_err=0.0))
    return rows


def test_bucket_edges():
    assert bucket_of(0.5) == "le2" and bucket_of(2.0) == "le2"
    assert bucket_of(2.01) == "2to5" and bucket_of(5.0) == "2to5"
    assert bucket_of(7.5) == "5to10" and bucket_of(10.0) == "5to10"
    assert bucket_of(10.1) == "gt10"
    assert [b[0] for b in BUCKETS] == ["le2", "2to5", "5to10", "gt10"]


def test_absence_events_bounded_and_min_length():
    rows = _rows("1" * 8 + "0" * 12 + "1" * 8)
    ev = absence_events(rows, min_absence=1.0)
    assert len(ev) == 1
    assert ev[0]["leave_index"] == 7 and ev[0]["return_index"] == 20
    assert abs(ev[0]["absence"] - (20 - 7) / 8) < 1e-9 and ev[0]["bucket"] == "le2"
    assert absence_events(_rows("1111" + "000" + "1111"), 1.0) == []
    assert absence_events(_rows("000" + "1" * 8 + "0" * 12), 1.0) == []
    assert len(absence_events(_rows("1111" + "000" + "1111"), 0.3)) == 1


def test_event_windows_and_stats():
    rows = _rows("1" * 16 + "0" * 40 + "mmmm" + "1" * 12)
    ev = absence_events(rows, 1.0)[0]
    assert ev["bucket"] == "5to10"
    pre, post = event_windows(rows, ev, window=1.0)
    assert pre == list(range(8, 16)) and post == list(range(56, 64))
    gt = dict(entities={"a": dict(label="person", dynamic=True)}, frames=[], width=1280, height=720)
    W, H = 1344, 768
    frames = []
    for k, r in enumerate(rows):
        dets = []
        if r.get("pred") is not None:
            dets.append(dict(track_id=1, label="person", score=1.0, box=[0.1 * W, 0.1 * H, 0.2 * W, 0.2 * H]))
        if 56 <= k < 60:
            dets.append(dict(track_id=2, label="person", score=1.0, box=[0.1 * W, 0.1 * H, 0.2 * W, 0.2 * H]))
        frames.append(dict(frame=k, detections=dets))
    pred = dict(width=W, height=H, fps=24.0, frame_count=len(rows), frames=frames)
    ent = dict(per_entity={"a": dict(label="person", track_id=1, instants=rows, mean_iou=0.9, miss_rate=0.1)}, assignment={"a": 1},
               summary=dict(mean_iou=0.9, mean_center_err=0.01, miss_rate=0.1, n_instants=len(rows), n_entities=1, tracker="t"),
               alignment=dict(instants=[r["t"] for r in rows], source_frames=[], output_frames=list(range(len(rows))),
                              crop_xywh=[0, 0, 1280, 720], out_size=[W, H]))
    s = GeneratedSample(sample_id="x/follow/white", episode="x", camera="follow", representation="white", method="m", run_id="r",
                        video="v", fps=24, width=W, height=H, frame_count=720)
    res = evaluate(gt, pred, s, entity_metric=ent)
    e = res["events"][0]
    assert e["post"]["n"] == 8 and e["post"]["n_found"] == 8
    assert abs(e["post"]["rate"] - 1.0) < 1e-9 and abs(e["post"]["iou"] - 1.0) < 1e-9 and abs(e["post"]["iou_present"] - 1.0) < 1e-9
    assert abs(e["post"]["center_err"]) < 1e-9 and e["post"]["boxes"][0] is not None
    assert abs(e["post"]["tracked_rate"] - 0.5) < 1e-9 and abs(e["post"]["tracked_iou"] - 0.5) < 1e-9
    assert abs(e["pre"]["rate"] - 1.0) < 1e-9 and abs(e["pre"]["iou"] - 1.0) < 1e-9
    sm = res["summary"]
    assert sm["n_events"] == 1 and abs(sm["reappear_rate"] - 1.0) < 1e-9 and abs(sm["tracked_reappear_iou"] - 0.5) < 1e-9
    assert sm["clip"]["tracked_iou"] == 0.9 and abs(sm["clip"]["found_rate"] - 1.0) < 1e-9
    assert res["per_entity"]["a"]["instants"][56]["best_track"] == 2 and res["per_entity"]["a"]["instants"][8]["best_track"] == 1
    assert res["by_bucket"]["5to10"]["n_events"] == 1 and res["by_bucket"]["le2"]["n_events"] == 0
    assert res["by_bucket"]["le2"]["reappear_rate"] is None
    assert abs(e["post"]["rate_lenient"] - 1.0) < 1e-9 and e["weak"] == [] and sm["strong"]["n_events"] == 1 and sm["n_weak"] == 0


def test_lenient_match_and_weak_tags():
    from prowbench_eval.metric_memory import best_match_rows, weak_reasons
    rows = _rows("1" * 8 + "0" * 8 + "1" * 3)
    W, H = 1344, 768
    for r in rows:
        if r.get("gt") is not None:
            r["gt"] = [0.1, 0.1, 0.12, 0.2]
    frames = [dict(frame=k, detections=([dict(track_id=3, label="person", score=1.0, box=[0.14 * W, 0.1 * H, 0.16 * W, 0.2 * H])] if k >= 16 else []))
              for k in range(len(rows))]
    pred = dict(width=W, height=H, fps=24.0, frame_count=len(rows), frames=frames)
    from prowbench_eval.metric_entity import _pred_index
    best = best_match_rows(rows, "person", _pred_index(pred), list(range(len(rows))))
    assert best[16]["best_box"] is None and best[16]["near_dist"] is not None and abs(best[16]["near_dist"] - 0.4) < 1e-9
    assert best[16]["near_box"] is not None and best[7]["near_dist"] is None
    from prowbench_eval.metric_memory import _window_stats
    ws = _window_stats(best, rows, [16, 17, 18])
    assert ws["n_found"] == 0 and ws["n_near"] == 3 and ws["near_boxes"][0] == best[16]["near_box"]
    ev = absence_events(rows, 1.0)[0]
    assert weak_reasons(rows, ev, 8.0, H) == ["short_return"]
    rows2 = _rows("1" * 8 + "0" * 8 + "1" * 8)
    rows2[16]["gt"] = [0.0, 0.1, 0.05, 0.2]
    ev2 = absence_events(rows2, 1.0)[0]
    assert weak_reasons(rows2, ev2, 8.0, H) == ["edge_reentry"]
    rows3 = _rows("1" * 8 + "0" * 8 + "1" * 8)
    rows3[16]["gt"] = [0.1, 0.1, 0.11, 0.115]
    assert weak_reasons(rows3, absence_events(rows3, 1.0)[0], 8.0, H) == ["small_box"]


def test_state_gate_not_judgeable():
    from prowbench_eval.memory_state import detection_zoom_box
    box = detection_zoom_box([[0.5, 0.5, 0.52, 0.55], None, [0.51, 0.5, 0.53, 0.56]], 1344, 768)
    assert box[0] < 0.5 * 1344 < box[2] and box[1] < 0.5 * 768 < box[3] and (box[2] - box[0]) >= 0.25 * 1344 - 1
    try:
        detection_zoom_box([None, None], 1344, 768)
        raise AssertionError("ValueError expected")
    except ValueError:
        pass


def test_evaluated_entities_dynamic_only():
    gt = dict(entities={"p": dict(label="person", dynamic=True, interaction_candidate=True),
                        "car": dict(label="vehicle", dynamic=False, interaction_candidate=True),
                        "ground": dict(label="terrain", dynamic=True)})
    assert evaluated_entities(gt, "dynamic") == ["p"]
    assert evaluated_entities(gt, "dynamic_or_candidate") == ["p", "car"]
    try:
        evaluated_entities(gt, "bogus")
        raise AssertionError("ValueError expected")
    except ValueError:
        pass


def test_eval_duration_env_override():
    code = ("import prowbench_eval.common as c; from prowbench_eval.common import eval_grid, source_frame_for_time; "
            "print(c.EVAL_DURATION, len(eval_grid(8.0)), source_frame_for_time(29.99))")
    env = dict(os.environ, PROWBENCH_EVAL_DURATION="30", PYTHONPATH=str(ROOT))
    out = subprocess.check_output([sys.executable, "-c", code], env=env, cwd=ROOT, text=True).split()
    assert out == ["30.0", "240", "899"], out
    env = dict(os.environ, PYTHONPATH=str(ROOT)); env.pop("PROWBENCH_EVAL_DURATION", None); env.pop("GAMEBENCH_EVAL_DURATION", None)
    out = subprocess.check_output([sys.executable, "-c", code], env=env, cwd=ROOT, text=True).split()
    assert out == ["5.0", "40", "149"], out


def test_state_judges_only_people():
    from types import SimpleNamespace
    from unittest.mock import patch
    import numpy as np
    from prowbench_eval import memory_state
    gt = dict(fps=30, source=dict(episode_dir="unused"), entities=dict(npc=dict(label="person"), cart=dict(label="cart")))
    post = dict(near_boxes=[[0.4, 0.4, 0.5, 0.6]], n_near=1)
    memory = dict(events=[dict(entity=e, leave_t=1.0, return_t=3.0, absence=2.0, bucket="le2", weak=[], post=post) for e in ("npc", "cart")])
    judged = []
    def judge_event(sample, gt, ev, judge, n_frames, window, crop_margin):
        judged.append(ev["entity"])
        return dict(answer="walking")
    with patch.object(memory_state, "entity_speed", return_value=np.full(300, 1.0)), \
         patch.object(memory_state, "judge_event", judge_event):
        out = memory_state.evaluate(dict(sample_id="s", method="m", run_id="r"), gt, memory, SimpleNamespace(model_name="j"))
    assert judged == ["npc"] and out["summary"]["excluded_entities"] == ["cart"]
    assert out["summary"]["n_events"] == 1 and out["summary"]["state_persist"] == 1.0


def test_state_without_people_is_undefined():
    from types import SimpleNamespace
    from unittest.mock import patch
    from prowbench_eval import memory_state
    gt = dict(fps=30, source=dict(episode_dir="unused"), entities=dict(cart=dict(label="cart")))
    post = dict(near_boxes=[[0.4, 0.4, 0.5, 0.6]], n_near=1)
    memory = dict(events=[dict(entity=e, leave_t=1.0, return_t=3.0, absence=2.0, bucket="le2", weak=[], post=post)
                          for e in ("cart", "not-in-gt")])
    with patch.object(memory_state, "judge_event", side_effect=AssertionError("judged")):
        out = memory_state.evaluate(dict(sample_id="s", method="m", run_id="r"), gt, memory, SimpleNamespace(model_name="j"))
    assert out["summary"]["excluded_entities"] == ["cart", "not-in-gt"]
    assert out["summary"]["n_events"] == 0 and out["summary"]["state_persist"] is None
