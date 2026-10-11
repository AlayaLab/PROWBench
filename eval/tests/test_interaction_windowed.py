"""Tests of the ISR judging module `interaction_windowed`."""
from types import SimpleNamespace
from unittest.mock import patch
import json
import tempfile
from pathlib import Path
import numpy as np

from prowbench_eval import interaction_windowed as m

REG = dict(instances=[
    dict(kind='phase', episode='ep', name='a', start_time=0.0, end_time=1.0, event_id='ep/a'),
    dict(kind='phase', episode='ep', name='b', start_time=1.0, end_time=1.5, event_id='ep/b'),
    dict(kind='phase', episode='ep', name='c', start_time=1.5, end_time=5.0, event_id='ep/c'),
    dict(kind='state_change', episode='ep', name='a', start_time=0.0, end_time=9.0, event_id='x')])
TABLE = dict(window_pad_seconds=0.1, episodes=dict(ep=[
    dict(name='first', phases=['a'], participants=['player', 'thing'], action='A happens', outcome='A done'),
    dict(name='merged', phases=['b', 'c'], participants=['player'], action='B then C', outcome='C done')]))


def gt(w=1280, h=720, fps=30, n=150):
    frames = []
    for i in range(n):
        t = i / fps
        frames.append(dict(frame=i, time=t, boxes={'player': dict(box=[600, 300, 640, 400], visible=True),
                                                  'thing': dict(box=[700 + 100 * t, 350, 720 + 100 * t, 380], visible=t < 2.0)}))
    return dict(width=w, height=h, fps=fps, frames=frames)


def _table(obj=TABLE):
    p = Path(tempfile.mkdtemp()) / 'visual.json'; p.write_text(json.dumps(obj)); return str(p)


def test_visual_events_merge_registered_phases_and_pad():
    ev = m.load_visual_events(_table(), 'ep', REG)
    assert [e['name'] for e in ev] == ['first', 'merged']
    assert ev[1]['start'] == 1.0 and ev[1]['end'] == 5.0 and ev[1]['phase_ids'] == ['ep/b', 'ep/c']
    assert ev[0]['window'] == (0.0, 1.1) and ev[1]['window'] == (0.9, 5.1)


def test_unregistered_phase_is_an_error():
    bad = dict(TABLE, episodes=dict(ep=[dict(name='x', phases=['zzz'], participants=[], action='', outcome='')]))
    try:
        m.load_visual_events(_table(bad), 'ep', REG)
    except ValueError:
        return
    raise AssertionError('unregistered phase accepted')


def test_zoom_box_uses_only_window_and_visible_boxes():
    g = gt()
    x0, y0, x1, y1 = m.zoom_box(g, ['player', 'thing'], 0.0, 1.0, margin=0.0)
    assert x0 <= 600 and x1 >= 820 and x1 < 900
    x0, y0, x1, y1 = m.zoom_box(g, ['thing'], 3.0, 4.0, margin=0.0)
    assert (x0, y0, x1, y1) == (0, 0, 1280, 720)
    x0, y0, x1, y1 = m.zoom_box(g, ['player'], 0.0, 1.0, margin=0.0, min_frac=0.25)
    assert x1 - x0 >= 0.25 * 1280 - 2 and y1 - y0 >= 0.25 * 720 - 2


def test_composite_scales_box_from_source_to_generated_resolution():
    frame = np.zeros((360, 640, 3), np.uint8); frame[150:200, 300:320] = 255
    out = m.composite(frame, (600, 300, 640, 400), (1280, 720), tile_h=100)
    assert out.shape[0] == 100 and out.shape[1] > int(640 * 100 / 360) + 6
    zoom = out[:, int(640 * 100 / 360) + 6:]
    assert zoom.mean() > 100


def test_judge_sees_only_window_frames_and_expectations():
    ev = m.load_visual_events(_table(), 'ep', REG)[0]
    seen = []
    judge = SimpleNamespace(ask=lambda frames, prompt: (seen.append((frames, prompt)) or dict(parsed=dict(
        description='d', end_state='e', action='yes', action_evidence=[0, 33, 999], outcome='no', outcome_evidence=[33], participants='yes'))))
    fake = [(i, np.zeros((72, 128, 3), np.uint8)) for i in (0, 11, 22, 33)]
    with patch.object(m, 'sample_frames', return_value=fake) as sf:
        r = m.judge_event('/v.mp4', 30, 150, ev, gt(), judge, n_frames=4)
    assert sf.call_args[0][1:4] == (0.0, 1.1, 4)
    frames, prompt = seen[0]
    assert [i for i, _ in frames] == [0, 11, 22, 33] and 'A happens' in prompt and 'A done' in prompt and '/v.mp4' not in prompt
    assert r['answers'] == dict(action='yes', outcome='no', participants='yes') and r['correct'] is False
    assert r['evidence']['action'] == [0, 33]


def test_invalid_answers_are_recorded_not_zeroed():
    from prowbench_eval.vlm_judge import JudgeResponseError
    events = m.load_visual_events(_table(), 'ep', REG)
    calls = iter([dict(parsed=dict(action='yes', outcome='yes', participants='maybe')), JudgeResponseError('judge answer is not JSON')])
    def ask(frames, prompt):
        c = next(calls)
        if isinstance(c, Exception): raise c
        return c
    judge = SimpleNamespace(ask=ask)
    fake = [(i, np.zeros((72, 128, 3), np.uint8)) for i in range(4)]
    sample = dict(sample_id='ep/cam/white', method='m', run_id='r', video=__file__, fps=30, frame_count=150)
    with patch.object(m, 'sample_frames', return_value=fake):
        out = m.evaluate_sample(sample, events, gt(), judge, n_frames=4)
    s = out['summary']
    assert s['n_events'] == 2 and s['n_invalid'] == 2 and s['n_correct'] == 0 and s['interaction_success'] == 0.0
    assert out['events'][0]['invalid'] and out['events'][1]['error'].startswith('JudgeResponseError') and len(out['failures']) == 1


def test_judge_execution_error_fails_the_sample():
    events = m.load_visual_events(_table(), 'ep', REG)
    def ask(frames, prompt):
        raise RuntimeError('model could not be loaded')
    fake = [(i, np.zeros((72, 128, 3), np.uint8)) for i in range(4)]
    sample = dict(sample_id='ep/cam/white', method='m', run_id='r', video=__file__, fps=30, frame_count=150)
    with patch.object(m, 'sample_frames', return_value=fake):
        try:
            m.evaluate_sample(sample, events, gt(), SimpleNamespace(ask=ask), n_frames=4)
        except RuntimeError:
            return
    raise AssertionError('an execution error was recorded as an answer')


def test_score_sample_rates():
    rs = [dict(invalid=False, correct=True, answers=dict(action='yes', outcome='yes', participants='yes')),
          dict(invalid=False, correct=False, answers=dict(action='yes', outcome='unknown', participants='yes')),
          dict(invalid=False, correct=False, answers=dict(action='no', outcome='no', participants='no')),
          dict(invalid=True, correct=False, answers=dict(action='invalid', outcome='invalid', participants='invalid'))]
    s = m.score_sample(rs)
    assert s['interaction_success'] == 0.25 and s['action_rate'] == 0.5 and s['unknown_outcome_rate'] == 0.25 and s['n_invalid'] == 1
    assert m.score_sample([])['interaction_success'] is None


