"""Tests of the input validation of the metric modules and the judge."""
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np

from prowbench_eval import metric_camera as cam, metric_vbench
from prowbench_eval.vlm_judge import require_bool, require_score, Judge


def raises_value(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError('Invalid input was accepted')


def test_camera_rejects_short_sparse_and_zero_quaternion():
    f = dict(frame=0, position=[0,0,0], quaternion=[0,0,0,1])
    raises_value(lambda: cam._sample_traj(dict(fps=24, frames=[f]), [0, 4.875]))
    raises_value(lambda: cam._sample_traj(dict(fps=24, frames=[f,dict(f,frame=2)]), [0]))
    raises_value(lambda: cam.quat_to_R([0,0,0,0]))
    raises_value(lambda: cam.quat_to_R([0,0,np.nan,1]))


def test_tum_timestamps_take_precedence_over_video_fps():
    frames=[dict(time=t, position=[t,0,0], quaternion=[0,0,0,1]) for t in [0, .2, 1.]]
    p,_=cam._sample_traj(dict(frames=frames), [0,.2,1.], fps=30)
    assert np.allclose(p[:,0], [0,.2,1.])


def test_judge_false_strings_and_missing_or_invalid_scores_rejected():
    assert require_bool({'x':False}, 'x') is False
    for value in ['false','true',0,1,None]:
        raises_value(lambda value=value: require_bool({'x':value},'x'))
    for value in [True, -1, 2, np.nan, None]:
        raises_value(lambda value=value: require_score({'score':value}))


def test_judge_parse_error_is_not_a_negative_vote_and_cache_settings_matter():
    with tempfile.TemporaryDirectory() as directory:
        judge=Judge(model='mock',cache_dir=directory)
        with patch.object(judge,'_ask_hf',return_value='not JSON'):
            raises_value(lambda:judge.ask([], 'p'))
        with patch.object(judge,'_ask_hf',return_value='{"x":true}'):
            raises_value(lambda:judge.ask([], 'p'))
            judge.max_new_tokens += 1
            assert judge.ask([], 'p')['parsed']['x'] is True


def test_judge_thinking_modes_do_not_share_cached_answers():
    with tempfile.TemporaryDirectory() as directory:
        judge = Judge(model='mock', cache_dir=directory, enable_thinking=False)
        with patch.object(judge, '_ask_hf', return_value='{"x":true}') as ask:
            result = judge.ask([], 'p')
            assert result['settings']['enable_thinking'] is False
            judge.ask([], 'p')
            assert ask.call_count == 1
            judge.enable_thinking = True
            judge.ask([], 'p')
            assert ask.call_count == 2


def test_aggregation_rejects_nonfinite_and_duplicate_paired_rows():
    row=dict(method='m',run_id='r',representation='x',sample_id='s',**{'entity.mean_iou':.5})
    raises_value(lambda:metric_vbench._summarise_rows([dict(row,**{'entity.mean_iou':np.nan})]))
    raises_value(lambda:metric_vbench._paired([row,row],('m','r'),('other','r')))


def test_tracker_duplicates_and_invalid_assignment_rejected():
    from prowbench_eval import metric_entity, common
    d=dict(track_id=1,label='person',box=[0,0,10,10])
    p=dict(width=100,height=100,frames=[dict(frame=0,detections=[d,d])])
    raises_value(lambda:metric_entity._pred_index(p))
    raises_value(lambda:common.hungarian(np.array([[np.nan]])))
    raises_value(lambda:common.eval_grid(0))


def test_vbench_bridge_includes_current_camera_keys_and_tau_a_ties():
    from prowbench_eval import common, metric_vbench
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        mapping=[dict(file=f'{i}.mp4',method='m',run_id='r',sample_id=str(i)) for i in range(3)]
        rows=[dict(method='m',run_id='r',sample_id=str(i),**{'camera.RotErr_mean':i}) for i in range(3)]
        vbench={'subject_consistency':[1,[dict(video_path=f'{i}.mp4',video_results=i) for i in range(3)]]}
        common.write_json(root/'mapping.json',mapping);common.write_json(root/'vbench.json',vbench)
        common.write_jsonl(root/'rows.jsonl',rows)
        metric_vbench.correlate(str(root/'vbench.json'),str(root/'mapping.json'),str(root/'rows.jsonl'),str(root/'out.json'))
        result=common.read_json(root/'out.json')['correlations']['camera.RotErr_mean ~ subject_consistency']
        assert result['n']==3 and result['spearman']==1
    assert np.isclose(metric_vbench.kendall([1,1,2],[1,2,2]),1/3)



def test_vbench_summary_of_single_samples_has_no_interval():
    from prowbench_eval import common, metric_vbench
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        mapping=[dict(file=f'{m}.mp4',method=m,run_id='r',representation='white',sample_id='s') for m in 'ab']
        vbench={'imaging_quality':[1,[dict(video_path=f'{m}.mp4',video_results=v) for m,v in (('a',50.0),('b',60.0))]]}
        common.write_json(root/'mapping.json',mapping);common.write_json(root/'vbench.json',vbench)
        metric_vbench.summarise(str(root/'vbench.json'),str(root/'mapping.json'),str(root/'out'),pairs=['a:r,b:r'])
        summary=common.read_json(root/'out'/'summary.json')
        assert [r['vbench.imaging_quality']['ci95'] for r in summary]==[[None,None]]*2
        assert common.read_json(root/'out'/'paired.json')['a:r,b:r']['metrics']['vbench.imaging_quality']['mean_diff']==10.0
        text=(root/'out'/'summary.md').read_text()
        assert '50.000 ± 0.000 [n/a]' in text and '+10.000 [n/a] 1/1' in text

def test_bbox_crossing_near_plane_has_analytic_clipped_extent():
    from prowbench_eval.gt_export import box_from_corners
    camera=dict(worldToCamera=np.eye(4).T.reshape(-1).tolist(),
                intrinsics=dict(fx=100,fy=100,cx=50,cy=50))
    corners=np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])
    box,front=box_from_corners(camera,corners,near_eps=.1)
    assert front==4
    assert np.allclose(box,[-950,-950,1050,1050])


def test_judge_ask_separates_invalid_responses_from_errors():
    import pytest
    from prowbench_eval import vlm_judge
    frames = [(0, np.zeros((8, 8, 3), np.uint8))]
    with tempfile.TemporaryDirectory() as tmp:
        judge = Judge(backend="hf", model="m", cache_dir=tmp)
        with patch.object(Judge, "_ask_hf", return_value="no json here"):
            with pytest.raises(vlm_judge.JudgeResponseError):
                judge.ask(frames, "p1")
            with pytest.raises(vlm_judge.JudgeResponseError):          # the cached invalid response
                judge.ask(frames, "p1")
        with patch.object(Judge, "_ask_hf", return_value='{"a": 1}'), \
             patch.object(vlm_judge, "extract_json", side_effect=MemoryError("out of memory")):
            with pytest.raises(MemoryError):
                judge.ask(frames, "p2")
        assert len(list(Path(tmp).glob("*.json"))) == 1                  # the error was not cached
        with patch.object(Judge, "_ask_hf", side_effect=RuntimeError("model could not be loaded")):
            with pytest.raises(RuntimeError):
                judge.ask(frames, "p3")


def test_outputs_are_written_atomically():
    import os
    import pytest
    from prowbench_eval import common
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sub" / "r.json"
        common.write_json(path, {"a": 1})
        assert common.read_json(path) == {"a": 1}
        mask = os.umask(0); os.umask(mask)
        assert path.stat().st_mode & 0o777 == 0o666 & ~mask           # as a file created with open()
        with pytest.raises(RuntimeError):
            with common.atomic_open(path) as f:
                f.write('{"a": 2')
                raise RuntimeError("interrupted")
        assert common.read_json(path) == {"a": 1}                      # the complete earlier result is kept
        with pytest.raises(RuntimeError):
            with common.atomic_open(Path(tmp) / "new.json") as f:
                f.write("{")
                raise RuntimeError("interrupted")
        assert sorted(p.name for p in Path(tmp).rglob("*")) == ["r.json", "sub"]   # no partial file, no temporary file


def test_atomic_writes_leave_the_umask_and_threads_alone():
    import os, threading
    from prowbench_eval import common
    with tempfile.TemporaryDirectory() as tmp:
        before = os.umask(0o022); os.umask(before)
        threads = [threading.Thread(target=common.write_json, args=(Path(tmp) / f"{i % 5}.json", {"i": i})) for i in range(100)]
        for th in threads: th.start()
        for th in threads: th.join()
        after = os.umask(before); os.umask(after)
        assert after == before
        assert sorted(p.name for p in Path(tmp).iterdir()) == [f"{i}.json" for i in range(5)]
        assert all(p.stat().st_mode & 0o777 == 0o666 & ~before for p in Path(tmp).iterdir())


def test_vbench_copy_is_not_left_partial():
    import shutil
    from unittest.mock import patch
    import pytest
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / "v.mp4"; src.write_bytes(b"0" * 1000)
        row = dict(sample_id="ep/cam/white", method="m", run_id="r", representation="white", video=str(src))
        (tmp / "s.jsonl").write_text(__import__("json").dumps(row) + "\n")
        def broken_copy(a, b):
            Path(b).write_bytes(b"0" * 10); raise KeyboardInterrupt
        with patch.object(metric_vbench, "video_meta", return_value=dict(fps=24.0, frame_count=120, width=64, height=36)), \
             patch.object(metric_vbench.os, "symlink", side_effect=OSError), patch.object(metric_vbench.shutil, "copy2", broken_copy):
            with pytest.raises(KeyboardInterrupt):
                metric_vbench.prepare(str(tmp / "s.jsonl"), str(tmp / "out"))
        assert list((tmp / "out").iterdir()) == []                      # retried on the next call
        with patch.object(metric_vbench, "video_meta", return_value=dict(fps=24.0, frame_count=120, width=64, height=36)), \
             patch.object(metric_vbench.os, "symlink", side_effect=OSError):
            metric_vbench.prepare(str(tmp / "s.jsonl"), str(tmp / "out"))
        assert (tmp / "out" / "m__r__ep__cam__white.mp4").read_bytes() == src.read_bytes()
