"""Synthetic checks for reusable multiview and long-horizon metrics."""
import numpy as np
from prowbench_eval import multiview_appearance as a, multiview_geometry as g, long_horizon_logic as l
from prowbench_eval import common

def test_appearance_missing_and_all_attributes():
    assert a.score_observation({'hair':'match','build':'mismatch'})['score']==0
    assert a.score_observation({'hair':'match','build':'unobservable'})['score']==1
    assert a.score_observation({'hair':'unspecified'})['score'] is None
    assert a.score_observation({'hair':'match'},present=False)['score'] is None
    assert a.score_observation({'hair':'consistent','build':'inconsistent'},consistency=True)['score']==0

def test_appearance_hierarchical_scene_weights():
    rows=[dict(episode='a',participant='p',t=0,score=v) for v in [1,1,0,0]]
    rows += [dict(episode='a',participant='q',t=0,score=1),dict(episode='b',participant='p',t=0,score=0)]
    assert a.aggregate_observations(rows,['a','b'])['score']==.375
    assert a.aggregate_observations(rows,['a','b','empty'])['score'] is None

def test_multiview_crop_resize():
    assert np.allclose(g.map_box([16,0,1264,720],[16,0,1248,720],832,480),[0,0,832,480])

def test_long_horizon_adjudication_and_scene_weights():
    assert l.adjudicated_interaction_success('yes','yes')
    assert not l.adjudicated_interaction_success('yes','unknown')
    assert l.adjudicated_interaction_success('no','yes','y','u')
    assert not l.adjudicated_alignment_success('no',False,'y')
    assert l.adjudicated_alignment_success('no',True,'y')
    assert l.scene_mean([dict(episode='a',correct=True)]*3+[dict(episode='b',correct=False)])==.5

def test_time_grid_fps_and_half_frame_ties():
    for fps in [16,24,30]:
        assert common.output_frame_for_time(2.5,fps)==round(2.5*fps)
        assert common.output_frame_for_time(.5/fps,fps)==0
        assert common.output_frame_for_time(1.5/fps,fps)==2
    assert len(common.eval_grid(8))==40

def test_joint_control_multiplies_before_averaging():
    from prowbench_eval.joint_control import score_samples
    score=score_samples([dict(iou=1,interaction=0,align=1),dict(iou=0,interaction=1,align=0)])
    assert score==dict(gisr=0,glra=.5)


def test_aggregation_rejects_mixed_methods():
    rows=[dict(method=m,episode='s',participant='p',t=0,score=1,correct=True) for m in ['a','b']]
    for func in [lambda: a.aggregate_observations(rows,['s']),lambda:l.scene_mean(rows)]:
        try:func()
        except ValueError:continue
        raise AssertionError('mixed methods accepted')

def test_met3r_empty_overlap_is_undefined():
    from prowbench_eval.multiview_met3r import aggregate_scenes
    rows=[dict(episode='a',score=.2,valid=True),dict(episode='b',score=.6,valid=True)]
    assert abs(aggregate_scenes(rows,['a','b'])['score']-.4)<1e-12
    rows[0].update(score=None,valid=False)
    assert aggregate_scenes(rows,['a','b'])['score'] is None


def test_met3r_rejects_mixed_kinds():
    from prowbench_eval.multiview_met3r import aggregate_scenes
    rows=[dict(method='m',kind=k,episode='s',score=v,valid=True) for k,v in [('a',0),('b',1)]]
    try:aggregate_scenes(rows,['s'])
    except ValueError:return
    raise AssertionError('mixed metric kinds accepted')
