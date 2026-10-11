import copy
import tempfile
from pathlib import Path

from prowbench_eval.common import write_json
from prowbench_eval.event_registry import inventory_episode, load_scoring_events


def fixture():
    manifest = dict(fps=12, frameCount=4, entities=[dict(id='a',semanticLabel='person'),dict(id='b',semanticLabel='box')])
    contact=dict(actorId='a',targetEntityId='b',type='grip',side=1,distance=0.1,
                 source='recorded IK target; not an exact collision truth')
    frames=[]
    for i in range(4):
        frames.append(dict(index=i,interaction=dict(phase='hold' if i<2 else 'release',participants=['a','b'],
                      contacts=[dict(contact,distance=.1+i*.01)] if i<2 else [],
                      relations=[dict(subject='a',predicate='holds',object='b')] if i<2 else []),
                      entities=dict(a=dict(state=dict(role='giver',holding='b' if i<2 else None)),b=dict(state={}))))
    return manifest,frames,dict(path='/recording/frames.jsonl.gz',sha256='a'*64,encoding='jsonl.gz')


def test_registry_keeps_short_initial_phases_and_uses_recorded_fps():
    m,f,s=fixture();events,frames=inventory_episode('ep',m,f,s)
    phases=[e for e in events if e['kind']=='phase']
    assert len(phases)==2 and phases[0]['end_time']==2/12
    assert phases[0]['payload']['initial'] and phases[0]['evidence']['before'] is None
    assert phases[-1]['evidence']['after'] is None
    assert phases[0]['participants'][0]['recorded_role']==dict(present=True,value='giver')
    assert not any(e['kind'].endswith('_start') for e in events)
    assert frames[0]['source']['line']==1


def test_registry_preserves_contact_caveat_without_float_jitter_events():
    m,f,s=fixture();events,_=inventory_episode('ep',m,f,s)
    contacts=[e for e in events if e['kind'].startswith('contacts_')]
    assert len(contacts)==1 and contacts[0]['kind']=='contacts_end'
    assert contacts[0]['payload']['record']['source']=='recorded IK target; not an exact collision truth'
    assert contacts[0]['start_frame']==2


def test_registry_distinguishes_missing_null_and_unknown_new_fields():
    m,f,s=fixture()
    f[1]['entities']['a']['state']['extra']=None
    f[2]['interaction']['new_engine_signal']={'active':True}
    events,_=inventory_episode('ep',m,f,s)
    added=next(e for e in events if e['name']=='a/extra' and e['start_frame']==1)
    assert added['payload']['before']==dict(present=False,value=None)
    assert added['payload']['after']==dict(present=True,value=None)
    assert any(e['name']=='new_engine_signal' for e in events)
    assert all(e['scoring_status']=='unapproved' for e in events)


def test_registry_deterministic_and_rejects_sparse_recordings():
    m,f,s=fixture();a=inventory_episode('ep',m,f,s);b=inventory_episode('ep',m,copy.deepcopy(f),s)
    assert a==b
    assert len({e['event_id'] for e in a[0]})==len(a[0])
    f[1]['index']=9
    try:inventory_episode('ep',m,f,s)
    except ValueError:pass
    else:raise AssertionError('Sparse recording accepted')


def test_registry_preserves_source_supports_and_recorded_timestamps():
    m,f,s=fixture()
    for i,frame in enumerate(f):
        frame['time']=10+i*.1
        frame['sourceFrame']={'supports':{'a':'ground' if i<2 else 'platform'}}
    events,snapshots=inventory_episode('ep',m,f,s)
    support=next(e for e in events if e['kind']=='source_frame_change' and e['name']=='supports')
    assert support['start_time']==10.2 and support['payload']['before']['value']=={'a':'ground'}
    assert snapshots[0]['source_frame']['value']=={'supports':{'a':'ground'}}


def test_registry_blocks_unreviewed_or_missing_event_tables_before_judge():
    m,f,s=fixture();events,_=inventory_episode('ep',m,f,s)
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'registry.json';write_json(p,dict(schema='prowbench.event_registry.v1',instances=events))
        try:load_scoring_events(p,'ep')
        except ValueError:pass
        else:raise AssertionError('Unapproved events allowed to score')
