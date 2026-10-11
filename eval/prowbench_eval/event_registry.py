"""Event registry of a dataset: the phase instances, contacts, relations and state changes recorded in the engine files
of every episode, with stable identifiers and source hashes."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

from .common import read_json, read_jsonl, sha256_file, write_json, write_text, atomic_open


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def observed(obj, key):
    return dict(present=key in obj, value=obj.get(key))


def contact_identity(c):
    keys = ('actorId', 'actorNodeId', 'targetEntityId', 'targetNodeId', 'type', 'side')
    return {k: c[k] for k in keys if k in c}


def inventory_episode(episode, manifest, frames, source):
    fps = float(manifest['fps'])
    if not fps > 0 or not frames or len(frames) != manifest['frameCount']:
        raise ValueError('Invalid recording length/fps')
    if [f['index'] for f in frames] != list(range(len(frames))):
        raise ValueError('Recording must have contiguous, ordered frame indices')
    times = [f.get('time', f['index']/fps) for f in frames]
    if not all(isinstance(t,(int,float)) and math.isfinite(t) for t in times) or any(b<=a for a,b in zip(times,times[1:])):
        raise ValueError('Invalid recorded timestamps')
    entities = {e['id']: e for e in manifest['entities']}
    snapshots = []
    for line, f in enumerate(frames, 1):
        snapshots.append(dict(frame=f['index'], time=times[line-1],
                              recorded_time=observed(f,'time'), source_time=observed(f,'sourceTime'),
                              source_frame=observed(f,'sourceFrame'),
                              interaction=observed(f, 'interaction'),
                              entities={eid: {k: observed(st, k) for k in
                                        ('state', 'position', 'quaternion', 'visible')}
                                        for eid, st in f.get('entities', {}).items()},
                              source=dict(**source, line=line)))
    events = []

    def ref(i):
        return None if i < 0 or i >= len(frames) else dict(**source, line=i+1, frame=i)

    def add(kind, name, start, end, payload, participants):
        definition = f'{episode}/{kind}/{name}'
        evidence = dict(before=ref(start-1), start=ref(start), last=ref(end-1), after=ref(end))
        actors = []
        for eid in sorted({x for x in participants if isinstance(x, str)}):
            st = frames[start].get('entities', {}).get(eid, {}).get('state') or {}
            actors.append(dict(entity_id=eid, label=entities.get(eid, {}).get('semanticLabel'),
                               recorded_role=observed(st, 'role'), known_entity=eid in entities))
        item = dict(definition_id=definition, kind=kind, name=name, episode=episode,
                    start_frame=start, end_frame_exclusive=end, start_time=times[start],
                    end_time=times[end] if end<len(frames) else times[-1]+1/fps,
                    participants=actors, payload=payload, evidence=evidence,
                    review_status='pending', visual_evaluability='unreviewed',
                    scoring_status='unapproved', scoring_rule=None)
        item['event_id'] = f'{episode}/{source["sha256"][:12]}/{start:04d}/{digest(item)[:16]}'
        events.append(item)

    start = 0
    for end in range(1, len(frames)+1):
        a = observed(frames[start].get('interaction') or {}, 'phase')
        b = observed(frames[end].get('interaction') or {}, 'phase') if end < len(frames) else None
        if a == b:
            continue
        participants = set()
        for f in frames[start:end]:
            participants.update((f.get('interaction') or {}).get('participants') or [])
        add('phase', str(a['value']) if a['present'] else '__missing__', start, end,
            dict(phase=a, initial=start == 0,
                 entry_interaction=frames[start].get('interaction'),
                 exit_interaction=frames[end-1].get('interaction'),
                 entry_states={k: observed(v, 'state') for k,v in frames[start].get('entities',{}).items()},
                 before_states=None if start==0 else {k: observed(v, 'state') for k,v in frames[start-1].get('entities',{}).items()},
                 exit_states={k: observed(v, 'state') for k,v in frames[end-1].get('entities',{}).items()}), participants)
        start = end

    for i in range(1, len(frames)):
        prev, cur = frames[i-1], frames[i]
        ia, ib = prev.get('interaction') or {}, cur.get('interaction') or {}
        for field, identity in [('relations', lambda x: x), ('contacts', contact_identity)]:
            aa, bb = ia.get(field), ib.get(field)
            if isinstance(aa, list) and isinstance(bb, list):
                aset = {canonical(identity(x)): x for x in aa}
                bset = {canonical(identity(x)): x for x in bb}
                for direction, removed in [('end', True), ('start', False)]:
                    left, right = (aset,bset) if removed else (bset,aset)
                    for key in sorted(set(left)-set(right)):
                        val = left[key]
                        pred = val.get('predicate', '__missing__') if field=='relations' else val.get('type', '__missing__')
                        ids = [val.get(k) for k in ('subject','object','actorId','targetEntityId')]
                        add(field+'_'+direction, str(pred), i, i+1,
                            dict(record=val, field=field, source_pointer=f'/interaction/{field}'), ids)
            elif observed(ia,field) != observed(ib,field):
                add('interaction_change', field, i, i+1,
                    dict(before=observed(ia,field), after=observed(ib,field)), ib.get('participants') or [])
        for eid in sorted(set(prev.get('entities',{})) | set(cur.get('entities',{}))):
            a = prev.get('entities',{}).get(eid,{}).get('state')
            b = cur.get('entities',{}).get(eid,{}).get('state')
            if isinstance(a,dict) and isinstance(b,dict):
                for key in sorted(set(a)|set(b)):
                    if observed(a,key) != observed(b,key):
                        add('state_change', eid+'/'+key, i, i+1,
                            dict(entity_id=eid, field=key, before=observed(a,key), after=observed(b,key),
                                 source_pointer='/entities/'+eid+'/state/'+key), [eid])
            elif observed(prev.get('entities',{}).get(eid,{}),'state') != observed(cur.get('entities',{}).get(eid,{}),'state'):
                add('state_change', eid+'/__state__', i, i+1,
                    dict(entity_id=eid, before=observed(prev.get('entities',{}).get(eid,{}),'state'),
                         after=observed(cur.get('entities',{}).get(eid,{}),'state')), [eid])
        for key in sorted((set(ia)|set(ib))-{'phase','relations','contacts'}):
            if observed(ia,key) != observed(ib,key):
                add('interaction_change', key, i, i+1,
                    dict(before=observed(ia,key), after=observed(ib,key)), ib.get('participants') or [])
        sa, sb = prev.get('sourceFrame') or {}, cur.get('sourceFrame') or {}
        for key in sorted((set(sa)|set(sb))-{'frame','time','subject','screen','camera'}):
            if observed(sa,key) != observed(sb,key):
                add('source_frame_change',key,i,i+1,
                    dict(before=observed(sa,key),after=observed(sb,key),source_pointer='/sourceFrame/'+key),
                    sb.get('participants') or [])
    events.sort(key=lambda e: (e['start_frame'],e['kind'],e['event_id']))
    return events, snapshots


def export_registry(dataset, out):
    dataset, out = Path(dataset).resolve(), Path(out)
    all_events, definitions, sources, snapshots_by_episode, views = [], {}, [], {}, {}
    for ep in sorted((dataset/'episodes').iterdir()):
        if not (ep/'manifest.json').is_file():
            continue
        manifest = read_json(ep/'manifest.json')
        states = ep/manifest['states']
        source = dict(path=str(states), sha256=sha256_file(states), encoding='jsonl.gz' if states.suffix=='.gz' else 'jsonl')
        sources.append(dict(episode=ep.name, manifest_path=str(ep/'manifest.json'),
                            manifest_sha256=sha256_file(ep/'manifest.json'), **source))
        frames = list(read_jsonl(states))
        events, snapshots = inventory_episode(ep.name, manifest, frames, source)
        snapshots_by_episode[ep.name] = snapshots
        views[ep.name] = [dict(camera=p.parent.name, path=str(p), sha256=sha256_file(p),
                               visibility_status='unreviewed') for p in sorted(ep.glob('*/camera.json'))]
        for e in events:
            e['views'] = views[ep.name]
            d = definitions.setdefault(e['definition_id'], dict(definition_id=e['definition_id'],
                episode=e['episode'], kind=e['kind'], name=e['name'], count=0,
                review_status='pending', scoring_rule=None, participant_ids=[], examples=[]))
            d['count'] += 1
            d['participant_ids'] = sorted(set(d['participant_ids'])|{a['entity_id'] for a in e['participants']})
            if len(d['examples']) < 3:
                d['examples'].append(e['event_id'])
        all_events.extend(events)
    if not sources:
        raise ValueError('No engine recordings found')
    defs = sorted(definitions.values(), key=lambda d:d['definition_id'])
    version = digest(dict(sources=sources, definitions=defs, instances=all_events))
    registry = dict(schema='prowbench.event_registry.v1', version=version, dataset=str(dataset),
                    sources=sources, definitions=defs, instances=all_events,
                    interpretation='Engine observations only; source IK contacts are not collision truth; no visual scoring rules approved.')
    target = out/'registry.json'
    if target.exists() and read_json(target)['version'] != version:
        raise ValueError('Registry changed: choose a new versioned output directory; preserve prior review/results')
    out.mkdir(parents=True, exist_ok=True)
    write_json(target, registry)
    for ep, snapshots in snapshots_by_episode.items():
        path=out/'evidence'/f'{ep}.jsonl';path.parent.mkdir(exist_ok=True)
        write_text(path, ''.join(canonical(s)+'\n' for s in snapshots))
    with atomic_open(out/'instances.csv', newline='') as f:
        fields=['event_id','definition_id','episode','kind','name','start_frame','end_frame_exclusive','start_time','end_time','participants','payload','evidence','visual_evaluability','scoring_status']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for e in all_events:
            writer.writerow({k:canonical(e[k]) if isinstance(e[k],(list,dict)) else e[k] for k in fields})
    def cell(x):
        return str(x).replace('|','\\|').replace('\n',' ')
    lines=['# Engine event definitions','',f'Version: `{version}`. This table is generated automatically from engine records; being listed does not mean the visual scoring rules have been reviewed.',
           '', '| Definition ID | Type | Instances | Participant entity IDs | Review |','|---|---|---:|---|---|']
    for d in defs:
        lines.append('| '+' | '.join(map(cell,[d['definition_id'],d['kind'],d['count'],', '.join(d['participant_ids']),'pending review']))+' |')
    write_text(out/'DEFINITIONS.md', '\n'.join(lines)+'\n')
    lines=['# Engine event instances','', 'This page lists all action phases; all relation/contact/state/other interaction changes are in [instances.csv](instances.csv), and the structured evidence is in [registry.json](registry.json).',
           'Time intervals are [start, end) in seconds; the initial state serves only as a factual baseline and is not turned into fabricated "start holding / start contact" events.',
           'Each engine event is recorded once; per-camera visibility is recorded separately, pending review, and is not counted as additional events.', '',
           '| Episode | Phase | Frame range | Time range | Participants (recorded role) | State changes (phase start → end) | Evidence |',
           '|---|---|---|---|---|---|---|']
    for e in all_events:
        if e['kind']!='phase':continue
        payload=e['payload'];changes=[]
        for label, before, after in [('entering phase',payload['before_states'],payload['entry_states']),
                                     ('within phase',payload['entry_states'],payload['exit_states'])]:
            if before is None:continue
            for eid in sorted(set(before)|set(after)):
                a=before.get(eid,{}).get('value') or {};b=after.get(eid,{}).get('value') or {}
                for key in sorted(set(a)|set(b)):
                    if key!='phase' and observed(a,key)!=observed(b,key):changes.append(f'{label} {eid}.{key}: {canonical(observed(a,key))} → {canonical(observed(b,key))}')
        actors=', '.join(a['entity_id']+(' ('+str(a['recorded_role']['value'])+')' if a['recorded_role']['present'] else '') for a in e['participants'])
        evidence=f"[frames {e['start_frame']}–{e['end_frame_exclusive']-1}](evidence/{e['episode']}.jsonl); raw JSONL lines {e['start_frame']+1}–{e['end_frame_exclusive']}"
        lines.append('| '+' | '.join(map(cell,[e['episode'],e['name'],f"[{e['start_frame']}, {e['end_frame_exclusive']})",f"[{e['start_time']:.3f}, {e['end_time']:.3f})",actors,'; '.join(changes) or 'no state field changes within the phase; see relations/contacts and the boundary snapshots',evidence]))+' |')
    write_text(out/'EVENTS.md', '\n'.join(lines)+'\n')
    lines=['# Relation and discrete state changes','',
           'This view is for checking interaction outcomes; per-frame changes in numeric telemetry are not listed as scoring events. The complete record remains in instances.csv.', '',
           '| Episode | Frame / s | Record type | Change / raw relation | Raw rows |','|---|---|---|---|---|']
    for e in all_events:
        p=e['payload'];kind=e['kind']
        categorical=kind=='state_change' and p.get('field')!='phase' and all(
            isinstance(p[x]['value'],(str,bool)) or p[x]['value'] is None for x in ('before','after'))
        if kind.startswith('relations_'):
            detail=canonical(p['record'])
        elif categorical or (kind=='source_frame_change' and e['name']=='supports'):
            detail=e['name']+': '+canonical(p['before'])+' → '+canonical(p['after'])
        else:continue
        lines.append('| '+' | '.join(map(cell,[e['episode'],f"{e['start_frame']} / {e['start_time']:.3f}",kind,detail,
                                  f"{e['start_frame']} → {e['start_frame']+1}"]))+' |')
    write_text(out/'TRANSITIONS.md', '\n'.join(lines)+'\n')
    counts=dict(Counter(e['kind'] for e in all_events))
    report=dict(status='passed',version=version,episodes=len(sources),frames=sum(map(len,snapshots_by_episode.values())),
                definitions=len(defs),instances=len(all_events),by_kind=counts,
                unapproved=len(all_events),source_files_verified=True)
    write_json(out/'coverage.json',report)
    return report


def load_scoring_events(path, episode):
    """Fail closed until a versioned visual review exists. Never fall back to prose."""
    reg=read_json(path)
    if reg.get('schema') not in ('prowbench.event_registry.v1', 'gamebench.event_registry.v1'):  # former name accepted
        raise ValueError('Unsupported event registry schema')
    phases=[e for e in reg['instances'] if e['episode']==episode and e['kind']=='phase']
    if not phases:
        raise ValueError(f'No registered phases for {episode}')
    pending=[e['event_id'] for e in phases if e['scoring_status']!='approved' or not e.get('scoring_rule')]
    if pending:
        raise ValueError(f'{len(pending)} registered events have no approved visual rule; scoring blocked: {pending[:3]}')
    raise ValueError('Visual scoring of reviewed registry events is not implemented; no fallback is used')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',required=True)
    args=ap.parse_args();cfg=read_json(args.config)
    print(json.dumps(export_registry(cfg['dataset'],cfg['output']),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
