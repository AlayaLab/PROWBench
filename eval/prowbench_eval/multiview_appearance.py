"""Appearance compliance and consistency (Comp. / Cons.) of multi-view participants: validation and scoring of judged
observations and their aggregation over views, participants and scenes."""
import itertools
import cv2
import numpy as np

ATTRS=['clothing_color','clothing_style','hair','accessories','build']


PROMPT='''Evaluate APPEARANCE COMPLIANCE with the provided authored description. The left image gives scene context with a thin white target rectangle added by the evaluator. The right image is its crop. Evaluate only the person corresponding to this rectangle. Do not choose a different person elsewhere because their outfit matches. If multiple people cannot be disambiguated, mark ambiguous. If the target is absent mark absent. Do not infer hidden details from the description. If target is absent or ambiguous, use only unobservable or unspecified for every attribute.
Authored appearance: {description}
For each attribute clothing_color, clothing_style, hair, accessories, build return one of match, mismatch, unobservable, unspecified. Unspecified means the description does not constrain that attribute. Unobservable means constrained but too small, occluded or otherwise not discernible. Ignore pose, perspective and reasonable illumination changes. Visible contradictions must be mismatch; do not demand identical shades under different lighting. Do not assess unspecified facial identity, ethnicity, age or gender. Mention concrete visual evidence briefly.
Return ONLY JSON: {{"target":"present|absent|ambiguous","attributes":{{"clothing_color":"...","clothing_style":"...","hair":"...","accessories":"...","build":"..."}},"reason":"brief evidence"}}'''


CONSIST='''Evaluate CROSS-VIEW VISUAL APPEARANCE CONSISTENCY using only the supplied images. Each supplied panel is a simultaneous camera view: left context with an evaluator-added thin white target rectangle, right its crop. The panel labels are {views}. They designate the intended same participant by engine location, NOT a guarantee that a corresponding person was actually generated. No authored appearance is supplied. Do not infer intended clothing or use any textual target appearance. Do not match a different person outside the marked region.
For every view, report target present, absent, or ambiguous (cannot select one person or too unclear). For EVERY unordered view pair, compare clothing_color, clothing_style, hair, accessories, build as consistent, inconsistent, or unobservable. Compare only attributes visibly discernible in BOTH views. Normal front/back differences, pose, perspective, shadows, illumination and occlusion alone are not inconsistencies. A clear incompatible garment, hairstyle, accessory or build is inconsistent. Do not require the same pixels. If either target is absent or ambiguous, all pair attributes must be unobservable. Do not assess ungrounded facial identity or ethnicity. Cite visible evidence, not a guessed intended identity.
Return ONLY JSON: {{"targets":{{"view label":"present|absent|ambiguous"}},"pairs":[{{"views":["view label","view label"],"attributes":{{"clothing_color":"...","clothing_style":"...","hair":"...","accessories":"...","build":"..."}},"reason":"brief evidence"}}]}}'''


def panel(c,frame,box,label):
    left=frame.copy(); cv2.rectangle(left,box[:2],box[2:],(255,255,255),2)
    def fit(im,w,h):
        z=min(w/im.shape[1],h/im.shape[0]); return cv2.resize(im,(max(1,round(im.shape[1]*z)),max(1,round(im.shape[0]*z))),interpolation=cv2.INTER_AREA if z<1 else cv2.INTER_CUBIC)
    left=fit(left,576,384); crop=fit(c,256,384)
    canvas=np.full((416,848,3),32,np.uint8)
    canvas[32:32+left.shape[0],:left.shape[1]]=left
    canvas[32:32+crop.shape[0],592:592+crop.shape[1]]=crop
    cv2.putText(canvas,label,(12,24),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA)
    return canvas


def validate(j,r):
    if j['kind']=='prompt':
      if not (r['target'] in ['present','absent','ambiguous']):
        raise ValueError("multiview_appearance: check failed: r['target'] in ['present','absent','ambiguous']")
      if not (set(r['attributes'])==set(ATTRS)):
        raise ValueError("multiview_appearance: check failed: set(r['attributes'])==set(ATTRS)")
      if not (all(x in ['match','mismatch','unobservable','unspecified'] for x in r['attributes'].values())):
        raise ValueError("multiview_appearance: check failed: all(x in ['match','mismatch','unobservable','unspecified'] for x in r['attributes'].values())")
      if r['target']!='present' and not all(x in ['unobservable','unspecified'] for x in r['attributes'].values()):
        raise ValueError("multiview_appearance: an absent or ambiguous target has observed attributes")
    else:
      if not (set(r['targets'])==set(j['views'])):
        raise ValueError("multiview_appearance: check failed: set(r['targets'])==set(j['views'])")
      if not (all(x in ['present','absent','ambiguous'] for x in r['targets'].values())):
        raise ValueError("multiview_appearance: check failed: all(x in ['present','absent','ambiguous'] for x in r['targets'].values())")
      expect={tuple(v) for v in itertools.combinations(sorted(j['views']),2)}
      seen=[]
      for p in r['pairs']:
        if not (len(p['views'])==2):
            raise ValueError("multiview_appearance: check failed: len(p['views'])==2")
        seen.append(tuple(sorted(p['views'])))
        if not (set(p['attributes'])==set(ATTRS)):
            raise ValueError("multiview_appearance: check failed: set(p['attributes'])==set(ATTRS)")
        if not (all(x in ['consistent','inconsistent','unobservable'] for x in p['attributes'].values())):
            raise ValueError("multiview_appearance: check failed: all(x in ['consistent','inconsistent','unobservable'] for x in p['attributes'].values())")
        if any(r['targets'][v]!='present' for v in p['views']) and set(p['attributes'].values())!={'unobservable'}:
            raise ValueError("multiview_appearance: a pair with an absent or ambiguous view has observed attributes")
      if not (len(seen)==len(expect) and set(seen)==expect):
        raise ValueError('multiview_appearance: check failed: len(seen)==len(expect) and set(seen)==expect')


def score_observation(attributes, present=True, consistency=False):
    """Return the all-judgeable-attributes score and judgeable-attribute mean."""
    positive, negative = ('consistent', 'inconsistent') if consistency else ('match', 'mismatch')
    values = [v == positive for v in attributes.values() if v in (positive, negative)]
    if not present:
        values = []
    return dict(score=float(all(values)) if values else None,
                attribute_score=float(np.mean(values)) if values else None,
                judgeable=bool(values), present=present)


def aggregate_observations(rows, scenes):
    """Average views, then participant-time groups, then equally weighted scenes."""
    from collections import defaultdict
    rows = list(rows)
    if len({(r.get('method'), r.get('kind')) for r in rows}) > 1:
        raise ValueError('Rows must belong to one method and one appearance metric')
    def avg(values):
        values = [v for v in values if v is not None]
        return float(np.mean(values)) if values else None
    scene_scores = {}
    for scene in scenes:
        groups = defaultdict(list)
        for row in rows:
            if row['episode'] == scene:
                groups[(row['participant'], row['t'])].append(row['score'])
        scene_scores[scene] = avg([avg(values) for values in groups.values()])
    return dict(score=avg(list(scene_scores.values())) if all(v is not None for v in scene_scores.values()) else None,
                scene_scores=scene_scores)
