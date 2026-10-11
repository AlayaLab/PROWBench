"""MEt3R geometric consistency between pairs of views and its aggregation over scenes."""
import cv2
import numpy as np

CONFIG=dict(backbone='dust3r',feature_backbone='dino16',use_norm=True,upsampler='featup',distance='cosine',img_size=None,input_wh=[336,192],normalization='RGB [-1,1]',interpolation='bilinear',pair_aggregation='both directions mean',times=[.5*k for k in range(1,10)],zero_overlap='undefined (not zero score)')


def load_model():
 import torch
 from met3r import MEt3R
 torch.set_num_threads(1)
 metric=MEt3R(img_size=None,use_norm=True,backbone='dust3r',feature_backbone='dino16',feature_backbone_weights='mhamilton723/FeatUp',upsampler='featup',distance='cosine',freeze=True).cuda().eval()
 return metric


def tensor(fr):
 import torch
 im=cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)
 im=cv2.resize(im,(336,192),interpolation=cv2.INTER_LINEAR)
 return torch.from_numpy(im.copy()).permute(2,0,1).float()/127.5-1


def measure(model,fa,fb,mask_path):
 import torch
 a,b=tensor(fa),tensor(fb)
 x=torch.stack([torch.stack([a,b]),torch.stack([b,a])]).cuda()
 with torch.inference_mode():scores,masks=model(x,return_overlap_mask=True)
 scores=scores.cpu().numpy().tolist();ov=masks.float().mean(dim=(-2,-1)).cpu().numpy().tolist()
 if not (len(scores)==len(ov)==2 and all(np.isfinite(scores)) and all(np.isfinite(ov))):
  raise ValueError('multiview_met3r: check failed: len(scores)==len(ov)==2 and all(np.isfinite(scores)) and all(np.isfinite(ov))')
 mask_np=masks.cpu().numpy().astype(np.uint8)
 np.savez_compressed(mask_path,forward=mask_np[0],reverse=mask_np[1])
 valid=all(o>0 for o in ov)
 directional=[s if o>0 else None for s,o in zip(scores,ov)]
 return dict(score=float(np.mean(scores)) if valid else None,directional_scores=directional,overlap=ov,valid=valid,mask_path=str(mask_path),directions=['a_to_reference_a','b_to_reference_b'])


def aggregate_scenes(rows, scenes):
    """Average pair-times within scenes; every supplied pair must have valid overlap."""
    rows = list(rows)
    if len({(r.get('method'), r.get('kind')) for r in rows}) > 1:
        raise ValueError('Rows must belong to one method and one metric')
    scores = {}
    for scene in scenes:
        subset = [r for r in rows if r['episode'] == scene]
        scores[scene] = float(np.mean([r['score'] for r in subset])) if subset and all(r['valid'] for r in subset) else None
    return dict(score=float(np.mean(list(scores.values()))) if scores and all(v is not None for v in scores.values()) else None,
                per_scene=scores)
