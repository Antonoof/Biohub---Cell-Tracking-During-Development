#!/usr/bin/env python3
import hashlib, json
from collections import defaultdict
from pathlib import Path
import numpy as np
import zarr
from tqdm import tqdm

ROOT=Path('/home/tweak/bio/train')
OUT=Path('/home/tweak/bio/cross_video_frame_reuse_audit.json')
exact=defaultdict(list)
projection=defaultdict(list)
videos=sorted(ROOT.glob('*.zarr'))
for p in tqdm(videos,desc='frame reuse audit'):
    a=zarr.open_group(str(p),mode='r')['0']
    T=int(a.shape[0])
    frames=sorted(set((0,T//4,T//2,(3*T)//4,T-1)))
    for t in frames:
        x=np.asarray(a[t])
        exact[hashlib.blake2b(x.tobytes(),digest_size=16).hexdigest()].append((p.stem,t))
        # Translation-tolerant coarse signature: sorted block means from a Z-MIP.
        mip=x.max(axis=0).astype(np.float32)
        h,w=mip.shape; by,bx=max(h//16,1),max(w//16,1)
        blocks=mip[:by*16,:bx*16].reshape(16,by,16,bx).mean((1,3))
        q=np.round(np.sort(blocks.ravel())/16).astype(np.int32)
        projection[hashlib.blake2b(q.tobytes(),digest_size=16).hexdigest()].append((p.stem,t))
def cross(groups):
    out=[]
    for h,items in groups.items():
        vids={v for v,_ in items}
        if len(vids)>1: out.append({'hash':h,'items':items})
    return sorted(out,key=lambda x:(-len(x['items']),x['hash']))
result={'videos':len(videos),'sampled_frames':sum(len(v) for v in exact.values()),
        'exact_cross_video_groups':cross(exact),'coarse_cross_video_groups':cross(projection)}
OUT.write_text(json.dumps(result,indent=2))
print('videos',result['videos'],'sampled',result['sampled_frames'])
print('exact cross-video groups',len(result['exact_cross_video_groups']))
print('coarse cross-video groups',len(result['coarse_cross_video_groups']))
for row in result['exact_cross_video_groups'][:20]: print('EXACT',row['items'])
