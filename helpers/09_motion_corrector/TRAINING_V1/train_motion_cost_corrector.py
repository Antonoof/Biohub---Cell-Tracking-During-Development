#!/usr/bin/env python3
"""Learn a bounded residual cost for the V8 sequential motion relinker."""
from __future__ import annotations
import argparse,json,random,time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader,TensorDataset
from tqdm import tqdm
import sys
sys.path.insert(0,str(Path(__file__).parent))
import finetune_edge_head_on_proposals as ft

FEATURES=["base_cost","raw_dist","registered_dist","motion_dist","abs_dz","abs_dy","abs_dx",
          "abs_reg_dz","abs_reg_dy","abs_reg_dx","velocity_z","velocity_y","velocity_x","velocity_mag",
          "shift_z","shift_y","shift_x","shift_mag","det_src","det_tgt","det_disagree_src","det_disagree_tgt",
          "density_src","density_tgt","z_boundary_src","z_boundary_tgt","has_predecessor","frozen"]
# These signals are available during proposal-cache construction but are not
# preserved as node properties in the submitted GEFF graph. Exclude them from
# the learned model so training and post-processing inference are identical.
RUNTIME_DROP={"det_src","det_tgt","det_disagree_src","det_disagree_tgt","frozen"}

def argspec():
 p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
 p.add_argument("--data",type=Path,default=Path('data/train'));p.add_argument("--proposals",type=Path,default=Path('data/ab_proposals_export/biohub_ab_proposals'))
 p.add_argument("--splits",type=Path,default=Path(__file__).resolve().parent/'splits_ensembleB.json');p.add_argument("--cache",type=Path,default=Path('data/motion_cost_cache'))
 p.add_argument("--output",type=Path,default=Path('artifacts/motion_cost_corrector'));p.add_argument("--tight",type=float,default=6.2);p.add_argument("--relaxed",type=float,default=9.5)
 p.add_argument("--velocity-weight",type=float,default=.52);p.add_argument("--residual-scale",type=float,default=2.0);p.add_argument("--epochs",type=int,default=40);p.add_argument("--batch-size",type=int,default=8192)
 p.add_argument("--lr",type=float,default=2e-3);p.add_argument("--patience",type=int,default=7);p.add_argument("--negative-ratio",type=int,default=20);p.add_argument("--seed",type=int,default=2028);p.add_argument("--rebuild-cache",action='store_true');p.add_argument("--max-videos",type=int,default=0);p.add_argument("--device",default='cuda:0')
 p.add_argument("--parent-mode",choices=["proposal_nn","gt"],default="proposal_nn",help="Velocity parents: proposal_nn=serve-matched (default), gt=teacher-forced leak")
 p.add_argument("--fold",type=int,default=0,help="Fold index when --splits is a list of folds (honest GKF5)")
 return p.parse_args()

class MotionResidual(nn.Module):
 def __init__(self,n):
  super().__init__();self.net=nn.Sequential(nn.Linear(n,64),nn.SiLU(),nn.Dropout(.05),nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
 def forward(self,x):return self.net(x).squeeze(-1)

def shifts_for_video(v):
 shifts=[]
 scale=np.asarray(v.voxel_scale_um,np.float64)
 for t in range(v.image_shape_raw[0]-1):
  a0,a1=int(v.offsets[t]),int(v.offsets[t+1]);b0,b1=int(v.offsets[t+1]),int(v.offsets[t+2])
  a=v.coords[a0:a1,1:].astype(np.float64)*scale;b=v.coords[b0:b1,1:].astype(np.float64)*scale
  if t in v.frozen_sources or not len(a) or not len(b):shifts.append(np.zeros(3));continue
  ta,tb=cKDTree(a),cKDTree(b);dab,jab=tb.query(a,k=1);dba,jba=ta.query(b,k=1)
  ia=np.arange(len(a));good=(dab<=10)&(jba[jab]==ia)
  shifts.append(np.median(b[jab[good]]-a[good],axis=0) if good.sum()>=5 else np.zeros(3))
 return shifts

def video_rows(v,group_start,train,args):
 scale=np.asarray(v.voxel_scale_um,np.float64);shifts=shifts_for_video(v)
 # Serve-matched velocity: proposal NN parents, NOT GT edges (fixes teacher forcing).
 parent_mode=getattr(args,'parent_mode','proposal_nn')
 if parent_mode=='gt':
  parent={int(d):int(s) for s,d in v.gt_edges}
 else:
  parent={}  # unused when proposal_nn velocity is used
 rows=[];group_meta=[];gid=group_start
 for t in range(v.image_shape_raw[0]-1):
  a0,a1=int(v.offsets[t]),int(v.offsets[t+1]);b0,b1=int(v.offsets[t+1]),int(v.offsets[t+2]);n0,n1=a1-a0,b1-b0
  group_meta.append((gid,n0,n1,len(v.edges_by_t.get(t,()))));
  if not n0 or not n1:gid+=1;continue
  c0=v.coords[a0:a1,1:].astype(np.float64)*scale;c1=v.coords[b0:b1,1:].astype(np.float64)*scale;m0=v.matches[a0:a1];m1=v.matches[b0:b1]
  shift=np.asarray(shifts[t]);prev_shift=np.asarray(shifts[t-1] if t else np.zeros(3));reg_delta=c1[None,:,:]-c0[:,None,:]-shift;reg=np.linalg.norm(reg_delta,axis=2);candidate=reg<=args.relaxed
  row_active=np.fromiter((int(x) in v.outgoing for x in m0),bool,n0);col_active=np.fromiter((int(x) in v.incoming for x in m1),bool,n1);sup=row_active[:,None]|col_active[None,:]
  label=np.zeros((n0,n1),bool);right={int(g):j for j,g in enumerate(m1) if g>=0}
  for i,g in enumerate(m0):
   if g>=0:
    for d in v.children.get(int(g),()):
     if d in right:label[i,right[d]]=True
  vel=np.zeros((n0,3));has=np.zeros(n0)
  if parent_mode=='gt':
   prev_lookup={}
   if t:
    p0,p1=int(v.offsets[t-1]),int(v.offsets[t]);
    for k,g in enumerate(v.matches[p0:p1]):
     if g>=0:prev_lookup[int(g)]=v.coords[p0+k,1:].astype(np.float64)*scale
   for i,g in enumerate(m0):
    pg=parent.get(int(g),-1) if g>=0 else -1
    if pg in prev_lookup:vel[i]=c0[i]-prev_lookup[pg]-prev_shift;has[i]=1
  elif t:
   # Proposal-index NN on previous frame (same signal available at serve).
   p0,p1=int(v.offsets[t-1]),int(v.offsets[t])
   prev=v.coords[p0:p1,1:].astype(np.float64)*scale
   if len(prev):
    tree=cKDTree(prev); dist,jix=tree.query(c0-shift,k=1)
    if np.ndim(dist)==0: dist=np.asarray([dist]); jix=np.asarray([jix])
    ok=dist<=args.relaxed
    vel[ok]=c0[ok]-prev[jix[ok]]-prev_shift; has[ok]=1
  predicted=c0+shift+args.velocity_weight*vel;motion_delta=c1[None,:,:]-predicted[:,None,:];motion=np.linalg.norm(motion_delta,axis=2);raw_delta=c1[None,:,:]-c0[:,None,:];raw=np.linalg.norm(raw_delta,axis=2);base_cost=motion+.05*reg
  dens0=(cdist(c0,c0)<=15).sum(1)-1;dens1=(cdist(c1,c1)<=15).sum(1)-1
  zmax=max((v.image_shape_raw[1]-1)*scale[0],1);zb0=np.minimum(c0[:,0]/zmax,1-c0[:,0]/zmax);zb1=np.minimum(c1[:,0]/zmax,1-c1[:,0]/zmax)
  ii,jj=np.nonzero(candidate if not train else (candidate&sup))
  if train and len(ii):
   pos=np.flatnonzero(label[ii,jj]);neg=np.flatnonzero(~label[ii,jj]);nk=min(len(neg),max(64,max(1,len(pos))*args.negative_ratio))
   if nk<len(neg):neg=neg[np.argpartition(base_cost[ii[neg],jj[neg]],nk-1)[:nk]]
   take=np.concatenate([pos,neg]);ii,jj=ii[take],jj[take]
  if len(ii):
   md0=v.member_det_prob[a0:a1];md1=v.member_det_prob[b0:b1];f=np.column_stack([base_cost[ii,jj],raw[ii,jj],reg[ii,jj],motion[ii,jj],np.abs(raw_delta[ii,jj,0]),np.abs(raw_delta[ii,jj,1]),np.abs(raw_delta[ii,jj,2]),np.abs(reg_delta[ii,jj,0]),np.abs(reg_delta[ii,jj,1]),np.abs(reg_delta[ii,jj,2]),vel[ii,0],vel[ii,1],vel[ii,2],np.linalg.norm(vel[ii],axis=1),np.full(len(ii),shift[0]),np.full(len(ii),shift[1]),np.full(len(ii),shift[2]),np.full(len(ii),np.linalg.norm(shift)),v.fused_det_prob[a0:a1][ii],v.fused_det_prob[b0:b1][jj],np.abs(md0[ii,0]-md0[ii,1]),np.abs(md1[jj,0]-md1[jj,1]),dens0[ii],dens1[jj],zb0[ii],zb1[jj],has[ii],np.full(len(ii),float(t in v.frozen_sources))]).astype(np.float32)
   rows.append((f,label[ii,jj].astype(np.uint8),sup[ii,jj].astype(np.uint8),np.full(len(ii),gid,np.int32),ii.astype(np.int32),jj.astype(np.int32),reg[ii,jj].astype(np.float32)))
  gid+=1
 return rows,group_meta,gid

def make_cache(stems,name,args):
 allrows=[];meta=[];gid=0
 for stem in tqdm(stems[:args.max_videos or None],desc=f'cache {name}'):
  v=ft.load_proposal_video(args.data,args.proposals,stem,7.0);r,m,gid=video_rows(v,gid,name=='train',args);allrows.extend(r);meta.extend(m)
 arrays=[np.concatenate([r[i] for r in allrows]) for i in range(7)];gm=np.asarray(meta,np.int32);args.cache.mkdir(parents=True,exist_ok=True);np.savez_compressed(args.cache/f'{name}.npz',features=arrays[0],labels=arrays[1],supervision=arrays[2],groups=arrays[3],src=arrays[4],tgt=arrays[5],registered=arrays[6],group_meta=gm,feature_names=np.asarray(FEATURES));print(name,len(arrays[1]),'positive',int(arrays[1].sum()),'groups',len(gm))

def assignments(cost,reg,groups,src,tgt,meta,args):
 selected=np.zeros(len(cost),bool)
 for gid,n0,n1,_ in meta:
  idx=np.flatnonzero(groups==gid)
  if not len(idx) or not n0 or not n1:continue
  used0=set();used1=set()
  for gate in (args.tight,args.relaxed):
   q=idx[(reg[idx]<=gate)&~np.isin(src[idx],list(used0))&~np.isin(tgt[idx],list(used1))]
   if not len(q):continue
   mat=np.full((n0,n1),1e6);mat[src[q],tgt[q]]=cost[q];ri,ci=linear_sum_assignment(mat)
   for a,b in zip(ri,ci):
    if mat[a,b]>=1e6:continue
    hit=q[(src[q]==a)&(tgt[q]==b)]
    if len(hit):selected[hit[0]]=True;used0.add(int(a));used1.add(int(b))
 return selected

def score(selected,y,sup,groups,meta):
 tp=int((selected&(y>0)).sum());fp=int((selected&(y==0)&(sup>0)).sum());hit=np.bincount(groups[selected&(y>0)],minlength=len(meta));fn=int(np.maximum(meta[:,3]-hit,0).sum());return {'tp':tp,'fp':fp,'fn':fn,'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'jaccard':tp/max(tp+fp+fn,1)}

@torch.no_grad()
def residuals(model,x,mean,std,args):
 out=[];model.eval()
 for i in range(0,len(x),args.batch_size):out.append((args.residual_scale*torch.tanh(model(torch.from_numpy((x[i:i+args.batch_size]-mean)/std).to(args.device))/args.residual_scale)).cpu().numpy())
 return np.concatenate(out)

def main():
 args=argspec();random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);args.device=str(torch.device(args.device if torch.cuda.is_available() else 'cpu'));tr,va=ft.load_split(args.splits,args.fold)
 print(f'Motion fold={args.fold} parent_mode={args.parent_mode} train={len(tr)} val={len(va)}',flush=True)
 if args.rebuild_cache or not (args.cache/'train.npz').exists():make_cache(tr,'train',args);make_cache(va,'val',args)
 a=np.load(args.cache/'train.npz');v=np.load(args.cache/'val.npz');names=[str(q) for q in a['feature_names']];keep=np.asarray([n not in RUNTIME_DROP for n in names]);runtime_features=[n for n,k in zip(names,keep) if k];x=a['features'][:,keep].astype(np.float32);y=a['labels'].astype(np.float32);xv=v['features'][:,keep].astype(np.float32);yv=v['labels'];sv=v['supervision'];gv=v['groups'];src=v['src'];tgt=v['tgt'];reg=v['registered'];meta=v['group_meta'];mean=x.mean(0);std=x.std(0).clip(1e-4);print('Runtime-safe features',len(runtime_features),runtime_features)
 model=MotionResidual(x.shape[1]).to(args.device);opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4);args.output.mkdir(parents=True,exist_ok=True)
 base_sel=assignments(xv[:,0],reg,gv,src,tgt,meta,args);baseline=score(base_sel,yv,sv,gv,meta);best=baseline['jaccard'];print('MOTION BASELINE',baseline)
 torch.save({'model':model.state_dict(),'mean':mean,'std':std,'features':runtime_features,'residual_scale':args.residual_scale,'metrics':baseline},args.output/'motion_corrector_best.pt')
 loader=DataLoader(TensorDataset(torch.from_numpy(x),torch.from_numpy(y)),batch_size=args.batch_size,shuffle=True,pin_memory=True);stale=0;hist=[];mt=torch.from_numpy(mean).to(args.device);st=torch.from_numpy(std).to(args.device)
 for ep in range(args.epochs):
  model.train();ls=[];t0=time.time()
  for xb,yb in loader:
   xb=xb.to(args.device);yb=yb.to(args.device);res=args.residual_scale*torch.tanh(model((xb-mt)/st)/args.residual_scale);logit=2.5-xb[:,0]+res;bce=F.binary_cross_entropy_with_logits(logit,yb,reduction='none');p=torch.sigmoid(logit);pt=p*yb+(1-p)*(1-yb);loss=(((1-pt)**2)*bce).mean();opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2);opt.step();ls.append(float(loss.detach()))
  corr=residuals(model,xv,mean,std,args);sel=assignments(xv[:,0]-corr,reg,gv,src,tgt,meta,args);m=score(sel,yv,sv,gv,meta);row={'epoch':ep,'loss':float(np.mean(ls)),'seconds':time.time()-t0,**m};hist.append(row);(args.output/'metrics.json').write_text(json.dumps(hist,indent=2));print('Epoch',ep,row)
  if m['jaccard']>best:best=m['jaccard'];stale=0;torch.save({'model':model.state_dict(),'mean':mean,'std':std,'features':runtime_features,'residual_scale':args.residual_scale,'metrics':m},args.output/'motion_corrector_best.pt');print(' NEW MOTION BEST',best)
  else:stale+=1
  if stale>=args.patience:print('Early stopping');break
 print('Done',best,args.output/'motion_corrector_best.pt')
if __name__=='__main__':main()
