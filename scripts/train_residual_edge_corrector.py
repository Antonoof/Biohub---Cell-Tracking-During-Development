#!/usr/bin/env python3
"""Cache hard A/B edge candidates and train a tiny residual graph corrector.

The corrector never replaces the proven detector or transformer. It starts as
an exact identity mapping on the production logit-fused A+B probability and
learns a bounded log-odds correction from confidence, disagreement, geometry,
detection confidence, rank/margin, density, boundary, and frozen-frame cues.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import finetune_edge_head_on_proposals as ft


FEATURE_NAMES = [
    "base_logit", "prob_a", "prob_b", "prob_fused", "prob_disagreement",
    "prob_min", "prob_max", "distance_um", "abs_dz_um", "abs_dy_um", "abs_dx_um",
    "det_src", "det_tgt", "det_a_src", "det_b_src", "det_a_tgt", "det_b_tgt",
    "det_disagree_src", "det_disagree_tgt", "parent_rank", "child_rank",
    "parent_margin", "child_margin", "density_src_15um", "density_tgt_15um",
    "z_boundary_src", "z_boundary_tgt", "frame_fraction", "frozen_transition",
]


def parse_args():
    p=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data",type=Path,default=Path("data/train"))
    p.add_argument("--proposals",type=Path,default=Path("data/ab_proposals_export/biohub_ab_proposals"))
    p.add_argument("--splits",type=Path,default=Path("data/splits_ensembleB.json"))
    p.add_argument("--weights-a",type=Path,default=Path("datahub_ensemble_2model/weights/unet_transformer/split_0/edge_predictor_best.pth"))
    p.add_argument("--weights-b",type=Path,default=Path("datahub_ensemble_2model/weights/unet_transformer/split_1/edge_predictor_best.pth"))
    p.add_argument("--config-a",type=Path,default=Path("datahub_ensemble_2model/weights/unet_transformer/split_0/config.json"))
    p.add_argument("--config-b",type=Path,default=Path("datahub_ensemble_2model/weights/unet_transformer/split_1/config.json"))
    p.add_argument("--cache",type=Path,default=Path("data/edge_corrector_cache"))
    p.add_argument("--output",type=Path,default=Path("external/bio_track_repo/weights/residual_edge_corrector"))
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--max-nodes",type=int,default=2800)
    p.add_argument("--max-distance-um",type=float,default=14.5)
    p.add_argument("--candidate-min-prob",type=float,default=0.01)
    p.add_argument("--candidate-topk",type=int,default=4)
    p.add_argument("--negative-ratio",type=int,default=16)
    p.add_argument("--max-negatives-per-window",type=int,default=512)
    p.add_argument("--edge-threshold",type=float,default=0.54)
    p.add_argument("--epochs",type=int,default=30)
    p.add_argument("--batch-size",type=int,default=8192)
    p.add_argument("--lr",type=float,default=2e-3)
    p.add_argument("--weight-decay",type=float,default=1e-4)
    p.add_argument("--delta-scale",type=float,default=2.0)
    p.add_argument("--patience",type=int,default=6)
    p.add_argument("--seed",type=int,default=2027)
    p.add_argument("--rebuild-cache",action="store_true")
    p.add_argument("--max-windows",type=int,default=0,help="Smoke-test limit; 0 uses all windows")
    return p.parse_args()


class ResidualCorrector(nn.Module):
    def __init__(self,n_features):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(n_features,64),nn.SiLU(),nn.Dropout(.05),
                               nn.Linear(64,32),nn.SiLU(),nn.Linear(32,1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self,x): return self.net(x).squeeze(-1)


def rank_and_margin(prob, dim):
    n=prob.shape[dim]
    k=min(2,n)
    vals,idx=torch.topk(prob,k=k,dim=dim)
    rank=torch.empty_like(prob,dtype=torch.float32)
    order=torch.argsort(prob,dim=dim,descending=True)
    shape=[1]*prob.ndim; shape[dim]=n
    ranks=torch.arange(n,device=prob.device,dtype=torch.float32).reshape(shape)+1
    rank.scatter_(dim,order,ranks.expand_as(order))
    best=vals.select(dim,0)
    second=vals.select(dim,1) if k>1 else torch.zeros_like(best)
    if dim==0:
        margin=prob-second.unsqueeze(0)
    else:
        margin=prob-second.unsqueeze(1)
    return rank,margin,idx


@torch.no_grad()
def window_features(model_a,model_b,batch,device,args,keep_all_negatives):
    with torch.autocast("cuda",dtype=torch.float16):
        la=ft.forward_logits(model_a,batch,device)[0].float()
        lb=ft.forward_logits(model_b,batch,device)[0].float()
    n0=int(batch["mask0"][0].sum()); n1=int(batch["mask1"][0].sum())
    la=la[:n0,:n1]; lb=lb[:n0,:n1]
    pa=torch.softmax(la,dim=0); pb=torch.softmax(lb,dim=0)
    pf=torch.softmax((la+lb)*.5,dim=0).clamp(1e-6,1-1e-6)
    y=batch["target"][0,:n0,:n1].to(device).bool()
    sup=batch["supervision"][0,:n0,:n1].to(device).bool()
    c0=batch["coords0"][0,:n0].to(device); c1=batch["coords1"][0,:n1].to(device)
    phys=batch["downsample"][0].to(device)*batch["voxel_scale"][0].to(device)
    u0=c0*phys; u1=c1*phys
    delta=u1.unsqueeze(0)-u0.unsqueeze(1); dist=torch.linalg.vector_norm(delta,dim=-1)
    pr,pm,ptop=rank_and_margin(pf,0); cr,cm,ctop=rank_and_margin(pf,1)
    top=torch.zeros_like(pf,dtype=torch.bool)
    top.scatter_(0,ptop[:min(args.candidate_topk,ptop.shape[0])],True)
    top.scatter_(1,ctop[:,:min(args.candidate_topk,ctop.shape[1])],True)
    cand=sup & (y | ((dist<=args.max_distance_um)&((pf>=args.candidate_min_prob)|top)))
    ij=cand.nonzero(as_tuple=False)
    if not len(ij): return None
    ii,jj=ij[:,0],ij[:,1]; labels=y[ii,jj]
    if not keep_all_negatives:
        pos=torch.nonzero(labels,as_tuple=False).flatten(); neg=torch.nonzero(~labels,as_tuple=False).flatten()
        nkeep=min(len(neg),max(64,min(args.max_negatives_per_window,max(1,len(pos))*args.negative_ratio)))
        if nkeep<len(neg):
            hardness=pf[ii[neg],jj[neg]]+.35*(pa[ii[neg],jj[neg]]-pb[ii[neg],jj[neg]]).abs()-.01*dist[ii[neg],jj[neg]]
            neg=neg[torch.topk(hardness,nkeep).indices]
        selected=torch.cat([pos,neg]); ii,jj,labels=ii[selected],jj[selected],labels[selected]
    d0=(torch.cdist(u0,u0)<=15).sum(1).float()-1; d1=(torch.cdist(u1,u1)<=15).sum(1).float()-1
    det0=batch["det0"][0,:n0].to(device); det1=batch["det1"][0,:n1].to(device)
    md0=batch["member_det0"][0,:n0].to(device); md1=batch["member_det1"][0,:n1].to(device)
    shape=batch["image_shape"][0,1:].to(device).float().clamp_min(2)
    z0=(c0[:,0]/(shape[0]-1)).clamp(0,1); z1=(c1[:,0]/(shape[0]-1)).clamp(0,1)
    zb0=torch.minimum(z0,1-z0); zb1=torch.minimum(z1,1-z1)
    frame=float(batch["frame"][0]); total=max(float(batch["image_shape"][0,0])-1,1)
    frozen=float(batch["frozen"][0])
    dab=(pa-pb).abs(); amin=torch.minimum(pa,pb); amax=torch.maximum(pa,pb)
    f=torch.stack([
        torch.logit(pf[ii,jj]),pa[ii,jj],pb[ii,jj],pf[ii,jj],dab[ii,jj],amin[ii,jj],amax[ii,jj],
        dist[ii,jj],delta[ii,jj,0].abs(),delta[ii,jj,1].abs(),delta[ii,jj,2].abs(),
        det0[ii],det1[jj],md0[ii,0],md0[ii,1],md1[jj,0],md1[jj,1],
        (md0[ii,0]-md0[ii,1]).abs(),(md1[jj,0]-md1[jj,1]).abs(),
        pr[ii,jj],cr[ii,jj],pm[ii,jj],cm[ii,jj],d0[ii],d1[jj],zb0[ii],zb1[jj],
        torch.full_like(pf[ii,jj],frame/total),torch.full_like(pf[ii,jj],frozen),
    ],dim=1)
    return f.cpu().numpy().astype(np.float32),labels.cpu().numpy().astype(np.uint8),pf[ii,jj].cpu().numpy().astype(np.float32)


def build_cache(stems,split_name,args,model_a,model_b,device):
    videos=[ft.load_proposal_video(args.data,args.proposals,s,7.0) for s in tqdm(stems,desc=f"{split_name} metadata")]
    ds=ft.ProposalWindowDataset(videos,args.max_nodes,False,None,1,args.seed)
    loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=0,collate_fn=ft.collate,pin_memory=True)
    feats=[]; labels=[]; base=[]; groups=[]; gt=[]
    limit=args.max_windows or len(loader)
    for group,batch in enumerate(tqdm(loader,total=min(limit,len(loader)),desc=f"cache {split_name}")):
        if group>=limit: break
        out=window_features(model_a,model_b,batch,device,args,split_name=="val")
        gt.append(int(batch["gt_edges_total"][0]))
        if out is None: continue
        f,y,p=out; feats.append(f); labels.append(y); base.append(p); groups.append(np.full(len(y),group,np.int32))
    result={"features":np.concatenate(feats),"labels":np.concatenate(labels),"base_prob":np.concatenate(base),
            "groups":np.concatenate(groups),"group_gt":np.asarray(gt,np.int32),"feature_names":np.asarray(FEATURE_NAMES)}
    args.cache.mkdir(parents=True,exist_ok=True); path=args.cache/f"{split_name}.npz"
    np.savez_compressed(path,**result)
    print(f"cached {split_name}: rows={len(result['labels']):,} positives={int(result['labels'].sum()):,} -> {path}")


def metric(prob,y,groups,group_gt,threshold):
    pred=prob>=threshold; tp=int((pred&(y>0)).sum()); fp=int((pred&(y==0)).sum())
    hit=np.bincount(groups[pred&(y>0)],minlength=len(group_gt)); fn=int(np.maximum(group_gt-hit,0).sum())
    return {"tp":tp,"fp":fp,"fn":fn,"precision":tp/max(tp+fp,1),"recall":tp/max(tp+fn,1),"jaccard":tp/max(tp+fp+fn,1)}


@torch.no_grad()
def evaluate(model,x,base,y,groups,group_gt,mean,std,args):
    model.eval(); out=[]
    for i in range(0,len(x),args.batch_size):
        xb=torch.from_numpy((x[i:i+args.batch_size]-mean)/std).to(args.device)
        delta=args.delta_scale*torch.tanh(model(xb)/args.delta_scale)
        bl=torch.from_numpy(np.log(base[i:i+args.batch_size]/(1-base[i:i+args.batch_size]))).to(args.device)
        out.append(torch.sigmoid(bl+delta).cpu().numpy())
    prob=np.concatenate(out)
    rows={float(t):metric(prob,y,groups,group_gt,float(t)) for t in np.arange(.48,.601,.01)}
    primary=metric(prob,y,groups,group_gt,args.edge_threshold); best=max(rows,key=lambda t:rows[t]["jaccard"])
    return primary,best,rows[best]


def main():
    args=parse_args(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    args.device=str(torch.device(args.device if torch.cuda.is_available() else "cpu"))
    train_stems,val_stems=ft.load_split(args.splits,0)
    train_path,val_path=args.cache/"train.npz",args.cache/"val.npz"
    if args.rebuild_cache or not (train_path.exists() and val_path.exists()):
        default={"unet_out_channels":32,"unet_layers":[32,64,128],"downsample":[1,4,4],"window_size":2,"pool_kernel_um":5.0}
        ca={**default,**json.loads(args.config_a.read_text())}; cb={**default,**json.loads(args.config_b.read_text())}
        ma=ft.build_model(ca,args.weights_a,torch.device(args.device)).eval(); mb=ft.build_model(cb,args.weights_b,torch.device(args.device)).eval()
        for p in ma.parameters():p.requires_grad=False
        for p in mb.parameters():p.requires_grad=False
        build_cache(train_stems,"train",args,ma,mb,torch.device(args.device)); build_cache(val_stems,"val",args,ma,mb,torch.device(args.device))
        del ma,mb; torch.cuda.empty_cache()
    tr=np.load(train_path); va=np.load(val_path)
    x=tr["features"].astype(np.float32); y=tr["labels"].astype(np.float32); base=tr["base_prob"].astype(np.float32)
    xv=va["features"].astype(np.float32); yv=va["labels"].astype(np.uint8); bv=va["base_prob"].astype(np.float32)
    gv=va["groups"].astype(np.int64); ggt=va["group_gt"].astype(np.int64)
    mean=x.mean(0); std=x.std(0).clip(1e-4)
    model=ResidualCorrector(x.shape[1]).to(args.device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    args.output.mkdir(parents=True,exist_ok=True)
    baseline=metric(bv,yv,gv,ggt,args.edge_threshold); best=baseline["jaccard"]; stale=0
    print(f"IDENTITY BASELINE J={best:.4f} P={baseline['precision']:.4f} R={baseline['recall']:.4f} th={args.edge_threshold}")
    state={"model":model.state_dict(),"mean":mean,"std":std,"feature_names":FEATURE_NAMES,"delta_scale":args.delta_scale,"threshold":args.edge_threshold,"metrics":baseline}
    torch.save(state,args.output/"edge_corrector_best.pt")
    dataset=TensorDataset(torch.from_numpy(x),torch.from_numpy(base),torch.from_numpy(y))
    loader=DataLoader(dataset,batch_size=args.batch_size,shuffle=True,num_workers=0,pin_memory=True)
    history=[]
    for epoch in range(args.epochs):
        model.train(); losses=[]; t0=time.time()
        for xb,pb,yb in loader:
            xb=((xb.to(args.device)-torch.from_numpy(mean).to(args.device))/torch.from_numpy(std).to(args.device))
            pb=pb.to(args.device).clamp(1e-6,1-1e-6); yb=yb.to(args.device)
            delta=args.delta_scale*torch.tanh(model(xb)/args.delta_scale); prob=torch.sigmoid(torch.logit(pb)+delta)
            bce=F.binary_cross_entropy(prob,yb,reduction="none"); pt=prob*yb+(1-prob)*(1-yb)
            focal=(((1-pt)**2)*bce).mean(); inter=(prob*yb).sum(); union=(prob+yb-prob*yb).sum()
            loss=focal+.02*(1-(inter+1e-6)/(union+1e-6))
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),2); opt.step(); losses.append(float(loss.detach()))
        primary,grid_t,grid=evaluate(model,xv,bv,yv,gv,ggt,mean,std,args)
        row={"epoch":epoch,"loss":float(np.mean(losses)),"seconds":time.time()-t0,"primary":primary,"grid_best_threshold":grid_t,"grid_best":grid}; history.append(row)
        (args.output/"metrics.json").write_text(json.dumps(history,indent=2))
        print(f"Epoch {epoch:02d}: loss={row['loss']:.6f} J@{args.edge_threshold}={primary['jaccard']:.4f} P={primary['precision']:.4f} R={primary['recall']:.4f} grid_best={grid_t:.2f} time={row['seconds']:.1f}s",flush=True)
        if primary["jaccard"]>best:
            best=primary["jaccard"]; stale=0; torch.save({"model":model.state_dict(),"mean":mean,"std":std,"feature_names":FEATURE_NAMES,"delta_scale":args.delta_scale,"threshold":args.edge_threshold,"metrics":primary},args.output/"edge_corrector_best.pt"); print(f" NEW CORRECTOR BEST {best:.4f}")
        else:
            stale+=1
            if stale>=args.patience: print("Early stopping"); break
    print(f"Done; best corrector J={best:.6f} -> {args.output/'edge_corrector_best.pt'}")


if __name__=="__main__": main()
