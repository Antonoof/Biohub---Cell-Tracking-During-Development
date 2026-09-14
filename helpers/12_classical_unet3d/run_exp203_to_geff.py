#!/usr/bin/env python3
"""Exp203 classical ensemble -> GEFF (final_combo from 0_917model notebook)."""
from __future__ import annotations
import argparse, gc, glob, json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import polars as pl
import torch, torch.nn as nn
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max

ROOT = Path(__file__).resolve().parents[2]
for p in (
    ROOT / "public_models/Biohub Tracking Support Pack/repo/src",
    ROOT / "helpers/01_p1_p2_base/shared_repo/src",
):
    if p.exists():
        sys.path.insert(0, str(p))
import tracksdata as td
from biohub_tracking.io import save_graph

SCALE = np.array([1.625, 0.40625, 0.40625]); POOL = 4
DEFAULT_WROOT = ROOT / "public_models" / "0_917model"

def pick_device(pref: str) -> str:
    pref = (pref or "auto").lower()
    if pref == "cpu":
        return "cpu"
    if pref == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if pref in ("cuda", "auto") and torch.cuda.is_available():
        return "cuda"
    if pref == "auto" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


REPAIR = True
CAND_THR = 0.05
GAP_DT = 0
GAP_GATE_UM = 10.0
SNAP_UM = 3.0
SHORT_MIN = 6
LINEFIT_WEIGHT = 0.8
LINEFIT_WINDOW = 2
UNET_THRESH = 0.15
NMS_UM = 4.0
DETECT_THRESH = min(UNET_THRESH, CAND_THR) if REPAIR else UNET_THRESH
MAX_LINK_UM = 10.0
TIGHT_UM = 6.0
DEVICE = "cpu"
import pandas as pd
def _block(ci, co):
    return nn.Sequential(nn.Conv3d(ci,co,3,padding=1), nn.BatchNorm3d(co), nn.ReLU(inplace=True),
                         nn.Conv3d(co,co,3,padding=1), nn.BatchNorm3d(co), nn.ReLU(inplace=True))
class UNet3D(nn.Module):
    def __init__(self, base=24):
        super().__init__()
        self.e1=_block(1,base); self.e2=_block(base,base*2); self.e3=_block(base*2,base*4)
        self.pool=nn.MaxPool3d(2); self.bott=_block(base*4,base*8)
        self.u3=nn.ConvTranspose3d(base*8,base*4,2,stride=2); self.d3=_block(base*8,base*4)
        self.u2=nn.ConvTranspose3d(base*4,base*2,2,stride=2); self.d2=_block(base*4,base*2)
        self.u1=nn.ConvTranspose3d(base*2,base,2,stride=2); self.d1=_block(base*2,base)
        self.out=nn.Conv3d(base,1,1)
    def forward(self,x):
        e1=self.e1(x); e2=self.e2(self.pool(e1)); e3=self.e3(self.pool(e2)); b=self.bott(self.pool(e3))
        d3=self.d3(torch.cat([self.u3(b),e3],1)); d2=self.d2(torch.cat([self.u2(d3),e2],1))
        d1=self.d1(torch.cat([self.u1(d2),e1],1)); return self.out(d1)

MODELS = []  # filled in main()
PREPROCS = []
_BRANCHES = []

def read_array_meta(zp):
    with open(Path(zp)/"0"/"zarr.json") as f: m=json.load(f)
    return dict(shape=tuple(m["shape"]), dtype=np.dtype(m["data_type"]))
_ZC={}
def load_volume(zp, t, meta=None):
    try:
        import zarr; k=str(zp)
        if k not in _ZC: _ZC[k]=zarr.open(k,mode="r")["0"]
        return np.asarray(_ZC[k][t])
    except Exception:
        import blosc2
        if meta is None: meta=read_array_meta(zp)
        buf=blosc2.decompress(open(Path(zp)/"0"/"c"/str(t)/"0"/"0"/"0","rb").read())
        return np.frombuffer(buf,dtype=meta["dtype"]).reshape(meta["shape"][1:])

def pool_xy(vol, f=POOL):
    Z,Y,X=vol.shape; Y2,X2=(Y//f)*f,(X//f)*f
    v=vol[:,:Y2,:X2].astype(np.float32,copy=False)
    return v.reshape(Z,Y2//f,f,X2//f,f).mean(axis=(2,4))
def pool_norm(vol, preproc=""):
    p=pool_xy(vol)
    if preproc=="tophat":
        from scipy.ndimage import grey_opening
        p=np.clip(p-grey_opening(p,size=(1,7,7)),0.0,None)
    lo=float(np.percentile(p,50)); hi=float(np.percentile(p,99.5))
    return np.clip((p-lo)/(hi-lo+1e-6),-0.5,6.0).astype(np.float32)

def _refine(vol, zyx, rz=2, ryx=5):
    Z,Y,X=vol.shape; z,y,x=(int(round(v)) for v in zyx)
    z0,z1=max(0,z-rz),min(Z,z+rz+1); y0,y1=max(0,y-ryx),min(Y,y+ryx+1); x0,x1=max(0,x-ryx),min(X,x+ryx+1)
    crop=vol[z0:z1,y0:y1,x0:x1].astype(np.float32); bg=float(crop.min())
    w=np.clip(crop-bg,0,None); s=float(w.sum())
    if s<=0: return np.array([z,y,x],float),0.0
    zz,yy,xx=np.mgrid[z0:z1,y0:y1,x0:x1]
    return np.array([(zz*w).sum(),(yy*w).sum(),(xx*w).sum()])/s, float(crop.max()-bg)
def _physical_nms(coords, scores, radius_um, scale=SCALE):
    if len(coords)<=1: return coords,scores
    pts=coords*scale[None,:]; order=np.argsort(-scores); tree=cKDTree(pts)
    killed=np.zeros(len(coords),bool); keep=[]
    for i in order:
        if killed[i]: continue
        keep.append(int(i)); killed[tree.query_ball_point(pts[i],r=radius_um)]=True
    keep=np.array(keep); return coords[keep],scores[keep]

UNET_THRESH=0.15; NMS_UM=4.0
DETECT_THRESH = min(UNET_THRESH, CAND_THR) if REPAIR else UNET_THRESH
def detect(vol):
    # each model sees the preprocessing it was TRAINED with, then AVERAGE the heatmaps
    # (never union the detections: DoG-union U-Net measured 0.661 -- over-detection
    #  blows past T_true and the node-count adjustment punishes it)
    hs=[]
    for _m,_pp in zip(MODELS, PREPROCS):
        x=pool_norm(vol,_pp)
        with torch.no_grad():
            hs.append(torch.sigmoid(_m(torch.from_numpy(x)[None,None].to(DEVICE)))[0,0].float().cpu().numpy())
    h = hs[0] if len(hs)==1 else np.mean(hs, axis=0)
    pk=peak_local_max(h, min_distance=1, threshold_abs=DETECT_THRESH, exclude_border=False)
    if len(pk)==0: return np.zeros((0,3)), np.zeros(0)
    sc=h[pk[:,0],pk[:,1],pk[:,2]].astype(float)
    coords=pk.astype(float); coords[:,1]=coords[:,1]*POOL+(POOL-1)/2; coords[:,2]=coords[:,2]*POOL+(POOL-1)/2
    ref=np.array([_refine(vol,c)[0] for c in coords])
    return _physical_nms(ref, sc, NMS_UM)

MAX_LINK_UM=10.0; TIGHT_UM=6.0
def _link(prev_xyz, curr_xyz, prev_vel):
    if len(prev_xyz)==0 or len(curr_xyz)==0: return []
    P=prev_xyz*SCALE[None,:]; C=curr_xyz*SCALE[None,:]
    pred=P+(0.5*prev_vel if prev_vel is not None else 0.0); N,M=len(P),len(C); BIG=1e9
    def _hun(pi,ci,gate):
        if len(pi)==0 or len(ci)==0: return []
        Draw=np.sqrt(((P[pi][:,None]-C[ci][None])**2).sum(2)); D=np.sqrt(((pred[pi][:,None]-C[ci][None])**2).sum(2))
        cost=np.where(Draw>gate,BIG,D); ri,rc=linear_sum_assignment(cost)
        return [(int(pi[r]),int(ci[c])) for r,c in zip(ri,rc) if cost[r,c]<BIG]
    links=_hun(np.arange(N),np.arange(M),min(TIGHT_UM,MAX_LINK_UM))
    up={p for p,_ in links}; uc={c for _,c in links}
    fp=np.array([i for i in range(N) if i not in up],int); fc=np.array([j for j in range(M) if j not in uc],int)
    return links+_hun(fp,fc,MAX_LINK_UM)

COLS=["dataset","row_type","node_id","t","z","y","x","source_id","target_id"]
def _dist_um(a,b):
    d=(np.asarray(a,float)-np.asarray(b,float))*SCALE
    return float(np.sqrt((d*d).sum()))

def _gap_support(pe, ps, te, dt, cand, cand_trees):
    out=[]
    for k in range(1, dt):
        tk=te+k; interp=pe+(ps-pe)*(k/dt); tree=cand_trees[tk] if 0 <= tk < len(cand_trees) else None
        if tree is None: return None
        dist,idx=tree.query(interp*SCALE)
        if dist > SNAP_UM: return None
        out.append(cand[tk][idx])
    return out

def _segments(nodes, succ, pred, vel):
    segs=[]
    for g in nodes:
        if g in pred: continue
        ch=[g]
        while ch[-1] in succ: ch.append(succ[ch[-1]])
        segs.append(ch)
    ends=[]; starts=[]
    for ch in segs:
        ge=ch[-1]; ve=vel.get(ge, np.zeros(3))
        if len(ch) >= 2: ve=(nodes[ch[-1]]["xyz"]-nodes[ch[-2]]["xyz"])*SCALE
        ends.append((ge, int(nodes[ge]["t"]), nodes[ge]["xyz"], ve))
        gs=ch[0]; starts.append((gs, int(nodes[gs]["t"]), nodes[gs]["xyz"]))
    return segs, ends, starts

def _gap_close(nodes, edges, succ, pred, vel, cand, cand_trees, next_id):
    if GAP_DT <= 0: return next_id
    segs, ends, starts = _segments(nodes, succ, pred, vel)
    seglen=[len(ch) for ch in segs]; props=[]
    for i,(ge,te,pe,ve_um) in enumerate(ends):
        for j,(gs,ts,ps) in enumerate(starts):
            dt=ts-te
            if i == j or dt < 1 or dt > GAP_DT: continue
            predpos=pe+(ve_um/SCALE)*dt
            cost=_dist_um(predpos, ps)
            if cost > GAP_GATE_UM: continue
            if dt >= 2 and _gap_support(pe, ps, te, dt, cand, cand_trees) is None: continue
            props.append((cost,i,j,dt))
    used_e=set(); used_s=set()
    for _,i,j,dt in sorted(props):
        if i in used_e or j in used_s: continue
        used_e.add(i); used_s.add(j)
        ge,te,pe,_=ends[i]; gs,_,ps=starts[j]
        if dt == 1:
            edges.append((ge,gs)); continue
        prev=ge
        for k in range(1,dt):
            tk=te+k; interp=pe+(ps-pe)*(k/dt); use=interp
            if cand_trees[tk] is not None:
                dist,idx=cand_trees[tk].query(interp*SCALE)
                if dist <= SNAP_UM: use=cand[tk][idx]
            ng=next_id; next_id += 1
            nodes[ng]={"t":tk, "xyz":np.asarray(use,float)}
            edges.append((prev,ng)); prev=ng
        edges.append((prev,gs))
    return next_id

def _short_filter(nodes, edges):
    if SHORT_MIN <= 1 or not edges: return nodes, edges
    parent={nid:nid for nid in nodes}
    def find(x):
        while parent[x] != x:
            parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b):
        if a not in parent or b not in parent: return
        ra,rb=find(a),find(b)
        if ra != rb: parent[ra]=rb
    out_count=defaultdict(int)
    for a,b in edges:
        union(a,b); out_count[a]+=1
    comps=defaultdict(list)
    for nid in nodes: comps[find(nid)].append(nid)
    keep=set()
    for members in comps.values():
        has_div=any(out_count[n] >= 2 for n in members)
        if len(members) >= SHORT_MIN or has_div: keep.update(members)
    nodes2={nid:n for nid,n in nodes.items() if nid in keep}
    edges2=[(a,b) for a,b in edges if a in nodes2 and b in nodes2]
    return nodes2, edges2

def _linefit(nodes, edges):
    if LINEFIT_WEIGHT <= 0: return
    pred=defaultdict(list); succ=defaultdict(list)
    for a,b in edges:
        if a in nodes and b in nodes and int(nodes[b]["t"]) == int(nodes[a]["t"]) + 1:
            succ[a].append(b); pred[b].append(a)
    orig={k:v["xyz"].copy() for k,v in nodes.items()}; updates={}
    W=int(LINEFIT_WINDOW)
    for nid in nodes:
        neigh=[(0,nid)]; cur=nid
        for step in range(1,W+1):
            ps=pred.get(cur, [])
            if len(ps) != 1: break
            cur=ps[0]; neigh.append((-step,cur))
        cur=nid
        for step in range(1,W+1):
            ss=succ.get(cur, [])
            if len(ss) != 1: break
            cur=ss[0]; neigh.append((step,cur))
        if len(neigh) < 3: continue
        dt=np.array([a for a,_ in neigh], float)
        xyz=np.stack([orig[n] for _,n in neigh])
        fit=np.array([np.polyval(np.polyfit(dt, xyz[:,ax], 1), 0.0) for ax in range(3)])
        if np.isfinite(fit).all():
            updates[nid]=(1.0-LINEFIT_WEIGHT)*orig[nid]+LINEFIT_WEIGHT*fit
    for nid,xyz in updates.items(): nodes[nid]["xyz"]=xyz

def _emit(ds, nodes, edges):
    edge_set=[]; seen=set()
    for a,b in edges:
        if a == b or a not in nodes or b not in nodes or (a,b) in seen: continue
        seen.add((a,b)); edge_set.append((a,b))
    used=set()
    for a,b in edge_set: used.add(a); used.add(b)
    nrows=[]; erows=[]
    for nid in sorted(used):
        n=nodes[nid]; z,y,x=n["xyz"]
        nrows.append((ds,"node",int(nid),int(n["t"]),float(z),float(y),float(x),-1,-1))
    for a,b in edge_set:
        if a in used and b in used: erows.append((ds,"edge",-1,-1,-1,-1,-1,int(a),int(b)))
    return pd.DataFrame(nrows,columns=COLS), pd.DataFrame(erows,columns=COLS)

def repair_track(dets, ds):
    nodes={}; frame_ids=[]; cand=[]; cand_trees=[]; nid=1
    for t,(coords,scores) in enumerate(dets):
        coords=np.asarray(coords,float).reshape(-1,3); scores=np.asarray(scores,float).reshape(-1)
        seeds=coords[scores >= UNET_THRESH]
        cands=coords[(scores >= CAND_THR) & (scores < UNET_THRESH)]
        cand.append(cands); cand_trees.append(cKDTree(cands*SCALE) if len(cands) else None)
        ids=[]
        for xyz in seeds:
            nodes[nid]={"t":t, "xyz":np.asarray(xyz,float)}; ids.append(nid); nid += 1
        frame_ids.append(ids)
    edges=[]; succ={}; pred={}; vel={}
    for t in range(len(dets)-1):
        P=np.asarray([nodes[g]["xyz"] for g in frame_ids[t]], float).reshape(-1,3)
        C=np.asarray([nodes[g]["xyz"] for g in frame_ids[t+1]], float).reshape(-1,3)
        if len(P) == 0 or len(C) == 0: continue
        prev_vel=np.array([vel.get(g, np.zeros(3)) for g in frame_ids[t]])
        for pi,ci in _link(P, C, prev_vel if len(prev_vel) else None):
            gp,gc=frame_ids[t][pi],frame_ids[t+1][ci]
            edges.append((gp,gc)); succ[gp]=gc; pred[gc]=gp; vel[gc]=(C[ci]-P[pi])*SCALE
    nid=_gap_close(nodes, edges, succ, pred, vel, cand, cand_trees, nid)
    nodes,edges=_short_filter(nodes, edges)
    _linefit(nodes, edges)
    return _emit(ds, nodes, edges)

def track_movie(zp, ds, T):
    if REPAIR:
        meta=read_array_meta(zp); dets=[]
        for t in range(T):
            dets.append(detect(load_volume(zp,t,meta))); gc.collect()
        return repair_track(dets, ds)
    meta=read_array_meta(zp); node_rows=[]; edge_rows=[]
    prev_ids=[]; prev_xyz=np.zeros((0,3)); prev_vel=None; nid=1
    for t in range(T):
        coords,scores=detect(load_volume(zp,t,meta)); gc.collect()
        ids=list(range(nid,nid+len(coords))); nid+=len(coords)
        for i,c in zip(ids,coords): node_rows.append((ds,"node",i,t,float(c[0]),float(c[1]),float(c[2]),-1,-1))
        if t>0 and len(prev_ids):
            links=_link(prev_xyz,coords,prev_vel); vel=np.zeros((len(prev_xyz),3))
            for p,c in links:
                edge_rows.append((ds,"edge",-1,-1,-1,-1,-1,prev_ids[p],ids[c])); vel[p]=(coords[c]-prev_xyz[p])*SCALE
            nv=np.zeros((len(coords),3))
            for p,c in links: nv[c]=vel[p]
            prev_vel=nv
        else: prev_vel=None
        prev_ids,prev_xyz=ids,coords
    nodes=pd.DataFrame(node_rows,columns=COLS); edges=pd.DataFrame(edge_rows,columns=COLS)
    if len(edges):
        used=set(edges.source_id)|set(edges.target_id); nodes=nodes[nodes.node_id.isin(used)].reset_index(drop=True)
    return nodes,edges

def avail_T(zp):
    meta=read_array_meta(zp); T=meta["shape"][0]
    present=[t for t in range(T) if (Path(zp)/"0"/"c"/str(t)/"0"/"0"/"0").exists()]
    return max(present)+1 if present else 0

# ======================================================================
# DETECTION VARIANT — N-model ensemble + flip-quartet TTA on pooled (Y,X)
# ----------------------------------------------------------------------
# Per branch: forward the 4 flip views {id, flipY, flipX, flipY.flipX} — the
# exact augmentations used in training (Z never touched: anisotropic) —
# inverse-transform the LOGITS (flips are self-inverse), average, sigmoid
# AFTER averaging. Branch heatmaps averaged (equal weights). Everything
# downstream (peak_local_max @0.15, refine, physical NMS 4.0 um) unchanged.
# ======================================================================
def _build_branches():
    global _BRANCHES
    _BRANCHES = []
    _BRANCHES.append(([MODELS[0]], [1.0], ''))
    _BRANCHES.append(([MODELS[1]], [1.0], 'tophat'))
    # third branch already appended as MODELS[2] with tophat
    _BRANCHES.append(([MODELS[2]], [1.0], 'tophat'))
# SHORT_MIN override: read by _short_filter (cell 3) at call time.
SHORT_MIN = 6
_FLIP_VIEWS = (0, 1, 2, 3)  # bit0 = flip Y, bit1 = flip X

def _flip(x, i):
    if i & 1: x = np.flip(x, -2)
    if i & 2: x = np.flip(x, -1)
    return x

detect_base = detect  # base detect kept for reference (A/B)
def detect(vol):
    hs = []
    for models, ws, pp in _BRANCHES:
        x = pool_norm(vol, pp)
        acc = None
        for i in _FLIP_VIEWS:
            xv = np.ascontiguousarray(_flip(x, i))
            with torch.no_grad():
                lg = None
                for m, w in zip(models, ws):
                    l = m(torch.from_numpy(xv)[None, None].to(DEVICE))[0, 0].float().cpu().numpy()
                    lg = w * l if lg is None else lg + w * l
            lg = _flip(lg, i)  # self-inverse
            acc = lg if acc is None else acc + lg
        hs.append(1.0 / (1.0 + np.exp(-(acc / len(_FLIP_VIEWS)))))
    h = hs[0] if len(hs) == 1 else np.mean(hs, axis=0)
    pk = peak_local_max(h, min_distance=1, threshold_abs=DETECT_THRESH, exclude_border=False)
    if len(pk) == 0: return np.zeros((0, 3)), np.zeros(0)
    sc = h[pk[:, 0], pk[:, 1], pk[:, 2]].astype(float)
    coords = pk.astype(float)
    coords[:, 1] = coords[:, 1] * POOL + (POOL - 1) / 2
    coords[:, 2] = coords[:, 2] * POOL + (POOL - 1) / 2
    ref = np.array([_refine(vol, c)[0] for c in coords])
    return _physical_nms(ref, sc, NMS_UM)

# ======================================================================
# POST-LINK GRAPH PATCHES  (final config "final_combo", VAL-24 verified)
# ----------------------------------------------------------------------
# Two conservative patches on the linker GRAPH (nodes + edges) only.
# Detection is UNTOUCHED: 2-model heatmap-mean ensemble @ UNET_THRESH=0.15,
# each model with its own preprocessing, physical NMS 4.0 um.
#
# VAL-24, official tracking_cellmot metric (24 held-out movies):
#   base 0.8387  ->  +safe divisions 0.8516  ->  +gap(snap-only) 0.8518
#   division TP/FP/FN = 6/31/8   (base M001: 0/0/14 — predicts no divisions)
#
# PATCH 1 — SAFE DIVISIONS (after short-filter + linefit, before emit).
#   The base linker is strictly 1:1, so a mitosis second daughter is never
#   linked. For a parent p (frame t) with exactly ONE child c1 at t+1, propose
#   a second daughter q among ORPHAN (in-degree 0) nodes at t+1:
#     d(p,q) <= 12 um,  d(c1,q) <= 15 um,  d(p,c1) <= 10 um (sanity),
#     q is the nearest orphan to c1 as well (sisters are mutual nearest
#     orphans), and the sisters DIVERGE after mitosis:
#     d(succ(c1),succ(q)) - d(c1,q) >= 2.25 um (both children continue at t+2).
#   Accepted 1:1 greedy by score = d(p,q) + 0.15*d(c1,q); safety caps
#   (0.76% of frame nodes, 0.375% of movie nodes) — never hit in practice.
#
# PATCH 2 — 1-FRAME GAP CLOSING, snap-only (before short-filter).
#   Track END at t vs track START at t+2, Hungarian under a 9 um gate; ends
#   must have a predecessor and starts a successor (no fragment tips). The
#   t+1 midpoint is filled ONLY by snapping to an unused low-score candidate
#   (0.10 <= score < UNET_THRESH) within 3.2 um — never a synthetic point.
#   Cap 0.3% of movie nodes.
# ======================================================================
DIV_PARENT_UM=12.0; DIV_SISTER_UM=15.0; DIV_CHILD_UM=10.0; DIV_DIVERGE_UM=2.25
DIV_W_SISTER=0.15; DIV_FRAME_CAP=0.0076; DIV_GLOBAL_CAP=0.00375
GAP1_GATE_UM=9.0; GAP1_SNAP_UM=3.2; GAP1_MIN_CAND_SCORE=0.10; GAP1_CAP_FRAC=0.003

def _add_safe_divisions(nodes, edges):
    succ=defaultdict(list); pred=defaultdict(list)
    for a,b in edges:
        succ[a].append(b); pred[b].append(a)
    by_t=defaultdict(list)
    for nid,n in nodes.items(): by_t[n["t"]].append(nid)
    orphan_tree={}; orphan_ids={}
    for t,ids in by_t.items():
        orph=[g for g in ids if not pred.get(g)]
        orphan_ids[t]=orph
        if orph: orphan_tree[t]=cKDTree(np.asarray([nodes[g]["xyz"] for g in orph],float)*SCALE)
    proposals=[]
    for p,n in nodes.items():
        ch=succ.get(p,[])
        if len(ch)!=1: continue
        c1=ch[0]; t=n["t"]
        if nodes[c1]["t"]!=t+1: continue
        if len(pred.get(p,[]))!=1: continue                       # parent must have a predecessor
        if _dist_um(n["xyz"],nodes[c1]["xyz"])>DIV_CHILD_UM: continue
        tree=orphan_tree.get(t+1)
        if tree is None: continue
        for idx in tree.query_ball_point(n["xyz"]*SCALE, r=DIV_PARENT_UM):
            q=orphan_ids[t+1][idx]
            if q==c1: continue
            d_pq=_dist_um(n["xyz"],nodes[q]["xyz"]); d_s=_dist_um(nodes[c1]["xyz"],nodes[q]["xyz"])
            if d_s>DIV_SISTER_UM: continue
            d_q,q_idx=tree.query(nodes[c1]["xyz"]*SCALE)          # sisters: mutual nearest orphans
            if orphan_ids[t+1][q_idx]!=q or d_q>DIV_SISTER_UM: continue
            c1n,qn=succ.get(c1,[]),succ.get(q,[])                 # both children continue at t+2
            if len(c1n)!=1 or len(qn)!=1: continue
            if _dist_um(nodes[c1n[0]]["xyz"],nodes[qn[0]]["xyz"])-d_s<DIV_DIVERGE_UM: continue
            proposals.append((d_pq+DIV_W_SISTER*d_s,t,p,q))
    proposals.sort()
    used_p=set(); used_q=set(); per_frame=defaultdict(int)
    n_frame={t:len(ids) for t,ids in by_t.items()}
    max_global=max(1,int(DIV_GLOBAL_CAP*len(nodes))); added=0
    for score,t,p,q in proposals:
        if added>=max_global: break
        if p in used_p or q in used_q: continue
        if per_frame[t]>=max(1,int(DIV_FRAME_CAP*n_frame[t])): continue
        edges.append((p,q)); succ[p].append(q); pred[q].append(p)
        used_p.add(p); used_q.add(q); per_frame[t]+=1; added+=1
    return added

def _gap_close_1f_snap(nodes, edges, cand, cand_sc, cand_trees, next_id):
    T=len(cand); succ={}; pred={}
    for a,b in edges:
        succ[a]=b; pred[b]=a
    by_t=defaultdict(list)
    for nid,n in nodes.items(): by_t[n["t"]].append(nid)
    cand_used=[np.zeros(len(c),bool) for c in cand]
    max_gaps=max(1,int(GAP1_CAP_FRAC*len(nodes))); n_added=0; BIG=1e9
    for t in range(T-2):
        if n_added>=max_gaps: break
        ends=[g for g in by_t.get(t,[]) if g not in succ and g in pred]      # track len >= 2 back
        starts=[g for g in by_t.get(t+2,[]) if g not in pred and g in succ]  # track len >= 2 forward
        if not ends or not starts: continue
        P=np.asarray([nodes[g]["xyz"] for g in ends],float)*SCALE
        C=np.asarray([nodes[g]["xyz"] for g in starts],float)*SCALE
        D=np.sqrt(((P[:,None]-C[None])**2).sum(2)); cost=np.where(D>GAP1_GATE_UM,BIG,D)
        ri,ci=linear_sum_assignment(cost)
        props=sorted((float(D[r,c]),ends[r],starts[c]) for r,c in zip(ri,ci) if cost[r,c]<BIG)
        for d,ge,gs in props:
            if n_added>=max_gaps: break
            if ge in succ or gs in pred: continue
            mid=0.5*(nodes[ge]["xyz"]+nodes[gs]["xyz"]); use=None
            tree=cand_trees[t+1]
            if tree is not None:
                dist,idx=tree.query(mid*SCALE)
                if (dist<=GAP1_SNAP_UM and not cand_used[t+1][idx]
                        and cand_sc[t+1][idx]>=GAP1_MIN_CAND_SCORE):
                    use=np.asarray(cand[t+1][idx],float); cand_used[t+1][idx]=True
            if use is None: continue                                       # snap-only: never synthesise
            ng=next_id; next_id+=1
            nodes[ng]={"t":t+1,"xyz":use}
            edges.append((ge,ng)); edges.append((ng,gs))
            succ[ge]=ng; pred[ng]=ge; succ[ng]=gs; pred[gs]=ng
            n_added+=1
    return next_id, n_added


# appearance cost in the linker: cost += 2.0 * |logit(s_prev) - logit(s_curr)| (um)
LINK_SIM_W = 2.0
def _logit(s):
    s = np.clip(s, 1e-4, 1 - 1e-4); return np.log(s / (1 - s))

def _link_sim(prev_xyz, curr_xyz, prev_vel, prev_sc, curr_sc):
    if len(prev_xyz)==0 or len(curr_xyz)==0: return []
    P=prev_xyz*SCALE[None,:]; C=curr_xyz*SCALE[None,:]
    pred=P+(0.5*prev_vel if prev_vel is not None else 0.0); N,M=len(P),len(C); BIG=1e9
    sim = LINK_SIM_W*np.abs(_logit(prev_sc)[:,None]-_logit(curr_sc)[None])
    def _hun(pi,ci,gate):
        if len(pi)==0 or len(ci)==0: return []
        Draw=np.sqrt(((P[pi][:,None]-C[ci][None])**2).sum(2)); D=np.sqrt(((pred[pi][:,None]-C[ci][None])**2).sum(2))
        cost=np.where(Draw>gate,BIG,D)+np.where(Draw>gate,0.0,sim[np.ix_(pi,ci)])
        ri,rc=linear_sum_assignment(cost)
        return [(int(pi[r]),int(ci[c])) for r,c in zip(ri,rc) if cost[r,c]<BIG]
    links=_hun(np.arange(N),np.arange(M),min(TIGHT_UM,MAX_LINK_UM))
    up={p for p,_ in links}; uc={c for _,c in links}
    fp=np.array([i for i in range(N) if i not in up],int); fc=np.array([j for j in range(M) if j not in uc],int)
    return links+_hun(fp,fc,MAX_LINK_UM)

repair_track_base=repair_track   # base pipeline kept for reference (A/B)
def repair_track(dets, ds):
    # same graph pipeline as the base kernel, with the two patch hooks marked
    nodes={}; frame_ids=[]; cand=[]; cand_sc=[]; cand_trees=[]; nid=1; NSC={}
    for t,(coords,scores) in enumerate(dets):
        coords=np.asarray(coords,float).reshape(-1,3); scores=np.asarray(scores,float).reshape(-1)
        seeds=coords[scores >= UNET_THRESH]
        cm=(scores >= CAND_THR) & (scores < UNET_THRESH)
        cand.append(coords[cm]); cand_sc.append(scores[cm])                # cand scores needed by patch 2
        cand_trees.append(cKDTree(cand[-1]*SCALE) if len(cand[-1]) else None)
        seed_sc=scores[scores >= UNET_THRESH]
        ids=[]
        for xyz,s_ in zip(seeds,seed_sc):
            nodes[nid]={"t":t, "xyz":np.asarray(xyz,float)}; NSC[nid]=float(s_); ids.append(nid); nid += 1
        frame_ids.append(ids)
    edges=[]; succ={}; pred={}; vel={}
    for t in range(len(dets)-1):
        P=np.asarray([nodes[g]["xyz"] for g in frame_ids[t]], float).reshape(-1,3)
        C=np.asarray([nodes[g]["xyz"] for g in frame_ids[t+1]], float).reshape(-1,3)
        if len(P)==0 or len(C)==0: continue
        prev_vel=np.array([vel.get(g, np.zeros(3)) for g in frame_ids[t]])
        psc=np.array([NSC[g] for g in frame_ids[t]]); csc=np.array([NSC[g] for g in frame_ids[t+1]])
        for pi,ci in _link_sim(P, C, prev_vel if len(prev_vel) else None, psc, csc):
            gp,gc=frame_ids[t][pi],frame_ids[t+1][ci]
            edges.append((gp,gc)); succ[gp]=gc; pred[gc]=gp; vel[gc]=(C[ci]-P[pi])*SCALE
    nid,n_gaps=_gap_close_1f_snap(nodes, edges, cand, cand_sc, cand_trees, nid)   # PATCH 2 (pre short-filter)
    nodes,edges=_short_filter(nodes, edges)
    _linefit(nodes, edges)
    n_div=_add_safe_divisions(nodes, edges)                                       # PATCH 1 (post linefit)
    print(f"    patches[{ds}]: gaps={n_gaps} divs={n_div}")
    return _emit(ds, nodes, edges)



def find_weight(name: str, roots: list[Path]) -> Path:
    for r in roots:
        for cand in [r / name, r / "biohub-unet3d-weights" / name, r / "biohub-unet3d-weights-v2models" / name]:
            if cand.exists():
                return cand
        hits = list(r.rglob(name))
        if hits:
            return hits[0]
    raise FileNotFoundError(name)


def load_models(device: str, weight_root: Path):
    global MODELS, PREPROCS, DEVICE
    DEVICE = device
    names = [
        ("unet3d_bright.pt", ""),
        ("unet3d_traintophat.pt", "tophat"),
        ("unet3d_v2_tophat_b32.pt", "tophat"),
    ]
    roots = [weight_root, weight_root / "biohub-unet3d-weights", weight_root / "biohub-unet3d-weights-v2models", DEFAULT_WROOT]
    MODELS = []
    PREPROCS = []
    for name, pp in names:
        path = find_weight(name, roots)
        ck = torch.load(path, map_location=device, weights_only=False)
        m = UNet3D(base=ck.get("base", 24)).to(device)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        MODELS.append(m)
        PREPROCS.append(pp)
        print("loaded", path.name, "preproc", pp or "none", "val_recall", ck.get("val_recall"), flush=True)
    _build_branches()


def nodes_edges_to_graph(nodes_df, edges_df) -> td.graph.InMemoryGraph:
    g = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        g.add_node_attr_key(key, pl.Float64, -999999.0)
    if nodes_df is None or len(nodes_df) == 0:
        return g
    # nodes_df columns from _emit
    id_map = {}
    rows = []
    for r in nodes_df.itertuples(index=False):
        # COLS dataset,row_type,node_id,t,z,y,x,source_id,target_id
        nid = int(r.node_id)
        rows.append({"t": int(r.t), "z": float(r.z), "y": float(r.y), "x": float(r.x)})
        id_map[nid] = len(rows) - 1
    graph_ids = g.bulk_add_nodes(rows)
    remap = {old: int(graph_ids[i]) for old, i in id_map.items()}
    if edges_df is not None and len(edges_df):
        erows = []
        for r in edges_df.itertuples(index=False):
            s, t = int(r.source_id), int(r.target_id)
            if s in remap and t in remap:
                erows.append({"source_id": remap[s], "target_id": remap[t]})
        if erows:
            g.bulk_add_edges(erows)
    return g


def process_movie(zp: Path, out_geff: Path, device: str):
    import pandas as pd
    ds = zp.name.replace(".zarr", "")
    T = avail_T(zp)
    if T <= 0:
        raise RuntimeError(f"no frames in {zp}")
    nodes, edges = track_movie(zp, ds, T)
    # track_movie returns dataframes
    graph = nodes_edges_to_graph(nodes, edges)
    out_geff.parent.mkdir(parents=True, exist_ok=True)
    save_graph(graph, out_geff, overwrite=True)
    return {"dataset": ds, "T": T, "n_nodes": graph.num_nodes(), "n_edges": graph.num_edges()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--weight-root", type=Path, default=DEFAULT_WROOT)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--slice", default=None, help="python slice on sorted stems, e.g. :10")
    ap.add_argument("--stems", default=None, help="comma-separated stems")
    ap.add_argument("--unet-thresh", type=float, default=None)
    ap.add_argument("--cand-thr", type=float, default=None)
    ap.add_argument("--nms-um", type=float, default=None)
    ap.add_argument("--max-link-um", type=float, default=None)
    ap.add_argument("--tight-um", type=float, default=None)
    args = ap.parse_args()
    global UNET_THRESH, CAND_THR, DETECT_THRESH, NMS_UM, MAX_LINK_UM, TIGHT_UM
    if args.unet_thresh is not None:
        UNET_THRESH = float(args.unet_thresh)
    if args.cand_thr is not None:
        CAND_THR = float(args.cand_thr)
    if args.nms_um is not None:
        NMS_UM = float(args.nms_um)
    if args.max_link_um is not None:
        MAX_LINK_UM = float(args.max_link_um)
    if args.tight_um is not None:
        TIGHT_UM = float(args.tight_um)
    DETECT_THRESH = min(UNET_THRESH, CAND_THR) if REPAIR else UNET_THRESH
    print(
        f"knobs unet_thresh={UNET_THRESH} cand_thr={CAND_THR} nms_um={NMS_UM} "
        f"max_link_um={MAX_LINK_UM} tight_um={TIGHT_UM}",
        flush=True,
    )
    device = pick_device(args.device)
    print("device", device, flush=True)
    load_models(device, args.weight_root)

    zarrs = sorted(args.data_dir.glob("*.zarr"))
    stems = [z.name.replace(".zarr", "") for z in zarrs]
    if args.stems:
        want = set(args.stems.split(","))
        stems = [s for s in stems if s in want]
    if args.slice:
        sl = slice(*[int(x) if x else None for x in args.slice.split(":")])
        stems = stems[sl]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, stem in enumerate(stems):
        zp = args.data_dir / f"{stem}.zarr"
        out = args.out_dir / f"{stem}.geff"
        t0 = time.time()
        print(f"[{i+1}/{len(stems)}] {stem}", flush=True)
        info = process_movie(zp, out, device)
        info["seconds"] = time.time() - t0
        rows.append(info)
        print(" ", info, flush=True)
        gc.collect()
    (args.out_dir / "manifest.json").write_text(json.dumps(rows, indent=2))
    print("Done", len(rows), "->", args.out_dir)


if __name__ == "__main__":
    main()
