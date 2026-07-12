#!/usr/bin/env python3
"""Render XY/Z projections and short 3-D trajectories from submission.csv."""
from __future__ import annotations
import argparse
from pathlib import Path
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Visualization dependencies are missing. Install requirements-validation.txt "
        f"(missing module: {exc.name})."
    ) from exc

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--submission",type=Path,required=True)
    p.add_argument("--dataset",required=True)
    p.add_argument("--frame",type=int,default=50)
    p.add_argument("--window",type=int,default=5)
    p.add_argument("--zarr",type=Path,default=None,help="Optional dataset .zarr for a Z-MIP background")
    p.add_argument("--output",type=Path,default=Path("tracking_visualization.png"))
    a=p.parse_args();df=pd.read_csv(a.submission);df=df[df.dataset.astype(str)==a.dataset]
    nodes=df[df.row_type=="node"].copy();edges=df[df.row_type=="edge"]
    nodes[["node_id","t"]]=nodes[["node_id","t"]].astype(int);lookup=nodes.set_index("node_id")
    frame_nodes=nodes[nodes.t==a.frame];lo,hi=a.frame-a.window,a.frame+a.window
    win=nodes[(nodes.t>=lo)&(nodes.t<=hi)]
    fig=plt.figure(figsize=(16,5));ax1=fig.add_subplot(131);ax2=fig.add_subplot(132,projection="3d");ax3=fig.add_subplot(133,projection="3d")
    if a.zarr is not None:
        import zarr
        arr=zarr.open_group(str(a.zarr),mode="r")["0"]
        ax1.imshow(np.asarray(arr[a.frame]).max(0),cmap="gray",origin="upper")
    ax1.scatter(frame_nodes.x,frame_nodes.y,s=7,c=frame_nodes.z,cmap="viridis",edgecolors="none")
    ax1.set(title=f"{a.dataset} t={a.frame}: XY projection",xlabel="x",ylabel="y");ax1.invert_yaxis()
    ax2.scatter(frame_nodes.x,frame_nodes.y,frame_nodes.z,s=7,c=frame_nodes.z,cmap="viridis")
    ax2.set(title="3-D detections",xlabel="x",ylabel="y",zlabel="z")
    cmap=plt.get_cmap("turbo");times=max(hi-lo,1)
    for _,e in edges.iterrows():
        try:s=lookup.loc[int(e.source_id)];t=lookup.loc[int(e.target_id)]
        except KeyError:continue
        if int(s.t)<lo or int(t.t)>hi:continue
        color=cmap((float(s.t)-lo)/times)
        ax3.plot([s.x,t.x],[s.y,t.y],[s.t,t.t],color=color,alpha=.45,linewidth=.65)
    ax3.scatter(win.x,win.y,win.t,s=3,c=win.t,cmap="turbo")
    ax3.set(title=f"Trajectories t={lo}..{hi}",xlabel="x",ylabel="y",zlabel="time")
    fig.tight_layout();a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=180,bbox_inches="tight");print("wrote",a.output)
if __name__=="__main__":main()
