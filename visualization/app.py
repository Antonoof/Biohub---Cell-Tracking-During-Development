#!/usr/bin/env python3
"""Interactive FastAPI viewer for tracking predictions on submission.csv.

Serves a Z-max-intensity-projection background per frame (decoded from the
matching .zarr volume) plus node/track overlays, so predictions on the
(label-free) test set can be stepped through frame by frame in the browser.

Run with:
    uvicorn visualization.app:app --reload --port 8000

Configure input locations with:
    BIOHUB_SUBMISSION_CSV   path to submission.csv (default: ./submission.csv)
    BIOHUB_RESULTS_DIR      directory containing {dataset}.zarr volumes (default: ./results)
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response

from . import tracking_data as data

app = FastAPI(title="Biohub Tracking Viewer")

_TEMPLATE_PATH = Path(__file__).parent / "templates" / "viewer.html"


@app.get("/", response_class=HTMLResponse)
def index():
    return _TEMPLATE_PATH.read_text(encoding="utf-8")


@app.get("/api/datasets")
def list_datasets():
    out = []
    for name in data.dataset_names():
        ds = data.get_dataset(name)
        out.append(
            {
                "name": name,
                "t_max": ds.t_max,
                "n_nodes": int(len(ds.nodes)),
                "n_edges": int(len(ds.edges)),
                "n_tracks": int(ds.track_id.max()) + 1 if len(ds.track_id) else 0,
                "shape": data.volume_shape(name),
            }
        )
    return out


@app.get("/api/frame/{dataset}/{t}.png")
def frame_png(dataset: str, t: int):
    try:
        png = data.frame_png(dataset, t)
    except data.DatasetNotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    except data.FrameOutOfRange as exc:
        raise HTTPException(404, str(exc)) from exc
    return Response(content=png, media_type="image/png")


@app.get("/api/nodes/{dataset}/{t}")
def nodes_at(dataset: str, t: int, tail: int = Query(default=10, ge=0, le=99)):
    try:
        ds = data.get_dataset(dataset)
    except data.DatasetNotFound as exc:
        raise HTTPException(404, str(exc)) from exc

    current = ds.nodes_at(t)
    nodes_out = [
        {"id": int(nid), "x": float(r.x), "y": float(r.y), "z": int(r.z), "track": int(ds.track_id[nid])}
        for nid, r in current.iterrows()
    ]

    trail = ds.edges_trailing(t, tail)
    edges_out = [
        {
            "x1": float(r.x_s), "y1": float(r.y_s), "t1": int(r.t_s),
            "x2": float(r.x_t), "y2": float(r.y_t), "t2": int(r.t_t),
            "track": int(r.track),
        }
        for r in trail.itertuples(index=False)
    ]

    return {
        "t": t,
        "tail_from": max(t - tail, 0),
        "nodes": nodes_out,
        "edges": edges_out,
    }