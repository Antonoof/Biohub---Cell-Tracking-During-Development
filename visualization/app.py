#!/usr/bin/env python3
"""Interactive FastAPI viewer for tracking predictions on submission.csv.

Serves a Z-max-intensity-projection background per frame (decoded from the
matching .zarr volume) plus node/track overlays, so predictions on the
(label-free) test set can be stepped through frame by frame in the browser.

Every submission-dependent route takes an optional `src`: empty means the live
submission.csv, any other value names a snapshot in save_files/. That is what
lets the compare view put two submissions side by side on one page.

Run with:
    uvicorn visualization.app:app --reload --port 8000

Configure input locations with:
    BIOHUB_SUBMISSION_CSV   path to submission.csv (default: ./submission.csv)
    BIOHUB_ZARR_DIR         directory containing {dataset}.zarr volumes (default: ./files_zarr)
    BIOHUB_SAVE_DIR         directory for saved submissions (default: ./save_files)
"""
from __future__ import annotations

from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response

from . import tracking_data as data

app = FastAPI(title="Biohub Tracking Viewer")

_TEMPLATE_PATH = Path(__file__).parent / "templates" / "viewer.html"

# `src=""` (the live submission) and `src=some_name` share every route, so this
# one dependency-free helper normalises the empty string to None.
Source = Query(default="", description="saved submission name; empty = submission.csv")


def _src(src: str) -> str | None:
    return src or None


def _resolve(src: str) -> str | None:
    """Normalise and validate a `src` before it reaches the loaders."""
    name = _src(src)
    if name is None:
        return None
    try:
        data.source_path(name)
    except (data.InvalidName, data.SourceNotFound) as exc:
        raise HTTPException(404, str(exc)) from exc
    return name


@app.get("/", response_class=HTMLResponse)
def index():
    # no-store: this template is actively edited during development, and a
    # stale cached copy in the browser silently hides layout/JS changes.
    return HTMLResponse(
        _TEMPLATE_PATH.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/datasets")
def list_datasets(src: str = Source):
    source = _resolve(src)
    out = []
    for name in data.dataset_names(source):
        ds = data.get_dataset(name, source)
        out.append(
            {
                "name": name,
                "t_max": ds.t_max,
                "n_nodes": int(len(ds.nodes)),
                "n_edges": int(len(ds.edges)),
                "n_tracks": int(ds.track_id.max()) + 1 if len(ds.track_id) else 0,
                "shape": data.volume_shape(name),
                "voxel_scale": data.voxel_scale(name),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# saved submissions
# --------------------------------------------------------------------------- #

@app.get("/api/sources")
def list_sources():
    """The live submission plus every saved snapshot, for the compare picker."""
    return {
        "current": {"src": "", "label": data.source_label(None)},
        "saved": [
            {"src": s["name"], "label": f"{s['name']}.csv", **s}
            for s in data.list_saved()
        ],
    }


@app.post("/api/save")
def save_submission(
    name: str = Body(..., embed=True),
    src: str = Body(default="", embed=True),
    overwrite: bool = Body(default=False, embed=True),
):
    """Copy the submission behind `src` into save_files/{name}.csv."""
    source = _resolve(src)
    try:
        saved = data.save_snapshot(name, source, overwrite=overwrite)
    except data.InvalidName as exc:
        raise HTTPException(400, str(exc)) from exc
    except data.NameTaken as exc:
        raise HTTPException(409, f"{name!r} already exists") from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    return saved


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

@app.get("/api/metrics")
def metrics_all(src: str = Source):
    """Every sample plus the leaderboard-style aggregate over all of them."""
    return data.evaluate_all(_resolve(src))


@app.get("/api/metrics/{dataset}")
def metrics_for(dataset: str, src: str = Source):
    try:
        return data.evaluate(dataset, _resolve(src))
    except (data.DatasetNotFound, data.GroundTruthNotFound) as exc:
        raise HTTPException(404, str(exc)) from exc


# --------------------------------------------------------------------------- #
# frames and overlays
# --------------------------------------------------------------------------- #

@app.get("/api/frame/{dataset}/{t}.png")
def frame_png(dataset: str, t: int, view: str = Query(default="xy", pattern="^(xy|xz|yz)$")):
    # The background image comes from the .zarr volume, which both compared
    # submissions share, so this route deliberately takes no `src`.
    try:
        png = data.frame_png(dataset, t, view)
    except data.DatasetNotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    except data.FrameOutOfRange as exc:
        raise HTTPException(404, str(exc)) from exc
    return Response(content=png, media_type="image/png")


@app.get("/api/nodes/{dataset}/{t}")
def nodes_at(
    dataset: str,
    t: int,
    tail: int = Query(default=10, ge=0, le=99),
    src: str = Source,
):
    try:
        ds = data.get_dataset(dataset, _resolve(src))
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
            "x1": float(r.x_s), "y1": float(r.y_s), "z1": int(r.z_s), "t1": int(r.t_s),
            "x2": float(r.x_t), "y2": float(r.y_t), "z2": int(r.z_t), "t2": int(r.t_t),
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
