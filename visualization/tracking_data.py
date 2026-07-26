"""Loading and caching for the tracking viewer service.

Wraps a submission.csv plus the matching .zarr image volumes so the FastAPI
app in app.py can serve frame images and node/edge overlays cheaply enough
for interactive frame-by-frame stepping.
"""
from __future__ import annotations

import io
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from PIL import Image

SUBMISSION_CSV = Path(os.environ.get("BIOHUB_SUBMISSION_CSV", "submission.csv"))
RESULTS_DIR = Path(os.environ.get("BIOHUB_RESULTS_DIR", "results"))


class DatasetNotFound(KeyError):
    pass


class FrameOutOfRange(IndexError):
    pass


class Dataset:
    """Nodes/edges/track-id lookup for one dataset, built once and reused."""

    def __init__(self, name: str, df: pd.DataFrame):
        nodes = df[df.row_type == "node"].copy()
        nodes[["node_id", "t", "z"]] = nodes[["node_id", "t", "z"]].astype(int)
        nodes = nodes.set_index("node_id", drop=False).sort_index()

        edges = df[df.row_type == "edge"][["source_id", "target_id"]].astype(int)
        edges = edges[edges.source_id.isin(nodes.index) & edges.target_id.isin(nodes.index)]

        self.name = name
        self.nodes = nodes
        self.t_max = int(nodes.t.max()) if len(nodes) else 0
        self.track_id = _assign_track_ids(nodes.index, edges)

        merged = edges.merge(
            nodes[["t", "x", "y", "z"]], left_on="source_id", right_index=True
        ).merge(
            nodes[["t", "x", "y", "z"]], left_on="target_id", right_index=True, suffixes=("_s", "_t")
        )
        merged["track"] = self.track_id.reindex(merged.source_id).to_numpy()
        self.edges = merged

    def nodes_at(self, t: int) -> pd.DataFrame:
        return self.nodes[self.nodes.t == t]

    def edges_trailing(self, t: int, tail: int) -> pd.DataFrame:
        lo = max(t - tail, 0)
        e = self.edges
        return e[(e.t_s >= lo) & (e.t_s <= t) & (e.t_t <= t)]


def _assign_track_ids(node_ids: pd.Index, edges: pd.DataFrame) -> pd.Series:
    """Union-find over the lineage graph so every node gets a stable small
    integer id shared with its ancestors/descendants (used for consistent
    per-lineage coloring across frames, so identity swaps stand out)."""
    parent = {nid: nid for nid in node_ids}

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for s, t in edges.itertuples(index=False):
        ra, rb = find(s), find(t)
        if ra != rb:
            parent[rb] = ra

    order: dict[int, int] = {}
    track_ids = np.empty(len(node_ids), dtype=np.int64)
    for i, nid in enumerate(node_ids):
        root = find(nid)
        track_ids[i] = order.setdefault(root, len(order))
    return pd.Series(track_ids, index=node_ids, name="track_id")


@lru_cache(maxsize=1)
def _submission() -> pd.DataFrame:
    if not SUBMISSION_CSV.exists():
        raise FileNotFoundError(
            f"Submission CSV not found at {SUBMISSION_CSV} "
            "(set BIOHUB_SUBMISSION_CSV to override)"
        )
    df = pd.read_csv(SUBMISSION_CSV)
    df["dataset"] = df["dataset"].astype(str)
    return df


def dataset_names() -> list[str]:
    return sorted(_submission().dataset.unique())


@lru_cache(maxsize=8)
def get_dataset(name: str) -> Dataset:
    df = _submission()
    df = df[df.dataset == name]
    if df.empty:
        raise DatasetNotFound(name)
    return Dataset(name, df)


@lru_cache(maxsize=8)
def _zarr_group(name: str):
    path = RESULTS_DIR / f"{name}.zarr"
    if not path.exists():
        raise DatasetNotFound(f"no .zarr volume for {name!r} at {path}")
    return zarr.open_group(str(path), mode="r")


def volume_shape(name: str) -> tuple[int, ...] | None:
    try:
        return tuple(_zarr_group(name)["0"].shape)
    except DatasetNotFound:
        return None


def voxel_scale(name: str) -> tuple[float, float, float] | None:
    """(z, y, x) physical size of one voxel, from the OME-Zarr multiscale
    transform, so the frontend can render an anisotropic volume (z spacing
    is usually much coarser than x/y) without looking squashed in 3D."""
    try:
        group = _zarr_group(name)
    except DatasetNotFound:
        return None
    try:
        scale = group.attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"]
        return tuple(float(s) for s in scale[-3:])
    except (KeyError, IndexError, TypeError):
        return None


# Max-intensity projections along each axis, keyed by the axes that remain
# after projecting: "xy" (project out z, the default top-down view), "xz"
# (project out y) and "yz" (project out x). Combined, these give three
# orthogonal viewing angles on the same frame.
_PROJECTION_AXES = {"xy": 0, "xz": 1, "yz": 2}


@lru_cache(maxsize=512)
def frame_png(name: str, t: int, view: str = "xy") -> bytes:
    if view not in _PROJECTION_AXES:
        raise ValueError(f"unknown view {view!r}, expected one of {sorted(_PROJECTION_AXES)}")

    group = _zarr_group(name)
    arr = group["0"]
    if not (0 <= t < arr.shape[0]):
        raise FrameOutOfRange(f"t={t} out of range for {name} (0..{arr.shape[0] - 1})")

    mip = np.asarray(arr[t]).max(axis=_PROJECTION_AXES[view])

    quantiles = group.attrs.get("image_statistics", {}).get("quantiles", {})
    lo = float(quantiles.get("0.01", mip.min()))
    hi = float(quantiles.get("0.99", mip.max()))
    scaled = np.clip((mip.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    img8 = (scaled * 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(img8, mode="L").save(buf, format="PNG")
    return buf.getvalue()