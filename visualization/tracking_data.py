"""Loading and caching for the tracking viewer service.

Wraps one or more submission.csv files plus the matching .zarr image volumes
so the FastAPI app in app.py can serve frame images and node/edge overlays
cheaply enough for interactive frame-by-frame stepping.

Two submissions can be open at once (the compare view), so everything that
depends on a submission is keyed by a *source*: `None` is the live
submission.csv, any other string names a snapshot in save_files/.
"""
from __future__ import annotations

import io
import os
import re
import shutil
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from PIL import Image

from . import metrics

SUBMISSION_CSV = Path(os.environ.get("BIOHUB_SUBMISSION_CSV", "submission.csv"))
# The image volumes. BIOHUB_RESULTS_DIR is the old name of this setting, kept
# working so existing shells/scripts do not silently point at nothing.
ZARR_DIR = Path(
    os.environ.get("BIOHUB_ZARR_DIR")
    or os.environ.get("BIOHUB_RESULTS_DIR")
    or "files_zarr"
)
LABELS_DIR = Path(os.environ.get("BIOHUB_LABELS_DIR", "labels_geff"))
# Named snapshots of a submission, written by the Save button.
SAVE_DIR = Path(os.environ.get("BIOHUB_SAVE_DIR", "save_files"))

CURRENT_LABEL = SUBMISSION_CSV.name

# Snapshot names become file names, so keep them boring: no separators, no
# leading dot, nothing that could escape SAVE_DIR.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}$")


class DatasetNotFound(KeyError):
    pass


class FrameOutOfRange(IndexError):
    pass


class GroundTruthNotFound(KeyError):
    pass


class SourceNotFound(KeyError):
    pass


class InvalidName(ValueError):
    pass


class NameTaken(FileExistsError):
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


# --------------------------------------------------------------------------- #
# sources: the live submission.csv, plus any saved snapshot
# --------------------------------------------------------------------------- #

def source_path(src: str | None) -> Path:
    """CSV backing a source. `None`/"" is the live submission."""
    if not src:
        return SUBMISSION_CSV
    if not _SAFE_NAME.match(src):
        raise InvalidName(f"invalid save name {src!r}")
    path = SAVE_DIR / f"{src}.csv"
    if not path.exists():
        raise SourceNotFound(f"no saved submission named {src!r}")
    return path


def source_label(src: str | None) -> str:
    return CURRENT_LABEL if not src else f"{src}.csv"


def list_saved() -> list[dict]:
    """Saved snapshots, newest first."""
    if not SAVE_DIR.exists():
        return []
    out = []
    for path in SAVE_DIR.glob("*.csv"):
        stat = path.stat()
        out.append({"name": path.stem, "size": stat.st_size, "modified": stat.st_mtime})
    return sorted(out, key=lambda s: -s["modified"])


def save_snapshot(name: str, src: str | None = None, overwrite: bool = False) -> dict:
    """Copy the CSV behind `src` into SAVE_DIR under `name`."""
    name = name.strip()
    if name.lower().endswith(".csv"):
        name = name[:-4].strip()
    if not _SAFE_NAME.match(name):
        raise InvalidName(
            "name must be 1-64 chars of letters, digits, space, dot, dash or "
            "underscore, and start with a letter or digit"
        )

    source = source_path(src)
    if not source.exists():
        raise FileNotFoundError(f"nothing to save: {source} does not exist")

    target = SAVE_DIR / f"{name}.csv"
    existed = target.exists()
    if existed and not overwrite:
        raise NameTaken(name)

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    if existed:
        # The name now points at different rows, so nothing cached under it is
        # still valid. The caches are small and cheap to refill.
        _clear_submission_caches()
    stat = target.stat()
    return {"name": name, "size": stat.st_size, "modified": stat.st_mtime}


def _clear_submission_caches() -> None:
    for cache in (_submission, get_dataset, _pred_graph, evaluate):
        cache.cache_clear()


@lru_cache(maxsize=8)
def _submission(src: str | None = None) -> pd.DataFrame:
    path = source_path(src)
    if not path.exists():
        raise FileNotFoundError(
            f"Submission CSV not found at {path} "
            "(set BIOHUB_SUBMISSION_CSV to override)"
        )
    df = pd.read_csv(path)
    df["dataset"] = df["dataset"].astype(str)
    return df


def dataset_names(src: str | None = None) -> list[str]:
    return sorted(_submission(src).dataset.unique())


@lru_cache(maxsize=16)
def get_dataset(name: str, src: str | None = None) -> Dataset:
    df = _submission(src)
    df = df[df.dataset == name]
    if df.empty:
        raise DatasetNotFound(name)
    return Dataset(name, df)


@lru_cache(maxsize=16)
def _pred_graph(name: str, src: str | None = None) -> metrics.TrackGraph:
    ds = get_dataset(name, src)
    nodes = ds.nodes
    edges = ds.edges[["source_id", "target_id"]].to_numpy()
    return metrics.build_graph(
        nodes.index.to_numpy(), nodes.t.to_numpy(),
        nodes.z.to_numpy(), nodes.y.to_numpy(), nodes.x.to_numpy(),
        edges,
    )


@lru_cache(maxsize=8)
def _gt_graph(name: str) -> tuple[metrics.TrackGraph, int | None]:
    """Ground-truth lineages for one dataset, read straight from its .geff.

    Returns the graph plus the organisers' coarse estimate of the *total*
    number of true nodes (the annotations themselves are sparse), which the
    edge metric needs for its over-prediction penalty.
    """
    path = LABELS_DIR / f"{name}.geff"
    if not path.exists():
        raise GroundTruthNotFound(f"no ground truth for {name!r} at {path}")

    group = zarr.open_group(str(path), mode="r")
    nodes, edges = group["nodes"], group["edges"]
    props = nodes["props"]
    graph = metrics.build_graph(
        np.asarray(nodes["ids"][:]).astype(np.int64),
        np.asarray(props["t"]["values"][:]),
        np.asarray(props["z"]["values"][:]),
        np.asarray(props["y"]["values"][:]),
        np.asarray(props["x"]["values"][:]),
        np.asarray(edges["ids"][:]).astype(np.int64),
    )
    extra = group.attrs.get("geff", {}).get("extra") or {}
    estimated = extra.get("estimated_number_of_nodes")
    return graph, int(estimated) if estimated else None


@lru_cache(maxsize=16)
def evaluate(name: str, src: str | None = None) -> dict:
    """Score the prediction for one dataset against its ground truth."""
    gt, estimated = _gt_graph(name)
    return metrics.evaluate(_pred_graph(name, src), gt, estimated)


def evaluate_all(src: str | None = None) -> dict:
    """Per-sample scores plus the leaderboard-style aggregate over all of them."""
    per_sample = {}
    for name in dataset_names(src):
        try:
            per_sample[name] = evaluate(name, src)
        except (GroundTruthNotFound, DatasetNotFound):
            continue
    return {"samples": per_sample, "overall": metrics.aggregate(per_sample)}


@lru_cache(maxsize=8)
def _zarr_group(name: str):
    path = ZARR_DIR / f"{name}.zarr"
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