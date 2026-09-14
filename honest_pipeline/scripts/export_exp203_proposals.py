#!/usr/bin/env python3
"""Dump 0_917 GEFF nodes into motion-corrector proposal NPZ format.

Single-detector stack: member_det_prob is duplicated so A/B disagreement is 0.
Coordinates are raw voxels [t,z,y,x]. downsample=(1,1,1) because Exp203 refines
on the full-resolution volume.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _pack_src() -> Path:
    for c in (
        ROOT / "public_models" / "support_pack" / "repo" / "src",
        ROOT / "public_models" / "Biohub" / "repo" / "src",
        ROOT / "public_models" / "Biohub Tracking Support Pack" / "repo" / "src",
        ROOT / "helpers" / "01_p1_p2_base" / "shared_repo" / "src",
    ):
        if (c / "biohub_tracking" / "io.py").exists():
            return c
    raise FileNotFoundError("biohub_tracking.io not found")


PACK_SRC = _pack_src()
SCALE = np.array([1.625, 0.40625, 0.40625], np.float32)


def _open_pred(pred: Path, data_dir: Path, stem: str):
    sys.path.insert(0, str(PACK_SRC))
    from biohub_tracking.io import open_dataset

    td = Path(tempfile.mkdtemp(prefix="exp203prop_"))
    os.symlink((data_dir / f"{stem}.zarr").resolve(), td / f"{stem}.zarr")
    os.symlink(pred.resolve(), td / f"{stem}.geff")
    ds = open_dataset(td / stem, load_image=False, require_tracks=True, normalize=False)
    return ds, td


def nodes_to_npz(ds, zp: Path, out: Path) -> dict:
    df = ds.tracks.node_attrs(attr_keys=["t", "z", "y", "x"])
    if hasattr(df, "select"):
        arr = np.asarray(df.select(["t", "z", "y", "x"]).to_numpy())
    else:
        arr = np.stack(
            [np.asarray(df[k]) for k in ("t", "z", "y", "x")],
            axis=1,
        )
    t = arr[:, 0].astype(np.int32)
    z, y, x = arr[:, 1].astype(np.float32), arr[:, 2].astype(np.float32), arr[:, 3].astype(np.float32)
    order = np.lexsort((x, y, z, t))
    t, z, y, x = t[order], z[order], y[order], x[order]
    coords = np.stack([t, np.rint(z), np.rint(y), np.rint(x)], axis=1).astype(np.int16)
    T = int(t.max()) + 1 if len(t) else 0
    if zp.exists():
        meta = json.loads((zp / "0" / "zarr.json").read_text()) if (zp / "0" / "zarr.json").exists() else {}
        shape = tuple(meta.get("shape") or [T, 0, 0, 0])
        if len(shape) == 4:
            image_shape = np.asarray(shape, np.int32)
        else:
            image_shape = np.asarray([T, int(z.max()) + 1 if len(z) else 1, int(y.max()) + 1 if len(y) else 1, int(x.max()) + 1 if len(x) else 1], np.int32)
        T = int(image_shape[0])
    else:
        image_shape = np.asarray(
            [max(T, 1), int(z.max()) + 1 if len(z) else 1, int(y.max()) + 1 if len(y) else 1, int(x.max()) + 1 if len(x) else 1],
            np.int32,
        )
        T = int(image_shape[0])
    counts = np.bincount(t, minlength=T).astype(np.int32) if len(t) else np.zeros(T, np.int32)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    n = len(coords)
    fused = np.ones(n, np.float32)
    member = np.stack([fused, fused], axis=1)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        coords=coords,
        fused_det_prob=fused,
        member_det_prob=member,
        frame_counts=counts,
        frame_offsets=offsets,
        frozen_sources=np.zeros(0, np.int16),
        image_shape=image_shape,
        voxel_scale_um=SCALE,
        downsample=np.asarray([1, 1, 1], np.int16),
        det_threshold=np.float32(0.15),
        pool_kernel_um=np.float32(0.0),
        detector=np.asarray("exp203_classical"),
    )
    return {"stem": zp.name.replace(".zarr", ""), "n_nodes": n, "T": T}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geff-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    geffs = sorted(args.geff_dir.glob("*.geff"))
    rows = []
    for i, g in enumerate(geffs):
        stem = g.name.replace(".geff", "")
        zp = args.data_dir / f"{stem}.zarr"
        dest = args.out_dir / f"{stem}.npz"
        print(f"[{i+1}/{len(geffs)}] {stem}", flush=True)
        try:
            ds, td = _open_pred(g, args.data_dir, stem)
            try:
                info = nodes_to_npz(ds, zp, dest)
            finally:
                shutil.rmtree(td, ignore_errors=True)
            rows.append(info)
        except Exception as e:
            rows.append({"stem": stem, "error": repr(e)})
            print(" FAIL", e, flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manifest.json").write_text(json.dumps(rows, indent=2))
    n_ok = sum(1 for r in rows if "error" not in r)
    print(f"Wrote {n_ok}/{len(rows)} -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
