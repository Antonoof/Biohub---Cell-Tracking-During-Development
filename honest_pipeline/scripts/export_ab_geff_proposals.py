#!/usr/bin/env python3
"""Fuse P1+P2 GEFF nodes into motion-corrector A/B proposal NPZ.

P1 (member 0) = Support Pack, P2 (member 1) = 0_917 classical.
Per frame: Hungarian match within --match-um, then union. Matched nodes keep
P1 coordinates. member_det_prob is 1 if that detector contributed the node.
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
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

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

    td = Path(tempfile.mkdtemp(prefix="abgeff_"))
    os.symlink((data_dir / f"{stem}.zarr").resolve(), td / f"{stem}.zarr")
    os.symlink(pred.resolve(), td / f"{stem}.geff")
    ds = open_dataset(td / stem, load_image=False, require_tracks=True, normalize=False)
    return ds, td


def _xyz(ds) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = ds.tracks.node_attrs(attr_keys=["t", "z", "y", "x"])
    if hasattr(df, "select"):
        arr = np.asarray(df.select(["t", "z", "y", "x"]).to_numpy())
    else:
        arr = np.stack([np.asarray(df[k]) for k in ("t", "z", "y", "x")], axis=1)
    t = arr[:, 0].astype(np.int32)
    z, y, x = arr[:, 1].astype(np.float32), arr[:, 2].astype(np.float32), arr[:, 3].astype(np.float32)
    return t, z, y, x


def _fuse_frame(a: np.ndarray, b: np.ndarray, match_um: float) -> tuple[np.ndarray, np.ndarray]:
    """a,b: (n,3) zyx voxels. Returns coords (n,3) and member (n,2)."""
    if len(a) == 0 and len(b) == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 2), np.float32)
    if len(a) == 0:
        return b.astype(np.float32), np.stack([np.zeros(len(b)), np.ones(len(b))], axis=1).astype(np.float32)
    if len(b) == 0:
        return a.astype(np.float32), np.stack([np.ones(len(a)), np.zeros(len(a))], axis=1).astype(np.float32)
    au = a.astype(np.float64) * SCALE
    bu = b.astype(np.float64) * SCALE
    # Hungarian on a padded cost; cap unmatched by match_um.
    n0, n1 = len(a), len(b)
    cost = np.full((n0, n1), 1e6, np.float64)
    tree = cKDTree(bu)
    dist, idx = tree.query(au, k=min(8, n1))
    if dist.ndim == 1:
        dist = dist[:, None]
        idx = idx[:, None]
    for i in range(n0):
        for d, j in zip(dist[i], idx[i]):
            if np.isfinite(d) and d <= match_um:
                cost[i, int(j)] = d
    ri, ci = linear_sum_assignment(cost)
    matched_a = set()
    matched_b = set()
    coords = []
    member = []
    for i, j in zip(ri, ci):
        if cost[i, j] >= 1e6:
            continue
        matched_a.add(int(i))
        matched_b.add(int(j))
        coords.append(a[i])
        member.append((1.0, 1.0))
    for i in range(n0):
        if i not in matched_a:
            coords.append(a[i])
            member.append((1.0, 0.0))
    for j in range(n1):
        if j not in matched_b:
            coords.append(b[j])
            member.append((0.0, 1.0))
    return np.asarray(coords, np.float32), np.asarray(member, np.float32)


def fuse_video(t1, z1, y1, x1, t2, z2, y2, x2, match_um: float, T: int):
    coords_t = []
    member_t = []
    for t in range(T):
        m1 = t1 == t
        m2 = t2 == t
        a = np.stack([z1[m1], y1[m1], x1[m1]], axis=1) if m1.any() else np.zeros((0, 3), np.float32)
        b = np.stack([z2[m2], y2[m2], x2[m2]], axis=1) if m2.any() else np.zeros((0, 3), np.float32)
        xyz, mem = _fuse_frame(a, b, match_um)
        n = len(xyz)
        tt = np.full(n, t, np.int32)
        if n:
            coords_t.append(np.column_stack([tt, xyz]))
            member_t.append(mem)
        else:
            coords_t.append(np.zeros((0, 4), np.float32))
            member_t.append(np.zeros((0, 2), np.float32))
    coords = np.concatenate(coords_t, axis=0) if coords_t else np.zeros((0, 4), np.float32)
    member = np.concatenate(member_t, axis=0) if member_t else np.zeros((0, 2), np.float32)
    counts = np.asarray([len(c) for c in coords_t], np.int32)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return coords, member, counts, offsets


def image_shape(zp: Path, T: int, z, y, x) -> np.ndarray:
    meta_p = zp / "0" / "zarr.json"
    if zp.exists() and meta_p.exists():
        meta = json.loads(meta_p.read_text())
        shape = tuple(meta.get("shape") or [])
        if len(shape) == 4:
            return np.asarray(shape, np.int32)
    return np.asarray(
        [
            max(T, 1),
            int(z.max()) + 1 if len(z) else 1,
            int(y.max()) + 1 if len(y) else 1,
            int(x.max()) + 1 if len(x) else 1,
        ],
        np.int32,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1-geff-dir", type=Path, required=True, help="Support Pack GEFFs")
    ap.add_argument("--p2-geff-dir", type=Path, required=True, help="0_917 classical GEFFs")
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--match-um", type=float, default=7.0)
    args = ap.parse_args()
    stems = sorted(
        {p.name.replace(".geff", "") for p in args.p1_geff_dir.glob("*.geff")}
        & {p.name.replace(".geff", "") for p in args.p2_geff_dir.glob("*.geff")}
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, stem in enumerate(stems):
        dest = args.out_dir / f"{stem}.npz"
        print(f"[{i+1}/{len(stems)}] {stem}", flush=True)
        if dest.exists():
            rows.append({"stem": stem, "skipped": True})
            continue
        try:
            ds1, td1 = _open_pred(args.p1_geff_dir / f"{stem}.geff", args.data_dir, stem)
            ds2, td2 = _open_pred(args.p2_geff_dir / f"{stem}.geff", args.data_dir, stem)
            try:
                t1, z1, y1, x1 = _xyz(ds1)
                t2, z2, y2, x2 = _xyz(ds2)
                T = int(max(int(t1.max()) if len(t1) else 0, int(t2.max()) if len(t2) else 0)) + 1
                zp = args.data_dir / f"{stem}.zarr"
                shape = image_shape(zp, T, np.concatenate([z1, z2]) if len(z1) + len(z2) else np.zeros(1),
                                    np.concatenate([y1, y2]) if len(y1) + len(y2) else np.zeros(1),
                                    np.concatenate([x1, x2]) if len(x1) + len(x2) else np.zeros(1))
                T = int(shape[0])
                coords, member, counts, offsets = fuse_video(t1, z1, y1, x1, t2, z2, y2, x2, args.match_um, T)
                out_coords = np.stack(
                    [
                        coords[:, 0],
                        np.rint(coords[:, 1]),
                        np.rint(coords[:, 2]),
                        np.rint(coords[:, 3]),
                    ],
                    axis=1,
                ).astype(np.int16) if len(coords) else np.zeros((0, 4), np.int16)
                fused = member.mean(axis=1).astype(np.float32) if len(member) else np.zeros(0, np.float32)
                np.savez_compressed(
                    dest,
                    coords=out_coords,
                    fused_det_prob=fused,
                    member_det_prob=member.astype(np.float32),
                    frame_counts=counts,
                    frame_offsets=offsets,
                    frozen_sources=np.zeros(0, np.int16),
                    image_shape=shape,
                    voxel_scale_um=SCALE,
                    downsample=np.asarray([1, 1, 1], np.int16),
                    det_threshold=np.float32(0.99),
                    pool_kernel_um=np.float32(0.0),
                    detector=np.asarray("p1_support_pack__p2_0917"),
                )
                rows.append(
                    {
                        "stem": stem,
                        "n_p1": int(len(t1)),
                        "n_p2": int(len(t2)),
                        "n_fused": int(len(out_coords)),
                        "T": T,
                    }
                )
            finally:
                shutil.rmtree(td1, ignore_errors=True)
                shutil.rmtree(td2, ignore_errors=True)
        except Exception as e:
            rows.append({"stem": stem, "error": repr(e)})
            print(" FAIL", e, flush=True)
    (args.out_dir / "manifest.json").write_text(json.dumps(rows, indent=2))
    n_ok = sum(1 for r in rows if "error" not in r)
    print(f"Wrote {n_ok}/{len(rows)} -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
