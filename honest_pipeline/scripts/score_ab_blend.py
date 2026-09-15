#!/usr/bin/env python3
"""Score P1∪P2 blend graphs vs GT (adj_edge_jaccard).

P1 = Support Pack GEFFs, P2 = 0_917 classical GEFFs.
Two blend constructions (same 7 um node identity):
  edge_union  — keep P1 graph, add unmatched P2 nodes + remapped P2 edges
  relink      — fused nodes, 1:1 Hungarian (tight 6 um then 10 um)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
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
SCALE = np.array([1.625, 0.40625, 0.40625], np.float64)
TIGHT = 6.0
RELAXED = 10.0
MATCH_UM = 7.0


def _open_pred(pred: Path, data_dir: Path, stem: str):
    sys.path.insert(0, str(PACK_SRC))
    from biohub_tracking.io import open_dataset

    td = Path(tempfile.mkdtemp(prefix="blendsc_"))
    os.symlink((data_dir / f"{stem}.zarr").resolve(), td / f"{stem}.zarr")
    os.symlink(pred.resolve(), td / f"{stem}.geff")
    ds = open_dataset(td / stem, load_image=False, require_tracks=True, normalize=False)
    return ds, td


def _nodes(ds):
    df = ds.tracks.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
    if hasattr(df, "select"):
        arr = np.asarray(df.select(["node_id", "t", "z", "y", "x"]).to_numpy())
    else:
        arr = np.stack([np.asarray(df[k]) for k in ("node_id", "t", "z", "y", "x")], axis=1)
    return (
        arr[:, 0].astype(np.int64),
        arr[:, 1].astype(np.int32),
        arr[:, 2].astype(np.float64),
        arr[:, 3].astype(np.float64),
        arr[:, 4].astype(np.float64),
    )


def _edges(ds):
    df = ds.tracks.edge_attrs(attr_keys=["source_id", "target_id"])
    if hasattr(df, "select"):
        arr = np.asarray(df.select(["source_id", "target_id"]).to_numpy())
    else:
        arr = np.stack([np.asarray(df[k]) for k in ("source_id", "target_id")], axis=1)
    if arr.size == 0:
        return np.zeros((0, 2), np.int64)
    return arr.astype(np.int64)


def _match_frame(a_zyx: np.ndarray, b_zyx: np.ndarray, match_um: float):
    """Return dict local_b -> local_a for matches within match_um."""
    if len(a_zyx) == 0 or len(b_zyx) == 0:
        return {}
    au = a_zyx * SCALE
    bu = b_zyx * SCALE
    n0, n1 = len(a_zyx), len(b_zyx)
    cost = np.full((n0, n1), 1e6)
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
    out = {}
    for i, j in zip(ri, ci):
        if cost[i, j] < 1e6:
            out[int(j)] = int(i)
    return out


def _hun_links(a: np.ndarray, b: np.ndarray, tight: float, relaxed: float):
    if len(a) == 0 or len(b) == 0:
        return []
    A = a * SCALE
    B = b * SCALE
    n0, n1 = len(a), len(b)
    BIG = 1e6

    def one(mask0, mask1, gate):
        pi = np.flatnonzero(mask0)
        ci = np.flatnonzero(mask1)
        if len(pi) == 0 or len(ci) == 0:
            return []
        d = np.sqrt(((A[pi][:, None] - B[ci][None]) ** 2).sum(2))
        cost = np.where(d > gate, BIG, d)
        ri, rc = linear_sum_assignment(cost)
        return [(int(pi[r]), int(ci[c])) for r, c in zip(ri, rc) if cost[r, c] < BIG]

    used0 = np.ones(n0, bool)
    used1 = np.ones(n1, bool)
    links = one(used0, used1, tight)
    for i, j in links:
        used0[i] = False
        used1[j] = False
    links += one(used0, used1, relaxed)
    return links


def _to_graph(t, z, y, x, edges):
    import tracksdata as td

    g = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        g.add_node_attr_key(key, __import__("polars").Float64, -999999.0)
    if len(t) == 0:
        return g
    rows = [{"t": int(tt), "z": float(zz), "y": float(yy), "x": float(xx)} for tt, zz, yy, xx in zip(t, z, y, x)]
    gids = g.bulk_add_nodes(rows)
    if len(edges):
        erows = [{"source_id": int(gids[s]), "target_id": int(gids[d])} for s, d in edges if 0 <= s < len(gids) and 0 <= d < len(gids)]
        if erows:
            g.bulk_add_edges(erows)
    return g


def _score_graph(pred, gt, n_total: float):
    from biohub_tracking.metrics import evaluate, node_recall, per_sample_metrics

    scale = tuple(gt.original_scale or gt.scale or (1.625, 0.40625, 0.40625))
    res = evaluate(pred, gt.tracks, scale=scale, max_distance=7.0)
    nr = float(node_recall(pred, gt.tracks))
    if not (n_total == n_total) or n_total <= 0:
        n_total = float(gt.tracks.num_nodes())
    return per_sample_metrics(res, n_total, nr)


def blend_one(stem: str, p1_dir: str, p2_dir: str, data_dir: str) -> dict:
    sys.path.insert(0, str(PACK_SRC))
    from biohub_tracking.io import open_dataset
    from geff import GeffMetadata

    data_dir_p = Path(data_dir)
    p1_p = Path(p1_dir) / f"{stem}.geff"
    p2_p = Path(p2_dir) / f"{stem}.geff"
    try:
        meta = GeffMetadata.read(data_dir_p / f"{stem}.geff")
        n_total = float((meta.extra or {}).get("estimated_number_of_nodes") or float("nan"))
    except Exception:
        n_total = float("nan")
    gt = open_dataset(data_dir_p / stem, load_image=False, require_tracks=True, normalize=False)
    ds1, td1 = _open_pred(p1_p, data_dir_p, stem)
    ds2, td2 = _open_pred(p2_p, data_dir_p, stem)
    try:
        id1, t1, z1, y1, x1 = _nodes(ds1)
        id2, t2, z2, y2, x2 = _nodes(ds2)
        e1 = _edges(ds1)
        e2 = _edges(ds2)
        T = int(max(int(t1.max()) if len(t1) else 0, int(t2.max()) if len(t2) else 0)) + 1
        extra_t, extra_z, extra_y, extra_x = [], [], [], []
        p2_id_to_idx = {}
        extra_k = 0
        for t in range(T):
            i1 = np.flatnonzero(t1 == t)
            i2 = np.flatnonzero(t2 == t)
            a = np.stack([z1[i1], y1[i1], x1[i1]], axis=1) if len(i1) else np.zeros((0, 3))
            b = np.stack([z2[i2], y2[i2], x2[i2]], axis=1) if len(i2) else np.zeros((0, 3))
            matched = _match_frame(a, b, MATCH_UM)
            for jb, loc in enumerate(i2):
                pid = int(id2[loc])
                if jb in matched:
                    p2_id_to_idx[pid] = int(i1[matched[jb]])
                else:
                    extra_t.append(int(t2[loc]))
                    extra_z.append(z2[loc])
                    extra_y.append(y2[loc])
                    extra_x.append(x2[loc])
                    p2_id_to_idx[pid] = len(t1) + extra_k
                    extra_k += 1

        n1 = len(t1)
        ft = np.concatenate([t1, np.asarray(extra_t, np.int32)]) if extra_t else t1
        fz = np.concatenate([z1, np.asarray(extra_z)]) if extra_z else z1
        fy = np.concatenate([y1, np.asarray(extra_y)]) if extra_y else y1
        fx = np.concatenate([x1, np.asarray(extra_x)]) if extra_x else x1
        p1_id_to_idx = {int(i): k for k, i in enumerate(id1)}

        union_edges = set()
        for s, d in e1:
            if int(s) in p1_id_to_idx and int(d) in p1_id_to_idx:
                union_edges.add((p1_id_to_idx[int(s)], p1_id_to_idx[int(d)]))
        for s, d in e2:
            if int(s) in p2_id_to_idx and int(d) in p2_id_to_idx:
                union_edges.add((p2_id_to_idx[int(s)], p2_id_to_idx[int(d)]))

        g_union = _to_graph(ft, fz, fy, fx, list(union_edges))
        row_union = _score_graph(g_union, gt, n_total)

        # relink fused nodes
        relink_edges = []
        for t in range(T - 1):
            ia = np.flatnonzero(ft == t)
            ib = np.flatnonzero(ft == t + 1)
            a = np.stack([fz[ia], fy[ia], fx[ia]], axis=1) if len(ia) else np.zeros((0, 3))
            b = np.stack([fz[ib], fy[ib], fx[ib]], axis=1) if len(ib) else np.zeros((0, 3))
            for i, j in _hun_links(a, b, TIGHT, RELAXED):
                relink_edges.append((int(ia[i]), int(ib[j])))
        g_relink = _to_graph(ft, fz, fy, fx, relink_edges)
        row_relink = _score_graph(g_relink, gt, n_total)

        return {
            "dataset": stem,
            "n_p1": int(len(t1)),
            "n_p2": int(len(t2)),
            "n_fused": int(len(ft)),
            "edge_union": {k: row_union.get(k) for k in ("adj_edge_jaccard", "edge_jaccard", "node_recall", "score")},
            "relink": {k: row_relink.get(k) for k in ("adj_edge_jaccard", "edge_jaccard", "node_recall", "score")},
        }
    finally:
        shutil.rmtree(td1, ignore_errors=True)
        shutil.rmtree(td2, ignore_errors=True)


def mean_key(rows, path):
    vals = []
    for r in rows:
        cur = r
        for k in path:
            cur = cur.get(k) if isinstance(cur, dict) else None
        if cur is not None and cur == cur:
            vals.append(float(cur))
    return float(np.mean(vals)) if vals else float("nan")


def fold_break(rows, splits, path):
    by = {r["dataset"]: r for r in rows}
    folds = []
    for s in splits:
        test = list(s.get("test") or s.get("val") or [])
        fr = [by[d] for d in test if d in by]
        folds.append(
            {
                "fold_id": s.get("fold_id"),
                "n_test": len(test),
                "n_scored": sum(1 for r in fr if "error" not in r),
                "mean_adj": mean_key(fr, path),
                "mean_node_recall": mean_key(fr, path[:-1] + ["node_recall"]),
            }
        )
    ms = [f["mean_adj"] for f in folds if f["mean_adj"] == f["mean_adj"]]
    return {"folds": folds, "mean_of_fold_means_adj": float(np.mean(ms)) if ms else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p1-geff-dir", type=Path, required=True)
    ap.add_argument("--p2-geff-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--splits", type=Path, default=ROOT / "honest_pipeline/splits/dataset_splits_gkf5_train175.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    stems = sorted(
        {p.name.replace(".geff", "") for p in args.p1_geff_dir.glob("*.geff")}
        & {p.name.replace(".geff", "") for p in args.p2_geff_dir.glob("*.geff")}
    )
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(blend_one, s, str(args.p1_geff_dir), str(args.p2_geff_dir), str(args.data_dir)): s
            for s in stems
        }
        for i, fut in enumerate(as_completed(futs), 1):
            stem = futs[fut]
            try:
                row = fut.result()
            except Exception as e:
                row = {"dataset": stem, "error": repr(e), "edge_union": {}, "relink": {}}
            rows.append(row)
            adj_u = (row.get("edge_union") or {}).get("adj_edge_jaccard")
            adj_r = (row.get("relink") or {}).get("adj_edge_jaccard")
            print(f"[{i}/{len(stems)}] {stem} union={adj_u} relink={adj_r}", flush=True)
    splits = json.loads(args.splits.read_text()) if args.splits.exists() else []
    ok = [r for r in rows if "error" not in r]
    summary = {
        "n_movies": len(stems),
        "n_ok": len(ok),
        "n_fail": len(rows) - len(ok),
        "edge_union": {
            "mean_adj_edge_jaccard": mean_key(ok, ["edge_union", "adj_edge_jaccard"]),
            "mean_edge_jaccard": mean_key(ok, ["edge_union", "edge_jaccard"]),
            "mean_node_recall": mean_key(ok, ["edge_union", "node_recall"]),
            "gkf5": fold_break(ok, splits, ["edge_union", "adj_edge_jaccard"]) if splits else None,
        },
        "relink": {
            "mean_adj_edge_jaccard": mean_key(ok, ["relink", "adj_edge_jaccard"]),
            "mean_edge_jaccard": mean_key(ok, ["relink", "edge_jaccard"]),
            "mean_node_recall": mean_key(ok, ["relink", "node_recall"]),
            "gkf5": fold_break(ok, splits, ["relink", "adj_edge_jaccard"]) if splits else None,
        },
        "reference": {
            "p1_support_pack_adj199": 0.9108443602845554,
            "p2_0917_adj199": 0.8696347517891077,
        },
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    print("Wrote", args.out)


if __name__ == "__main__":
    main()
