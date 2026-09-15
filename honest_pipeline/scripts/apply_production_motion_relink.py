#!/usr/bin/env python3
"""Production motion relink on P1 Support Pack ILP graphs.

This is the original notebook stage, not a graph rebuild from proposals:

  P1 ILP nodes/edges -> sanitize consecutive edges -> motion_relink_edges
  (Hungarian, production gates) -> single-parent repair -> write GEFF.

Learned residual uses the frozen OOF geom_tight5_rel8 checkpoints. Det features
the checkpoint was trained with are copied from the fused proposal bank by
nearest-neighbour match (7 um). Geometric knobs are the production notebook
defaults (tight 6 / relaxed 10 / joint assignment / max_match 7.5).
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
HP = Path(__file__).resolve().parents[1]
BIO = HP.parent
if not (BIO / "kaggle").exists() and (Path("/data/projects/ryzhichkin/biohub") / "kaggle").exists():
    BIO = Path("/data/projects/ryzhichkin/biohub")
    HP = BIO / "honest_pipeline"
    ROOT = BIO

TRAINER_DIR = (
    BIO
    / "william-duckworth-reproducible-training-pipeline"
    / "helpers/09_motion_corrector/TRAINING_V1"
)
if not TRAINER_DIR.exists():
    TRAINER_DIR = ROOT / "helpers/09_motion_corrector/TRAINING_V1"

SWEEP = HP / "runs/04_motion_corrector/20260914T200230Z_motion_sp_p2_0917_geom_mine_feat_loss"
P1_GEFF = HP / "runs/p1_candidate_compare/kaggle_train_all/support_pack"
PROPOSALS = BIO / "data/ab_proposals_sp_0917"
DATA = BIO / "kaggle/input/competitions/biohub-cell-tracking-during-development/train"
SPLITS = HP / "splits/dataset_splits_gkf5_train175.json"
WINNER = "geom_tight5_rel8"

SCALE = np.asarray([1.625, 0.40625, 0.40625], dtype=np.float64)
Z_EXTENT_UM = 102.375
MATCH_UM = 7.0

# Production notebook CFG["graph"] motion relink knobs.
RELINK_TIGHT_UM = 6.0
RELINK_RELAXED_UM = 10.0
RELINK_VELOCITY_AXES = np.asarray([0.0, 0.45, 0.47], dtype=np.float64)
RELINK_FRAME_REGISTRATION = True
RELINK_REGISTRATION_WEIGHT = 1.0
RELINK_LEARNED_BONUS = 1.0
RELINK_MAX_MATCH_COST = 7.5
RELINK_CAP_INCLUDES_LEARNED = False
RELINK_MOTION_STEPS = 3
RELINK_CORRECTOR_ONE_STEP = True
RELINK_JOINT_ASSIGNMENT = True
RELINK_TIGHT_BONUS_UM = 3.0
RELINK_ORPHAN_PRIOR = False
RELINK_RAW_WEIGHT = 0.05
RELINK_MAX_FRAME_NODES = 4000
MOTION_CORRECTOR_STRENGTH = 1.0
EDGE_MAX_UM = 0.0  # production also has a cap; 0 = off (P1 ILP already gated)


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


def movie_fold(splits: list[dict]) -> dict[str, int]:
    out = {}
    for i, s in enumerate(splits):
        for m in s.get("test") or s.get("val") or []:
            out[m] = i
    return out


def _open_pred(pred: Path, data_dir: Path, stem: str):
    sys.path.insert(0, str(_pack_src()))
    from biohub_tracking.io import open_dataset

    td = Path(tempfile.mkdtemp(prefix="relink_"))
    os.symlink((data_dir / f"{stem}.zarr").resolve(), td / f"{stem}.zarr")
    os.symlink(pred.resolve(), td / f"{stem}.geff")
    ds = open_dataset(td / stem, load_image=False, require_tracks=True, normalize=False)
    return ds, td


def _table(df, keys):
    if hasattr(df, "select"):
        return np.asarray(df.select(keys).to_numpy())
    return np.stack([np.asarray(df[k]) for k in keys], axis=1)


def load_p1_graph(pred: Path, data_dir: Path, stem: str):
    ds, td = _open_pred(pred, data_dir, stem)
    na = ds.tracks.node_attrs(attr_keys=["node_id", "t", "z", "y", "x"])
    nodes = _table(na, ["node_id", "t", "z", "y", "x"])
    ea = ds.tracks.edge_attrs(attr_keys=["source_id", "target_id", "edge_prob"])
    if hasattr(ea, "select"):
        edges = np.asarray(ea.select(["source_id", "target_id", "edge_prob"]).to_numpy())
    else:
        src = np.asarray(ea["source_id"])
        tgt = np.asarray(ea["target_id"])
        if "edge_prob" in getattr(ea, "columns", []):
            pr = np.asarray(ea["edge_prob"])
        else:
            pr = np.zeros(len(src), np.float64)
        edges = np.stack([src, tgt, pr], axis=1) if len(src) else np.zeros((0, 3))
    import shutil

    shutil.rmtree(td, ignore_errors=True)
    return nodes, edges


def load_proposal_det(proposals: Path, stem: str):
    path = proposals / f"{stem}.npz"
    if not path.exists():
        return None
    item = np.load(path, allow_pickle=False)
    coords = np.asarray(item["coords"])
    fused = np.asarray(item["fused_det_prob"], np.float64)
    member = np.asarray(item["member_det_prob"], np.float64)
    return coords, fused, member


def attach_det(nodes, proposals, stem):
    """nodes: (N,5) node_id,t,z,y,x in voxels. Returns det, disagree (N,)."""
    n = len(nodes)
    det = np.zeros(n, np.float64)
    disagree = np.zeros(n, np.float64)
    packed = load_proposal_det(proposals, stem)
    if packed is None or n == 0:
        return det, disagree
    coords, fused, member = packed
    t_nodes = nodes[:, 1].astype(np.int32)
    for t in np.unique(t_nodes):
        ni = np.flatnonzero(t_nodes == t)
        pi = np.flatnonzero(coords[:, 0].astype(np.int32) == int(t))
        if ni.size == 0 or pi.size == 0:
            continue
        src = nodes[ni][:, 2:5].astype(np.float64) * SCALE
        tgt = coords[pi][:, 1:4].astype(np.float64) * SCALE
        dist, jix = cKDTree(tgt).query(src, k=1)
        if np.ndim(dist) == 0:
            dist = np.asarray([dist])
            jix = np.asarray([jix])
        ok = dist <= MATCH_UM
        det[ni[ok]] = fused[pi[jix[ok]]]
        md = member[pi[jix[ok]]]
        if md.ndim == 2 and md.shape[1] >= 2:
            disagree[ni[ok]] = np.abs(md[:, 0] - md[:, 1])
    return det, disagree


def _ids_by_frame(nodes_by_id):
    ids_by_t = {}
    for node_id, node in nodes_by_id.items():
        ids_by_t.setdefault(int(node["t"]), []).append(node_id)
    for ids in ids_by_t.values():
        ids.sort()
    return ids_by_t


def _coords_um(nodes_by_id, ids):
    out = np.empty((len(ids), 3), dtype=np.float64)
    for index, node_id in enumerate(ids):
        node = nodes_by_id[node_id]
        out[index] = (node["z"], node["y"], node["x"])
    return out * SCALE


def _frame_registration(source_pos, target_pos, gate_um):
    if len(source_pos) < 5 or len(target_pos) < 5:
        return np.zeros(3)
    distance, index = cKDTree(target_pos).query(source_pos, k=1)
    keep = distance <= gate_um
    if int(keep.sum()) < 5:
        return np.zeros(3)
    return np.median(target_pos[index[keep]] - source_pos[keep], axis=0)


def sanitized_edge_probs(edges):
    out = {}
    for edge in edges:
        prob = edge.get("edge_prob")
        if prob is None:
            continue
        try:
            value = float(prob)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(value):
            continue
        if value < 0.0 or value > 1.0:
            value = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, value))))
        key = (int(edge["source_id"]), int(edge["target_id"]))
        if value > out.get(key, -1.0):
            out[key] = value
    return out


def load_corrector(ckpt_path: Path):
    sys.path.insert(0, str(TRAINER_DIR))
    import train_motion_cost_corrector as tr

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    feat_names = [str(x) for x in ckpt["features"]]
    args = SimpleNamespace(
        arch="mlp",
        hidden=0,
        dropout=0.05,
        tabm_k=8,
        residual_scale=float(ckpt.get("residual_scale", 2.0)),
        batch_size=8192,
        device="cpu",
        feat_mode=str((ckpt.get("meta") or {}).get("feat_mode", "raw")),
    )
    model = tr.build_model("mlp", len(feat_names), args)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return {
        "model": model,
        "mean": np.asarray(ckpt["mean"], np.float32),
        "std": np.asarray(ckpt["std"], np.float32),
        "features": feat_names,
        "args": args,
        "tr": tr,
    }


def motion_residual(corrector, features: np.ndarray) -> np.ndarray:
    if corrector is None or features.size == 0:
        return np.zeros(len(features), dtype=np.float64)
    tr = corrector["tr"]
    args = corrector["args"]
    names = corrector["features"]
    x, _ = tr.transform_features(features.astype(np.float32), names, args.feat_mode)
    return tr.residuals(corrector["model"], x, corrector["mean"], corrector["std"], args).astype(
        np.float64
    )


def motion_relink_edges(nodes_by_id, stats, learned_edge_probs, corrector, det_by_id):
    if not nodes_by_id:
        return []
    ids_by_t = _ids_by_frame(nodes_by_id)
    if max((len(ids) for ids in ids_by_t.values()), default=0) > RELINK_MAX_FRAME_NODES:
        stats["motion_relink_skipped_large_frame"] = 1
        return []

    probs = learned_edge_probs or {}
    position = {}
    density = {}
    for t, ids in ids_by_t.items():
        coords = _coords_um(nodes_by_id, ids)
        position[t] = coords
        counts = cKDTree(coords).query_ball_point(coords, 15.0, return_length=True)
        density[t] = np.maximum(counts.astype(np.float64) - 1.0, 0.0)

    previous_position = {}
    previous_history = {}
    selected = []
    feat_names = corrector["features"] if corrector is not None else []

    for t in sorted(ids_by_t):
        source_ids = ids_by_t.get(t, [])
        target_ids = ids_by_t.get(t + 1, [])
        if not source_ids or not target_ids:
            continue
        source_pos = position[t]
        target_pos = position[t + 1]
        velocity = np.zeros_like(source_pos)
        step_velocity = np.zeros_like(source_pos)
        has_predecessor = np.zeros(len(source_ids), dtype=np.float64)
        for index, node_id in enumerate(source_ids):
            previous = previous_position.get(node_id)
            if previous is not None:
                step_velocity[index] = source_pos[index] - previous
                velocity[index] = step_velocity[index]
                has_predecessor[index] = 1.0
            if RELINK_MOTION_STEPS > 1:
                chain = previous_history.get(node_id)
                if chain:
                    points = [source_pos[index], *chain]
                    deltas = [points[offset] - points[offset + 1] for offset in range(len(points) - 1)]
                    velocity[index] = np.mean(deltas[:RELINK_MOTION_STEPS], axis=0)
                    has_predecessor[index] = 1.0
        shift = (
            _frame_registration(source_pos, target_pos, RELINK_RELAXED_UM)
            if RELINK_FRAME_REGISTRATION
            else np.zeros(3)
        )
        extrapolation = RELINK_VELOCITY_AXES * velocity
        predicted = source_pos + extrapolation + RELINK_REGISTRATION_WEIGHT * shift
        corrector_velocity = step_velocity if RELINK_CORRECTOR_ONE_STEP else velocity
        velocity_magnitude = np.linalg.norm(corrector_velocity, axis=1)
        z_boundary_src = np.clip(source_pos[:, 0] / Z_EXTENT_UM, 0.0, 1.0)
        z_boundary_src = np.minimum(z_boundary_src, 1.0 - z_boundary_src)
        z_boundary_tgt = np.clip(target_pos[:, 0] / Z_EXTENT_UM, 0.0, 1.0)
        z_boundary_tgt = np.minimum(z_boundary_tgt, 1.0 - z_boundary_tgt)
        det_src = np.asarray([det_by_id.get(i, (0.0, 0.0))[0] for i in source_ids], np.float64)
        det_tgt = np.asarray([det_by_id.get(i, (0.0, 0.0))[0] for i in target_ids], np.float64)
        dis_src = np.asarray([det_by_id.get(i, (0.0, 0.0))[1] for i in source_ids], np.float64)
        dis_tgt = np.asarray([det_by_id.get(i, (0.0, 0.0))[1] for i in target_ids], np.float64)

        def assign_pass(source_sel, target_sel, gate_um, tight_bonus=0.0):
            if source_sel.size == 0 or target_sel.size == 0:
                return np.empty(0, np.int64), np.empty(0, np.int64)
            local_source = source_pos[source_sel]
            local_target = target_pos[target_sel]
            neighbours = cKDTree(local_target).query_ball_point(local_source, gate_um)
            counts = np.fromiter((len(item) for item in neighbours), np.int64, len(neighbours))
            if not counts.any():
                return np.empty(0, np.int64), np.empty(0, np.int64)
            rows = np.repeat(np.arange(len(source_sel), dtype=np.int64), counts)
            cols = np.fromiter(
                (index for item in neighbours for index in item), np.int64, int(counts.sum())
            )
            delta = local_target[cols] - local_source[rows]
            raw = np.linalg.norm(delta, axis=1)
            inside = raw <= gate_um
            rows, cols, delta, raw = rows[inside], cols[inside], delta[inside], raw[inside]
            if rows.size == 0:
                return np.empty(0, np.int64), np.empty(0, np.int64)
            global_source = source_sel[rows]
            global_target = target_sel[cols]
            motion = np.linalg.norm(local_target[cols] - predicted[global_source], axis=1)
            probability = np.fromiter(
                (
                    probs.get((source_ids[s], target_ids[g]), 0.0)
                    for s, g in zip(global_source, global_target)
                ),
                np.float64,
                rows.size,
            )
            base_cost = motion + RELINK_RAW_WEIGHT * raw
            values = base_cost - RELINK_LEARNED_BONUS * probability
            if corrector is not None:
                absolute = np.abs(delta)
                registered_delta = delta - shift
                registered = np.linalg.norm(registered_delta, axis=1)
                blocks = {
                    "base_cost": base_cost,
                    "raw_dist": raw,
                    "registered_dist": registered,
                    "motion_dist": motion,
                    "abs_dz": absolute[:, 0],
                    "abs_dy": absolute[:, 1],
                    "abs_dx": absolute[:, 2],
                    "abs_reg_dz": np.abs(registered_delta[:, 0]),
                    "abs_reg_dy": np.abs(registered_delta[:, 1]),
                    "abs_reg_dx": np.abs(registered_delta[:, 2]),
                    "velocity_z": corrector_velocity[global_source, 0],
                    "velocity_y": corrector_velocity[global_source, 1],
                    "velocity_x": corrector_velocity[global_source, 2],
                    "velocity_mag": velocity_magnitude[global_source],
                    "shift_z": np.full(rows.size, shift[0]),
                    "shift_y": np.full(rows.size, shift[1]),
                    "shift_x": np.full(rows.size, shift[2]),
                    "shift_mag": np.full(rows.size, float(np.linalg.norm(shift))),
                    "det_src": det_src[global_source],
                    "det_tgt": det_tgt[global_target],
                    "det_disagree_src": dis_src[global_source],
                    "det_disagree_tgt": dis_tgt[global_target],
                    "density_src": density[t][global_source],
                    "density_tgt": density[t + 1][global_target],
                    "z_boundary_src": z_boundary_src[global_source],
                    "z_boundary_tgt": z_boundary_tgt[global_target],
                    "has_predecessor": has_predecessor[global_source],
                }
                features = np.column_stack([blocks[n] for n in feat_names])
                residual = motion_residual(corrector, features) * MOTION_CORRECTOR_STRENGTH
                values = values - residual
                stats["motion_corrector_candidates"] += int(rows.size)
            if tight_bonus > 0.0:
                values = values - tight_bonus * (raw <= RELINK_TIGHT_UM)
            if RELINK_MAX_MATCH_COST > 0.0:
                judged = values if RELINK_CAP_INCLUDES_LEARNED else base_cost
                keep = judged <= RELINK_MAX_MATCH_COST
                rejected = int(keep.size - keep.sum())
                if rejected:
                    stats["relink_cost_cap_rejected"] += rejected
                    rows, cols, values = rows[keep], cols[keep], values[keep]
                    if rows.size == 0:
                        stats["relink_cost_cap_emptied_frames"] += 1
                        return np.empty(0, np.int64), np.empty(0, np.int64)
            big = gate_um * 1000.0 + 1.0
            cost = np.full((source_sel.size, target_sel.size), big, dtype=np.float64)
            cost[rows, cols] = values
            row_index, col_index = linear_sum_assignment(cost)
            matched = cost[row_index, col_index] < big
            return source_sel[row_index[matched]], target_sel[col_index[matched]]

        open_sources = np.ones(len(source_ids), dtype=bool)
        open_targets = np.ones(len(target_ids), dtype=bool)
        if RELINK_JOINT_ASSIGNMENT:
            schedule = (("joint", RELINK_RELAXED_UM, RELINK_TIGHT_BONUS_UM),)
        else:
            schedule = (("tight", RELINK_TIGHT_UM, 0.0), ("relaxed", RELINK_RELAXED_UM, 0.0))
        for pass_name, gate_um, tight_bonus in schedule:
            matched_source, matched_target = assign_pass(
                np.flatnonzero(open_sources),
                np.flatnonzero(open_targets),
                gate_um,
                tight_bonus,
            )
            if matched_source.size == 0:
                continue
            open_sources[matched_source] = False
            open_targets[matched_target] = False
            raw = np.linalg.norm(target_pos[matched_target] - source_pos[matched_source], axis=1)
            motion = np.linalg.norm(target_pos[matched_target] - predicted[matched_source], axis=1)
            for offset in range(matched_source.size):
                source_id = source_ids[int(matched_source[offset])]
                target_id = target_ids[int(matched_target[offset])]
                selected.append(
                    {
                        "source_id": source_id,
                        "target_id": target_id,
                        "edge_prob": probs.get((source_id, target_id), 0.0),
                        "distance_um": float(raw[offset]),
                        "motion_distance_um": float(motion[offset]),
                        "motion_relinked": 1,
                        "motion_pass": pass_name,
                    }
                )
                source_point = source_pos[int(matched_source[offset])]
                previous_position[target_id] = source_point
                if RELINK_MOTION_STEPS > 1:
                    previous_history[target_id] = [
                        source_point,
                        *previous_history.get(source_id, []),
                    ][:RELINK_MOTION_STEPS]
        stats["motion_relink_frames"] += 1
    stats["motion_relink_edges"] = len(selected)
    return selected


def single_parent_repair(edges):
    best_by_target = {}

    def sort_key(edge):
        return (float(edge.get("edge_prob") or 0.0), -float(edge.get("distance_um") or 0.0))

    for edge in edges:
        target_id = int(edge["target_id"])
        previous = best_by_target.get(target_id)
        if previous is None or sort_key(edge) > sort_key(previous):
            best_by_target[target_id] = edge
    kept = {id(edge) for edge in best_by_target.values()}
    return [edge for edge in edges if id(edge) in kept]


def apply_one(stem: str, fold: int, ckpt: str, p1_dir: str, out_dir: str, proposals: str, data: str):
    sys.path.insert(0, str(_pack_src()))
    from biohub_tracking.io import save_graph
    import tracksdata as td
    import polars as pl

    nodes_arr, edges_arr = load_p1_graph(Path(p1_dir) / f"{stem}.geff", Path(data), stem)
    nodes_by_id = {}
    for row in nodes_arr:
        nid = int(row[0])
        nodes_by_id[nid] = {
            "node_id": nid,
            "t": int(row[1]),
            "z": float(row[2]),
            "y": float(row[3]),
            "x": float(row[4]),
        }
    raw_edges = []
    for row in edges_arr:
        src, tgt = int(row[0]), int(row[1])
        if src not in nodes_by_id or tgt not in nodes_by_id:
            continue
        if int(nodes_by_id[tgt]["t"]) != int(nodes_by_id[src]["t"]) + 1:
            continue
        raw_edges.append({"source_id": src, "target_id": tgt, "edge_prob": float(row[2])})

    det, disagree = attach_det(nodes_arr, Path(proposals), stem)
    det_by_id = {
        int(nodes_arr[i, 0]): (float(det[i]), float(disagree[i])) for i in range(len(nodes_arr))
    }
    corrector = load_corrector(Path(ckpt))
    stats = defaultdict(int)
    motion_edges = motion_relink_edges(
        nodes_by_id, stats, sanitized_edge_probs(raw_edges), corrector, det_by_id
    )
    edges = motion_edges if motion_edges else raw_edges
    edges = single_parent_repair(edges)

    g = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        g.add_node_attr_key(key, pl.Float64, -999999.0)
    node_rows = [
        {"t": n["t"], "z": n["z"], "y": n["y"], "x": n["x"]}
        for n in sorted(nodes_by_id.values(), key=lambda x: (x["t"], x["node_id"]))
    ]
    old_ids = [n["node_id"] for n in sorted(nodes_by_id.values(), key=lambda x: (x["t"], x["node_id"]))]
    gids = g.bulk_add_nodes(node_rows) if node_rows else []
    remap = {old: int(new) for old, new in zip(old_ids, gids)}
    if edges and remap:
        erows = [
            {"source_id": remap[int(e["source_id"])], "target_id": remap[int(e["target_id"])]}
            for e in edges
            if int(e["source_id"]) in remap and int(e["target_id"]) in remap
        ]
        if erows:
            g.bulk_add_edges(erows)
    dest = Path(out_dir) / f"{stem}.geff"
    dest.parent.mkdir(parents=True, exist_ok=True)
    save_graph(g, dest, overwrite=True)
    return {
        "dataset": stem,
        "fold": fold,
        "n_nodes": int(g.num_nodes()),
        "n_edges": int(g.num_edges()),
        "n_raw_edges": len(raw_edges),
        "n_relink_edges": int(stats.get("motion_relink_edges") or 0),
        "relink_cost_cap_rejected": int(stats.get("relink_cost_cap_rejected") or 0),
        "fallback_raw": int(not bool(motion_edges)),
    }


def score_dir(pred_dir: Path, data_dir: Path, stems: list[str]) -> dict:
    sys.path.insert(0, str(HP / "scripts"))
    from compare_p1_candidates import fold_breakdown, mean_metric, score_stems

    rows = score_stems(pred_dir, data_dir, stems)
    splits = json.loads(SPLITS.read_text()) if SPLITS.exists() else []
    return {
        "mean_adj_edge_jaccard": mean_metric(rows),
        "mean_node_recall": mean_metric(rows, "node_recall"),
        "n_scored": sum(1 for r in rows if r.get("adj_edge_jaccard") == r.get("adj_edge_jaccard")),
        "gkf5": fold_breakdown(rows, splits) if splits else None,
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=Path, default=SWEEP)
    ap.add_argument("--winner", default=WINNER)
    ap.add_argument("--p1-geff-dir", type=Path, default=P1_GEFF)
    ap.add_argument("--proposals", type=Path, default=PROPOSALS)
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=HP / "runs/oof_graphs/p1_production_motion_relink",
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--stems", default="")
    ap.add_argument("--skip-score", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    splits = json.loads(SPLITS.read_text())
    fold_of = movie_fold(splits)
    stems = sorted(p.name.replace(".geff", "") for p in args.p1_geff_dir.glob("*.geff"))
    if args.stems.strip():
        want = {s.strip() for s in args.stems.split(",") if s.strip()}
        stems = [s for s in stems if s in want]
    ckpts = {f: args.sweep / args.winner / f"fold_{f}" / "motion_corrector_best.pt" for f in range(5)}
    missing = [f for f, p in ckpts.items() if not p.exists()]
    if missing:
        raise SystemExit(f"Missing motion ckpts for folds {missing}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for stem in stems:
        dest = args.out_dir / f"{stem}.geff"
        if args.resume and dest.exists():
            continue
        fold = int(fold_of.get(stem, 0))
        jobs.append((stem, fold, str(ckpts[fold])))
    print(f"relink {len(jobs)}/{len(stems)} movies workers={args.workers}", flush=True)
    rows = []
    if jobs:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {
                ex.submit(
                    apply_one,
                    stem,
                    fold,
                    ckpt,
                    str(args.p1_geff_dir),
                    str(args.out_dir),
                    str(args.proposals),
                    str(args.data_dir),
                ): stem
                for stem, fold, ckpt in jobs
            }
            for i, fut in enumerate(as_completed(futs), 1):
                stem = futs[fut]
                try:
                    row = fut.result()
                except Exception as e:
                    row = {"dataset": stem, "error": repr(e)}
                    print("FAIL", stem, e, flush=True)
                rows.append(row)
                print(f"[{i}/{len(jobs)}] {stem} {row}", flush=True)
    if args.resume:
        existing = [
            {"dataset": p.stem, "resumed": True}
            for p in sorted(args.out_dir.glob("*.geff"))
            if p.stem not in {r.get("dataset") for r in rows}
        ]
        rows = existing + rows
    (args.out_dir / "apply_manifest.json").write_text(json.dumps(rows, indent=2) + "\n")
    ok_stems = [r["dataset"] for r in rows if "error" not in r]
    if args.skip_score:
        print("skip score, wrote", args.out_dir)
        return
    print("Scoring", len(ok_stems), flush=True)
    metrics = score_dir(args.out_dir, args.data_dir, ok_stems)
    slim = {k: v for k, v in metrics.items() if k != "rows"}
    (args.out_dir / "metrics.json").write_text(json.dumps(slim, indent=2) + "\n")
    p1m = json.loads((args.p1_geff_dir / "metrics.json").read_text()) if (args.p1_geff_dir / "metrics.json").exists() else {}
    print(
        json.dumps(
            {
                "p1_adj199": p1m.get("mean_adj_edge_jaccard"),
                "p1_gkf5": (p1m.get("gkf5") or {}).get("mean_of_fold_means_adj"),
                "relink": slim,
            },
            indent=2,
        )
    )
    print("Wrote", args.out_dir)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
