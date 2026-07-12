from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


COMPETITION = "biohub-cell-tracking-during-development"
COMP_DIR_CANDIDATES = [
    Path(f"/kaggle/input/competitions/{COMPETITION}"),
    Path(f"/kaggle/input/{COMPETITION}"),
]
COMP_DIR = next((p for p in COMP_DIR_CANDIDATES if p.exists()), COMP_DIR_CANDIDATES[0])
TRAIN_DIR = Path(os.environ.get("BIOHUB_TRAIN_DIR", COMP_DIR / "train"))
SUBMISSION_PATH = Path(os.environ.get("BIOHUB_SUBMISSION_PATH", "/kaggle/working/submission.csv"))
OUT_PATH = Path(os.environ.get("BIOHUB_LOCAL_GT_EVAL_PATH", "/kaggle/working/local_gt_eval.csv"))
SHORT7_COMPONENTS_PATH = Path(os.environ.get("BIOHUB_SHORT7_COMPONENTS_PATH", "/kaggle/working/dense_short7_components.csv"))
SHORT7_NODES_PATH = Path(os.environ.get("BIOHUB_SHORT7_NODES_PATH", "/kaggle/working/dense_short7_nodes.csv"))

VOXEL_SCALE_UM = np.asarray((1.625, 0.40625, 0.40625), dtype=np.float32)
MATCH_UM = float(os.environ.get("BIOHUB_MATCH_UM", "7.0"))
NODE_PENALTY_A = float(os.environ.get("BIOHUB_NODE_PENALTY_A", "0.1"))


def graph_from_geff(path: Path):
    try:
        import tracksdata as td
    except Exception as e:
        raise RuntimeError(
            "tracksdata is required. Run this after the .901 notebook dependency install, "
            "or install the Biohub support-pack wheels first."
        ) from e
    graph = td.graph.IndexedRXGraph.from_geff(path)
    return graph[0] if isinstance(graph, tuple) else graph


def graph_nodes_edges(graph):
    nodes = {}
    for row in graph.node_attrs().iter_rows(named=True):
        nid = int(row["node_id"])
        nodes[nid] = {
            "node_id": nid,
            "t": int(row["t"]),
            "z": float(row["z"]),
            "y": float(row["y"]),
            "x": float(row["x"]),
        }
    edges = []
    for row in graph.edge_attrs().iter_rows(named=True):
        edges.append((int(row["source_id"]), int(row["target_id"])))
    return nodes, edges


def load_gt(dataset: str):
    geff_path = TRAIN_DIR / f"{dataset}.geff"
    if not geff_path.exists():
        raise FileNotFoundError(f"GT geff not found for {dataset}: {geff_path}")
    return graph_nodes_edges(graph_from_geff(geff_path))


def estimated_true_nodes_from_geff(dataset: str, fallback_gt_nodes: int) -> int:
    """Read the coarse total-node estimate used by the adjusted metric when present.

    The sparse GEFF contains annotated nodes only, but its metadata may include a
    coarse estimate of all true nodes. If unavailable, fall back to annotated GT
    count so the column remains defined; fallback is only a local-practice proxy.
    """
    geff_path = TRAIN_DIR / f"{dataset}.geff"
    # Optional manual override for quick Kaggle experiments:
    # BIOHUB_ESTIMATED_TRUE_NODES_JSON='{"44b6_0113de3b":33900,...}'
    override_json = os.environ.get("BIOHUB_ESTIMATED_TRUE_NODES_JSON", "").strip()
    if override_json:
        try:
            import json

            override = json.loads(override_json)
            value = override.get(dataset)
            if value is not None:
                value = int(round(float(value)))
                if value > 0:
                    return value
        except Exception:
            pass

    meta_path = geff_path / "zarr.json"
    if meta_path.exists():
        try:
            import json

            meta = json.loads(meta_path.read_text())
            # Known competition metadata path used by the Biohub notebooks.
            value = (
                meta.get("attributes", {})
                .get("geff", {})
                .get("extra", {})
                .get("estimated_number_of_nodes")
            )
            if value is not None:
                value = int(round(float(value)))
                if value > 0:
                    return value

            attrs = meta.get("attributes", {})
            # Accept a few likely names; GEFF metadata has varied across versions.
            stack = [attrs]
            if isinstance(attrs.get("geff"), dict):
                stack.append(attrs["geff"])
                if isinstance(attrs["geff"].get("extra"), dict):
                    stack.append(attrs["geff"]["extra"])
            if isinstance(attrs.get("metadata"), dict):
                stack.append(attrs["metadata"])
            keys = (
                "estimated_number_of_nodes",
                "estimated_nodes",
                "estimated_total_nodes",
                "total_nodes_estimate",
                "node_count_estimate",
                "true_node_count_estimate",
                "estimated_true_nodes",
            )
            for obj in stack:
                for key in keys:
                    value = obj.get(key)
                    if value is not None:
                        value = int(round(float(value)))
                        if value > 0:
                            return value

            # Last-resort recursive search for any key containing both
            # "estimated" and "node". This avoids silently using sparse count
            # when GEFF metadata shape changes.
            def walk(obj):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        lk = str(k).lower()
                        if "estimated" in lk and "node" in lk and v is not None:
                            try:
                                iv = int(round(float(v)))
                                if iv > 0:
                                    return iv
                            except Exception:
                                pass
                        found = walk(v)
                        if found is not None:
                            return found
                elif isinstance(obj, list):
                    for item in obj:
                        found = walk(item)
                        if found is not None:
                            return found
                return None

            found = walk(meta)
            if found is not None:
                return found
        except Exception:
            pass
    print(
        f"[warn] {dataset}: full-node estimate not found in {geff_path}/zarr.json; "
        f"falling back to sparse GT node count={fallback_gt_nodes}. "
        "Adjusted node penalty will be too harsh for this dataset.",
        flush=True,
    )
    return int(fallback_gt_nodes)


def load_pred(df: pd.DataFrame, dataset: str):
    g = df[df["dataset"] == dataset]
    node_rows = g[g["row_type"] == "node"]
    edge_rows = g[g["row_type"] == "edge"]
    nodes = {}
    for row in node_rows.itertuples(index=False):
        nid = int(row.node_id)
        nodes[nid] = {
            "node_id": nid,
            "t": int(row.t),
            "z": float(row.z),
            "y": float(row.y),
            "x": float(row.x),
        }
    edges = []
    for row in edge_rows.itertuples(index=False):
        s, t = int(row.source_id), int(row.target_id)
        if s in nodes and t in nodes:
            edges.append((s, t))
    return nodes, edges


def node_xyz_um(nodes: dict[int, dict], ids: list[int]) -> np.ndarray:
    arr = np.asarray([[nodes[i]["z"], nodes[i]["y"], nodes[i]["x"]] for i in ids], dtype=np.float32)
    return arr * VOXEL_SCALE_UM[None, :]


def match_nodes(pred_nodes: dict[int, dict], gt_nodes: dict[int, dict], gate_um: float = MATCH_UM):
    pred_by_t, gt_by_t = {}, {}
    for nid, node in pred_nodes.items():
        pred_by_t.setdefault(int(node["t"]), []).append(nid)
    for nid, node in gt_nodes.items():
        gt_by_t.setdefault(int(node["t"]), []).append(nid)

    pred_to_gt = {}
    gt_to_pred = {}
    dists = []
    for t in sorted(set(pred_by_t) | set(gt_by_t)):
        pids = pred_by_t.get(t, [])
        gids = gt_by_t.get(t, [])
        if not pids or not gids:
            continue
        P = node_xyz_um(pred_nodes, pids)
        G = node_xyz_um(gt_nodes, gids)
        C = np.linalg.norm(P[:, None, :] - G[None, :, :], axis=2)
        ri, ci = linear_sum_assignment(C)
        for r, c in zip(ri, ci):
            dist = float(C[r, c])
            if dist <= gate_um:
                pid, gid = pids[int(r)], gids[int(c)]
                pred_to_gt[pid] = gid
                gt_to_pred[gid] = pid
                dists.append(dist)
    return pred_to_gt, gt_to_pred, dists


def eval_dataset(dataset: str, pred_nodes: dict[int, dict], pred_edges: list[tuple[int, int]]):
    gt_nodes, gt_edges = load_gt(dataset)
    pred_to_gt, gt_to_pred, dists = match_nodes(pred_nodes, gt_nodes)

    gt_edge_set = set(gt_edges)
    pred_edge_set = set(pred_edges)
    gt_out = {}
    gt_in = {}
    for s, t in gt_edge_set:
        gt_out.setdefault(s, set()).add(t)
        gt_in.setdefault(t, set()).add(s)

    mapped_pred_edges = []
    edge_tp = 0
    edge_fp = 0
    ignored_pred_edges = 0

    for s, t in pred_edge_set:
        ms = pred_to_gt.get(s)
        mt = pred_to_gt.get(t)
        if ms is None and mt is None:
            ignored_pred_edges += 1
            continue
        mapped = (ms, mt)
        if ms is not None and mt is not None and mapped in gt_edge_set:
            edge_tp += 1
            mapped_pred_edges.append(mapped)
            continue
        # Metric FP rule: wrong edge is only penalized if it conflicts with an
        # annotated source or target neighborhood. Edges entirely outside the
        # annotated graph are ignored.
        source_conflict = ms is not None and ms in gt_out
        target_conflict = mt is not None and mt in gt_in
        if source_conflict or target_conflict:
            edge_fp += 1
            if ms is not None and mt is not None:
                mapped_pred_edges.append(mapped)
        else:
            ignored_pred_edges += 1

    edge_fn = max(0, len(gt_edge_set) - edge_tp)

    node_tp = len(pred_to_gt)
    pred_n, gt_n = len(pred_nodes), len(gt_nodes)
    pred_e, gt_e = len(pred_edge_set), len(gt_edge_set)
    t_true_est = estimated_true_nodes_from_geff(dataset, fallback_gt_nodes=gt_n)

    node_precision = node_tp / pred_n if pred_n else 0.0
    node_recall = node_tp / gt_n if gt_n else 0.0
    node_f1 = 2 * node_precision * node_recall / (node_precision + node_recall) if (node_precision + node_recall) else 0.0

    # The visible/practice GEFF annotations are sparse: most real cells are not marked.
    # Metric-style edge Jaccard penalizes only edges that conflict with annotated GT
    # neighborhoods, while ignoring edges outside annotated regions.
    strict_edge_jaccard_debug = edge_tp / (gt_e + pred_e - edge_tp) if (gt_e + pred_e - edge_tp) else 0.0
    metric_edge_jaccard_proxy = (
        edge_tp / (edge_tp + edge_fp + edge_fn)
        if (edge_tp + edge_fp + edge_fn)
        else 0.0
    )
    strict_edge_precision_debug = edge_tp / pred_e if pred_e else 0.0
    sparse_edge_recall = edge_tp / gt_e if gt_e else 0.0
    node_penalty_factor = max(0.0, 1.0 - NODE_PENALTY_A * max(0, pred_n - t_true_est) / max(1, t_true_est))
    adjusted_edge_jaccard_proxy = metric_edge_jaccard_proxy * node_penalty_factor

    gt_div_sources = {}
    pred_div_sources = {}
    for s, t in gt_edge_set:
        gt_div_sources[s] = gt_div_sources.get(s, 0) + 1
    for s, t in pred_edge_set:
        pred_div_sources[s] = pred_div_sources.get(s, 0) + 1
    gt_div_count = sum(1 for v in gt_div_sources.values() if v >= 2)
    pred_div_count = sum(1 for v in pred_div_sources.values() if v >= 2)

    return {
        "dataset": dataset,
        "pred_nodes": pred_n,
        "gt_nodes": gt_n,
        "node_tp": node_tp,
        "node_precision": node_precision,
        "node_recall": node_recall,
        "node_f1": node_f1,
        "mean_match_um": float(np.mean(dists)) if dists else math.nan,
        "p95_match_um": float(np.percentile(dists, 95)) if dists else math.nan,
        "pred_edges": pred_e,
        "gt_edges": gt_e,
        "edge_tp": edge_tp,
        "edge_fp_metric": edge_fp,
        "edge_fn_metric": edge_fn,
        "ignored_pred_edges_metric": ignored_pred_edges,
        "sparse_edge_recall": sparse_edge_recall,
        "metric_edge_jaccard_proxy": metric_edge_jaccard_proxy,
        "estimated_true_nodes": t_true_est,
        "node_penalty_factor_proxy": node_penalty_factor,
        "adjusted_edge_jaccard_proxy": adjusted_edge_jaccard_proxy,
        "strict_edge_precision_debug": strict_edge_precision_debug,
        "strict_edge_jaccard_debug": strict_edge_jaccard_debug,
        "pred_division_sources": pred_div_count,
        "gt_division_sources": gt_div_count,
    }


def annotate_dense_short7_components(submission_df: pd.DataFrame) -> None:
    if not SHORT7_COMPONENTS_PATH.exists() or not SHORT7_NODES_PATH.exists():
        print("[short7] component diagnostics not found; skipping GT annotation")
        return
    components = pd.read_csv(SHORT7_COMPONENTS_PATH)
    component_nodes = pd.read_csv(SHORT7_NODES_PATH)
    if components.empty or component_nodes.empty:
        print("[short7] no dense seven-node components found")
        return

    for column, default in (
        ("visible_gt_matched_nodes", 0),
        ("visible_gt_edge_tp", 0),
        ("visible_gt_edge_fp", 0),
        ("visible_gt_mean_match_um", np.nan),
        ("contains_visible_gt", 0),
    ):
        components[column] = default

    for dataset in sorted(set(components["dataset"].astype(str))):
        gt_path = TRAIN_DIR / f"{dataset}.geff"
        if not gt_path.exists():
            continue
        full_pred_nodes, _ = load_pred(submission_df, dataset)
        gt_nodes, gt_edges = load_gt(dataset)
        pred_to_gt, _, _ = match_nodes(full_pred_nodes, gt_nodes)
        gt_edge_set = set(gt_edges)
        gt_out: dict[int, set[int]] = {}
        gt_in: dict[int, set[int]] = {}
        for source_id, target_id in gt_edge_set:
            gt_out.setdefault(source_id, set()).add(target_id)
            gt_in.setdefault(target_id, set()).add(source_id)

        dataset_components = components.index[components["dataset"].astype(str) == dataset]
        for idx in dataset_components:
            component_id = str(components.at[idx, "component_id"])
            node_rows = component_nodes[component_nodes["component_id"].astype(str) == component_id]
            component_node_ids = {int(value) for value in node_rows["node_id"].tolist()}
            matched_node_ids = [node_id for node_id in component_node_ids if node_id in pred_to_gt]
            match_distances = []
            for pred_id in matched_node_ids:
                gt_id = pred_to_gt[pred_id]
                pred_pos = node_xyz_um(full_pred_nodes, [pred_id])[0]
                gt_pos = node_xyz_um(gt_nodes, [gt_id])[0]
                match_distances.append(float(np.linalg.norm(pred_pos - gt_pos)))

            edge_tp = 0
            edge_fp = 0
            edge_pairs_text = components.at[idx, "edge_pairs"]
            if pd.notna(edge_pairs_text):
                for pair in str(edge_pairs_text).split(";"):
                    if not pair or ":" not in pair:
                        continue
                    source_text, target_text = pair.split(":", 1)
                    source_id, target_id = int(source_text), int(target_text)
                    mapped_source = pred_to_gt.get(source_id)
                    mapped_target = pred_to_gt.get(target_id)
                    if mapped_source is not None and mapped_target is not None and (mapped_source, mapped_target) in gt_edge_set:
                        edge_tp += 1
                    elif (mapped_source is not None and mapped_source in gt_out) or (mapped_target is not None and mapped_target in gt_in):
                        edge_fp += 1

            components.at[idx, "visible_gt_matched_nodes"] = len(matched_node_ids)
            components.at[idx, "visible_gt_edge_tp"] = edge_tp
            components.at[idx, "visible_gt_edge_fp"] = edge_fp
            components.at[idx, "visible_gt_mean_match_um"] = float(np.mean(match_distances)) if match_distances else np.nan
            components.at[idx, "contains_visible_gt"] = int(bool(matched_node_ids or edge_tp or edge_fp))

    components.to_csv(SHORT7_COMPONENTS_PATH, index=False)
    print(f"[short7] annotated {len(components):,} components -> {SHORT7_COMPONENTS_PATH}")
    print(components[["contains_visible_gt", "visible_gt_matched_nodes", "visible_gt_edge_tp", "visible_gt_edge_fp"]].sum().to_string())


def main():
    if not SUBMISSION_PATH.exists():
        raise FileNotFoundError(f"submission.csv not found: {SUBMISSION_PATH}")
    df = pd.read_csv(SUBMISSION_PATH)
    datasets = sorted(set(df["dataset"].astype(str)))
    rows = []
    for dataset in datasets:
        gt_path = TRAIN_DIR / f"{dataset}.geff"
        if not gt_path.exists():
            print(f"[skip] {dataset}: no local GT at {gt_path}")
            continue
        pred_nodes, pred_edges = load_pred(df, dataset)
        row = eval_dataset(dataset, pred_nodes, pred_edges)
        rows.append(row)
        print(
            f"{dataset}: node_recall={row['node_recall']:.4f} "
            f"metric_edge_jaccard={row['metric_edge_jaccard_proxy']:.4f} "
            f"adjusted_edge_jaccard={row['adjusted_edge_jaccard_proxy']:.4f} "
            f"sparse_edge_recall={row['sparse_edge_recall']:.4f} "
            f"FP={row['edge_fp_metric']} FN={row['edge_fn_metric']} "
            f"mean_match_um={row['mean_match_um']:.3f}"
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        total = {
            "dataset": "TOTAL",
            "pred_nodes": int(out["pred_nodes"].sum()),
            "gt_nodes": int(out["gt_nodes"].sum()),
            "node_tp": int(out["node_tp"].sum()),
            "pred_edges": int(out["pred_edges"].sum()),
            "gt_edges": int(out["gt_edges"].sum()),
            "edge_tp": int(out["edge_tp"].sum()),
            "edge_fp_metric": int(out["edge_fp_metric"].sum()),
            "edge_fn_metric": int(out["edge_fn_metric"].sum()),
            "ignored_pred_edges_metric": int(out["ignored_pred_edges_metric"].sum()),
            "estimated_true_nodes": int(out["estimated_true_nodes"].sum()),
            "pred_division_sources": int(out["pred_division_sources"].sum()),
            "gt_division_sources": int(out["gt_division_sources"].sum()),
        }
        total["node_precision"] = total["node_tp"] / total["pred_nodes"] if total["pred_nodes"] else 0.0
        total["node_recall"] = total["node_tp"] / total["gt_nodes"] if total["gt_nodes"] else 0.0
        total["node_f1"] = (
            2 * total["node_precision"] * total["node_recall"] / (total["node_precision"] + total["node_recall"])
            if (total["node_precision"] + total["node_recall"])
            else 0.0
        )
        total["strict_edge_precision_debug"] = total["edge_tp"] / total["pred_edges"] if total["pred_edges"] else 0.0
        total["sparse_edge_recall"] = total["edge_tp"] / total["gt_edges"] if total["gt_edges"] else 0.0
        total["metric_edge_jaccard_proxy"] = (
            total["edge_tp"] / (total["edge_tp"] + total["edge_fp_metric"] + total["edge_fn_metric"])
            if (total["edge_tp"] + total["edge_fp_metric"] + total["edge_fn_metric"])
            else 0.0
        )
        sample_weights = (
            out["edge_tp"].astype(float)
            + out["edge_fp_metric"].astype(float)
            + out["edge_fn_metric"].astype(float)
        )
        adjusted_weight_sum = float(sample_weights.sum())
        aggregate_node_penalty_debug = max(
            0.0,
            1.0
            - NODE_PENALTY_A
            * max(0, total["pred_nodes"] - total["estimated_true_nodes"])
            / max(1, total["estimated_true_nodes"]),
        )
        total["node_penalty_factor_proxy"] = (
            float(np.average(out["node_penalty_factor_proxy"].astype(float), weights=sample_weights))
            if adjusted_weight_sum > 0
            else 0.0
        )
        total["adjusted_edge_jaccard_proxy"] = (
            float(np.average(out["adjusted_edge_jaccard_proxy"].astype(float), weights=sample_weights))
            if adjusted_weight_sum > 0
            else 0.0
        )
        total["adjusted_edge_weight_sum"] = adjusted_weight_sum
        total["aggregate_node_penalty_debug"] = aggregate_node_penalty_debug
        total["aggregate_adjusted_edge_jaccard_debug"] = total["metric_edge_jaccard_proxy"] * aggregate_node_penalty_debug
        total["strict_edge_jaccard_debug"] = (
            total["edge_tp"] / (total["gt_edges"] + total["pred_edges"] - total["edge_tp"])
            if (total["gt_edges"] + total["pred_edges"] - total["edge_tp"])
            else 0.0
        )
        total["mean_match_um"] = np.nan
        total["p95_match_um"] = np.nan
        out = pd.concat([out, pd.DataFrame([total])], ignore_index=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    annotate_dense_short7_components(df)
    print(f"\nwrote {OUT_PATH}")
    try:
        from IPython.display import display

        display(out)
    except Exception:
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
