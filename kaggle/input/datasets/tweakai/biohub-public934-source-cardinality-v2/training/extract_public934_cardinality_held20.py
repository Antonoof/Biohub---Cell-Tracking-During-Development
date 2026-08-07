#!/usr/bin/env python3
"""Score the full held-20 source/pair population with the option head.

The compact event cache is never used as the inference population here.  Each
video's complete V2 14/20-um candidate set is regenerated, aligned to Model C
and native P1/P2 evidence, and reduced only after the grouped option head has
scored every daughter pair.  Persisting one winner and one calibrated source
probability keeps the replay cache small without changing inference.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

import extract_ctc_full_population_source_bank as common
import train_model_c_v2_event_decoder as decoder


WORKSPACE = Path("/mnt/c/Users/sk8fu/Documents/Codex/2026-07-01/c")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("/home/tweak/bio/public914_backbone_matched_v1"))
    p.add_argument("--artifact", type=Path, default=WORKSPACE / "artifacts/biohub-model-c-combined-primary-v1")
    p.add_argument("--head", type=Path, default=Path("/home/tweak/bio/public934_source_cardinality_head_v2/source_cardinality_head.pt"))
    p.add_argument("--split", type=Path, default=Path("/home/tweak/bio/division_balanced_175_20_split.json"))
    p.add_argument("--data", type=Path, default=Path("/home/tweak/bio/train"))
    p.add_argument("--output", type=Path, default=Path("/home/tweak/bio/public934_source_cardinality_held20_population_v2"))
    p.add_argument("--worker-id", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def evidence_path(stem: str, roots: tuple[Path, ...]) -> Path:
    matches = [root / f"{stem}.npz" for root in roots if (root / f"{stem}.npz").exists()]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one evidence file for {stem}, found {len(matches)}")
    return matches[0]


def load_head(path: Path):
    head_module = load_module(
        "public934_cardinality_head_model",
        WORKSPACE / "scripts/train_public934_source_cardinality_head.py",
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload["config"]
    model = head_module.OptionHead(
        int(config["source_dim"]), int(config["pair_dim"]),
        int(config["hidden_source"]), int(config["hidden_pair"]),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload


def cap_pair_options(
    owner: np.ndarray,
    max_pairs: int,
) -> np.ndarray:
    """Return original row indices for the first N stable rows per source."""
    order = np.argsort(owner, kind="stable")
    if not len(order):
        return order
    sorted_owner = owner[order]
    starts = np.flatnonzero(np.r_[True, sorted_owner[1:] != sorted_owner[:-1]])
    ends = np.r_[starts[1:], len(order)]
    keep = np.concatenate(
        [order[left : min(right, left + max_pairs)] for left, right in zip(starts, ends)]
    )
    return np.sort(keep)


@torch.no_grad()
def option_scores(
    model,
    payload: dict,
    source_x: np.ndarray,
    pair_blocks: list[np.ndarray],
    owner: np.ndarray,
    batch: int = 250_000,
) -> tuple[np.ndarray, np.ndarray]:
    source_mean = np.asarray(payload["source_mean"], np.float32)
    source_scale = np.asarray(payload["source_scale"], np.float32)
    pair_mean = np.asarray(payload["pair_mean"], np.float32)
    pair_scale = np.asarray(payload["pair_scale"], np.float32)
    source_norm = np.clip((source_x - source_mean) / source_scale, -10.0, 10.0)
    source_h = model.source_encoder(torch.from_numpy(source_norm)).cpu()
    continue_logit = model.continue_head(source_h).squeeze(-1).numpy()
    pair_logits = np.empty(len(owner), np.float32)
    for left in range(0, len(owner), batch):
        right = min(len(owner), left + batch)
        block = np.concatenate([value[left:right] for value in pair_blocks], axis=1)
        block = np.clip((block - pair_mean) / pair_scale, -10.0, 10.0)
        pair_h = model.pair_encoder(torch.from_numpy(block))
        source_block = source_h[torch.from_numpy(owner[left:right].astype(np.int64))]
        value = model.divide_head(torch.cat([source_block, pair_h], dim=1)).squeeze(-1)
        pair_logits[left:right] = (value + model.division_bias).numpy()

    n_sources = len(source_x)
    maximum = np.full(n_sources, -np.inf, np.float32)
    np.maximum.at(maximum, owner, pair_logits)
    total = np.zeros(n_sources, np.float64)
    valid_pair = np.isfinite(maximum[owner])
    np.add.at(total, owner[valid_pair], np.exp(pair_logits[valid_pair] - maximum[owner[valid_pair]]))
    log_total = np.full(n_sources, -np.inf, np.float32)
    valid_source = total > 0
    log_total[valid_source] = maximum[valid_source] + np.log(total[valid_source]).astype(np.float32)
    delta = np.clip(log_total - continue_logit, -40.0, 40.0)
    probability = np.zeros(n_sources, np.float32)
    probability[valid_source] = 1.0 / (1.0 + np.exp(-delta[valid_source]))
    best = decoder.grouped_best_pair(owner, pair_logits, n_sources)
    return probability, best


def process(stem: str, args, v2, model, payload) -> dict:
    destination = args.output / stem
    manifest = destination / "manifest.json"
    if manifest.exists() and not args.overwrite:
        value = json.loads(manifest.read_text())
        if value.get("complete"):
            return value
    temporary = args.output / f".{stem}.worker{args.worker_id}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    started = time.time()

    graphs = args.root / "division_candidate_audit_v1" / "pre_safe_graphs"
    shifts = args.root / "division_candidate_audit_v1" / "registration_shifts"
    nodes, edges = common.graph_dicts(graphs / f"{stem}.geff")
    scored = v2.score(
        args.data / f"{stem}.zarr", nodes, edges,
        common.load_shifts(shifts / f"{stem}.csv"),
        (1.625, 0.40625, 0.40625),
    )
    source_ids = np.asarray(scored["source_ids"], np.int64)
    source_x = np.asarray(scored["_source_x"], np.float32)
    pair_x = np.asarray(scored["_pair_x"], np.float32)
    owner = np.asarray(scored["_pair_owner"], np.int32)
    pair_nodes = np.asarray(scored["pair_nodes"], np.int64)
    all_pair_count = len(owner)
    max_pairs = int(payload["config"]["max_pairs"])
    retained = cap_pair_options(owner, max_pairs)
    pair_x = pair_x[retained]
    owner = owner[retained]
    pair_nodes = pair_nodes[retained]
    graph_coordinates = decoder.load_graph_coordinates(graphs / f"{stem}.geff")

    c_roots = (
        args.root / "model_c_native_evidence_train175",
        args.root / "model_c_native_evidence_held20",
        args.root / "model_c_native_evidence_practice4",
    )
    c_x = decoder.model_c_features(
        evidence_path(stem, c_roots), graph_coordinates, source_ids, owner,
        pair_nodes[:, 0], pair_nodes[:, 1],
    )
    p1_x = decoder.model_c_features(
        args.root / "public_primary_native_evidence_all199_v1" / f"{stem}.npz",
        graph_coordinates, source_ids, owner, pair_nodes[:, 0], pair_nodes[:, 1],
    )
    p2_x = decoder.model_c_features(
        args.root / "public_secondary_native_evidence_all199_v1" / f"{stem}.npz",
        graph_coordinates, source_ids, owner, pair_nodes[:, 0], pair_nodes[:, 1],
    )
    score, best = option_scores(model, payload, source_x, [pair_x, c_x, p1_x, p2_x], owner)
    best_nodes = np.full((len(source_ids), 2), -1, np.int64)
    valid = best >= 0
    best_nodes[valid] = pair_nodes[best[valid]]

    tube_root = args.root / "public914_source_tubes" / stem
    tube_ids = np.load(tube_root / "source_id.npy")
    tube_values = np.load(tube_root / "source_tube.npy")
    order = np.argsort(tube_ids, kind="stable")
    sorted_ids = tube_ids[order]
    where = np.searchsorted(sorted_ids, source_ids)
    present = where < len(sorted_ids)
    present &= sorted_ids[np.minimum(where, len(sorted_ids) - 1)] == source_ids
    if not bool(np.all(present)):
        raise RuntimeError(
            f"Public source/tube ID alignment failed for {stem}: "
            f"missing={int(np.count_nonzero(~present))}/{len(source_ids)}"
        )
    tubes = tube_values[order[where]].astype(np.int64, copy=False)
    for name, value in (
        ("source_id", source_ids), ("source_tube", tubes),
        ("source_score", score), ("best_pair_nodes", best_nodes),
    ):
        common.atomic_save(temporary / f"{name}.npy", value)
    result = {
        "version": "public934-source-cardinality-held20-population-v1",
        "dataset": stem,
        "complete": True,
        "sources": int(len(source_ids)),
        "pairs": int(len(owner)),
        "pairs_before_cap": int(all_pair_count),
        "max_pairs_per_source": max_pairs,
        "threshold": float(payload["config"]["threshold"]),
        "threshold_passing_sources": int(np.count_nonzero(score >= float(payload["config"]["threshold"]))),
        "seconds": time.time() - started,
    }
    (temporary / "manifest.json").write_text(json.dumps(result, indent=2))
    if destination.exists():
        shutil.rmtree(destination)
    temporary.replace(destination)
    return result


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    split = json.loads(args.split.read_text())
    stems = list(map(str, split["held"]))[args.worker_id :: args.workers]
    runtime_module = load_module(
        f"public934_cardinality_v2_runtime_{args.worker_id}",
        args.artifact / "division_gbm_runtime.py",
    )
    v2 = runtime_module.DivisionGBMRuntime(args.artifact)
    model, payload = load_head(args.head)
    rows = []
    for index, stem in enumerate(stems, 1):
        row = process(stem, args, v2, model, payload)
        rows.append(row)
        print(
            f"[{index:2d}/{len(stems)}] {stem}: sources={row['sources']:,} "
            f"pairs={row['pairs']:,} pass={row['threshold_passing_sources']:,} "
            f"time={row['seconds']/60:.1f}m",
            flush=True,
        )
    (args.output / f"worker_{args.worker_id}.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
