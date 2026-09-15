#!/usr/bin/env python3
"""Materialize Model C / cardinality graph-audit layout from frozen oof_graphs.

Creates:
  runs/oof_graphs/model_c_audit/division_candidate_audit_v1/pre_safe_graphs
  runs/oof_graphs/model_c_audit/division_candidate_audit_v1/registration_shifts
  runs/oof_graphs/model_c_audit/audit_907_v1/metrics_all.csv
  runs/oof_graphs/full_population_cache/<stem>/{source_id,source_tube}.npy

Event-cache parts are built next by train_division_pair_model.py --cache-only.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
import tracksdata as td

HP = Path(__file__).resolve().parents[1]
OUT = HP / "runs/oof_graphs"
MANIFEST = OUT / "manifest.json"
AUDIT = OUT / "model_c_audit"
GRAPHS = AUDIT / "division_candidate_audit_v1" / "pre_safe_graphs"
SHIFTS = AUDIT / "division_candidate_audit_v1" / "registration_shifts"
METRICS = AUDIT / "audit_907_v1" / "metrics_all.csv"
POP = OUT / "full_population_cache"


def _load_graph(path: Path):
    loaded = td.graph.IndexedRXGraph.from_geff(path)
    return loaded[0] if isinstance(loaded, tuple) else loaded


def _components(graph) -> dict[int, int]:
    ids = [int(n) for n in graph.node_ids()]
    parent = {n: n for n in ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edges = graph.edge_attrs()
    for s, t in edges.select(["source_id", "target_id"]).iter_rows():
        a, b = find(int(s)), find(int(t))
        if a != b:
            parent[b] = a
    return {n: find(n) for n in ids}


def main() -> None:
    man = json.loads(MANIFEST.read_text())
    src = Path(man["selected_graph_dir"])
    geffs = sorted(src.glob("*.geff"))
    if not geffs:
        raise SystemExit(f"No GEFFs in {src}")
    GRAPHS.mkdir(parents=True, exist_ok=True)
    SHIFTS.mkdir(parents=True, exist_ok=True)
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    POP.mkdir(parents=True, exist_ok=True)

    stems = []
    for g in geffs:
        dest = GRAPHS / g.name
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        os.symlink(g.resolve(), dest)
        stems.append(g.stem)
        graph = _load_graph(g)
        attrs = graph.node_attrs(attr_keys=["node_id", "t"])
        ids = [int(x) for x in attrs["node_id"].to_list()]
        ts = [int(x) for x in attrs["t"].to_list()]
        tmax = max(ts) if ts else 0
        shift_path = SHIFTS / f"{g.stem}.csv"
        with shift_path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t", "shift_z_um", "shift_y_um", "shift_x_um"])
            for t in range(tmax + 1):
                w.writerow([t, 0.0, 0.0, 0.0])
        comp = _components(graph) if ids else {}
        part = POP / g.stem
        part.mkdir(parents=True, exist_ok=True)
        np.save(part / "source_id.npy", np.asarray(ids, np.int64))
        np.save(part / "source_tube.npy", np.asarray([comp.get(i, i) for i in ids], np.int64))

    with METRICS.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset"])
        w.writeheader()
        for stem in stems:
            w.writerow({"dataset": stem})

    info = {
        "selected_graph_dir": str(src),
        "n_geff": len(stems),
        "pre_safe_graphs": str(GRAPHS),
        "division_audit": str(GRAPHS.parent),
        "full_population_cache": str(POP),
        "event_cache_parts_will_be": str(
            AUDIT / "division_training_cache_v1_parts"
        ),
        "cache_cmd": [
            "train_division_pair_model.py",
            "--cache-only",
            "--rebuild-cache",
            f"--data={man.get('kaggle_train')}",
            f"--division-audit={GRAPHS.parent}",
            f"--cache={AUDIT / 'division_training_cache_v1.npz'}",
        ],
    }
    (AUDIT / "layout.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
