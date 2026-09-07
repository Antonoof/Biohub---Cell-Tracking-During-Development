from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import tracksdata as td

from biohub.features.decoder import (
    C_FEATURE_NAMES,
    CTC_FEATURE_NAMES,
    PUBLIC_EVIDENCE_LABELS,
    model_c_features,
)


@dataclass
class VideoRows:
    stem: str
    embryo: str
    source_x: np.ndarray
    source_y: np.ndarray
    source_node: np.ndarray
    source_tube: np.ndarray
    source_event: np.ndarray
    pair_x: np.ndarray
    pair_y: np.ndarray
    pair_owner: np.ndarray
    pair_a: np.ndarray
    pair_b: np.ndarray
    c_pair_x: np.ndarray
    ctc_pair_x: np.ndarray
    public_pair_x: np.ndarray
    event_total: int


def source_tubes(full_cache: Path, stem: str, source_nodes: np.ndarray) -> np.ndarray:
    root = full_cache / stem
    ids = np.load(root / 'source_id.npy', mmap_mode='r')
    tubes = np.load(root / 'source_tube.npy', mmap_mode='r')
    order = np.argsort(ids, kind='stable')
    sorted_ids = np.asarray(ids[order])
    where = np.searchsorted(sorted_ids, source_nodes)
    valid = where < len(sorted_ids)
    valid &= sorted_ids[np.minimum(where, len(sorted_ids) - 1)] == source_nodes
    result = source_nodes.astype(np.int64, copy=True)
    result[valid] = np.asarray(tubes[order[where[valid]]], np.int64)
    return result


def load_graph_coordinates(path: Path) -> dict[int, tuple[int, np.ndarray]]:
    loaded = td.graph.IndexedRXGraph.from_geff(path)
    graph = loaded[0] if isinstance(loaded, tuple) else loaded
    attrs = graph.node_attrs(attr_keys=['node_id', 't', 'z', 'y', 'x'])
    return {
        int(row['node_id']): (
            int(row['t']),
            np.asarray([row['z'], row['y'], row['x']], np.float32),
        )
        for row in attrs.iter_rows(named=True)
    }


def load_video(
    stem: str,
    args,
    evidence_root: Path | None,
    cache: dict | None = None,
    graph_coordinates: dict[int, tuple[int, np.ndarray]] | None = None,
) -> VideoRows:
    if cache is None:
        cache = torch.load(args.event_cache / f'{stem}.pt', map_location='cpu', weights_only=False)
    source_node = cache['source_node'].astype(np.int64)
    owner = cache['pair_source_idx'].astype(np.int32)
    needs_coordinates = (
        evidence_root is not None
        or args.public_primary_evidence is not None
        or args.public_secondary_evidence is not None
    )
    if graph_coordinates is None:
        graph_coordinates = (
            load_graph_coordinates(args.graph_dir / f'{stem}.geff') if needs_coordinates else {}
        )
    if evidence_root is None:
        c_pair_x = np.zeros((len(owner), 0), np.float32)
    else:
        latent_node = cache.get('latent_node')
        latent_coords = cache.get('latent_node_coords')
        if latent_node is not None and latent_coords is not None:
            latent_node = np.asarray(latent_node, np.int64)
            latent_coords = np.asarray(latent_coords, np.float32)
            if latent_coords.shape != (len(latent_node), 4):
                raise RuntimeError(
                    f'Latent-node coordinate mismatch for {stem}: '
                    f'{latent_node.shape} / {latent_coords.shape}'
                )
            for node, coordinate in zip(latent_node, latent_coords):
                graph_coordinates[int(node)] = (
                    int(coordinate[0]),
                    coordinate[1:].astype(np.float32),
                )
        c_pair_x = model_c_features(
            evidence_root / f'{stem}.npz',
            graph_coordinates,
            source_node,
            owner,
            cache['pair_a'].astype(np.int64),
            cache['pair_b'].astype(np.int64),
        )
    public_roots = (
        args.public_primary_evidence,
        args.public_secondary_evidence,
    )
    if all(root is None for root in public_roots):
        public_pair_x = np.zeros((len(owner), 0), np.float32)
    elif any(root is None for root in public_roots):
        raise ValueError(
            'Public evidence is atomic: both primary and secondary roots must be supplied.'
        )
    else:
        public_blocks = []
        for label, root in zip(PUBLIC_EVIDENCE_LABELS, public_roots):
            path = root / f'{stem}.npz'
            if not path.exists():
                raise FileNotFoundError(f'Missing {label} evidence for {stem}: {path}')
            block = model_c_features(
                path,
                graph_coordinates,
                source_node,
                owner,
                cache['pair_a'].astype(np.int64),
                cache['pair_b'].astype(np.int64),
            )
            if block.shape != (len(owner), len(C_FEATURE_NAMES)):
                raise RuntimeError(
                    f'{stem}: {label} feature shape {block.shape}, expected '
                    f'({len(owner)}, {len(C_FEATURE_NAMES)})'
                )
            public_blocks.append(block)
        public_pair_x = np.concatenate(public_blocks, axis=1)
    if args.ctc_evidence is None:
        ctc_pair_x = np.zeros((len(owner), 0), np.float32)
    else:
        ctc_path = args.ctc_evidence / f'{stem}.npz'
        if not ctc_path.exists():
            raise FileNotFoundError(f'Missing CTC evidence: {ctc_path}')
        with np.load(ctc_path, allow_pickle=False) as ctc:
            ctc_pair_x = ctc['ctc_pair_features'].astype(np.float32)
            ctc_names = list(map(str, ctc['feature_names']))
            if ctc_names != CTC_FEATURE_NAMES:
                raise RuntimeError(f'CTC feature schema mismatch for {stem}: {ctc_names}')
            for key, expected in (
                ('source_node', source_node),
                ('pair_source_idx', owner),
                ('pair_a', cache['pair_a'].astype(np.int64)),
                ('pair_b', cache['pair_b'].astype(np.int64)),
            ):
                if not np.array_equal(ctc[key], expected):
                    raise RuntimeError(f'CTC evidence row alignment failed for {stem}: {key}')
    return VideoRows(
        stem=stem,
        embryo=stem.split('_', 1)[0],
        source_x=np.nan_to_num(cache['source_x'].astype(np.float32), nan=0.0),
        source_y=cache['source_y'].astype(np.int8),
        source_node=source_node,
        source_tube=source_tubes(args.full_population_cache, stem, source_node),
        source_event=cache['source_event'].astype(np.int32),
        pair_x=np.nan_to_num(cache['pair_x'].astype(np.float32), nan=0.0),
        pair_y=cache['pair_y'].astype(np.int8),
        pair_owner=owner,
        pair_a=cache['pair_a'].astype(np.int64),
        pair_b=cache['pair_b'].astype(np.int64),
        c_pair_x=c_pair_x,
        ctc_pair_x=np.nan_to_num(ctc_pair_x, nan=0.0, posinf=0.0, neginf=0.0),
        public_pair_x=np.nan_to_num(public_pair_x, nan=0.0, posinf=0.0, neginf=0.0),
        event_total=len(cache['event_rows']),
    )
