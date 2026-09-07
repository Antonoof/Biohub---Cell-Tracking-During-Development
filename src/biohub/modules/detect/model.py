import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import tracksdata as td

from biohub.features.position import POS_EMBED_DIM
from biohub.models.detector import UNetNodeTransformer
from biohub.models.temporal_unet import TemporalUNet3D

_DEFAULT_CONFIG: dict[str, Any] = {
    'unet_out_channels': 32,
    'unet_layers': [32, 64, 128],
    'downsample': [1, 4, 4],
    'window_size': 2,
}


def build_graph(
    coords: np.ndarray,
    edges: list[tuple[int, int, float, float]],
) -> td.graph.InMemoryGraph:
    graph: Any = td.graph.InMemoryGraph()

    for key in ['z', 'y', 'x']:
        graph.add_node_attr_key(key, pl.Float64, -999999.0)

    node_ids = graph.bulk_add_nodes(
        [{'t': int(t), 'z': float(z), 'y': float(y), 'x': float(x)} for t, z, y, x in coords]
    )

    if edges:
        graph.add_edge_attr_key('edge_prob', pl.Float64, 0.0)
        graph.add_edge_attr_key('edge_dist', pl.Float64, 0.0)
        graph.bulk_add_edges(
            [
                {
                    'source_id': node_ids[src],
                    'target_id': node_ids[tgt],
                    'edge_prob': prob,
                    'edge_dist': dist,
                }
                for src, tgt, prob, dist in edges
            ]
        )

    return graph


def load_model(
    weights_path: Path,
    device: torch.device,
) -> tuple[UNetNodeTransformer, int, tuple[int, ...]]:
    config_path = weights_path.parent / 'config.json'
    config: dict[str, Any] = dict(_DEFAULT_CONFIG)
    if config_path.exists():
        config.update(json.loads(config_path.read_text()))
    else:
        print(f'Warning: config.json not found at {config_path}, using defaults.', flush=True)

    if 'downsample_factor' in config and 'downsample' not in config:
        df = config['downsample_factor']
        config['downsample'] = [df, df, df]

    downsample = tuple(int(v) for v in config['downsample'])
    out_channels = int(config['unet_out_channels'])
    layers = tuple(int(v) for v in config['unet_layers'])

    unet = TemporalUNet3D(
        in_channels=1,
        out_channels=out_channels,
        layers=layers,
    )
    model = UNetNodeTransformer(
        unet=unet,
        unet_out_channels=out_channels,
        pos_feat_dim=4 * POS_EMBED_DIM,
    )
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, int(config['window_size']), downsample


def load_frame(
    zarr_arr: Any,
    t: int,
    target_shape: list[int],
    downsample: tuple[int, ...] = (1, 1, 1),
) -> torch.Tensor:
    dz, dy, dx = downsample
    raw = zarr_arr[t, ::dz, ::dy, ::dx].astype(np.float32)
    frame = torch.from_numpy(raw)
    if list(frame.shape) != target_shape:
        frame = F.interpolate(
            frame[None, None],
            size=target_shape,
            mode='trilinear',
            align_corners=False,
        )[0, 0]
    return frame
