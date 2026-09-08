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
from biohub.models.temporal_unet import TemporalUNet3D, unet_in_channels

_DEFAULT_CONFIG: dict[str, Any] = {
    'unet_out_channels': 32,
    'unet_layers': [32, 64, 128],
    'downsample': [1, 4, 4],
    'window_size': 2,
    'hidden_dim': 128,
    'n_heads': 4,
    'n_blocks': 4,
    'dropout': 0.3,
    'pos_feat_dim': 4 * POS_EMBED_DIM,
    'mlp_ratio': 2.0,
    'pair_chunk_size': 32,
    'skip_fullres_temporal': True,
    'unet_n_heads': 4,
    'drop_path': 0.0,
    'use_self_attn': False,
    'norm': 'layernorm',
    'rel_coord_scale': 100.0,
    'pair_head': 'mlp',
    'layer_scale_init': 0.0,
    'ffn_act': 'gelu',
    'attn_dropout': 0.3,
    'drop_path_decay': False,
    'pair_geom': 'rel',
    'se_ratio': 0.0,
    'unet_block': 'plain',
    'unet_norm': 'batchnorm',
    'unet_gn_groups': 8,
    'unet_deform': False,
    'temporal_mix': 'attn',
    'coord_kind': 'none',
    'fourier_bands': 4,
    'flow_input': 'none',
    'extra_encoder': 'none',
    'extra_encoder_channels': 8,
    'extra_encoder_freeze': True,
    'extra_encoder_weights': None,
    'feature_sample': 'nearest',
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
    hidden_dim = int(config.get('hidden_dim', 128))
    n_heads = int(config.get('n_heads', 4))
    n_blocks = int(config.get('n_blocks', 4))
    dropout = float(config.get('dropout', 0.3))
    pos_feat_dim = int(config.get('pos_feat_dim', 4 * POS_EMBED_DIM))
    mlp_ratio = float(config.get('mlp_ratio', 2.0))
    raw_chunk = config.get('pair_chunk_size', 32)
    pair_chunk_size = None if raw_chunk in (None, 0) else int(raw_chunk)
    skip_fullres_temporal = bool(config.get('skip_fullres_temporal', True))
    unet_n_heads = int(config.get('unet_n_heads', 4))
    drop_path = float(config.get('drop_path', 0.0))
    use_self_attn = bool(config.get('use_self_attn', False))
    norm = str(config.get('norm', 'layernorm'))
    rel_coord_scale = float(config.get('rel_coord_scale', 100.0))
    pair_head = str(config.get('pair_head', 'mlp'))
    layer_scale_init = float(config.get('layer_scale_init', 0.0))
    ffn_act = str(config.get('ffn_act', 'gelu'))
    attn_dropout = float(config.get('attn_dropout', dropout))
    drop_path_decay = bool(config.get('drop_path_decay', False))
    pair_geom = str(config.get('pair_geom', 'rel'))
    se_ratio = float(config.get('se_ratio', 0.0))
    unet_block = str(config.get('unet_block', 'plain'))
    unet_norm = str(config.get('unet_norm', 'batchnorm'))
    unet_gn_groups = int(config.get('unet_gn_groups', 8))
    unet_deform = bool(config.get('unet_deform', False))
    temporal_mix = str(config.get('temporal_mix', 'attn'))
    coord_kind = str(config.get('coord_kind', 'none'))
    fourier_bands = int(config.get('fourier_bands', 4))
    flow_input = str(config.get('flow_input', 'none'))
    extra_encoder = str(config.get('extra_encoder', 'none'))
    extra_encoder_channels = int(config.get('extra_encoder_channels', 8))
    extra_encoder_freeze = bool(config.get('extra_encoder_freeze', True))
    extra_weights = config.get('extra_encoder_weights')
    extra_encoder_weights = None if extra_weights in (None, '', 0) else str(extra_weights)
    feature_sample = str(config.get('feature_sample', 'nearest'))

    unet = TemporalUNet3D(
        in_channels=unet_in_channels(
            coord_kind=coord_kind,
            fourier_bands=fourier_bands,
            flow_input=flow_input,
            extra_encoder=extra_encoder,
            extra_encoder_channels=extra_encoder_channels,
        ),
        out_channels=out_channels,
        layers=layers,
        skip_fullres_temporal=skip_fullres_temporal,
        temporal_n_heads=unet_n_heads,
        se_ratio=se_ratio,
        unet_block=unet_block,
        unet_norm=unet_norm,
        unet_gn_groups=unet_gn_groups,
        unet_deform=unet_deform,
        temporal_mix=temporal_mix,
    )
    model = UNetNodeTransformer(
        unet=unet,
        unet_out_channels=out_channels,
        pos_feat_dim=pos_feat_dim,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        n_blocks=n_blocks,
        dropout=dropout,
        mlp_ratio=mlp_ratio,
        pair_chunk_size=pair_chunk_size,
        drop_path=drop_path,
        use_self_attn=use_self_attn,
        norm=norm,
        rel_coord_scale=rel_coord_scale,
        pair_head=pair_head,
        layer_scale_init=layer_scale_init,
        ffn_act=ffn_act,
        attn_dropout=attn_dropout,
        drop_path_decay=drop_path_decay,
        pair_geom=pair_geom,
        feature_sample=feature_sample,
        coord_kind=coord_kind,
        fourier_bands=fourier_bands,
        flow_input=flow_input,
        extra_encoder=extra_encoder,
        extra_encoder_channels=extra_encoder_channels,
        extra_encoder_freeze=extra_encoder_freeze,
        extra_encoder_weights=extra_encoder_weights,
    )
    state = torch.load(weights_path, map_location=device, weights_only=True)
    saved_arch = state.get('_arch')
    if saved_arch is not None:
        saved = [int(value) for value in saved_arch.detach().cpu().tolist()]
        constructed = [hidden_dim, n_heads, n_blocks]
        if saved != constructed:
            raise RuntimeError(
                f'Detector architecture mismatch: checkpoint {saved} vs config {constructed}'
            )
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = {'_arch', 'offset_head.weight', 'offset_head.bias'}
    bad_missing = [key for key in missing if key not in allowed_missing]
    if bad_missing or unexpected:
        raise RuntimeError(
            f'Detector state_dict mismatch missing={bad_missing} unexpected={list(unexpected)}'
        )
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
