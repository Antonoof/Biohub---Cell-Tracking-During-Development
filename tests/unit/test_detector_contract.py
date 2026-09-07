import json
from pathlib import Path

import pytest
import torch

from biohub.losses.association import compute_batch_loss
from biohub.models.detector import UNetNodeTransformer
from biohub.models.division import DivisionMLP, load_division_checkpoint
from biohub.models.temporal_unet import TemporalUNet3D
from biohub.modules.detect.model import load_model
from biohub.train.detector import (
    build_matched_edge_targets,
    detect_and_match,
    prepare_detector_output,
)
from biohub.train.motion_cache import build_model


def _tiny_detector(
    *, hidden_dim: int = 32, n_heads: int = 4, n_blocks: int = 1
) -> UNetNodeTransformer:
    unet = TemporalUNet3D(
        in_channels=1,
        out_channels=4,
        layers=(8, 16),
        gradient_checkpointing=False,
    )
    model = UNetNodeTransformer(
        unet=unet,
        unet_out_channels=4,
        pos_feat_dim=8,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        n_blocks=n_blocks,
        dropout=0.0,
    )
    model.eval()
    return model


def _edge_logits(
    model: UNetNodeTransformer, imgs: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    unet_out, det_logits = model.encode(imgs)
    b, c, _z, _y, _x = unet_out[:, 0].shape
    src = unet_out[:, 0].reshape(b, c, -1)[:, :, :3].transpose(1, 2)
    tgt = unet_out[:, 1].reshape(b, c, -1)[:, :, :4].transpose(1, 2)
    coords_src = torch.zeros(b, 3, 3)
    coords_tgt = torch.zeros(b, 4, 3)
    pos_src = torch.zeros(b, 3, 8)
    pos_tgt = torch.zeros(b, 4, 8)
    mask_src = torch.ones(b, 3, dtype=torch.bool)
    mask_tgt = torch.ones(b, 4, dtype=torch.bool)
    edges = model.predict_edges(
        src, tgt, coords_src, coords_tgt, pos_src, pos_tgt, mask_src, mask_tgt
    )
    return det_logits[0], edges


def _write_detector(tmp_path: Path, model: UNetNodeTransformer, config: dict) -> Path:
    weights = tmp_path / 'edge_predictor_best.pth'
    torch.save(model.state_dict(), weights)
    (tmp_path / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    return weights


@torch.no_grad()
def test_load_model_restores_nondefault_heads(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = _tiny_detector(n_heads=8)
    imgs = torch.rand(1, 2, 4, 8, 8)
    det, edges = _edge_logits(model, imgs)
    config = {
        'unet_out_channels': 4,
        'unet_layers': [8, 16],
        'downsample': [1, 1, 1],
        'window_size': 2,
        'hidden_dim': 32,
        'n_heads': 8,
        'n_blocks': 1,
        'dropout': 0.0,
        'pos_feat_dim': 8,
    }
    loaded, window, _downsample = load_model(
        _write_detector(tmp_path, model, config), torch.device('cpu')
    )
    assert window == 2
    loaded_det, loaded_edges = _edge_logits(loaded, imgs)
    assert torch.allclose(det, loaded_det, atol=1e-5, rtol=1e-5)
    assert torch.allclose(edges, loaded_edges, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_load_model_restores_nondefault_hidden_dim(tmp_path: Path) -> None:
    torch.manual_seed(1)
    model = _tiny_detector(hidden_dim=64, n_heads=4)
    imgs = torch.rand(1, 2, 4, 8, 8)
    det, edges = _edge_logits(model, imgs)
    config = {
        'unet_out_channels': 4,
        'unet_layers': [8, 16],
        'downsample': [1, 1, 1],
        'window_size': 2,
        'hidden_dim': 64,
        'n_heads': 4,
        'n_blocks': 1,
        'dropout': 0.0,
        'pos_feat_dim': 8,
    }
    loaded, _window, _downsample = load_model(
        _write_detector(tmp_path, model, config), torch.device('cpu')
    )
    loaded_det, loaded_edges = _edge_logits(loaded, imgs)
    assert torch.allclose(det, loaded_det, atol=1e-5, rtol=1e-5)
    assert torch.allclose(edges, loaded_edges, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_load_model_rejects_eight_head_weights_with_default_config(tmp_path: Path) -> None:
    torch.manual_seed(2)
    model = _tiny_detector(n_heads=8)
    config = {
        'unet_out_channels': 4,
        'unet_layers': [8, 16],
        'downsample': [1, 1, 1],
        'window_size': 2,
        'hidden_dim': 32,
        'n_blocks': 1,
        'dropout': 0.0,
        'pos_feat_dim': 8,
    }
    weights = _write_detector(tmp_path, model, config)
    with pytest.raises(RuntimeError, match='architecture mismatch'):
        load_model(weights, torch.device('cpu'))


@torch.no_grad()
def test_division_checkpoint_roundtrip(tmp_path: Path) -> None:
    torch.manual_seed(3)
    source = DivisionMLP(6, (32, 16), 0.0, 0.0)
    pair = DivisionMLP(5, (24, 12), 0.0, 0.0)
    payload = {
        'source_model': source.state_dict(),
        'pair_model': pair.state_dict(),
        'source_hidden': [32, 16],
        'pair_hidden': [24, 12],
        'dropout_1': 0.0,
        'dropout_2': 0.0,
    }
    path = tmp_path / 'division_pair_model_best.pt'
    torch.save(payload, path)
    features_s = torch.randn(4, 6)
    features_p = torch.randn(4, 5)
    loaded = load_division_checkpoint(path)
    assert torch.allclose(source(features_s), loaded['source_model'](features_s), atol=1e-5)
    assert torch.allclose(pair(features_p), loaded['pair_model'](features_p), atol=1e-5)


@torch.no_grad()
def test_division_checkpoint_rejects_hardcoded_hidden(tmp_path: Path) -> None:
    torch.manual_seed(4)
    source = DivisionMLP(6, (32, 16), 0.0, 0.0)
    pair = DivisionMLP(5, (24, 12), 0.0, 0.0)
    payload = {
        'source_model': source.state_dict(),
        'pair_model': pair.state_dict(),
        'source_hidden': [96, 48],
        'pair_hidden': [128, 64],
        'dropout_1': 0.0,
        'dropout_2': 0.0,
    }
    path = tmp_path / 'division_pair_model_best.pt'
    torch.save(payload, path)
    with pytest.raises(RuntimeError):
        load_division_checkpoint(path)


@torch.no_grad()
def test_motion_cache_build_model_restores_nondefault_heads(tmp_path: Path) -> None:
    torch.manual_seed(5)
    model = _tiny_detector(n_heads=8)
    imgs = torch.rand(1, 2, 4, 8, 8)
    det, edges = _edge_logits(model, imgs)
    config = {
        'unet_out_channels': 4,
        'unet_layers': [8, 16],
        'downsample': [1, 1, 1],
        'window_size': 2,
        'hidden_dim': 32,
        'n_heads': 8,
        'n_blocks': 1,
        'dropout': 0.0,
        'pos_feat_dim': 8,
    }
    loaded = build_model(_write_detector(tmp_path, model, config), 'cpu')
    loaded_det, loaded_edges = _edge_logits(loaded, imgs)
    assert torch.allclose(det, loaded_det, atol=1e-5, rtol=1e-5)
    assert torch.allclose(edges, loaded_edges, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_motion_cache_build_model_allows_missing_arch(tmp_path: Path) -> None:
    torch.manual_seed(6)
    model = _tiny_detector()
    state = model.state_dict()
    state.pop('_arch', None)
    weights = tmp_path / 'edge_predictor_best.pth'
    torch.save(state, weights)
    (tmp_path / 'config.json').write_text(
        json.dumps(
            {
                'unet_out_channels': 4,
                'unet_layers': [8, 16],
                'downsample': [1, 1, 1],
                'window_size': 2,
                'hidden_dim': 32,
                'n_heads': 4,
                'n_blocks': 1,
                'dropout': 0.0,
                'pos_feat_dim': 8,
            },
            indent=2,
        )
        + '\n'
    )
    loaded = build_model(weights, 'cpu')
    imgs = torch.rand(1, 2, 4, 8, 8)
    _edge_logits(loaded, imgs)


def test_empty_target_frame_train_step_has_finite_grads() -> None:
    torch.manual_seed(7)
    unet = TemporalUNet3D(
        in_channels=1,
        out_channels=4,
        layers=(8, 16),
        gradient_checkpointing=False,
    )
    model = UNetNodeTransformer(
        unet=unet,
        unet_out_channels=4,
        pos_feat_dim=32,
        hidden_dim=32,
        n_heads=4,
        n_blocks=1,
        dropout=0.0,
    )
    model.train()
    imgs = torch.rand(2, 2, 4, 8, 8)
    unet_out, det_logits = model.encode(imgs)
    gt_coords = torch.zeros(2, 1, 3)
    gt_mask = torch.ones(2, 1, dtype=torch.bool)
    image_shape = (2, 4, 8, 8)
    voxel = (1.0, 1.0, 1.0)
    src = detect_and_match(
        det_logits[0],
        gt_coords,
        gt_mask,
        image_shape,
        voxel_size=voxel,
        frame_index=0,
        window_size=2,
    )
    empty_logits = torch.full_like(det_logits[1], -10.0)
    tgt = detect_and_match(
        empty_logits,
        gt_coords,
        gt_mask,
        image_shape,
        voxel_size=voxel,
        frame_index=1,
        window_size=2,
    )
    feat_src = model.index_features(unet_out[:, 0], src[0], src[2])
    feat_tgt = model.index_features(unet_out[:, 1], tgt[0], tgt[2])
    edge_logits = model.predict_edges(
        feat_src,
        feat_tgt,
        src[0],
        tgt[0],
        src[1],
        tgt[1],
        src[2],
        tgt[2],
    )
    assert torch.isfinite(edge_logits).all()
    targets = torch.zeros(2, 1, 1)
    pair_target = build_matched_edge_targets(
        src[3], tgt[3], targets, src[0].shape[1], tgt[0].shape[1]
    )
    loss = compute_batch_loss(edge_logits, pair_target, src[2], tgt[2])
    assert torch.isfinite(loss).all()
    loss.backward()
    for param in model.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all()


def test_prepare_detector_output_refuses_nonempty_without_overwrite(tmp_path: Path) -> None:
    occupied = tmp_path / 'split_0'
    occupied.mkdir()
    (occupied / 'edge_predictor_best.pth').write_bytes(b'x')
    with pytest.raises(RuntimeError, match='Refusing to overwrite'):
        prepare_detector_output(occupied, overwrite=False)
    prepare_detector_output(occupied, overwrite=True)
    empty = tmp_path / 'split_1'
    prepare_detector_output(empty, overwrite=False)
    assert empty.is_dir()
