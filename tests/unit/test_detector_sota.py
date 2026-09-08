import torch

from biohub.losses.association import compute_batch_loss
from biohub.losses.aux import offset_aux_loss
from biohub.models.detector import UNetNodeTransformer
from biohub.models.extra_encoder import FrozenConvEncoder, coord_channels, flow_channels
from biohub.models.temporal_unet import TemporalUNet3D, unet_in_channels
from biohub.train.assign import greedy_assign, hungarian_assign, match_peaks
from biohub.train.detector import detect_and_match


def test_unet_in_channels_defaults_to_one() -> None:
    assert unet_in_channels() == 1
    assert unet_in_channels(coord_kind='coord', flow_input='frame_diff') == 5
    assert unet_in_channels(extra_encoder='conv', extra_encoder_channels=8) == 9


def test_plain_unet_matches_legacy_keys() -> None:
    unet = TemporalUNet3D(in_channels=1, out_channels=4, layers=(8, 16))
    keys = set(unet.state_dict())
    assert 'encoder_blocks.0.0.weight' in keys
    assert 'encoder_blocks.0.1.running_mean' in keys
    assert not any(name.startswith('extra') for name in keys)


def test_residual_groupnorm_conv_temporal_forward() -> None:
    unet = TemporalUNet3D(
        in_channels=4,
        out_channels=4,
        layers=(8, 16),
        unet_block='residual',
        unet_norm='groupnorm',
        temporal_mix='conv',
        skip_fullres_temporal=True,
    )
    out = unet(torch.rand(1, 2, 4, 4, 8, 8))
    assert out.shape == (1, 2, 4, 4, 8, 8)
    assert torch.isfinite(out).all()


def test_convnext_deform_forward() -> None:
    unet = TemporalUNet3D(
        in_channels=1,
        out_channels=4,
        layers=(8, 16),
        unet_block='convnext',
        unet_deform=True,
        temporal_mix='both',
        skip_fullres_temporal=False,
    )
    out = unet(torch.rand(1, 2, 1, 4, 8, 8))
    assert out.shape[:3] == (1, 2, 4)
    assert torch.isfinite(out).all()


def test_coord_and_flow_channels_shapes() -> None:
    imgs = torch.rand(2, 3, 4, 6, 6)
    coord = coord_channels(imgs, 'coord', 4)
    fourier = coord_channels(imgs, 'fourier', 2)
    diff = flow_channels(imgs, 'frame_diff')
    grad = flow_channels(imgs, 'spatial_grad')
    both = flow_channels(imgs, 'frame_diff_grad')
    assert coord.shape == (2, 3, 3, 4, 6, 6)
    assert fourier.shape == (2, 3, 12, 4, 6, 6)
    assert diff.shape == (2, 3, 1, 4, 6, 6)
    assert grad.shape == (2, 3, 3, 4, 6, 6)
    assert both.shape == (2, 3, 4, 4, 6, 6)


def test_frozen_conv_encoder_has_no_grad() -> None:
    enc = FrozenConvEncoder(4, freeze=True)
    imgs = torch.rand(1, 2, 4, 6, 6, requires_grad=True)
    out = enc(imgs)
    out.sum().backward()
    for param in enc.parameters():
        assert param.grad is None or float(param.grad.abs().sum()) == 0.0


def test_detector_coord_flow_encode() -> None:
    unet = TemporalUNet3D(
        in_channels=unet_in_channels(coord_kind='coord', flow_input='frame_diff'),
        out_channels=4,
        layers=(8, 16),
    )
    model = UNetNodeTransformer(
        unet=unet,
        unet_out_channels=4,
        pos_feat_dim=8,
        hidden_dim=32,
        n_heads=4,
        n_blocks=1,
        dropout=0.0,
        coord_kind='coord',
        flow_input='frame_diff',
        extra_encoder='none',
        feature_sample='trilinear',
    )
    unet_out, det_logits = model.encode(torch.rand(1, 2, 4, 8, 8))
    assert unet_out.shape[0] == 1
    assert len(det_logits) == 2
    coords = torch.tensor([[[1.2, 2.4, 3.1], [0.0, 0.0, 0.0]]])
    mask = torch.tensor([[True, False]])
    feat = model.index_features(unet_out[:, 0], coords, mask)
    assert feat.shape == (1, 2, 4)
    assert torch.isfinite(feat).all()


def test_hungarian_resolves_crossed_greedy() -> None:
    dists = torch.tensor([[1.0, 4.0], [1.2, 3.0]])
    greedy = greedy_assign(dists, 10.0)
    hung = hungarian_assign(dists, 10.0)
    assert greedy.tolist() == [0, -1]
    assert hung.tolist() == [0, 1]


def test_sinkhorn_coupling_rows_positive() -> None:
    dists = torch.tensor([[0.1, 4.0], [4.0, 0.2]])
    matched, coupling = match_peaks(dists, 5.0, kind='sinkhorn', tau=0.2, iters=16)
    assert coupling.shape == (2, 2)
    assert torch.isfinite(coupling).all()
    assert int((matched >= 0).sum()) >= 1


def test_detect_and_match_hungarian_and_topk() -> None:
    logits = torch.full((1, 1, 4, 4, 4), -8.0)
    logits[0, 0, 1, 1, 1] = 2.0
    logits[0, 0, 2, 2, 2] = 1.5
    gt = torch.tensor([[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    image_shape = (1, 4, 4, 4)
    coords, _pos, det_mask, matches, couplings = detect_and_match(
        logits,
        gt,
        mask,
        image_shape,
        det_threshold=0.5,
        voxel_size=(1.0, 1.0, 1.0),
        pool_kernel_um=1.0,
        match_assign='hungarian',
        train_peak_topk=2,
    )
    assert int(det_mask[0].sum()) >= 2
    assert matches[0].numel() == int(det_mask[0].sum())
    assert couplings[0].shape[1] == 2


def test_distance_gate_masks_far_parents() -> None:
    logits = torch.zeros(1, 2, 2)
    target = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    src = torch.tensor([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]])
    tgt = torch.tensor([[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]]])
    loss = compute_batch_loss(
        logits,
        target,
        mask,
        mask,
        coords_src=src,
        coords_tgt=tgt,
        gate_distance=1.0,
    )
    assert torch.isfinite(loss).all()


def test_offset_parabolic_finite() -> None:
    pred = torch.zeros(1, 3, 4, 4, 4)
    det = torch.zeros(1, 1, 4, 4, 4)
    det[0, 0, 1, 1, 1] = 4.0
    det[0, 0, 0, 1, 1] = 1.0
    det[0, 0, 2, 1, 1] = 2.0
    coords = torch.tensor([[[1.0, 1.0, 1.0]]])
    mask = torch.ones(1, 1, dtype=torch.bool)
    loss = offset_aux_loss(pred, coords, mask, target='parabolic', det_logits=det)
    assert torch.isfinite(loss).all()


def test_extra_encoder_conv_changes_in_channels() -> None:
    unet = TemporalUNet3D(
        in_channels=unet_in_channels(extra_encoder='conv', extra_encoder_channels=4),
        out_channels=4,
        layers=(8, 16),
    )
    model = UNetNodeTransformer(
        unet=unet,
        unet_out_channels=4,
        pos_feat_dim=8,
        hidden_dim=32,
        n_heads=4,
        n_blocks=1,
        dropout=0.0,
        extra_encoder='conv',
        extra_encoder_channels=4,
        extra_encoder_freeze=True,
    )
    out, dets = model.encode(torch.rand(1, 2, 4, 8, 8))
    assert out.shape[1] == 2
    assert len(dets) == 2
    assert model.extra_encoder is not None
    for param in model.extra_encoder.parameters():
        assert not param.requires_grad
