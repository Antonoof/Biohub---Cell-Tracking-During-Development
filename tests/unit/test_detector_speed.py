from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from biohub.augmentations.blur import blur_augment
from biohub.augmentations.rot90 import rot90_augment
from biohub.data.windows import (
    FrameWindowData,
    FrameWindowDataset,
    VideoMeta,
    collate_windows,
    compact_window,
    pad_window,
)
from biohub.features.position import POS_EMBED_DIM
from biohub.losses.association import association_loss
from biohub.losses.aux import division_aux_loss
from biohub.losses.detection import _binary_target, gaussian_heatmap_target
from biohub.models.detector import UNetNodeTransformer, _index_nearest
from biohub.models.node_transformer import SimpleNodeTransformer
from biohub.models.temporal_unet import TemporalUNet3D, pack_conv3d_channels_last
from biohub.train.assign import greedy_assign, match_peaks
from biohub.train.detector import (
    build_matched_edge_targets,
    detect_and_match,
    detector_optimizer_step,
    train_epoch,
)
from biohub.train.optim import _muon_update
from biohub.train.schedule import ModelEma, build_optimizer


def _window(n=3, start=0):
    return FrameWindowData(
        start, 2, [torch.zeros(n, 32)] * 2, [torch.ones(n, 3)] * 2, [n, n], [torch.eye(n)]
    )


class RepeatedSample(Dataset):
    def __init__(self, sample, count=1):
        self.sample = sample
        self.count = count
        self.reads = 0

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        self.reads += 1
        return self.sample


@pytest.mark.parametrize('k', [1, 2, 3])
def test_rot90_all_turns_track_image(k):
    seed = next(s for s in range(100) if np.random.default_rng(s).integers(0, 4) == k)
    image = torch.zeros(1, 1, 5, 5)
    image[0, 0, 1, 3] = 1
    coords = torch.tensor([[[0.0, 1.0, 3.0]]])
    rotated, points, _ = rot90_augment(
        image, coords, torch.ones(1, 1, dtype=torch.bool), rng=np.random.default_rng(seed)
    )
    z, y, x = points[0, 0].long().tolist()
    assert rotated[0, z, y, x] == 1


def test_compact_window_reindexes_both_edge_axes():
    meta = pad_window(_window(), 5)
    mask = torch.tensor([[False, True, True, False, False], [True, False, True, False, False]])
    compact = compact_window(meta, meta['coords'], mask)
    assert compact['masks'].tolist() == [[True, True, False, False, False]] * 2
    assert compact['node_counts'].tolist() == [2, 2]
    torch.testing.assert_close(
        compact['targets'][0, :2, :2], torch.tensor([[0.0, 0.0], [0.0, 1.0]])
    )
    assert meta['targets'][0].trace() == 3


def test_binary_target_handles_nonprefix_masks():
    coords = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]])
    _, target = _binary_target(
        torch.zeros(1, 1, 3, 3, 3), coords, torch.tensor([[False, True, False]])
    )
    assert target.sum() == 1 and target[0, 1, 1, 1] == 1


def test_soft_matching_accepts_padded_gt_and_hard_gather_matches_gemm():
    gt = torch.zeros(1, 7, 7)
    gt[0, :2, :3] = torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    left, right = torch.tensor([1, -1, 0]), torch.tensor([2, 0])
    hard = build_matched_edge_targets([left], [right], gt, 4, 3)
    torch.testing.assert_close(hard[0, :3, :2], torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]))
    c0, c1 = torch.rand(3, 2), torch.rand(2, 3)
    soft = build_matched_edge_targets(
        [left], [right], gt, 4, 3, match_soft=True, couplings_t=[c0], couplings_t1=[c1]
    )
    torch.testing.assert_close(soft[0, :3, :2], c0 @ gt[0, :2, :3] @ c1.T)


def test_sinkhorn_rectangular_transport_is_subprobability():
    _, coupling = match_peaks(torch.zeros(2, 7), 5, kind='sinkhorn')
    assert (coupling.sum(1) <= 1 + 1e-6).all()
    assert (coupling.sum(0) <= 1 + 1e-6).all()


def test_vectorized_greedy_matches_loop_including_ties():
    for seed in range(10):
        torch.manual_seed(seed)
        dists = torch.randint(0, 8, (31, 17)).float()
        val, idx = dists.min(1)
        expected = torch.full((31,), -1)
        used = set()
        for row in val.argsort().tolist():
            col = int(idx[row])
            if val[row] <= 4 and col not in used:
                expected[row] = col
                used.add(col)
        assert torch.equal(greedy_assign(dists, 4), expected)


def test_batched_greedy_matches_per_sample_loop():
    torch.manual_seed(11)
    dists = torch.rand(5, 9, 6)
    dists[0, 3:] = 1e6
    expected = torch.stack([greedy_assign(row, 0.4) for row in dists])
    assert torch.equal(greedy_assign(dists, 0.4), expected)


def test_detect_and_match_caps_dense_peaks():
    logits = torch.ones(2, 1, 8, 20, 20)
    coords = torch.rand(2, 3, 3)
    mask = torch.ones(2, 3, dtype=torch.bool)
    det_c, _, det_m, matches, _ = detect_and_match(
        logits,
        coords,
        mask,
        (2, 8, 20, 20),
        det_threshold=0.0,
        window_size=2,
    )
    assert det_c.shape[1] == 512
    assert int(det_m.sum().item()) == 2 * 512
    assert all(m.shape[0] == 512 for m in matches)


def test_detect_and_match_keeps_sparse_peaks():
    logits = torch.full((1, 1, 4, 8, 8), -10.0)
    logits[0, 0, 1, 2, 3] = 5.0
    logits[0, 0, 2, 4, 5] = 5.0
    coords = torch.tensor([[[1.0, 2.0, 3.0], [2.0, 4.0, 5.0]]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    det_c, _, det_m, matches, _ = detect_and_match(
        logits, coords, mask, (2, 4, 8, 8), det_threshold=0.5, window_size=2
    )
    assert int(det_m.sum().item()) == 2
    assert matches[0].shape[0] == 2
    found = {tuple(row.tolist()) for row in det_c[0, :2]}
    assert found == {(1.0, 2.0, 3.0), (2.0, 4.0, 5.0)}


def test_detect_and_match_packed_skips_couplings():
    logits = torch.full((1, 1, 4, 8, 8), -10.0)
    logits[0, 0, 1, 2, 3] = 5.0
    logits[0, 0, 2, 4, 5] = 5.0
    coords = torch.tensor([[[1.0, 2.0, 3.0], [2.0, 4.0, 5.0]]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    det_c, _, det_m, matches, couplings = detect_and_match(
        logits,
        coords,
        mask,
        (2, 4, 8, 8),
        det_threshold=0.5,
        window_size=2,
        packed_matches=True,
        return_couplings=False,
    )
    assert torch.is_tensor(matches)
    assert matches.shape == det_m.shape
    assert couplings == []
    assert int((matches[0, : int(det_m[0].sum())] >= 0).sum()) == 2
    cropped = detect_and_match(
        logits, coords, mask, (2, 4, 8, 8), det_threshold=0.5, window_size=2
    )[3]
    assert cropped[0].shape[0] == 2


def test_detect_and_match_tensor_frame_index_matches_int():
    logits = torch.full((2, 1, 4, 8, 8), -10.0)
    logits[0, 0, 1, 2, 3] = 5.0
    logits[1, 0, 2, 4, 5] = 5.0
    coords = torch.tensor([[[1.0, 2.0, 3.0]], [[2.0, 4.0, 5.0]]])
    mask = torch.ones(2, 1, dtype=torch.bool)
    frame_ids = torch.tensor([0, 3])
    per_sample = []
    for i in range(2):
        _, pos_i, _, _, _ = detect_and_match(
            logits[i : i + 1],
            coords[i : i + 1],
            mask[i : i + 1],
            (4, 4, 8, 8),
            window_size=4,
            frame_index=int(frame_ids[i]),
        )
        per_sample.append(pos_i[0])
    _, pos, _, _, _ = detect_and_match(
        logits, coords, mask, (4, 4, 8, 8), window_size=4, frame_index=frame_ids
    )
    torch.testing.assert_close(pos, torch.stack(per_sample))


@pytest.mark.parametrize('kind', ['mlp', 'bilinear'])
@pytest.mark.parametrize('geom', ['rel', 'dist', 'rel_dist'])
def test_factorized_pair_head_output_and_gradients(kind, geom):
    torch.manual_seed(1)
    model = SimpleNodeTransformer(4, 8, 2, 1, dropout=0, pair_head=kind, pair_geom=geom)
    ref = deepcopy(model).eval()
    values = [
        torch.randn(2, 3, 8),
        torch.randn(2, 5, 8),
        torch.randn(2, 3, 3),
        torch.randn(2, 5, 3),
    ]
    left = [v.clone().requires_grad_() for v in values]
    right = [v.clone().requires_grad_() for v in values]
    actual = model._pair_scores(*left)
    expected = ref._pair_scores(*right)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    actual.sum().backward()
    expected.sum().backward()
    for a, b in zip(
        left + list(model.pair_mlp.parameters()),
        right + list(ref.pair_mlp.parameters()),
        strict=True,
    ):
        torch.testing.assert_close(a.grad, b.grad, atol=3e-6, rtol=2e-5)


def test_temporal_attention_window_five_backward():
    torch.manual_seed(12)
    model = TemporalUNet3D(1, 4, layers=(8, 16), temporal_mix='attn')
    loss = model(torch.randn(1, 5, 1, 4, 8, 8)).square().mean()
    loss.backward()
    assert torch.isfinite(loss).all()


def test_checkpoint_batchnorm_buffers_and_gradients_match():
    torch.manual_seed(2)
    model = TemporalUNet3D(1, 4, layers=(8, 16), temporal_mix='none')
    checkpointed = deepcopy(model)
    checkpointed.gradient_checkpointing = True
    x = torch.randn(2, 2, 1, 4, 8, 8)
    out = model(x)
    chk = checkpointed(x)
    out.square().mean().backward()
    chk.square().mean().backward()
    torch.testing.assert_close(out, chk)
    for key, val in model.state_dict().items():
        torch.testing.assert_close(val, checkpointed.state_dict()[key])
    for a, b in zip(model.parameters(), checkpointed.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad)


def test_frame_cache_bounded_matches_uncached_and_batch_padding(monkeypatch):
    reads = []
    raw = np.arange(5 * 4 * 8 * 8, dtype=np.float32).reshape(5, 4, 8, 8) / 1000

    class Array:
        def __getitem__(self, key):
            reads.append(key)
            return raw[key]

    monkeypatch.setattr('biohub.data.windows._zarr_array', lambda _: Array())
    vm = VideoMeta(Path('/fake'), raw.shape, (1, 1, 1), (1.0, 1.0, 1.0), 0.0, 1.0)
    data = [(vm, [_window(2, 0), _window(3, 1), _window(2, 3)])]
    cached = FrameWindowDataset(
        data, max_nodes=99, batch_padding=True, frame_cache_mb=2 * raw[0].nbytes / 1024**2
    )
    uncached = FrameWindowDataset(data, batch_padding=True)
    first = cached[0]
    cached[0]['imgs'].zero_()
    torch.testing.assert_close(first['imgs'], cached[0]['imgs'])
    assert len(reads) == 2
    for i in range(3):
        torch.testing.assert_close(cached[i]['imgs'], uncached[i]['imgs'])
        assert cached._frame_bytes <= cached.frame_cache_bytes
    batch = collate_windows([first, cached[1]])
    assert batch['targets'].shape == (2, 1, 3, 3)
    assert batch['coords'].shape == (2, 2, 3, 3)


def test_gaussian_nearest_distance_equals_dense_max():
    torch.manual_seed(4)
    coords = torch.rand(2, 7, 3) * 6
    mask = torch.rand(2, 7) > 0.3
    shape = (8, 9, 10)
    grid = torch.stack(torch.meshgrid(*(torch.arange(s) for s in shape), indexing='ij'), -1)
    expected = []
    for c, m in zip(coords, mask, strict=True):
        blobs = torch.exp(-((grid[None] - c[m, None, None, None]) ** 2).sum(-1) / (2 * 1.3**2))
        expected.append(blobs.max(0).values)
    actual = gaussian_heatmap_target(coords, mask, shape, 1.3)
    torch.testing.assert_close(actual, torch.stack(expected), atol=2e-7, rtol=2e-6)


def test_separable_blur_matches_full_kernel():
    import math

    import torch.nn.functional as F

    imgs = torch.rand(2, 3, 11, 13)
    sigma = 1.3
    r = math.ceil(3 * sigma)
    x = torch.arange(-r, r + 1).float()
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    expected = F.conv2d(
        F.pad(imgs.reshape(-1, 1, 11, 13), (r,) * 4, mode='replicate'),
        torch.outer(k, k)[None, None],
    ).reshape_as(imgs)
    actual, _, _ = blur_augment(
        imgs,
        torch.zeros(2, 1, 3),
        torch.ones(2, 1, dtype=torch.bool),
        sigma=sigma,
        rng=np.random.default_rng(1),
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize('kind', ['weighted_bce', 'focal', 'gaussian_heatmap'])
def test_train_matched_sinkhorn_soft_with_aux_losses(kind):
    torch.manual_seed(7)
    unet = TemporalUNet3D(1, 4, layers=(4, 8), temporal_mix='conv')
    model = UNetNodeTransformer(unet, 4, 32, hidden_dim=8, n_heads=2, n_blocks=1, dropout=0)
    sample = {
        **pad_window(_window(2), 7),
        'imgs': torch.rand(2, 4, 8, 8),
        'image_shape': torch.tensor([2, 4, 8, 8]),
        'voxel_size': torch.ones(3),
        'downsample': torch.ones(3),
    }
    if kind == 'gaussian_heatmap':
        sample['heatmap_target'] = gaussian_heatmap_target(
            sample['coords'], sample['masks'], (4, 8, 8)
        )
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    losses = train_epoch(
        model,
        DataLoader(RepeatedSample(sample), batch_size=1),
        opt,
        torch.device('cpu'),
        det_loss_kind=kind,
        match_assign='sinkhorn',
        match_soft=True,
        train_peak_topk=3,
        aux_division_weight=0.1,
        aux_contrastive_weight=0.1,
        aux_offset_weight=0.1,
    )
    assert all(np.isfinite(losses))


def test_gaussian_heatmap_train_without_precomputed_target():
    torch.manual_seed(8)
    unet = TemporalUNet3D(1, 4, layers=(4, 8), temporal_mix='none')
    model = UNetNodeTransformer(unet, 4, 32, hidden_dim=8, n_heads=2, n_blocks=1, dropout=0)
    sample = {
        **pad_window(_window(2), 7),
        'imgs': torch.rand(2, 4, 8, 8),
        'image_shape': torch.tensor([2, 4, 8, 8]),
        'voxel_size': torch.ones(3),
        'downsample': torch.ones(3),
    }
    assert 'heatmap_target' not in sample
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    losses = train_epoch(
        model,
        DataLoader(RepeatedSample(sample), batch_size=1),
        opt,
        torch.device('cpu'),
        det_loss_kind='gaussian_heatmap',
        match_assign='greedy',
        match_soft=False,
    )
    assert all(np.isfinite(losses))


def test_train_epoch_flushes_short_accumulation_and_does_not_cache_batches(monkeypatch):
    torch.manual_seed(3)
    unet = TemporalUNet3D(1, 4, layers=(4, 8), temporal_mix='none')
    model = UNetNodeTransformer(
        unet, 4, 4 * POS_EMBED_DIM, hidden_dim=8, n_heads=2, n_blocks=1, dropout=0
    )
    sample = {
        **pad_window(_window(2), 2),
        'imgs': torch.rand(2, 4, 8, 8),
        'image_shape': torch.tensor([2, 4, 8, 8]),
        'voxel_size': torch.ones(3),
        'downsample': torch.ones(3),
    }

    ds = RepeatedSample(sample, 2)
    loader = DataLoader(ds, batch_size=1)
    opt = torch.optim.SGD(model.parameters(), lr=0.001)
    steps = []
    original_step = opt.step

    def step(*args, **kwargs):
        steps.append(1)
        return original_step(*args, **kwargs)

    monkeypatch.setattr(opt, 'step', step)
    losses = train_epoch(
        model, loader, opt, torch.device('cpu'), max_iters=3, accum_steps=2, target_mode='gt_nodes'
    )
    assert len(steps) == 2
    assert ds.reads == 3
    assert all(np.isfinite(losses))
    assert all(p.grad is None for p in model.parameters())


def test_scaler_overflow_skips_optimizer_and_ema_then_recovers():
    model = torch.nn.Linear(2, 1)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler('cpu', init_scale=8.0)
    ema = ModelEma(model, 0.5)
    before = deepcopy(model.state_dict())
    loss = model(torch.ones(1, 2)).sum()
    scaler.scale(loss).backward()
    assert model.weight.grad is not None
    model.weight.grad.fill_(float('inf'))
    detector_optimizer_step(model, opt, scaler, 1.0, ema)
    assert scaler.get_scale() == 4.0
    for key, value in before.items():
        torch.testing.assert_close(model.state_dict()[key], value)
        torch.testing.assert_close(ema.state_dict()[key], value)
    scaler.scale(model(torch.ones(1, 2)).sum()).backward()
    detector_optimizer_step(model, opt, scaler, 1.0, ema)
    assert not torch.equal(model.weight, before['weight'])


def test_muon_conv3d_equals_flattened_matrix_update():
    grad = torch.randn(8, 4, 3, 3, 3)
    flat = grad.flatten(1).clone()
    actual = _muon_update(grad.clone(), torch.zeros_like(grad))
    expected = _muon_update(flat, torch.zeros_like(flat))
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA autocast')
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_cuda_amp_bce_losses_backward(dtype):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip()
    logits = torch.randn(3, 4, device='cuda', dtype=dtype, requires_grad=True)
    target = torch.zeros(3, 4, device='cuda')
    target[0, :2] = 1
    with torch.autocast('cuda', dtype=dtype):
        loss = sum(
            association_loss(kind, logits, target)
            for kind in ('focal_softmax', 'ce_softmax', 'asl_softmax')
        )
        loss = loss + division_aux_loss(logits, target)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_train_epoch_mixed_targets_peak_topk_and_edge_gate():
    torch.manual_seed(5)
    unet = TemporalUNet3D(1, 4, layers=(4, 8), temporal_mix='attn')
    model = UNetNodeTransformer(unet, 4, 32, hidden_dim=8, n_heads=2, n_blocks=1, dropout=0)
    sample = {
        **pad_window(_window(2), 7),
        'imgs': torch.rand(2, 4, 8, 8),
        'image_shape': torch.tensor([2, 4, 8, 8]),
        'voxel_size': torch.ones(3),
        'downsample': torch.ones(3),
    }
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    losses = train_epoch(
        model,
        DataLoader(RepeatedSample(sample), batch_size=1),
        opt,
        torch.device('cpu'),
        target_mode='mixed',
        target_gt_frac=0.5,
        train_peak_topk=3,
        edge_gate_distance=20.0,
        match_assign='hungarian',
    )
    assert all(np.isfinite(losses))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_cuda_bf16_short_temporal_and_node_attention():
    if not torch.cuda.is_bf16_supported():
        pytest.skip()
    torch.manual_seed(0)
    unet = TemporalUNet3D(
        1, 8, layers=(8, 16), temporal_mix='attn', skip_fullres_temporal=True
    ).cuda()
    volume = torch.randn(2, 2, 1, 8, 32, 32, device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        encoded = unet(volume)
        loss = encoded.float().pow(2).mean()
    loss.backward()
    assert torch.isfinite(loss).all()

    transformer = SimpleNodeTransformer(8, 32, 4, 1, dropout=0).cuda()
    feat = torch.randn(2, 1, 8, device='cuda')
    coords = torch.rand(2, 1, 3, device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        scores = transformer(feat, feat, coords, coords)
        node_loss = scores.float().pow(2).mean()
    node_loss.backward()
    assert torch.isfinite(node_loss).all()

    sample = {
        **pad_window(_window(2), 4),
        'imgs': torch.rand(2, 4, 8, 8),
        'image_shape': torch.tensor([2, 4, 8, 8]),
        'voxel_size': torch.ones(3),
        'downsample': torch.ones(3),
    }
    model = UNetNodeTransformer(
        TemporalUNet3D(1, 8, layers=(8, 16), temporal_mix='attn').cuda(),
        8,
        4 * POS_EMBED_DIM,
        hidden_dim=32,
        n_heads=4,
        n_blocks=1,
        dropout=0,
        use_self_attn=True,
    ).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    losses = train_epoch(
        model,
        DataLoader(RepeatedSample(sample), batch_size=1),
        opt,
        torch.device('cuda'),
        amp_kind='bf16',
        match_assign='sinkhorn',
        match_soft=True,
        train_peak_topk=4,
    )
    assert all(np.isfinite(losses))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_cuda_bf16_temporal_attn_huge_spatial_batch():
    if not torch.cuda.is_bf16_supported():
        pytest.skip()
    torch.manual_seed(0)
    unet = TemporalUNet3D(
        1, 8, layers=(8, 16), temporal_mix='attn', skip_fullres_temporal=False
    ).cuda()
    volume = torch.randn(2, 2, 1, 16, 64, 64, device='cuda')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        encoded = unet(volume)
        loss = encoded.float().pow(2).mean()
    loss.backward()
    assert torch.isfinite(loss).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_channels_last_nearest_index_matches_nchw():
    torch.manual_seed(8)
    feat = torch.randn(2, 4, 3, 5, 6, device='cuda')
    coords = torch.tensor([[[1.2, 2.4, 3.1], [0.1, 4.8, 5.2]]], device='cuda').expand(2, -1, -1)
    mask = torch.ones(2, 2, dtype=torch.bool, device='cuda')
    expected = _index_nearest(feat.contiguous(), coords, mask)
    actual = _index_nearest(feat.contiguous(memory_format=torch.channels_last_3d), coords, mask)
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
@pytest.mark.parametrize('name', ['adamp', 'muonwithauxadam'])
def test_adamp_muon_step_channels_last_conv3d(name):
    torch.manual_seed(9)
    conv = torch.nn.Conv3d(2, 4, 3, padding=1).cuda()
    pack_conv3d_channels_last(conv)
    opt = build_optimizer(conv, name=name, lr=0.05, weight_decay=0.0)
    x = torch.randn(1, 2, 4, 5, 6, device='cuda').contiguous(memory_format=torch.channels_last_3d)
    loss = conv(x).float().pow(2).mean()
    loss.backward()
    before = conv.weight.detach().clone()
    opt.step()
    assert not torch.equal(conv.weight, before)


def test_epoch_callback_and_fixed_validation_protocol(tmp_path, monkeypatch):
    from biohub.train import detector

    vm = VideoMeta(Path('/fake'), (3, 4, 8, 8), (1, 1, 1), (1.0, 1.0, 1.0), 0.0, 1.0)
    monkeypatch.setattr(detector, 'load_dataset_windows', lambda *a, **k: (vm, [_window(2)]))
    monkeypatch.setattr(detector, 'train_epoch', lambda *a, **k: (1.0, 1.0))
    eval_kwargs = []

    def evaluate(*args, **kwargs):
        eval_kwargs.append(kwargs)
        return {key: 0.0 for key in detector.CHECKPOINT_METRICS} | {
            'loss': 1.0,
            **{
                detector.score_threshold_key(threshold): 0.0
                for threshold in detector.SCORE_THRESHOLDS
            },
        }

    monkeypatch.setattr(detector, 'evaluate', evaluate)
    epochs = []

    def callback(epoch, metrics):
        epochs.append(epoch)
        return False

    detector.train(
        tmp_path,
        0,
        tmp_path / 'splits.json',
        tmp_path,
        debug_video=Path('/fake'),
        n_epochs=5,
        num_workers=0,
        unet_layers=[4, 8],
        unet_out_channels=4,
        hidden_dim=8,
        n_heads=2,
        n_blocks=1,
        device='cpu',
        match_assign='sinkhorn',
        match_soft=True,
        max_match_distance=8,
        train_peak_topk=64,
        epoch_callback=callback,
    )
    assert epochs == [0]
    assert len(eval_kwargs) == 1
    assert eval_kwargs[0]['max_match_distance'] == 5.0
    assert eval_kwargs[0]['match_assign'] == 'greedy'
    assert eval_kwargs[0]['match_soft'] is False
    assert eval_kwargs[0]['train_peak_topk'] == 0
    assert (tmp_path / 'unet_transformer/split_0/edge_predictor_best.pth').exists()
