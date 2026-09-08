from pathlib import Path

import numpy as np
import optuna
import pytest
import torch
import torch.nn.functional as F
import yaml

from biohub.augmentations.gamma import gamma_augment
from biohub.augmentations.noise import noise_augment
from biohub.data.windows import FrameWindowData, FrameWindowDataset, VideoMeta, load_dataset_windows
from biohub.losses.association import (
    compute_batch_loss,
    evaluate_pair,
    evaluate_pairs_batched,
    pair_event_counts,
)
from biohub.losses.aux import contrastive_aux_loss, division_aux_loss
from biohub.losses.detection import detection_loss, gaussian_heatmap_target
from biohub.models.temporal_unet import DeformConv3d
from biohub.train.detector import (
    _as_coupling,
    _augmentations_from_cfg,
    detect_and_match,
    detector_optimizer_step,
    train_from_config,
    validate_config,
)
from biohub.train.detector_search import apply_search_params, load_base_config, sample_search_params
from biohub.utils.seed import configure_determinism


def test_gaussian_empty_padded_gt_is_zero_and_pushes_logits_down():
    coords = torch.randn(3, 8, 3)
    mask = torch.zeros(3, 8, dtype=torch.bool)
    target = gaussian_heatmap_target(coords, mask, (4, 5, 6))
    assert target.count_nonzero() == 0
    logits = torch.zeros(3, 1, 4, 5, 6, requires_grad=True)
    detection_loss('gaussian_heatmap', logits, coords, mask).backward()
    assert logits.grad is not None
    assert (logits.grad > 0).all()


def test_deform_cache_validation_then_backward():
    model = DeformConv3d(2, 2)
    # Validation changes shape, thus overwrites even a cache first filled in training.
    model(torch.randn(1, 2, 3, 3, 3)).sum().backward()
    x = torch.randn(1, 2, 4, 5, 6)
    with torch.inference_mode():
        model(x)
    model.zero_grad(set_to_none=True)
    model(x).sum().backward()
    assert model.offset.weight.grad is not None
    assert torch.isfinite(model.offset.weight.grad).all()


@pytest.mark.parametrize('clip', [0.0, 2.0])
def test_nonfinite_grad_never_steps_even_between_periodic_checks(clip):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    before = model.weight.detach().clone()
    model.weight.grad = torch.full_like(model.weight, float('nan'))
    with pytest.raises(RuntimeError, match='finite'):
        detector_optimizer_step(model, optimizer, None, clip, check_finite=False)
    torch.testing.assert_close(model.weight, before)
    assert not optimizer.state


def test_aux_batch_loss_and_gradient_match_unpadded_reference():
    torch.manual_seed(1)
    b, n, m, d = 5, 6, 7, 8
    logits = torch.randn(b, n, m, requires_grad=True)
    query = torch.randn(b, n, d, requires_grad=True)
    key = torch.randn(b, m, d, requires_grad=True)
    target = (torch.rand(b, n, m) > 0.75).float()
    src_counts, tgt_counts = [3, 0, 6, 2, 5], [4, 5, 0, 1, 7]
    src = torch.arange(n)[None] < torch.tensor(src_counts)[:, None]
    tgt = torch.arange(m)[None] < torch.tensor(tgt_counts)[:, None]
    expected = logits.sum() * 0 + query.sum() * 0 + key.sum() * 0
    for i, (ns, nt) in enumerate(zip(src_counts, tgt_counts)):
        if ns == 0 or nt == 0:
            continue
        t = target[i, :ns, :nt]
        mass = logits[i, :ns, :nt].softmax(0).sum(1)
        expected = (
            expected + F.binary_cross_entropy_with_logits(mass - 1, (t.sum(1) > 1).float()) / b
        )
        positives = t > 0.5
        rows = positives.any(1)
        if rows.any():
            scores = F.normalize(query[i, :ns], dim=-1) @ F.normalize(key[i, :nt], dim=-1).T / 0.1
            nll = -(scores.log_softmax(-1) * positives).sum(-1) / positives.sum(-1).clamp_min(1)
            expected = expected + nll[rows].mean() / b
    actual = division_aux_loss(logits, target, src, tgt) + contrastive_aux_loss(
        query, key, target, src, tgt
    )
    torch.testing.assert_close(actual, expected)
    expected_grads = torch.autograd.grad(expected, (logits, query, key), retain_graph=True)
    actual_grads = torch.autograd.grad(actual, (logits, query, key))
    for a, e in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, e, atol=2e-7, rtol=2e-5)


@pytest.mark.parametrize('threshold', [0, 0.2, 0.5, 1])
def test_validation_batched_statistics_match_unpadded_reference(threshold):
    torch.manual_seed(12)
    logits = torch.randn(5, 6, 7)
    target = (torch.rand(5, 6, 7) > 0.8).float()
    src_counts, tgt_counts = [3, 0, 6, 2, 5], [4, 5, 0, 1, 7]
    src = torch.arange(6)[None] < torch.tensor(src_counts)[:, None]
    tgt = torch.arange(7)[None] < torch.tensor(tgt_counts)[:, None]
    expected = torch.zeros(9, dtype=torch.float64)
    for i, (ns, nt) in enumerate(zip(src_counts, tgt_counts)):
        a, b = logits[i, :ns, :nt], target[i, :ns, :nt]
        expected += torch.tensor(
            (*evaluate_pair(a, b, threshold), *pair_event_counts(a, b, threshold)),
            dtype=torch.float64,
        )
    actual = evaluate_pairs_batched(logits, target, src, tgt, threshold)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)


def test_nms_predictions_independent_of_gt_padding_and_no_fake_large_distance_matches():
    logits = torch.arange(1000).float().reshape(1, 1, 10, 10, 10) / 100
    results = []
    for n in (1, 256):
        results.append(
            detect_and_match(
                logits,
                torch.zeros(1, n, 3),
                torch.zeros(1, n, dtype=torch.bool),
                (2, 10, 10, 10),
                pool_kernel_um=1,
                max_match_distance=1e8,
                return_couplings=False,
                packed_matches=True,
            )
        )
    torch.testing.assert_close(results[0][0], results[1][0])
    matches = results[0][3]
    assert isinstance(matches, torch.Tensor)
    assert (matches == -1).all()
    assert _as_coupling(torch.tensor([-1, -1]), 0).shape == (2, 0)


@pytest.mark.parametrize(
    'cfg',
    [
        {'batch_size': 0},
        {'batch_size': 1.5},
        {'batch_size': True},
        {'hidden_dim': 7, 'n_heads': 4},
        {'lr': float('nan')},
        {'min_lr': 1},
        {'amp': 'bad'},
        {'match_soft': True, 'match_assign': 'greedy'},
        {'time_stretch_aug': True},
        {'extra_encoder': 'sam'},
        {'scale_aug_range': 1},
        {'drop_path': 1},
        {'device': 'cuda:-1'},
        {'split': 5, 'n_folds': 5},
        {'seed': 2**32},
        {'downsample': [1, 4]},
        {'accum_steps': 0},
        {'max_frames': 2, 'window_size': 3},
        {'unet_layers': [8, 10]},
        {'batch_szie': 16},
        {'noise_aug': 'false'},
        {'sinkhorn_tau': 0},
    ],
)
def test_invalid_configuration_rejected_before_files_or_training(cfg, tmp_path):
    output = tmp_path / 'untouched'
    with pytest.raises(ValueError):
        train_from_config(dict(data_dir='/missing', weights_dir=str(output)) | cfg)
    assert not output.exists()


def test_existing_detector_configs_and_random_search_parameters_validate():
    for pattern in ('01_p*.yaml', '02_model_c.yaml'):
        for directory in (Path('configs'), Path('configs/smoke')):
            for path in directory.glob(pattern):
                validate_config(yaml.safe_load(path.read_text()))
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=17))
    for _ in range(100):
        trial = study.ask()
        cfg = load_base_config() | apply_search_params(sample_search_params(trial))
        checked = validate_config(cfg)
        assert checked['batch_size'] == 16
        study.tell(trial, 0.0)


def test_zero_probability_temporal_augmentation_is_noop():
    cfg = validate_config(
        {
            'time_stretch_aug': True,
            'time_stretch_aug_proba': 0,
            'time_warp_aug': True,
            'time_warp_aug_proba': 0,
            'pair_chunk_size': None,
        }
    )
    assert not any('time_' in aug.func.__name__ for aug in _augmentations_from_cfg(cfg))


def test_cpu_gamma_fast_path_matches_torch_and_does_not_mutate_input():
    image = torch.rand(2, 3, 7, 9) * 4 - 0.2
    before = image.clone()
    coords, mask = torch.zeros(2, 1, 3), torch.ones(2, 1, dtype=torch.bool)
    actual = gamma_augment(image, coords, mask, rng=np.random.default_rng(11))[0]
    expected = gamma_augment(
        image.clone().requires_grad_(), coords, mask, rng=np.random.default_rng(11)
    )[0]
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(image, before)


def test_fast_noise_repeatability_distribution_and_parent_rng():
    image = torch.zeros(2, 16, 64, 64)
    coords, mask = torch.zeros(2, 1, 3), torch.ones(2, 1, dtype=torch.bool)
    rng_a, rng_b = np.random.default_rng(11), np.random.default_rng(11)
    a = noise_augment(image, coords, mask, rng=rng_a, std=0.1)[0]
    b = noise_augment(image, coords, mask, rng=np.random.default_rng(11), std=0.1)[0]
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    rng_b.integers(0, 2**31 - 1)
    assert rng_a.random() == rng_b.random()
    assert abs(float(a.mean())) < 0.001
    assert abs(float(a.std()) - 0.1) < 0.001
    assert image.count_nonzero() == 0


def test_custom_inplace_mask_augmentation_reindexes_edges(monkeypatch):
    raw = np.zeros((2, 4, 4, 4), dtype=np.float32)
    monkeypatch.setattr('biohub.data.windows._zarr_array', lambda path: raw)
    vm = VideoMeta(Path('/unused.zarr'), raw.shape, (1, 1, 1), (1.0, 1.0, 1.0), 0.0, 1.0)
    window = FrameWindowData(0, 2, [], [torch.ones(3, 3)] * 2, [3, 3], [torch.eye(3)])

    def custom_aug(imgs, coords, mask, *, rng):
        mask[0, 0] = False
        mask[1, 1] = False
        return imgs, coords, mask

    sample = FrameWindowDataset([(vm, [window])], augmentations=[custom_aug])[0]
    assert sample['node_counts'].tolist() == [2, 2]
    assert sample['masks'].tolist() == [[True, True, False]] * 2
    torch.testing.assert_close(sample['targets'][0, :2, :2], torch.tensor([[0.0, 0.0], [0.0, 1.0]]))


def test_unsafe_time_reverse_fails_before_reading_data():
    with pytest.raises(ValueError, match='image frames and GT'):
        load_dataset_windows(Path('/missing.zarr'), invert_time=True)


def test_deterministic_flag_can_be_disabled_between_python_trials():
    original = torch.are_deterministic_algorithms_enabled()
    try:
        configure_determinism(True)
        assert torch.are_deterministic_algorithms_enabled()
        configure_determinism(False)
        assert not torch.are_deterministic_algorithms_enabled()
        assert not torch.backends.cudnn.deterministic
    finally:
        configure_determinism(original)


@pytest.mark.parametrize('gamma', [0, 0.25, 0.5, 1, 3])
@pytest.mark.parametrize('kind', ['focal_softmax', 'asl_softmax'])
def test_fractional_focal_gamma_has_finite_saturated_gradients(kind, gamma):
    logits = torch.tensor([[[1000.0, -1000.0], [-1000.0, 1000.0]]], requires_grad=True)
    target = torch.eye(2).unsqueeze(0)
    mask = torch.ones(1, 2, dtype=torch.bool)
    loss = compute_batch_loss(logits, target, mask, mask, kind=kind, focal_gamma=gamma)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_gaussian_matches_cpu_and_handles_empty_masks():
    torch.manual_seed(3)
    coords = torch.rand(3, 300, 3) * 7
    mask = torch.rand(3, 300) > 0.4
    mask[0] = False
    actual = gaussian_heatmap_target(coords.cuda(), mask.cuda(), (7, 8, 9))
    expected = gaussian_heatmap_target(coords, mask, (7, 8, 9))
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-6, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_cuda_gaussian_matches_cpu_on_downsampled_volume():
    torch.manual_seed(4)
    coords = torch.rand(2, 20, 3) * 31
    mask = torch.rand(2, 20) > 0.3
    actual = gaussian_heatmap_target(coords.cuda(), mask.cuda(), (32, 32, 32))
    expected = gaussian_heatmap_target(coords, mask, (32, 32, 32))
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-6, rtol=2e-5)
