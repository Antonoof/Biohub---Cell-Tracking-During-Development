import torch

from biohub.losses.association import association_loss, compute_batch_loss, compute_loss
from biohub.losses.aux import contrastive_aux_loss, division_aux_loss, offset_aux_loss
from biohub.losses.detection import detection_loss
from biohub.modules.detect.peaks import subvoxel_offsets


def test_empty_association_is_zero_and_finite() -> None:
    logits = torch.randn(3, 4, requires_grad=True)
    target = torch.zeros(3, 4)
    for kind in ('focal_softmax', 'ce_softmax', 'asl_softmax'):
        loss = association_loss(kind, logits, target)
        assert torch.isfinite(loss).all()
        assert float(loss) == 0.0


def test_ce_softmax_identity_near_zero() -> None:
    logits = torch.tensor([[20.0, -20.0], [-20.0, 20.0]])
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    loss = association_loss('ce_softmax', logits, target)
    assert float(loss) < 1e-4


def test_div_weight_increases_division_row() -> None:
    logits = torch.tensor([[2.0, 2.0], [0.0, 0.0]], dtype=torch.float32, requires_grad=False)
    target = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    base = compute_loss(logits, target, div_weight=1.0)
    heavy = compute_loss(logits, target, div_weight=10.0)
    assert float(heavy) > float(base)


def test_gaussian_heatmap_empty_nodes() -> None:
    logits = torch.zeros(2, 1, 3, 3, 3)
    coords = torch.zeros(2, 0, 3)
    mask = torch.zeros(2, 0, dtype=torch.bool)
    loss = detection_loss('gaussian_heatmap', logits, coords, mask)
    assert torch.isfinite(loss).all()


def test_gaussian_heatmap_peaks_on_gt() -> None:
    logits = torch.zeros(1, 1, 4, 4, 4)
    coords = torch.tensor([[[1.0, 2.0, 1.0]]])
    mask = torch.ones(1, 1, dtype=torch.bool)
    loss = detection_loss('gaussian_heatmap', logits, coords, mask, heatmap_sigma=1.0)
    assert torch.isfinite(loss).all()
    target_like = torch.sigmoid(logits)[0, 0]
    assert float(target_like[1, 2, 1]) >= 0.0
    peaked = logits.clone()
    peaked[0, 0, 1, 2, 1] = 8.0
    better = detection_loss('gaussian_heatmap', peaked, coords, mask, heatmap_sigma=1.0)
    assert float(better) < float(loss)


def test_gaussian_heatmap_does_not_prefer_background_collapse() -> None:
    coords = torch.tensor([[[8.0, 8.0, 8.0]]])
    mask = torch.ones(1, 1, dtype=torch.bool)
    collapsed = torch.full((1, 1, 16, 16, 16), -8.0)
    peaked = torch.full((1, 1, 16, 16, 16), -2.0)
    peaked[0, 0, 8, 8, 8] = 8.0
    assert float(detection_loss('gaussian_heatmap', peaked, coords, mask)) < float(
        detection_loss('gaussian_heatmap', collapsed, coords, mask)
    )


def test_aux_zero_weight_does_not_change_total() -> None:
    logits = torch.randn(3, 4)
    target = torch.zeros(3, 4)
    target[0, 1] = 1.0
    edge = association_loss('focal_softmax', logits, target)
    total = edge + 0.0 * division_aux_loss(logits, target)
    assert float(total) == float(edge)


def test_contrastive_prefers_true_pairs() -> None:
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    key = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    good = contrastive_aux_loss(query, key, target, temperature=0.1)
    shuffled = contrastive_aux_loss(query, key.flip(0), target, temperature=0.1)
    assert float(good) < float(shuffled)


def test_offset_aux_is_finite() -> None:
    pred = torch.zeros(1, 3, 4, 4, 4)
    coords = torch.tensor([[[1.2, 2.4, 1.1]]])
    mask = torch.ones(1, 1, dtype=torch.bool)
    loss = offset_aux_loss(pred, coords, mask)
    assert torch.isfinite(loss).all()
    assert float(loss) > 0


def test_offset_aux_empty_sample_is_zero() -> None:
    pred = torch.zeros(2, 3, 4, 4, 4)
    coords = torch.tensor([[[1.2, 2.4, 1.1], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    mask = torch.tensor([[True, False], [False, False]])
    loss = offset_aux_loss(pred, coords, mask)
    assert torch.isfinite(loss).all()
    only = offset_aux_loss(pred[:1], coords[:1], mask[:1])
    torch.testing.assert_close(loss, only / 2)


def test_offset_parabolic_matches_masked_points() -> None:
    torch.manual_seed(0)
    pred = torch.randn(3, 3, 5, 6, 7)
    det = torch.randn(3, 1, 5, 6, 7)
    coords = torch.rand(3, 4, 3) * torch.tensor([4.0, 5.0, 6.0])
    mask = torch.tensor(
        [[True, False, True, False], [False, False, False, False], [True, True, True, True]]
    )
    actual = offset_aux_loss(pred, coords, mask, target='parabolic', det_logits=det)
    spatial = pred.shape[2:]
    losses = []
    for b in range(3):
        idx = mask[b].nonzero(as_tuple=False)[:, 0]
        if idx.numel() == 0:
            losses.append(pred.new_zeros(()))
            continue
        gt = coords[b, idx]
        zi = gt[:, 0].long().clamp(0, spatial[0] - 1)
        yi = gt[:, 1].long().clamp(0, spatial[1] - 1)
        xi = gt[:, 2].long().clamp(0, spatial[2] - 1)
        frac = subvoxel_offsets(det[b, 0], torch.stack((zi, yi, xi), dim=-1))
        sample_pred = pred[b, :, zi, yi, xi].T
        losses.append((sample_pred - frac).abs().mean())
    torch.testing.assert_close(actual, torch.stack(losses).mean())


def test_offset_aux_accepts_bf16_under_autocast() -> None:
    pred = torch.randn(2, 3, 4, 4, 4, dtype=torch.bfloat16)
    det = torch.randn(2, 1, 4, 4, 4, dtype=torch.bfloat16)
    coords = torch.tensor([[[1.2, 2.4, 1.1], [0.0, 0.0, 0.0]], [[2.1, 1.4, 0.8], [0.0, 0.0, 0.0]]])
    mask = torch.tensor([[True, False], [True, False]])
    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=True):
        frac = offset_aux_loss(pred, coords, mask, target='frac')
        para = offset_aux_loss(pred, coords, mask, target='parabolic', det_logits=det)
    assert torch.isfinite(frac).all()
    assert torch.isfinite(para).all()


def _gate_pair(logits, coords_src, coords_tgt, gate_distance):
    if gate_distance <= 0 or logits.numel() == 0:
        return logits
    dists = torch.cdist(coords_src, coords_tgt)
    keep = dists <= gate_distance
    if logits.shape[1] > 0:
        nearest = dists.argmin(dim=0)
        keep[nearest, torch.arange(logits.shape[1], device=logits.device)] = True
    return logits.masked_fill(~keep, -1.0e4)


def _loop_batch_loss(
    logits,
    target,
    mask_t,
    mask_t1,
    *,
    kind='focal_softmax',
    focal_gamma=2.0,
    div_weight=1.0,
    coords_src=None,
    coords_tgt=None,
    gate_distance=0.0,
    source_counts=None,
    target_counts=None,
):
    if source_counts is None:
        source_counts = [int(v) for v in mask_t.sum(dim=1).tolist()]
    if target_counts is None:
        target_counts = [int(v) for v in mask_t1.sum(dim=1).tolist()]
    losses = []
    for b in range(logits.shape[0]):
        nt = int(source_counts[b])
        nt1 = int(target_counts[b])
        pair = logits[b, :nt, :nt1]
        if coords_src is not None and coords_tgt is not None:
            pair = _gate_pair(pair, coords_src[b, :nt], coords_tgt[b, :nt1], gate_distance)
        losses.append(
            association_loss(
                kind,
                pair,
                target[b, :nt, :nt1],
                focal_gamma=focal_gamma,
                div_weight=div_weight,
            )
        )
    return torch.stack(losses).mean()


def test_compute_batch_loss_matches_per_sample_loop() -> None:
    torch.manual_seed(0)
    logits = torch.randn(4, 5, 6)
    target = torch.zeros(4, 5, 6)
    target[0, 0, 1] = 1.0
    target[0, 2, 3] = 1.0
    target[1, 1, 0] = 1.0
    target[1, 1, 2] = 1.0
    target[3, 0, 0] = 1.0
    mask_t = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, False],
            [False, False, False, False, False],
            [True, True, False, False, False],
        ]
    )
    mask_t1 = torch.tensor(
        [
            [True, True, True, True, False, False],
            [True, True, True, False, False, False],
            [True, False, False, False, False, False],
            [True, True, True, True, True, False],
        ]
    )
    src = torch.randn(4, 5, 3)
    tgt = torch.randn(4, 6, 3)
    for kind in ('focal_softmax', 'ce_softmax', 'asl_softmax'):
        torch.testing.assert_close(
            compute_batch_loss(
                logits,
                target,
                mask_t,
                mask_t1,
                kind=kind,
                focal_gamma=1.5,
                div_weight=2.0,
            ),
            _loop_batch_loss(
                logits,
                target,
                mask_t,
                mask_t1,
                kind=kind,
                focal_gamma=1.5,
                div_weight=2.0,
            ),
            atol=1e-5,
            rtol=1e-5,
        )
        torch.testing.assert_close(
            compute_batch_loss(
                logits,
                target,
                mask_t,
                mask_t1,
                kind=kind,
                focal_gamma=1.5,
                div_weight=2.0,
                coords_src=src,
                coords_tgt=tgt,
                gate_distance=1.5,
            ),
            _loop_batch_loss(
                logits,
                target,
                mask_t,
                mask_t1,
                kind=kind,
                focal_gamma=1.5,
                div_weight=2.0,
                coords_src=src,
                coords_tgt=tgt,
                gate_distance=1.5,
            ),
            atol=1e-5,
            rtol=1e-5,
        )


def test_compute_batch_loss_flat_pairs_matches_mean_over_pairs() -> None:
    torch.manual_seed(1)
    batch, pairs, nodes = 3, 4, 5
    logits = torch.randn(batch, pairs, nodes, nodes)
    target = torch.zeros(batch, pairs, nodes, nodes)
    target[0, 0, 0, 1] = 1.0
    target[1, 2, 1, 0] = 1.0
    target[2, 3, 2, 2] = 1.0
    mask = torch.zeros(batch, pairs + 1, nodes, dtype=torch.bool)
    mask[:, :, :3] = True
    mask[:, 0, 3] = True
    per = [
        compute_batch_loss(
            logits[:, i],
            target[:, i],
            mask[:, i],
            mask[:, i + 1],
            kind='focal_softmax',
        )
        for i in range(pairs)
    ]
    flat = compute_batch_loss(
        logits.reshape(batch * pairs, nodes, nodes),
        target.reshape(batch * pairs, nodes, nodes),
        mask[:, :-1].reshape(batch * pairs, nodes),
        mask[:, 1:].reshape(batch * pairs, nodes),
        kind='focal_softmax',
    )
    torch.testing.assert_close(flat, torch.stack(per).mean(), atol=1e-5, rtol=1e-5)
