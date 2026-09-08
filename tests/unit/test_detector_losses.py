import torch

from biohub.losses.association import association_loss, compute_loss
from biohub.losses.aux import contrastive_aux_loss, division_aux_loss, offset_aux_loss
from biohub.losses.detection import detection_loss


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
