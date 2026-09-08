import pytest
import torch

from biohub.losses.association import compute_loss, pair_event_counts
from biohub.models import SimpleNodeTransformer, TemporalUNet3D
from biohub.models.option_head import OptionHead
from biohub.train.detector import require_finite, require_finite_grads


def test_temporal_unet_output_shape():
    model = TemporalUNet3D(in_channels=1, out_channels=8, layers=(8, 16))
    x = torch.randn(1, 2, 1, 8, 16, 16)
    out = model(x)
    assert out.shape == (1, 2, 8, 8, 16, 16)


def test_simple_node_transformer_output_shape():
    model = SimpleNodeTransformer(feat_dim=16, hidden_dim=32, n_blocks=1, pair_chunk_size=4)
    feat_t = torch.randn(3, 16)
    feat_t1 = torch.randn(4, 16)
    coords_t = torch.randn(3, 3)
    coords_t1 = torch.randn(4, 3)
    out = model(feat_t, feat_t1, coords_t, coords_t1)
    assert out.shape == (3, 4)


def test_option_head_output_shape():
    model = OptionHead(4, 6, 8, 12)
    source = torch.randn(2, 4)
    pair = torch.randn(2, 3, 6)
    mask = torch.ones(2, 3, dtype=torch.bool)
    continue_logit, pair_logits = model(source, pair, mask)
    assert continue_logit.shape == (2,)
    assert pair_logits.shape == (2, 3)


def test_transformer_all_masked_keys_stay_finite() -> None:
    torch.manual_seed(0)
    model = SimpleNodeTransformer(feat_dim=16, hidden_dim=32, n_heads=4, n_blocks=1)
    feat_t = torch.randn(2, 3, 16)
    feat_t1 = torch.randn(2, 4, 16)
    coords_t = torch.randn(2, 3, 3)
    coords_t1 = torch.randn(2, 4, 3)
    mask_t = torch.ones(2, 3, dtype=torch.bool)
    mask_t1 = torch.zeros(2, 4, dtype=torch.bool)
    mask_t1[1, 0] = True
    out = model(feat_t, feat_t1, coords_t, coords_t1, mask_t, mask_t1)
    assert torch.isfinite(out).all()
    out.sum().backward()
    for param in model.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all()


def test_association_loss_empty_target_stays_finite() -> None:
    logits = torch.randn(3, 4, requires_grad=True)
    target = torch.zeros(3, 4)
    loss = compute_loss(logits, target)
    assert torch.isfinite(loss).all()
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_pair_event_counts_edges_and_divisions() -> None:
    identity = torch.tensor([[20.0, -20.0], [-20.0, 20.0]])
    target = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert pair_event_counts(identity, target) == (2, 0, 0, 0, 0, 0)
    split = torch.tensor([[20.0, 20.0], [-20.0, -20.0]])
    split_target = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
    assert pair_event_counts(split, split_target) == (2, 0, 0, 1, 0, 0)


def test_require_finite_rejects_nan_loss_and_grads() -> None:
    with pytest.raises(RuntimeError, match='Detector training loss is not finite'):
        require_finite(torch.tensor(float('nan')), 'Detector training loss')
    model = torch.nn.Linear(2, 2)
    model.weight.grad = torch.full_like(model.weight, float('nan'))
    with pytest.raises(RuntimeError, match='Detector gradient'):
        require_finite_grads(model)
