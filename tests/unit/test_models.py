import torch

from biohub.models import SimpleNodeTransformer, TemporalUNet3D
from biohub.models.option_head import OptionHead


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
