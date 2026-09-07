import torch

from biohub.models.detector import UNetNodeTransformer
from biohub.models.temporal_unet import TemporalUNet3D
from biohub.modules.detect.config import PredictConfig
from biohub.modules.detect.predict import _encode, encode_detection_views


def _tiny_detector() -> UNetNodeTransformer:
    unet = TemporalUNet3D(
        in_channels=1,
        out_channels=4,
        layers=(8, 16),
        gradient_checkpointing=False,
    )
    model = UNetNodeTransformer(unet=unet, unet_out_channels=4, pos_feat_dim=4)
    model.eval()
    return model


def _sequential_tta(model, imgs, cfg, device, kind: str):
    unet_out, det_logits = _encode(model, imgs, cfg, device)
    width = len(det_logits)
    if kind == 'none' or (kind == 'full' and not cfg.det_tta):
        native = [value.clone() for value in det_logits]
        return unet_out, det_logits, native
    views = 1
    for dims in [(-1,), (-2,), (-2, -1)]:
        flipped = imgs.flip(dims)
        _, det_flip = _encode(model, flipped, cfg, device)
        for frame in range(width):
            det_logits[frame] = det_logits[frame] + det_flip[frame].flip(dims)
        views += 1
    native = [value / views for value in det_logits]
    if kind == 'flips':
        det_logits = [value / views for value in det_logits]
        return unet_out, det_logits, native
    for k in (1, 3):
        rotated = torch.rot90(imgs, k, dims=(-2, -1))
        _, det_rot = _encode(model, rotated, cfg, device)
        for frame in range(width):
            det_logits[frame] = det_logits[frame] + torch.rot90(det_rot[frame], -k, dims=(-2, -1))
        views += 1
    transposed = imgs.transpose(-1, -2)
    _, det_t = _encode(model, transposed, cfg, device)
    for frame in range(width):
        det_logits[frame] = det_logits[frame] + det_t[frame].transpose(-1, -2)
    views += 1
    alt = torch.rot90(imgs, 1, dims=(-2, -1)).transpose(-1, -2)
    _, det_at = _encode(model, alt, cfg, device)
    for frame in range(width):
        det_logits[frame] = det_logits[frame] + torch.rot90(
            det_at[frame].transpose(-1, -2), -1, dims=(-2, -1)
        )
    views += 1
    det_logits = [value / views for value in det_logits]
    return unet_out, det_logits, native


@torch.no_grad()
def test_batched_tta_matches_sequential_full_and_flips() -> None:
    torch.manual_seed(0)
    model = _tiny_detector()
    cfg = PredictConfig(det_tta=True, amp_fp16=False, show_progress=False)
    device = torch.device('cpu')
    imgs = torch.rand(1, 2, 8, 16, 16)
    for kind in ('full', 'flips', 'none'):
        batched = encode_detection_views(model, imgs, cfg, device, kind)
        sequential = _sequential_tta(model, imgs, cfg, device, kind)
        assert torch.allclose(batched[0], sequential[0], atol=1e-5, rtol=1e-5)
        for left, right in zip(batched[1], sequential[1]):
            assert torch.allclose(left, right, atol=1e-5, rtol=1e-5)
        for left, right in zip(batched[2], sequential[2]):
            assert torch.allclose(left, right, atol=1e-5, rtol=1e-5)


@torch.no_grad()
def test_tta_unet_features_drop_other_view_storage() -> None:
    torch.manual_seed(2)
    model = _tiny_detector()
    cfg = PredictConfig(det_tta=True, amp_fp16=False, show_progress=False)
    device = torch.device('cpu')
    imgs = torch.rand(1, 2, 8, 16, 16)
    unet_out, _det, _native = encode_detection_views(model, imgs, cfg, device, 'full')
    logical = unet_out.numel() * unet_out.element_size()
    storage = unet_out.untyped_storage().nbytes()
    assert storage <= logical * 2
    assert storage < logical * 8


@torch.no_grad()
def test_window_batch_matches_per_window_encode() -> None:
    torch.manual_seed(1)
    model = _tiny_detector()
    cfg = PredictConfig(det_tta=True, amp_fp16=False, show_progress=False)
    device = torch.device('cpu')
    windows = torch.rand(3, 2, 8, 16, 16)
    batched = encode_detection_views(model, windows, cfg, device, 'full')
    for index in range(windows.shape[0]):
        single = encode_detection_views(model, windows[index : index + 1], cfg, device, 'full')
        assert torch.allclose(batched[0][index : index + 1], single[0], atol=1e-5, rtol=1e-5)
        for left, right in zip(batched[1], single[1]):
            assert torch.allclose(left[index : index + 1], right, atol=1e-5, rtol=1e-5)
        for left, right in zip(batched[2], single[2]):
            assert torch.allclose(left[index : index + 1], right, atol=1e-5, rtol=1e-5)
