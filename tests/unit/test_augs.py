import numpy as np
import torch

from biohub.augmentations.bleach import bleach_augment
from biohub.augmentations.blur import blur_augment
from biohub.augmentations.contrast import contrast_augment
from biohub.augmentations.cutout import cutout_augment
from biohub.augmentations.flip import flip_augment
from biohub.augmentations.gamma import gamma_augment
from biohub.augmentations.haze import haze_augment
from biohub.augmentations.noise import noise_augment
from biohub.augmentations.poisson import poisson_augment
from biohub.augmentations.rot90 import rot90_augment
from biohub.augmentations.scale import scale_augment
from biohub.augmentations.time import time_stretch_augment, time_warp_augment
from biohub.augmentations.translate import translate_augment


class _FixedRng:
    def __init__(self, random_value: float = 0.0, integers_value: int = 2) -> None:
        self.random_value = random_value
        self.integers_value = integers_value

    def random(self, *args, **kwargs):
        if args:
            return np.full(args[0], self.random_value)
        return self.random_value

    def integers(self, low, high=None, size=None):
        return self.integers_value

    def normal(self, loc, scale, size):
        return np.ones(size, dtype=np.float32) * scale

    def uniform(self, low, high):
        return (low + high) / 2

    def poisson(self, lam):
        return np.asarray(lam, dtype=np.float32)


def test_proba_zero_is_identity() -> None:
    imgs = torch.rand(2, 2, 4, 4)
    coords = torch.tensor([[[1.0, 1.0, 2.0], [1.0, 2.0, 1.0]]])
    masks = torch.ones(1, 2, dtype=torch.bool)
    rng = np.random.default_rng(0)
    for aug in (
        lambda **kw: noise_augment(proba=0.0, **kw),
        lambda **kw: contrast_augment(proba=0.0, **kw),
        lambda **kw: gamma_augment(proba=0.0, **kw),
        lambda **kw: rot90_augment(proba=0.0, **kw),
        lambda **kw: translate_augment(proba=0.0, **kw),
        lambda **kw: cutout_augment(proba=0.0, **kw),
        lambda **kw: flip_augment(proba=0.0, **kw),
        lambda **kw: time_stretch_augment(proba=0.0, **kw),
        lambda **kw: time_warp_augment(proba=0.0, **kw),
        lambda **kw: blur_augment(proba=0.0, **kw),
        lambda **kw: scale_augment(proba=0.0, **kw),
        lambda **kw: bleach_augment(proba=0.0, **kw),
        lambda **kw: poisson_augment(proba=0.0, **kw),
        lambda **kw: haze_augment(proba=0.0, **kw),
    ):
        out_i, out_c, out_m = aug(
            imgs=imgs.clone(), coords=coords.clone(), masks=masks.clone(), rng=rng
        )
        torch.testing.assert_close(out_i, imgs)
        torch.testing.assert_close(out_c, coords)
        assert torch.equal(out_m, masks)


def test_noise_and_cutout_keep_coords() -> None:
    imgs = torch.ones(1, 2, 4, 4)
    coords = torch.tensor([[[0.0, 1.0, 1.0]]])
    masks = torch.ones(1, 1, dtype=torch.bool)
    rng = _FixedRng(integers_value=0)
    noisy, ncoords, _ = noise_augment(imgs, coords, masks, rng=rng, proba=1.0, std=0.2)
    assert not torch.equal(noisy, imgs)
    torch.testing.assert_close(ncoords, coords)
    cut, ccoords, _ = cutout_augment(imgs, coords, masks, rng=rng, proba=1.0, holes=1, size=2)
    torch.testing.assert_close(ccoords, coords)
    assert float(cut.min()) == 0.0


def test_rot90_k2_agrees_with_coords() -> None:
    imgs = torch.zeros(1, 2, 4, 4)
    imgs[0, 0, 1, 2] = 1.0
    coords = torch.tensor([[[0.0, 1.0, 2.0]]])
    masks = torch.ones(1, 1, dtype=torch.bool)
    out_i, out_c, _ = rot90_augment(imgs, coords, masks, rng=_FixedRng(integers_value=2), proba=1.0)
    assert float(out_i[0, 0, 2, 1]) == 1.0
    torch.testing.assert_close(out_c[0, 0], torch.tensor([0.0, 2.0, 1.0]))


def test_flip_all_axes_agrees_with_coords() -> None:
    imgs = torch.zeros(1, 2, 4, 4)
    imgs[0, 0, 1, 2] = 1.0
    coords = torch.tensor([[[0.0, 1.0, 2.0]]])
    masks = torch.ones(1, 1, dtype=torch.bool)
    _out_i, out_c, _ = flip_augment(imgs, coords, masks, rng=_FixedRng(random_value=0.0), proba=1.0)
    torch.testing.assert_close(out_c[0, 0], torch.tensor([1.0, 2.0, 1.0]))


def test_translate_masks_nodes_outside() -> None:
    imgs = torch.ones(1, 2, 4, 4)
    coords = torch.tensor([[[0.0, 0.0, 0.0]]])
    masks = torch.ones(1, 1, dtype=torch.bool)
    rng = _FixedRng(integers_value=-3)
    _out_i, out_c, out_m = translate_augment(imgs, coords, masks, rng=rng, px=4, proba=1.0)
    assert not bool(out_m[0, 0])
    assert float(out_c[0, 0, 1]) == -3.0


def test_seeded_noise_repeats() -> None:
    imgs = torch.ones(1, 2, 4, 4)
    coords = torch.zeros(1, 1, 3)
    masks = torch.ones(1, 1, dtype=torch.bool)
    a = noise_augment(imgs, coords, masks, rng=np.random.default_rng(3), proba=1.0)
    b = noise_augment(imgs, coords, masks, rng=np.random.default_rng(3), proba=1.0)
    torch.testing.assert_close(a[0], b[0])


def test_time_domain_and_blur_keep_coords() -> None:
    imgs = torch.rand(3, 2, 4, 4)
    coords = torch.tensor([[[0.0, 1.0, 2.0]]])
    masks = torch.ones(1, 1, dtype=torch.bool)
    rng = np.random.default_rng(4)
    for fn in (
        time_stretch_augment,
        time_warp_augment,
        blur_augment,
        bleach_augment,
        poisson_augment,
        haze_augment,
    ):
        _out_i, out_c, out_m = fn(imgs.clone(), coords.clone(), masks.clone(), rng=rng, proba=1.0)
        torch.testing.assert_close(out_c, coords)
        assert torch.equal(out_m, masks)


def test_scale_aug_rounded_identity_preserves_coords() -> None:
    imgs = torch.zeros(1, 2, 4, 4)
    imgs[0, 0, 1, 2] = 1.0
    coords = torch.tensor([[[0.0, 1.0, 2.0]]])
    masks = torch.ones(1, 1, dtype=torch.bool)

    class _HighRng(_FixedRng):
        def uniform(self, low, high):
            return high

    out_i, out_c, _ = scale_augment(
        imgs, coords, masks, rng=_HighRng(), proba=1.0, scale_range=0.2
    )
    # floor(4 * 1.2) == 4: interpolate is identity, so labels must not move.
    torch.testing.assert_close(out_i, imgs)
    torch.testing.assert_close(out_c, coords)
