from pathlib import Path

import numpy as np
import torch

from biohub.augmentations.brightness import brightness_augment
from biohub.data.windows import (
    FrameWindowData,
    FrameWindowDataset,
    VideoGroupedSampler,
    VideoMeta,
    collate_windows,
    pad_window,
)


def _fake_volume(_path: Path) -> np.ndarray:
    return np.full((4, 4, 8, 8), 0.5, dtype=np.float32)


def _window() -> FrameWindowData:
    return FrameWindowData(
        t_start=0,
        n_frames=2,
        pos_feats=[torch.zeros(2, 4), torch.zeros(2, 4)],
        coords=[torch.zeros(2, 3), torch.zeros(2, 3)],
        node_counts=[2, 2],
        targets=[torch.zeros(2, 2)],
    )


def _dataset(seed: int) -> FrameWindowDataset:
    meta = VideoMeta(
        zarr_path=Path('/tmp/fake.zarr'),
        image_shape=(4, 4, 8, 8),
        downsample=(1, 1, 1),
        voxel_size=(1.0, 1.0, 1.0),
        q_low=0.0,
        q_high=1.0,
    )
    return FrameWindowDataset(
        [(meta, [_window()])],
        max_nodes=2,
        augmentations=[brightness_augment],
        seed=seed,
    )


def test_frame_window_augs_are_seeded(monkeypatch) -> None:
    monkeypatch.setattr('biohub.data.windows._zarr_array', _fake_volume)
    first = _dataset(7)
    second = _dataset(7)
    left = first[0]['imgs']
    right = second[0]['imgs']
    torch.testing.assert_close(left, right)
    torch.testing.assert_close(first[0]['imgs'], first[0]['imgs'])
    first.set_epoch(1)
    later = first[0]['imgs']
    assert not torch.equal(left, later)
    other_seed = _dataset(8)
    assert not torch.equal(right, other_seed[0]['imgs'])


def test_getitem_omits_unused_pos_feats(monkeypatch) -> None:
    monkeypatch.setattr('biohub.data.windows._zarr_array', _fake_volume)
    sample = _dataset(0)[0]
    assert 'pos_feats' not in sample
    assert sample['imgs'].dtype == torch.float32
    batch = collate_windows([sample, sample])
    assert 'pos_feats' not in batch


def test_pad_window_omits_pos_feats() -> None:
    padded = pad_window(_window(), 2)
    assert 'pos_feats' not in padded
    assert padded['coords'].shape == (2, 2, 3)
    assert padded['targets'].shape == (1, 2, 2)


def test_video_grouped_sampler_covers_all_indices() -> None:
    meta_a = VideoMeta(Path('/a.zarr'), (4, 4, 8, 8), (1, 1, 1), (1.0, 1.0, 1.0), 0.0, 1.0)
    meta_b = VideoMeta(Path('/b.zarr'), (4, 4, 8, 8), (1, 1, 1), (1.0, 1.0, 1.0), 0.0, 1.0)
    window = _window()
    dataset = FrameWindowDataset(
        [(meta_a, [window, window]), (meta_b, [window])],
        max_nodes=2,
    )
    generator = torch.Generator()
    generator.manual_seed(0)
    order = list(VideoGroupedSampler(dataset, generator=generator, num_workers=0))
    assert sorted(order) == list(range(len(dataset)))
    for group in dataset.video_index_groups():
        start = order.index(group[0])
        assert order[start : start + len(group)] == group
    interleaved = list(VideoGroupedSampler(dataset, generator=generator, num_workers=2))
    assert sorted(interleaved) == list(range(len(dataset)))
    for group in dataset.video_index_groups():
        seen = [index for index in interleaved if index in set(group)]
        assert seen == group
