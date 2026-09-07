from pathlib import Path

import numpy as np
import torch

from biohub.augmentations.brightness import brightness_augment
from biohub.data.windows import FrameWindowData, FrameWindowDataset, VideoMeta


def _fake_volume(_path: Path) -> np.ndarray:
    return np.full((4, 4, 8, 8), 0.5, dtype=np.float32)


def _dataset(seed: int) -> FrameWindowDataset:
    meta = VideoMeta(
        zarr_path=Path('/tmp/fake.zarr'),
        image_shape=(4, 4, 8, 8),
        downsample=(1, 1, 1),
        voxel_size=(1.0, 1.0, 1.0),
        q_low=0.0,
        q_high=1.0,
    )
    window = FrameWindowData(
        t_start=0,
        n_frames=2,
        pos_feats=[torch.zeros(2, 4), torch.zeros(2, 4)],
        coords=[torch.zeros(2, 3), torch.zeros(2, 3)],
        node_counts=[2, 2],
        targets=[torch.zeros(2, 2)],
    )
    return FrameWindowDataset(
        [(meta, [window])],
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
