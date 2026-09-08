import multiprocessing as mp
import os
import random

import numpy as np
import torch


class SharedEpoch:
    def __init__(self, epoch: int = 0) -> None:
        self._epoch = mp.Value('i', int(epoch), lock=False)

    def set(self, epoch: int) -> None:
        self._epoch.value = int(epoch)

    def get(self) -> int:
        return int(self._epoch.value)


def sample_rng(seed: int, epoch: int, index: int) -> random.Random:
    return random.Random(_sample_seed(seed, epoch, index))


def sample_numpy_rng(seed: int, epoch: int, index: int) -> np.random.Generator:
    return np.random.default_rng(_sample_seed(seed, epoch, index))


def _sample_seed(seed: int, epoch: int, index: int) -> int:
    return (int(seed) + 1) * 1_000_003 + int(epoch) * 1_000_033 + int(index)


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def seed_worker(worker_id: int) -> None:
    torch.set_num_threads(1)
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def dataloader_generator(seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator
