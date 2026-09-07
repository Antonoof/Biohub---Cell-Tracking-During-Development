import multiprocessing as mp
import os
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import TypeVar

T = TypeVar('T')
R = TypeVar('R')


def worker_count(n_items: int, cap: int = 32) -> int:
    return max(1, min(cap, n_items, os.cpu_count() or 1))


def ordered_process_map(
    fn: Callable[[T], R],
    items: Sequence[T],
    max_workers: int | None = None,
) -> list[R]:
    if not items:
        return []
    workers = worker_count(len(items) if max_workers is None else min(len(items), max_workers))
    if workers == 1 or len(items) == 1:
        return [fn(item) for item in items]
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context('spawn'),
    ) as pool:
        return list(pool.map(fn, items))


def ordered_thread_map(
    fn: Callable[[T], R],
    items: Iterable[T],
    max_workers: int | None = None,
) -> list[R]:
    sequence = list(items)
    if not sequence:
        return []
    workers = worker_count(
        len(sequence) if max_workers is None else min(len(sequence), max_workers)
    )
    if workers == 1 or len(sequence) == 1:
        return [fn(item) for item in sequence]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, sequence))
