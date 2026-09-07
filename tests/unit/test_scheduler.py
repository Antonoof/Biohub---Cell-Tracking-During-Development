import multiprocessing as mp
import time
from types import SimpleNamespace

import pytest

from biohub.infer.run import (
    assign_deepcenter_devices,
    logical_cuda_device,
    reap_dead_workers,
    scheduler_should_stop,
    stop_upgrade_workers,
    upgrade_worker_count,
)


def test_reap_dead_workers_clears_busy_slot() -> None:
    workers = [
        SimpleNamespace(is_alive=lambda: False),
        SimpleNamespace(is_alive=lambda: True),
    ]
    worker_task: dict[int, tuple[str, str] | None] = {
        0: ('movie_a', 'combined'),
        1: ('movie_b', 'combined'),
    }
    dead = reap_dead_workers(workers, worker_task)
    assert dead == [(0, 'movie_a', 'combined')]
    assert worker_task[0] is None
    assert worker_task[1] == ('movie_b', 'combined')


def test_scheduler_stops_on_deadline_even_if_busy() -> None:
    assert scheduler_should_stop(
        remaining=-1.0,
        producer_done=False,
        busy=True,
        pending=True,
        complete=False,
    )
    assert not scheduler_should_stop(
        remaining=10.0,
        producer_done=False,
        busy=True,
        pending=True,
        complete=False,
    )
    assert scheduler_should_stop(
        remaining=10.0,
        producer_done=True,
        busy=False,
        pending=False,
        complete=True,
    )


def test_cuda_upgrade_workers_are_capped_to_gpu_pool() -> None:
    assert upgrade_worker_count(cpu_workers=64, graph_device='cuda', deepcenter_gpu_workers=1) == 1
    assert upgrade_worker_count(cpu_workers=64, graph_device='cpu', deepcenter_gpu_workers=1) == 64


def test_assign_deepcenter_devices_pins_explicit_pool() -> None:
    cpu = assign_deepcenter_devices(
        n_workers=4,
        graph_device='cpu',
        gpu_workers=8,
        explicit=('cuda:1',),
        cuda_tokens=['0', '1'],
    )
    assert cpu == ['cpu', 'cpu', 'cpu', 'cpu']
    pinned = assign_deepcenter_devices(
        n_workers=2,
        graph_device='cuda',
        gpu_workers=2,
        explicit=('cuda:1',),
        cuda_tokens=['0', '1'],
    )
    assert pinned == ['cuda:1', 'cuda:1']
    remapped = assign_deepcenter_devices(
        n_workers=1,
        graph_device='cuda',
        gpu_workers=1,
        explicit=(),
        cuda_tokens=['1'],
    )
    assert remapped == ['cuda:0']
    uuid_pool = assign_deepcenter_devices(
        n_workers=1,
        graph_device='cuda',
        gpu_workers=1,
        explicit=(),
        cuda_tokens=['GPU-abc'],
    )
    assert uuid_pool == ['cuda:0']
    physical = assign_deepcenter_devices(
        n_workers=2,
        graph_device='cuda',
        gpu_workers=2,
        explicit=(),
        cuda_tokens=['1', '2'],
    )
    assert physical == ['cuda:0', 'cuda:1']


def test_explicit_cuda_ids_are_logical_when_physical_ids_overlap() -> None:
    tokens = ['1', '2']
    assert assign_deepcenter_devices(
        n_workers=1,
        graph_device='cuda',
        gpu_workers=1,
        explicit=('cuda:1',),
        cuda_tokens=tokens,
    ) == ['cuda:1']
    assert assign_deepcenter_devices(
        n_workers=2,
        graph_device='cuda',
        gpu_workers=2,
        explicit=('cuda:0', 'cuda:1'),
        cuda_tokens=tokens,
    ) == ['cuda:0', 'cuda:1']
    assert assign_deepcenter_devices(
        n_workers=1,
        graph_device='cuda',
        gpu_workers=1,
        explicit=('1',),
        cuda_tokens=tokens,
    ) == ['cuda:0']
    with pytest.raises(ValueError, match='outside visible pool'):
        logical_cuda_device('cuda:2', tokens)
    assert logical_cuda_device('2', tokens) == 'cuda:1'


def _hang_forever() -> None:
    while True:
        time.sleep(30)


def test_stop_upgrade_workers_kills_hung_process() -> None:
    context = mp.get_context('spawn')
    task_queue = context.Queue()
    process = context.Process(target=_hang_forever, name='hung-upgrade', daemon=False)
    process.start()
    try:
        assert process.is_alive()
        stop_upgrade_workers([process], [task_queue], timeout=0.2)
        assert not process.is_alive()
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=2)
