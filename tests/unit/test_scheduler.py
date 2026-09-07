import multiprocessing as mp
import time
from types import SimpleNamespace

from biohub.infer.run import (
    assign_deepcenter_devices,
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
    worker_task = {0: ('movie_a', 'combined'), 1: ('movie_b', 'combined')}
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
