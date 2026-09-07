import argparse
import csv
import importlib
import json
import multiprocessing as mp
import os
import queue as stdlib_queue
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import polars as pl
import torch

from biohub.infer.config import TrackingConfig, load_tracking_config
from biohub.log import LOGGER, log_event, setup_logging
from biohub.modules.graph.upgrade import GraphUpgrade
from biohub.paths import PROJECT_ROOT
from biohub.utils.runs import finish_run, init_run, new_run_id
from biohub.utils.yaml_config import load_yaml, resolve_path
from biohub.validation.splits import panel_movie_ids


class UpgradeDeadline(TimeoutError):
    pass


def _on_alarm(signum, frame) -> None:
    raise UpgradeDeadline('upgrade reached the submission-safe deadline')


def deepcenter_uses_cuda(device: str) -> bool:
    return str(device).startswith('cuda')


def logical_cuda_pool(cuda_tokens: list[str]) -> list[str]:
    return [f'cuda:{index}' for index in range(len(cuda_tokens))]


def logical_cuda_device(device: str, cuda_tokens: list[str]) -> str:
    text = str(device).strip()
    if not text or text == 'cpu' or not cuda_tokens:
        return 'cpu'
    if text == 'cuda':
        return 'cuda:0'
    if text.startswith('cuda:') and text[5:].isdigit():
        index = int(text[5:])
        if index < 0 or index >= len(cuda_tokens):
            raise ValueError(
                f'Logical CUDA device {text} is outside visible pool '
                f'cuda:0..cuda:{len(cuda_tokens) - 1}'
            )
        return f'cuda:{index}'
    token = text[5:] if text.startswith('cuda:') else text
    if token in cuda_tokens:
        return f'cuda:{cuda_tokens.index(token)}'
    raise ValueError(f'CUDA device {text} is not in the visible mask {cuda_tokens}')


def assign_deepcenter_devices(
    *,
    n_workers: int,
    graph_device: str,
    gpu_workers: int,
    explicit: tuple[str, ...] | list[str],
    cuda_tokens: list[str],
) -> list[str]:
    if n_workers <= 0:
        return []
    if not deepcenter_uses_cuda(graph_device):
        return ['cpu'] * n_workers
    n_gpu = max(0, int(gpu_workers))
    named = [str(item) for item in explicit if str(item)]
    if named:
        pool = [logical_cuda_device(item, cuda_tokens) for item in named]
    elif graph_device not in {'cuda', 'cpu'}:
        pool = [logical_cuda_device(graph_device, cuda_tokens)]
    else:
        pool = logical_cuda_pool(cuda_tokens)
    pool = [item for item in pool if item != 'cpu']
    if n_gpu <= 0 or not pool:
        return ['cpu'] * n_workers
    pool = pool[:n_gpu]
    return [pool[index % len(pool)] for index in range(n_workers)]


def upgrade_worker_count(
    *,
    cpu_workers: int,
    graph_device: str,
    deepcenter_gpu_workers: int,
) -> int:
    wanted = max(1, int(cpu_workers))
    if not deepcenter_uses_cuda(graph_device):
        return wanted
    return max(1, min(wanted, max(1, int(deepcenter_gpu_workers))))


def stop_upgrade_workers(workers: list, task_queues: list, timeout: float = 5.0) -> None:
    for worker_id, task_queue in enumerate(task_queues):
        if worker_id < len(workers) and workers[worker_id].is_alive():
            try:
                task_queue.put_nowait(None)
            except Exception:
                pass
    deadline = time.monotonic() + timeout
    for process in workers:
        remaining = max(0.0, deadline - time.monotonic())
        process.join(timeout=remaining)
        if process.is_alive():
            process.terminate()
            process.join(timeout=timeout)
        if process.is_alive() and hasattr(process, 'kill'):
            process.kill()
            process.join(timeout=timeout)


def _thread_budget(thread_boost: bool, cpu_total: int, active, live) -> int:
    if not thread_boost:
        return 1
    active_count = max(1, int(active.value))
    reserved = max(0, int(live.value))
    return max(1, (cpu_total - reserved) // active_count)


def _upgrade_worker_entry(
    worker_id: int,
    task_queue,
    results,
    active,
    live,
    tracking: TrackingConfig,
    deepcenter_device: str,
    deadline: float,
    started: float,
    thread_boost: bool,
    cpu_total: int,
) -> None:
    signal.signal(signal.SIGALRM, _on_alarm)
    worker_upgrade = GraphUpgrade(tracking)
    worker_upgrade.deepcenter_device = deepcenter_device
    run_upgrade(worker_upgrade)
    worker_upgrade.set_thread_retune_hook(
        lambda: worker_upgrade.set_thread_budget(
            _thread_budget(thread_boost, cpu_total, active, live)
        )
    )
    worker_upgrade.set_thread_budget(1)
    while True:
        task = task_queue.get()
        if task is None:
            return
        dataset, graph_path, mode = task
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            results.put(
                {
                    'worker_id': worker_id,
                    'dataset': dataset,
                    'mode': mode,
                    'status': 'deadline',
                    'error': 'no time left',
                }
            )
            continue
        with active.get_lock():
            active.value += 1
        signal.setitimer(signal.ITIMER_REAL, max(remaining, 1.0))
        try:
            worker_upgrade.retune_threads()
            if mode == 'emergency':
                row = worker_upgrade.emergency_shard(Path(graph_path))
            else:
                row = worker_upgrade.process_graph(Path(graph_path), mode)
            results.put(
                {
                    'worker_id': worker_id,
                    'dataset': dataset,
                    'mode': mode,
                    'status': 'ok',
                    'seconds': time.monotonic() - started,
                    'row': row,
                }
            )
        except UpgradeDeadline as error:
            results.put(
                {
                    'worker_id': worker_id,
                    'dataset': dataset,
                    'mode': mode,
                    'status': 'deadline',
                    'error': str(error),
                }
            )
        except BaseException as error:
            results.put(
                {
                    'worker_id': worker_id,
                    'dataset': dataset,
                    'mode': mode,
                    'status': 'error',
                    'error': f'{type(error).__name__}: {error}',
                    'traceback': traceback.format_exc(limit=12),
                }
            )
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            worker_upgrade.set_thread_budget(1)
            with active.get_lock():
                active.value -= 1


def _video_cost(test_dir: Path, stem: str) -> float:
    for meta_path in (
        test_dir / f'{stem}.zarr' / '0' / 'zarr.json',
        test_dir / f'{stem}.zarr' / '0' / '.zarray',
    ):
        try:
            shape = json.loads(meta_path.read_text())['shape']
        except Exception:
            continue
        cost = 1.0
        for extent in shape:
            cost *= float(extent)
        return cost
    return 1.0


def scheduler_should_stop(
    *,
    remaining: float,
    producer_done: bool,
    busy: bool,
    pending: bool,
    complete: bool,
) -> bool:
    if remaining <= 0:
        return True
    if complete and not busy and not pending:
        return True
    if producer_done and not busy and not pending:
        return True
    return False


def reap_dead_workers(
    workers: list,
    worker_task: dict[int, tuple[str, str] | None],
) -> list[tuple[int, str, str]]:
    dead: list[tuple[int, str, str]] = []
    for worker_id, task in list(worker_task.items()):
        if task is None:
            continue
        process = workers[worker_id]
        alive = process.is_alive() if hasattr(process, 'is_alive') else True
        if not alive:
            worker_task[worker_id] = None
            dead.append((worker_id, task[0], task[1]))
    return dead


def assemble_submission(
    stems: list[str],
    shard_dir: Path,
    submission_path: Path,
    csv_columns: list[str],
    missing_ok: bool = False,
) -> tuple[int, int]:
    row_id = node_rows = edge_rows = 0
    with submission_path.open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=csv_columns)
        writer.writeheader()
        for dataset in sorted(stems):
            shard = shard_dir / f'{dataset}.csv'
            if not shard.exists():
                if missing_ok:
                    continue
                raise FileNotFoundError(f'missing shard: {shard}')
            with shard.open(newline='') as handle:
                for row in csv.DictReader(handle):
                    row['id'] = row_id
                    writer.writerow(row)
                    row_id += 1
                    node_rows += row['row_type'] == 'node'
                    edge_rows += row['row_type'] == 'edge'
    if node_rows <= 0 or row_id != node_rows + edge_rows:
        if missing_ok and row_id == node_rows + edge_rows:
            return node_rows, edge_rows
        raise AssertionError('submission assembly produced an empty or inconsistent csv')
    return node_rows, edge_rows


def build_tracking(
    *,
    config_path: Path | None = None,
    bundle_dir: Path,
    test_dir: Path,
    work_dir: Path,
    bundle_paths: dict[str, Path] | None = None,
) -> TrackingConfig:
    return load_tracking_config(
        config_path=config_path,
        bundle_dir=bundle_dir,
        test_dir=test_dir,
        work_dir=work_dir,
        bundle_paths=bundle_paths,
    )


UPGRADE_STAGES = (
    'biohub.infer.04_unigraft',
    'biohub.infer.05_live_v2',
    'biohub.infer.06_ownership',
    'biohub.infer.07_edgegraft',
    'biohub.infer.08_candidategraft',
    'biohub.infer.09_motion',
    'biohub.infer.10_deepcenter',
)


def run_upgrade(upgrade: GraphUpgrade) -> GraphUpgrade:
    for stage in UPGRADE_STAGES:
        importlib.import_module(stage).apply(upgrade)
    return upgrade


def _predictor_environment(tracking: TrackingConfig) -> dict[str, str]:
    speed = tracking.speed
    association = tracking.association
    max_bytes = int(speed.uncompressed_evidence_max_mb * 1024 * 1024)
    src = str(PROJECT_ROOT / 'src')
    pythonpath = os.environ.get('PYTHONPATH', '')
    return {
        'BIOHUB_UNCOMPRESSED_EVIDENCE': '1' if speed.uncompressed_evidence else '0',
        'BIOHUB_UNCOMPRESSED_EVIDENCE_MAX_BYTES': str(max_bytes),
        'BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION': str(association.minimum_candidate_retention),
        'BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT': str(association.bidirectional_edge_weight),
        'BIOHUB_RETENTION_GUARD_SECONDARY_EDGE_WEIGHT': str(
            association.guarded_secondary_edge_weight
        ),
        'BIOHUB_CUDNN_BENCHMARK': '1' if speed.cudnn_benchmark else '0',
        'BIOHUB_ROOT': str(PROJECT_ROOT),
        'PYTHONPATH': src if not pythonpath else src + os.pathsep + pythonpath,
    }


def detect_job(
    *,
    movie_ids: list[str],
    test_dir: Path,
    predictions_dir: Path,
    tracking: TrackingConfig,
    method: str = 'unet_transformer',
    helper_deadline_epoch: float = float('inf'),
    gpu_shard: str | None = None,
    show_progress: bool = True,
) -> dict:
    bundle = tracking.bundle
    detection = tracking.detection
    association = tracking.association
    ilp = tracking.ilp
    paths = tracking.paths
    speed = tracking.speed
    shard = gpu_shard if gpu_shard is not None else method
    return {
        'movie_ids': list(movie_ids),
        'data_dir': str(Path(test_dir).resolve()),
        'output_dir': str(Path(predictions_dir).resolve()),
        'p1_weights': str(Path(bundle.p1_checkpoint).resolve()),
        'p2_weights': str(Path(bundle.p2_checkpoint).resolve()),
        'model_c_weights': str(Path(bundle.model_c_checkpoint).resolve()),
        'p1_evidence_dir': str(Path(paths.p1_evidence_dir).resolve()),
        'p2_evidence_dir': str(Path(paths.p2_evidence_dir).resolve()),
        'model_c_evidence_dir': str(Path(paths.model_c_evidence_dir).resolve()),
        'working_dir': str(Path(paths.working_dir).resolve()),
        'det_threshold': detection.threshold,
        'det_tta': detection.det_tta,
        'edge_activation': detection.edge_activation,
        'subvoxel_refinement': detection.subvoxel_refinement,
        'amp_fp16': speed.amp_fp16,
        'pool_kernel_um': detection.pool_kernel_um,
        'use_ilp': ilp.use,
        'ilp_edge_weight': ilp.edge_weight,
        'ilp_appearance_weight': ilp.appearance,
        'ilp_disappearance_weight': ilp.disappearance,
        'ilp_division_weight': ilp.division,
        'unet_batch_size': detection.unet_batch_size,
        'method': method,
        'gpu_shard': shard,
        'show_progress': show_progress,
        'helper_deadline_epoch': helper_deadline_epoch,
        'secondary_edge_weight': association.secondary_edge_weight,
        'secondary_detection_weight': association.secondary_detection_weight,
        'secondary_link_mode': association.secondary_link_mode,
        'secondary_mix_temperature': association.secondary_mix_temperature,
        'secondary_low_margin_max': association.secondary_low_margin_max,
        'division_det_threshold': detection.division_det_threshold,
        'division_pool_um': detection.division_pool_um,
        'division_radius_um': detection.division_radius_um,
        'division_topk': detection.division_topk,
        'division_min_probability': detection.division_min_probability,
        'native_evidence_det_threshold': detection.threshold,
        'native_evidence_pool_um': detection.native_evidence_pool_um,
        'native_evidence_radius_um': detection.native_evidence_radius_um,
        'native_evidence_topk': detection.native_evidence_topk,
        'native_evidence_min_probability': detection.native_evidence_min_probability,
        'uncompressed_evidence': speed.uncompressed_evidence,
        'uncompressed_evidence_max_bytes': int(speed.uncompressed_evidence_max_mb * 1024 * 1024),
        'dual_seed_min_candidate_retention': association.minimum_candidate_retention,
        'bidirectional_edge_weight': association.bidirectional_edge_weight,
        'retention_guard_secondary_edge_weight': association.guarded_secondary_edge_weight,
        'cudnn_benchmark': speed.cudnn_benchmark,
        'environment': _predictor_environment(tracking),
    }


def run_detect_jobs(
    *,
    movie_ids: list[str],
    test_dir: Path,
    predictions_dir: Path,
    tracking: TrackingConfig,
    gpu_tokens: list[str],
    gpu_workers: int,
    longest_first: bool,
    deadline: float,
) -> tuple[dict[int, subprocess.Popen], list[str], float]:
    predictions_dir.mkdir(parents=True, exist_ok=True)
    video_cost = {stem: _video_cost(test_dir, stem) for stem in movie_ids}
    cost_order = sorted(movie_ids, key=lambda stem: (-video_cost[stem], stem))
    shard_count = max(1, min(gpu_workers, len(gpu_tokens) or 1, len(movie_ids) or 1))
    if shard_count < 2:
        shard_stems = [list(movie_ids)]
    elif longest_first:
        load = [0.0] * shard_count
        shard_stems = [[] for _ in range(shard_count)]
        for stem in cost_order:
            index = min(range(shard_count), key=lambda key: (load[key], key))
            shard_stems[index].append(stem)
            load[index] += video_cost[stem]
    else:
        shard_stems = [list(movie_ids[index::shard_count]) for index in range(shard_count)]

    helper_deadline_epoch = time.time() + max(0.0, deadline - time.monotonic())
    predict_processes: dict[int, subprocess.Popen] = {}
    predict_methods: list[str] = []
    predict_started = time.monotonic()
    method = 'unet_transformer'
    environment_knobs = _predictor_environment(tracking)
    for index, stems in enumerate(shard_stems):
        if not stems:
            continue
        gpu_method = f'{method}_gpu{index}' if shard_count >= 2 else method
        job_path = predictions_dir / f'predict_job_{index}.json'
        job = detect_job(
            movie_ids=stems,
            test_dir=test_dir,
            predictions_dir=predictions_dir,
            tracking=tracking,
            method=gpu_method,
            helper_deadline_epoch=helper_deadline_epoch,
            gpu_shard=gpu_method,
        )
        job_path.write_text(json.dumps(job, indent=2) + '\n')
        command = [sys.executable, '-m', 'biohub.infer.01_detect', '--job', str(job_path)]
        environment = dict(os.environ)
        environment.update(environment_knobs)
        if gpu_tokens:
            environment['CUDA_VISIBLE_DEVICES'] = gpu_tokens[index % len(gpu_tokens)]
        predict_methods.append(gpu_method)
        predict_processes[index] = subprocess.Popen(command, cwd=predictions_dir, env=environment)
    return predict_processes, predict_methods, predict_started


def run(
    *,
    movie_ids: list[str],
    bundle_dir: Path,
    test_dir: Path,
    train_dir: Path,
    run_dir: Path,
    gpu_count: int,
    config: TrackingConfig | None = None,
) -> dict:
    run_dir = run_dir.resolve()
    setup_logging(run_dir)
    work_dir = run_dir / 'workdir'
    work_dir.mkdir(parents=True, exist_ok=True)
    tracking = config or load_tracking_config(
        pipeline='baseline',
        bundle_dir=bundle_dir,
        test_dir=test_dir,
        work_dir=work_dir,
    )

    predictions_dir = work_dir / 'predictions'
    predictions_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    runtime = tracking.runtime
    speed = tracking.speed
    deadline = started + runtime.hard_limit_seconds - runtime.finalize_reserve_seconds

    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    cuda_tokens = (
        [token.strip() for token in visible.split(',') if token.strip()]
        if visible and visible != '-1'
        else [str(index) for index in range(torch.cuda.device_count())]
    )
    gpu_workers = min(int(gpu_count or runtime.gpu_workers), len(cuda_tokens) or 1)
    predict_processes, predict_methods, predict_started = run_detect_jobs(
        movie_ids=movie_ids,
        test_dir=test_dir,
        predictions_dir=predictions_dir,
        tracking=tracking,
        gpu_tokens=cuda_tokens,
        gpu_workers=gpu_workers,
        longest_first=speed.longest_first,
        deadline=deadline,
    )
    upgrade = GraphUpgrade(tracking)
    run_upgrade(upgrade)

    cpu_total = int(speed.thread_pool_total or 0) or (os.cpu_count() or 4)
    cpu_workers = upgrade_worker_count(
        cpu_workers=max(1, int(runtime.cpu_workers or 0) or cpu_total),
        graph_device=str(tracking.graph.deepcenter_device),
        deepcenter_gpu_workers=int(runtime.deepcenter_gpu_workers),
    )
    while_predicting = int(runtime.cpu_workers_while_predicting or 0)
    cpu_workers_while_predicting = max(
        1,
        min(while_predicting or max(1, cpu_total - len(predict_processes)), cpu_workers),
    )
    devices = assign_deepcenter_devices(
        n_workers=cpu_workers,
        graph_device=str(tracking.graph.deepcenter_device),
        gpu_workers=int(runtime.deepcenter_gpu_workers),
        explicit=tuple(runtime.deepcenter_devices),
        cuda_tokens=cuda_tokens,
    )

    context = mp.get_context('spawn')
    result_queue = context.Queue()
    task_queues = [context.Queue(maxsize=1) for _ in range(cpu_workers)]
    active_tasks = context.Value('i', 0)
    live_producers = context.Value('i', len(predict_processes))

    workers = []
    for worker_id in range(cpu_workers):
        process = context.Process(
            target=_upgrade_worker_entry,
            args=(
                worker_id,
                task_queues[worker_id],
                result_queue,
                active_tasks,
                live_producers,
                tracking,
                devices[worker_id],
                deadline,
                started,
                bool(speed.thread_boost),
                cpu_total,
            ),
            name=f'biohub-upgrade-{worker_id}',
            daemon=False,
        )
        process.start()
        workers.append(process)

    video_cost = {stem: _video_cost(test_dir, stem) for stem in movie_ids}
    stats_by_dataset: dict[str, dict] = {}
    graph_by_dataset: dict[str, Path] = {}
    shard_done: set[str] = set()
    attempted: dict[str, set[str]] = {}
    upgraded: list[str] = []
    worker_task: dict[int, tuple[str, str] | None] = {
        worker_id: None for worker_id in range(cpu_workers)
    }
    evidence_roots = [
        tracking.paths.model_c_evidence_dir,
        tracking.paths.p1_evidence_dir,
        tracking.paths.p2_evidence_dir,
    ]
    eager_baseline = not speed.lazy_baseline

    def ready_graphs() -> dict[str, Path]:
        ready = {}
        methods = predict_methods or ['unet_transformer']
        patterns = [f'{gpu_method}/split_0/*.ready' for gpu_method in methods] + [
            f'*/{gpu_method}/split_0/*.ready' for gpu_method in methods
        ]
        for pattern in patterns:
            for marker in predictions_dir.glob(pattern):
                graph = marker.with_suffix('.geff')
                if graph.exists() and marker.stem in movie_ids:
                    ready[marker.stem] = graph
        return ready

    def has_evidence(dataset: str) -> bool:
        return all((root / f'{dataset}.npz').exists() for root in evidence_roots)

    def queue_task(dataset: str, mode: str, worker_id: int) -> None:
        attempted.setdefault(dataset, set()).add(mode)
        worker_task[worker_id] = (dataset, mode)
        task_queues[worker_id].put((dataset, str(graph_by_dataset[dataset]), mode))
        log_event(run_dir, 'upgrade_queued', dataset=dataset, mode=mode, worker=worker_id)

    def next_mode(dataset: str, producer_done: bool) -> str | None:
        tried = attempted.setdefault(dataset, set())
        if 'combined' not in tried:
            if has_evidence(dataset):
                return 'combined'
            if not producer_done:
                return None
            tried.add('combined')
        if dataset in shard_done:
            return None
        if 'baseline' not in tried:
            return 'baseline'
        if runtime.emergency_shard and 'emergency' not in tried:
            return 'emergency'
        return None

    def collect_results() -> None:
        while True:
            try:
                result = result_queue.get_nowait()
            except stdlib_queue.Empty:
                return
            worker_task[int(result['worker_id'])] = None
            dataset = str(result['dataset'])
            mode = str(result.get('mode', 'combined'))
            if result['status'] == 'ok':
                row = result['row']
                stats_by_dataset[dataset] = row
                shard_done.add(dataset)
                if mode == 'combined':
                    upgraded.append(dataset)
                log_event(
                    run_dir,
                    'upgrade_ok',
                    dataset=dataset,
                    mode=mode,
                    seconds=result['seconds'],
                    nodes=row.get('nodes'),
                    edges=row.get('edges'),
                )
            else:
                log_event(
                    run_dir,
                    'upgrade_failed',
                    dataset=dataset,
                    mode=mode,
                    status=result['status'],
                    error=result.get('error'),
                    traceback=result.get('traceback'),
                )

    producer_finished_at = None
    deadline_hit = False
    while True:
        graph_by_dataset.update(ready_graphs())
        collect_results()
        for worker_id, dataset, mode in reap_dead_workers(workers, worker_task):
            log_event(
                run_dir,
                'upgrade_failed',
                dataset=dataset,
                mode=mode,
                status='dead',
                worker_id=worker_id,
            )
        producer_done = all(process.poll() is not None for process in predict_processes.values())
        if producer_done and producer_finished_at is None:
            producer_finished_at = time.monotonic()
            with live_producers.get_lock():
                live_producers.value = 0
        remaining = deadline - time.monotonic()
        busy_datasets = {task[0] for task in worker_task.values() if task is not None}
        pending: list[tuple[str, str]] = []
        for dataset in graph_by_dataset:
            if dataset in busy_datasets:
                continue
            mode = next_mode(dataset, producer_done)
            if mode is None:
                continue
            if eager_baseline and mode == 'combined' and dataset not in shard_done:
                mode = 'baseline'
            pending.append((dataset, mode))
        pending.sort(key=lambda item: (-video_cost.get(item[0], 0.0), item[0]))
        if remaining > runtime.min_upgrade_seconds and pending:
            active_limit = cpu_workers if producer_done else cpu_workers_while_predicting
            idle = [
                worker_id
                for worker_id in range(active_limit)
                if worker_task[worker_id] is None and workers[worker_id].is_alive()
            ]
            for worker_id, (dataset, mode) in zip(idle, pending, strict=False):
                queue_task(dataset, mode, worker_id)
        busy = any(task is not None for task in worker_task.values())
        complete = producer_done and len(shard_done) == len(movie_ids)
        if scheduler_should_stop(
            remaining=remaining,
            producer_done=producer_done,
            busy=busy,
            pending=bool(pending),
            complete=complete,
        ):
            deadline_hit = remaining <= 0
            break
        time.sleep(2.0)

    collect_results()
    stop_upgrade_workers(workers, task_queues, timeout=10.0)

    for index, process in list(predict_processes.items()):
        if process.poll() is None:
            process.terminate()
    for index, process in predict_processes.items():
        try:
            code = process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            code = process.wait(timeout=5)
        if code != 0 and not deadline_hit:
            raise subprocess.CalledProcessError(code, predict_processes[index].args)

    upgrade.set_thread_budget(cpu_total)
    submission_path = tracking.paths.submission or (work_dir / 'submission.csv')
    node_rows, edge_rows = assemble_submission(
        movie_ids,
        tracking.paths.shard_dir,
        submission_path,
        upgrade.csv_columns,
        missing_ok=deadline_hit,
    )
    predict_hours = (
        (producer_finished_at or time.monotonic()) - (predict_started or started)
    ) / 3600.0
    rows = [stats_by_dataset[dataset] for dataset in sorted(stats_by_dataset)]
    stats = pl.DataFrame(rows).sort('dataset')
    stats = stats.with_columns(
        pl.lit(tracking.experiment_tag).alias('experiment_tag'),
        pl.lit(predict_hours).alias('predict_hours'),
        pl.lit(len(upgraded)).alias('upgraded_total'),
        pl.lit((time.monotonic() - started) / 3600.0).alias('wall_hours'),
    )
    if tracking.paths.run_stats is not None:
        stats.write_csv(tracking.paths.run_stats)
    summary = {
        'submission': str(submission_path),
        'n_movies': len(movie_ids),
        'node_rows': node_rows,
        'edge_rows': edge_rows,
        'upgraded': len(upgraded),
        'predict_hours': predict_hours,
        'wall_hours': (time.monotonic() - started) / 3600.0,
    }
    log_event(run_dir, 'schedule_done', **summary)
    LOGGER.info('inference complete: %s', submission_path)
    return summary


def run_inference(
    *,
    config: Path | str | None = None,
    panel: str | None = None,
    run_id: str | None = None,
    gpu_workers: int | None = None,
    movies_dir: Path | None = None,
    movie_ids: list[str] | None = None,
    movie: str | None = None,
    runs_root: Path | None = None,
) -> dict:
    config_path = Path(config) if config is not None else PROJECT_ROOT / 'configs' / 'infer.yaml'
    raw = load_yaml(config_path)
    panel = panel or str(raw.get('panel', 'smoke'))
    train_dir = resolve_path(raw['train_dir'])
    if movies_dir is not None:
        movies_root = resolve_path(movies_dir)
    else:
        movies_root = resolve_path(raw['test_dir'])
    if movie:
        movie_ids = [movie]
    elif movie_ids is not None:
        movie_ids = list(movie_ids)
    else:
        movie_ids = panel_movie_ids(panel)
    bundle_dir = resolve_path(raw['bundle_dir'])
    resolved_run_id = run_id or new_run_id('infer')
    run_path = init_run(
        resolved_run_id,
        config=raw,
        extra={'panel': panel, 'command': 'infer', 'movies': movie_ids},
        runs_root=runs_root,
    )
    setup_logging(run_path)
    work_dir = run_path / 'workdir'
    tracking_config = build_tracking(
        config_path=config_path,
        bundle_dir=bundle_dir,
        test_dir=movies_root,
        work_dir=work_dir,
    )
    if tracking_config.runtime.slice.startswith(':'):
        stop_token = tracking_config.runtime.slice[1:]
        if stop_token:
            movie_ids = movie_ids[: int(stop_token)]
    try:
        summary = run(
            movie_ids=movie_ids,
            bundle_dir=bundle_dir,
            test_dir=movies_root,
            train_dir=train_dir,
            run_dir=run_path,
            gpu_count=gpu_workers
            if gpu_workers is not None
            else tracking_config.runtime.gpu_workers,
            config=tracking_config,
        )
    except Exception as exc:
        finish_run(run_path, 'failed', extra={'error': str(exc)})
        raise
    payload = {'run': str(run_path), **summary}
    finish_run(run_path, 'ok', extra={'summary': summary})
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Biohub inference')
    parser.add_argument('--config', type=Path, default=None)
    parser.add_argument('--movie', action='append', dest='movies', default=None)
    parser.add_argument('--panel', default=None)
    parser.add_argument('--run-id', default=None)
    parser.add_argument('--gpu-workers', type=int, default=None)
    parser.add_argument('--movies-dir', type=Path, default=None)
    parser.add_argument('--runs-root', type=Path, default=None)
    args = parser.parse_args(argv)
    result = run_inference(
        config=args.config,
        panel=args.panel,
        run_id=args.run_id,
        gpu_workers=args.gpu_workers,
        movies_dir=args.movies_dir,
        movie_ids=args.movies,
        runs_root=args.runs_root,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
