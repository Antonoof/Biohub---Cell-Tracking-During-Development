import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tracksdata as td
import zarr
from tqdm import tqdm

import biohub.modules.detect.evidence as detect_evidence
from biohub.data.volume import open_dataset, save_graph
from biohub.features.position import extract_pos_features
from biohub.models.detector import UNetNodeTransformer
from biohub.modules.detect.config import PredictConfig
from biohub.modules.detect.evidence import (
    append_native_evidence_pair,
    biohub_savez,
    detect_native_evidence_frames,
    edgegraft_append_graph_ids,
    new_native_evidence_state,
    save_native_evidence,
)
from biohub.modules.detect.model import build_graph, load_frame, load_model
from biohub.modules.detect.peaks import detect_cells_pooled, pool_kernel_from_um
from biohub.modules.ilp import apply_ilp


def _encode(model, imgs, cfg, device):
    if cfg.amp_fp16 and device.type == 'cuda':
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            return model.encode(imgs)
    return model.encode(imgs)


def _safe_encode_batch(imgs: torch.Tensor) -> int:
    pooled = 1
    for size in imgs.shape[-3:]:
        pooled *= max(int(size) // 2, 1)
    return max(1, 65535 // pooled)


def _encode_all(model, imgs, cfg, device):
    limit = min(int(imgs.shape[0]), _safe_encode_batch(imgs))
    if int(imgs.shape[0]) <= limit:
        return _encode(model, imgs, cfg, device)
    unet_parts = []
    det_parts: list[list[torch.Tensor]] | None = None
    for start in range(0, int(imgs.shape[0]), limit):
        unet_out, det_logits = _encode(model, imgs[start : start + limit], cfg, device)
        unet_parts.append(unet_out)
        if det_parts is None:
            det_parts = [[frame] for frame in det_logits]
        else:
            for buckets, frame in zip(det_parts, det_logits):
                buckets.append(frame)
    assert det_parts is not None
    return torch.cat(unet_parts, 0), [torch.cat(frames, 0) for frames in det_parts]


def tta_view_specs(kind: str) -> list[tuple]:
    specs = [
        (lambda x: x, lambda x: x, True),
        (lambda x: x.flip((-1,)), lambda x: x.flip((-1,)), True),
        (lambda x: x.flip((-2,)), lambda x: x.flip((-2,)), True),
        (lambda x: x.flip((-2, -1)), lambda x: x.flip((-2, -1)), True),
    ]
    if kind == 'full':
        specs.extend(
            [
                (
                    lambda x: torch.rot90(x, 1, dims=(-2, -1)),
                    lambda x: torch.rot90(x, -1, dims=(-2, -1)),
                    False,
                ),
                (
                    lambda x: torch.rot90(x, 3, dims=(-2, -1)),
                    lambda x: torch.rot90(x, -3, dims=(-2, -1)),
                    False,
                ),
                (lambda x: x.transpose(-1, -2), lambda x: x.transpose(-1, -2), False),
                (
                    lambda x: torch.rot90(x, 1, dims=(-2, -1)).transpose(-1, -2),
                    lambda x: torch.rot90(x.transpose(-1, -2), -1, dims=(-2, -1)),
                    False,
                ),
            ]
        )
    return specs


def encode_detection_views(model, imgs, cfg, device, kind: str):
    if kind == 'none' or (kind == 'full' and not cfg.det_tta):
        unet_out, det_logits = _encode_all(model, imgs, cfg, device)
        native = [value.clone() for value in det_logits]
        return unet_out, det_logits, native
    specs = tta_view_specs(kind)
    n_views = len(specs)
    apply0, _invert0, native0 = specs[0]
    viewed0 = apply0(imgs)
    unet_out, orig_det = _encode_all(model, viewed0, cfg, device)
    unet_out = unet_out.contiguous()
    width = len(orig_det)
    det_sum = [frame.contiguous() for frame in orig_det]
    native_sum: list[torch.Tensor | None]
    n_native = 0
    if native0:
        native_sum = [frame.clone() for frame in det_sum]
        n_native = 1
    else:
        native_sum = [None] * width
    del viewed0, orig_det
    for apply, invert, native in specs[1:]:
        viewed = apply(imgs)
        unused_features, det_view = _encode_all(model, viewed, cfg, device)
        del unused_features, viewed
        for frame in range(width):
            inverted = invert(det_view[frame])
            det_sum[frame] = det_sum[frame] + inverted
            if native:
                current = native_sum[frame]
                native_sum[frame] = inverted if current is None else current + inverted
        if native:
            n_native += 1
        del det_view
    det_logits = [value / n_views for value in det_sum]
    native_frames: list[torch.Tensor] = []
    for value in native_sum:
        if value is None:
            raise RuntimeError('native detection view is missing')
        native_frames.append(value)
    native_detection = [value / n_native for value in native_frames]
    return unet_out, det_logits, native_detection


@torch.no_grad()
def predict_video(
    model: UNetNodeTransformer,
    ds_path: Path,
    device: torch.device,
    cfg: PredictConfig,
    window_size: int = 2,
    max_frames: int | None = None,
    unet_batch_size: int = 4,
    downsample: tuple[int, ...] = (1, 4, 4),
    secondary_model: UNetNodeTransformer | None = None,
    secondary_edge_weight: float = 0.0,
    secondary_detection_weight: float = 0.0,
    secondary_link_mode: str = 'raw',
    secondary_mix_temperature: float = 1.0,
    secondary_low_margin_max: float = 0.2,
    division_model: UNetNodeTransformer | None = None,
    division_evidence_path: Path | None = None,
    division_det_threshold: float = 0.99,
    division_pool_um: float = 3.0,
    division_radius_um: float = 20.0,
    division_topk: int = 16,
    division_min_probability: float = 0.01,
    primary_native_evidence_path: Path | None = None,
    secondary_native_evidence_path: Path | None = None,
    native_evidence_det_threshold: float = 0.96875,
    native_evidence_pool_um: float = 5.0,
    native_evidence_radius_um: float = 20.0,
    native_evidence_topk: int = 16,
    native_evidence_min_probability: float = 0.01,
) -> tuple[np.ndarray, list[tuple[int, int, float, float]]]:
    ds = open_dataset(ds_path, normalize=False, load_image=False, downsample=downsample)
    if '0.001' not in ds.quantiles or '0.999' not in ds.quantiles:
        raise ValueError(f'Zarr attrs missing image_statistics.quantiles for {ds_path}')
    if ds.zarr_path is None or ds.image_shape is None:
        raise ValueError(f'Missing zarr path or image shape for {ds_path}')
    zarr_arr: Any = zarr.open_group(str(ds.zarr_path), mode='r')['0']
    q_low = float(ds.quantiles['0.001'])
    q_high = float(ds.quantiles['0.999'])

    T = ds.image_shape[0] if max_frames is None else min(ds.image_shape[0], max_frames)
    image_shape = (T,) + ds.image_shape[1:]
    target_shape = list(image_shape[1:])

    ds_arr = np.array(downsample, dtype=np.float32)
    ds_arr_t = torch.from_numpy(ds_arr).to(device)
    W = window_size
    voxel_size = tuple(s * d for s, d in zip(ds.scale, downsample))
    pool_k = pool_kernel_from_um(cfg.pool_kernel_um, voxel_size)
    division_pool_k = pool_kernel_from_um(division_pool_um, voxel_size)
    division_ds_arr_t = torch.from_numpy(ds_arr).to(device)
    original_spacing = np.asarray(ds.scale, dtype=np.float32)
    division_seen_frames: set[int] = set()
    division_seen_pairs: set[tuple[int, int]] = set()
    division_coord_by_t: dict[int, np.ndarray] = {}
    division_offset: dict[int, tuple[int, int]] = {}
    division_node_count = 0
    division_evidence = {
        key: []
        for key in (
            'source_id',
            'target_id',
            'probability',
            'alternative_parent_probability',
            'is_target_winner',
            'distance_um',
            'source_target_rank',
        )
    }
    native_evidence_pool_kernel = pool_kernel_from_um(
        native_evidence_pool_um,
        voxel_size,
    )
    primary_native_state = (
        new_native_evidence_state() if primary_native_evidence_path is not None else None
    )
    secondary_native_state = (
        new_native_evidence_state() if secondary_native_evidence_path is not None else None
    )

    seen_frames: set[int] = set()
    seen_pairs: set[tuple[int, int]] = set()
    retention_guard_frames: set[int] = set()
    coord_lists: list[np.ndarray] = []
    coord_offset: dict[int, tuple[int, int]] = {}
    global_node_count: int = 0
    all_edges: list[tuple[int, int, float, float]] = []

    stride = max(W - 1, 1)
    window_starts = list(range(0, T - W + 1, stride))
    if not window_starts or window_starts[-1] + W < T:
        last = max(T - W, 0)
        if not window_starts or last != window_starts[-1]:
            window_starts.append(last)

    frame_cache: dict[int, torch.Tensor] = {}

    def cached_norm_frame(frame_t: int) -> torch.Tensor:
        cached = frame_cache.get(frame_t)
        if cached is not None:
            return cached
        image = load_frame(zarr_arr, frame_t, target_shape, downsample)
        image = ((image - q_low) / (q_high - q_low + 1e-6)).clamp(0.0)
        frame_cache[frame_t] = image
        return image

    n_tta_views = len(tta_view_specs('full')) if cfg.det_tta else 1
    windows_per_fwd = max(1, int(unet_batch_size) // n_tta_views)
    chunk_starts = list(range(0, len(window_starts), windows_per_fwd))
    for chunk_index in tqdm(
        chunk_starts,
        desc='  windows',
        leave=False,
        disable=not cfg.show_progress,
    ):
        chunk = window_starts[chunk_index : chunk_index + windows_per_fwd]
        chunk_frames = [list(range(ws, ws + W)) for ws in chunk]
        stacked = torch.stack(
            [torch.stack([cached_norm_frame(t) for t in frames]) for frames in chunk_frames]
        ).to(device)
        unet_b, det_b, native_b = encode_detection_views(
            model, stacked, cfg, device, 'full' if cfg.det_tta else 'none'
        )
        secondary_unet_b = None
        secondary_det_b = None
        secondary_native_b = None
        if secondary_model is not None:
            secondary_unet_b, secondary_det_b, secondary_native_b = encode_detection_views(
                secondary_model,
                stacked,
                cfg,
                device,
                'full' if cfg.det_tta else 'none',
            )
        division_unet_b = None
        division_det_b = None
        if division_model is not None:
            division_unet_b, division_det_b, _division_native = encode_detection_views(
                division_model, stacked, cfg, device, 'flips'
            )
        del stacked
        for window_i, frame_indices in enumerate(chunk_frames):
            unet_out = unet_b[window_i : window_i + 1]
            det_logits = [det[window_i : window_i + 1] for det in det_b]
            primary_native_detection = [det[window_i : window_i + 1] for det in native_b]
            secondary_unet_out = None
            secondary_native_detection = None
            secondary_det_logits = None
            if secondary_unet_b is not None:
                if secondary_det_b is None or secondary_native_b is None:
                    raise RuntimeError('Secondary detection tensors missing')
                secondary_unet_out = secondary_unet_b[window_i : window_i + 1]
                secondary_det_logits = [det[window_i : window_i + 1] for det in secondary_det_b]
                secondary_native_detection = [
                    det[window_i : window_i + 1] for det in secondary_native_b
                ]
            division_unet_out: Any = None
            division_det_logits: Any = None
            if division_unet_b is not None:
                if division_det_b is None:
                    raise RuntimeError('Division detection tensors missing')
                division_unet_out = division_unet_b[window_i : window_i + 1]
                division_det_logits = [det[window_i : window_i + 1] for det in division_det_b]
            if secondary_model is not None and secondary_detection_weight > 0.0:
                if secondary_det_logits is None:
                    raise RuntimeError('Secondary detection logits missing')
                if secondary_native_detection is None:
                    secondary_native_detection = [value.clone() for value in secondary_det_logits]
                for f in range(W):
                    primary_det = det_logits[f]
                    secondary_det = secondary_det_logits[f]
                    primary_mean = primary_det.mean()
                    secondary_mean = secondary_det.mean()
                    primary_scale = primary_det.float().std(unbiased=False).clamp_min(1e-4)
                    secondary_scale = secondary_det.float().std(unbiased=False).clamp_min(1e-4)
                    scale_ratio = (primary_scale / secondary_scale).clamp(0.5, 2.0)
                    secondary_det_aligned = (
                        secondary_det - secondary_mean
                    ) * scale_ratio + primary_mean
                    blended_det = (
                        1.0 - secondary_detection_weight
                    ) * primary_det + secondary_detection_weight * secondary_det_aligned
                    primary_candidates = len(
                        detect_cells_pooled(
                            primary_det[0],
                            int(frame_indices[f]),
                            cfg.det_threshold,
                            pool_k,
                            refine=cfg.subvoxel_refinement,
                        )
                    )
                    blended_candidates = len(
                        detect_cells_pooled(
                            blended_det[0],
                            int(frame_indices[f]),
                            cfg.det_threshold,
                            pool_k,
                            refine=cfg.subvoxel_refinement,
                        )
                    )
                    minimum_retention = float(cfg.dual_seed_min_candidate_retention)
                    candidate_retention = (
                        blended_candidates / primary_candidates if primary_candidates else 1.0
                    )
                    use_primary_detection = bool(
                        primary_candidates > 0 and candidate_retention < minimum_retention
                    )
                    det_logits[f] = primary_det if use_primary_detection else blended_det
                    if int(frame_indices[f]) not in seen_frames:
                        shard = (cfg.gpu_shard or 'single').replace('/', '_')
                        log_root = cfg.working_dir if cfg.working_dir is not None else Path('.')
                        guard_log = log_root / f'retention_guard_{shard}.jsonl'
                        guard_record = {
                            'dataset': ds_path.stem,
                            'frame': int(frame_indices[f]),
                            'primary_candidates': int(primary_candidates),
                            'blended_candidates': int(blended_candidates),
                            'retention': float(candidate_retention),
                            'minimum_retention': float(minimum_retention),
                            'use_primary': bool(use_primary_detection),
                        }
                        with guard_log.open('a') as guard_handle:
                            guard_handle.write(json.dumps(guard_record, sort_keys=True) + '\n')
                        if use_primary_detection:
                            retention_guard_frames.add(int(frame_indices[f]))
                            print(
                                'BIOHUB_RETENTION_GUARD '
                                + json.dumps(guard_record, sort_keys=True),
                                flush=True,
                            )
                del secondary_det_logits

            if primary_native_state is not None:
                detect_native_evidence_frames(
                    primary_native_state,
                    primary_native_detection,
                    frame_indices,
                    native_evidence_det_threshold,
                    native_evidence_pool_kernel,
                    downsample,
                )
            if secondary_native_state is not None:
                if secondary_native_detection is None:
                    raise RuntimeError('Secondary native evidence requested without model')
                detect_native_evidence_frames(
                    secondary_native_state,
                    secondary_native_detection,
                    frame_indices,
                    native_evidence_det_threshold,
                    native_evidence_pool_kernel,
                    downsample,
                )

            for f_idx, t in enumerate(frame_indices):
                if t not in seen_frames:
                    arr = detect_cells_pooled(
                        det_logits[f_idx][0],
                        t,
                        cfg.det_threshold,
                        pool_k,
                        refine=cfg.subvoxel_refinement,
                    )
                    coord_offset[t] = (global_node_count, global_node_count + len(arr))
                    global_node_count += len(arr)
                    coord_lists.append(arr)
                    seen_frames.add(t)

            if division_model is not None:
                for division_f_idx, division_t in enumerate(frame_indices):
                    if division_t in division_seen_frames:
                        continue
                    native = detect_cells_pooled(
                        division_det_logits[division_f_idx][0],
                        division_t,
                        division_det_threshold,
                        division_pool_k,
                        refine=cfg.subvoxel_refinement,
                    ).astype(np.float32)
                    native[:, 1:] *= np.asarray(downsample, np.float32)
                    division_coord_by_t[division_t] = native
                    division_offset[division_t] = (
                        division_node_count,
                        division_node_count + len(native),
                    )
                    division_node_count += len(native)
                    division_seen_frames.add(division_t)

            coords_so_far = (
                np.concatenate(coord_lists) if coord_lists else np.empty((0, 4), dtype=np.float32)
            )

            for f_idx in range(W - 1):
                t_src, t_tgt = frame_indices[f_idx], frame_indices[f_idx + 1]
                if (t_src, t_tgt) in seen_pairs:
                    continue
                seen_pairs.add((t_src, t_tgt))

                native_window_shape = (W,) + image_shape[1:]
                if primary_native_state is not None:
                    append_native_evidence_pair(
                        primary_native_state,
                        model,
                        unet_out,
                        f_idx,
                        t_src,
                        t_tgt,
                        downsample,
                        ds.scale,
                        native_window_shape,
                        device,
                        native_evidence_radius_um,
                        native_evidence_topk,
                        native_evidence_min_probability,
                    )
                if secondary_native_state is not None:
                    if secondary_model is None or secondary_unet_out is None:
                        raise RuntimeError('Secondary native evidence feature map missing')
                    append_native_evidence_pair(
                        secondary_native_state,
                        secondary_model,
                        secondary_unet_out,
                        f_idx,
                        t_src,
                        t_tgt,
                        downsample,
                        ds.scale,
                        native_window_shape,
                        device,
                        native_evidence_radius_um,
                        native_evidence_topk,
                        native_evidence_min_probability,
                    )

                if division_model is not None and (t_src, t_tgt) not in division_seen_pairs:
                    division_seen_pairs.add((t_src, t_tgt))
                    native_src = division_coord_by_t.get(t_src, np.empty((0, 4), np.float32))
                    native_tgt = division_coord_by_t.get(t_tgt, np.empty((0, 4), np.float32))
                    if len(native_src) and len(native_tgt):
                        source_raw = native_src[:, 1:].astype(np.float32)
                        target_raw = native_tgt[:, 1:].astype(np.float32)
                        source_ds = np.rint(source_raw / ds_arr).astype(np.float32)
                        target_ds = np.rint(target_raw / ds_arr).astype(np.float32)
                        n_div_source, n_div_target = len(source_ds), len(target_ds)
                        div_src = torch.from_numpy(source_ds).unsqueeze(0).to(device)
                        div_tgt = torch.from_numpy(target_ds).unsqueeze(0).to(device)
                        source_rel = np.column_stack(
                            [np.full(n_div_source, f_idx, np.float32), source_ds]
                        )
                        target_rel = np.column_stack(
                            [np.full(n_div_target, f_idx + 1, np.float32), target_ds]
                        )
                        div_window_shape = (W,) + image_shape[1:]
                        div_pos_src = (
                            torch.from_numpy(extract_pos_features(source_rel, div_window_shape))
                            .unsqueeze(0)
                            .to(device)
                        )
                        div_pos_tgt = (
                            torch.from_numpy(extract_pos_features(target_rel, div_window_shape))
                            .unsqueeze(0)
                            .to(device)
                        )
                        div_mask_src = torch.ones(1, n_div_source, dtype=torch.bool, device=device)
                        div_mask_tgt = torch.ones(1, n_div_target, dtype=torch.bool, device=device)
                        div_feat_src = division_model.index_features(
                            division_unet_out[:, f_idx], div_src, div_mask_src
                        )
                        div_feat_tgt = division_model.index_features(
                            division_unet_out[:, f_idx + 1], div_tgt, div_mask_tgt
                        )
                        div_logits = division_model.predict_edges(
                            div_feat_src,
                            div_feat_tgt,
                            div_src * division_ds_arr_t,
                            div_tgt * division_ds_arr_t,
                            div_pos_src,
                            div_pos_tgt,
                            div_mask_src,
                            div_mask_tgt,
                        )[0]
                        div_probability = torch.softmax(div_logits, dim=0).float()
                        div_source_um = torch.as_tensor(
                            source_raw * original_spacing, device=device
                        ).float()
                        div_target_um = torch.as_tensor(
                            target_raw * original_spacing, device=device
                        ).float()
                        div_distance = torch.cdist(div_source_um, div_target_um)
                        div_eligible = div_distance <= division_radius_um
                        div_masked = div_probability.masked_fill(~div_eligible, -1.0)
                        div_k = min(division_topk, n_div_target)
                        div_values, div_targets = torch.topk(div_masked, div_k, dim=1, sorted=True)
                        div_keep = div_values >= division_min_probability
                        if n_div_source == 1:
                            div_best_prob = div_probability[0]
                            div_best_row = torch.zeros_like(div_best_prob, dtype=torch.long)
                            div_second_prob = torch.zeros_like(div_best_prob)
                        else:
                            div_top2 = torch.topk(div_probability, 2, dim=0, sorted=True)
                            div_best_prob = div_top2.values[0]
                            div_best_row = div_top2.indices[0]
                            div_second_prob = div_top2.values[1]
                        div_source_rows = (
                            torch.arange(n_div_source, device=device)
                            .unsqueeze(1)
                            .expand_as(div_targets)
                        )
                        picked_source = div_source_rows[div_keep]
                        picked_target = div_targets[div_keep]
                        target_best = div_best_row[picked_target]
                        div_alternative = torch.where(
                            target_best == picked_source,
                            div_second_prob[picked_target],
                            div_best_prob[picked_target],
                        )
                        source_start = division_offset[t_src][0]
                        target_start = division_offset[t_tgt][0]
                        division_evidence['source_id'].append(
                            (picked_source + source_start).cpu().numpy().astype(np.int64)
                        )
                        division_evidence['target_id'].append(
                            (picked_target + target_start).cpu().numpy().astype(np.int64)
                        )
                        division_evidence['probability'].append(
                            div_values[div_keep].cpu().numpy().astype(np.float32)
                        )
                        division_evidence['alternative_parent_probability'].append(
                            div_alternative.cpu().numpy().astype(np.float32)
                        )
                        division_evidence['is_target_winner'].append(
                            (target_best == picked_source).cpu().numpy().astype(np.uint8)
                        )
                        division_evidence['distance_um'].append(
                            div_distance[picked_source, picked_target]
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        )
                        div_rank_grid = (
                            torch.arange(div_k, device=device).unsqueeze(0).expand_as(div_targets)
                        )
                        division_evidence['source_target_rank'].append(
                            div_rank_grid[div_keep].cpu().numpy().astype(np.int16)
                        )

                if t_src not in coord_offset or t_tgt not in coord_offset:
                    continue
                s_src, e_src = coord_offset[t_src]
                s_tgt, e_tgt = coord_offset[t_tgt]
                if e_src == s_src or e_tgt == s_tgt:
                    continue

                c_src = coords_so_far[s_src:e_src]
                c_tgt = coords_so_far[s_tgt:e_tgt]
                n_src, n_tgt = len(c_src), len(c_tgt)
                idx_src = np.arange(s_src, e_src, dtype=np.int64)
                idx_tgt = np.arange(s_tgt, e_tgt, dtype=np.int64)

                p_coords_src = (
                    torch.from_numpy(np.rint(c_src[:, 1:]).astype(np.float32))
                    .unsqueeze(0)
                    .to(device)
                )
                p_coords_tgt = (
                    torch.from_numpy(np.rint(c_tgt[:, 1:]).astype(np.float32))
                    .unsqueeze(0)
                    .to(device)
                )
                window_shape = (W,) + image_shape[1:]
                c_src_rel = np.rint(c_src).astype(np.float32)
                c_src_rel[:, 0] = f_idx
                c_tgt_rel = np.rint(c_tgt).astype(np.float32)
                c_tgt_rel[:, 0] = f_idx + 1
                p_pos_src = (
                    torch.from_numpy(extract_pos_features(c_src_rel, window_shape))
                    .unsqueeze(0)
                    .to(device)
                )
                p_pos_tgt = (
                    torch.from_numpy(extract_pos_features(c_tgt_rel, window_shape))
                    .unsqueeze(0)
                    .to(device)
                )
                p_mask_src = torch.ones(1, n_src, dtype=torch.bool, device=device)
                p_mask_tgt = torch.ones(1, n_tgt, dtype=torch.bool, device=device)

                unet_feat_src = model.index_features(
                    unet_out[:, f_idx],
                    p_coords_src,
                    p_mask_src,
                )
                unet_feat_tgt = model.index_features(
                    unet_out[:, f_idx + 1],
                    p_coords_tgt,
                    p_mask_tgt,
                )
                edge_logits_pair = model.predict_edges(
                    unet_feat_src,
                    unet_feat_tgt,
                    p_coords_src * ds_arr_t,
                    p_coords_tgt * ds_arr_t,
                    p_pos_src,
                    p_pos_tgt,
                    p_mask_src,
                    p_mask_tgt,
                )

                _bidirectional_weight = float(cfg.bidirectional_edge_weight)
                if _bidirectional_weight > 0.0:
                    reverse_logits_native = model.predict_edges(
                        unet_feat_tgt,
                        unet_feat_src,
                        p_coords_tgt * ds_arr_t,
                        p_coords_src * ds_arr_t,
                        p_pos_tgt,
                        p_pos_src,
                        p_mask_tgt,
                        p_mask_src,
                    )
                    reverse_logits_pair = reverse_logits_native.transpose(1, 2)

                    forward_center = edge_logits_pair.mean(dim=1, keepdim=True)
                    forward_scale = (
                        edge_logits_pair.float()
                        .std(dim=1, keepdim=True, unbiased=False)
                        .clamp_min(1e-4)
                    )
                    reverse_center = reverse_logits_pair.mean(dim=1, keepdim=True)
                    reverse_scale = (
                        reverse_logits_pair.float()
                        .std(dim=1, keepdim=True, unbiased=False)
                        .clamp_min(1e-4)
                    )
                    reverse_scale_ratio = (forward_scale / reverse_scale).clamp(0.5, 2.0)
                    reverse_scale_ratio = reverse_scale_ratio.to(reverse_logits_pair.dtype)
                    reverse_aligned = (
                        reverse_logits_pair - reverse_center
                    ) * reverse_scale_ratio + forward_center
                    forward_prob = torch.softmax(edge_logits_pair.float(), dim=1).clamp_min(1e-8)
                    reverse_prob = torch.softmax(reverse_aligned.float(), dim=1).clamp_min(1e-8)
                    harmonic_prob = 1.0 / (
                        (1.0 - _bidirectional_weight) / forward_prob
                        + _bidirectional_weight / reverse_prob
                    )
                    harmonic_prob = harmonic_prob / harmonic_prob.sum(
                        dim=1, keepdim=True
                    ).clamp_min(1e-8)
                    harmonic_logits = torch.log(harmonic_prob.clamp_min(1e-8))
                    harmonic_center = harmonic_logits.mean(dim=1, keepdim=True)
                    harmonic_scale = harmonic_logits.std(
                        dim=1, keepdim=True, unbiased=False
                    ).clamp_min(1e-4)
                    harmonic_scale_ratio = (forward_scale / harmonic_scale).clamp(0.5, 2.0)
                    edge_logits_pair = (
                        (harmonic_logits - harmonic_center) * harmonic_scale_ratio + forward_center
                    ).to(reverse_aligned.dtype)
                    del (
                        reverse_logits_native,
                        reverse_logits_pair,
                        reverse_aligned,
                        forward_prob,
                        reverse_prob,
                        harmonic_prob,
                        harmonic_logits,
                    )

                if secondary_model is not None:
                    if secondary_unet_out is None:
                        raise RuntimeError(
                            'Secondary model is loaded but its feature map is missing'
                        )
                    secondary_feat_src = secondary_model.index_features(
                        secondary_unet_out[:, f_idx],
                        p_coords_src,
                        p_mask_src,
                    )
                    secondary_feat_tgt = secondary_model.index_features(
                        secondary_unet_out[:, f_idx + 1],
                        p_coords_tgt,
                        p_mask_tgt,
                    )
                    secondary_logits_pair = secondary_model.predict_edges(
                        secondary_feat_src,
                        secondary_feat_tgt,
                        p_coords_src * ds_arr_t,
                        p_coords_tgt * ds_arr_t,
                        p_pos_src,
                        p_pos_tgt,
                        p_mask_src,
                        p_mask_tgt,
                    )

                    transition_uses_primary_detection = bool(
                        t_src in retention_guard_frames or t_tgt in retention_guard_frames
                    )
                    guarded_secondary_edge_weight = (
                        float(cfg.retention_guard_secondary_edge_weight)
                        if cfg.retention_guard_secondary_edge_weight is not None
                        else float(secondary_edge_weight)
                    )
                    active_secondary_edge_weight = (
                        guarded_secondary_edge_weight
                        if transition_uses_primary_detection
                        else secondary_edge_weight
                    )
                    if transition_uses_primary_detection:
                        print(
                            'BIOHUB_GUARD_EDGE_OWNERSHIP '
                            + json.dumps(
                                {
                                    'dataset': ds_path.stem,
                                    'source_frame': int(t_src),
                                    'target_frame': int(t_tgt),
                                    'ordinary_p2_ceiling': float(secondary_edge_weight),
                                    'active_p2_ceiling': float(active_secondary_edge_weight),
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )

                    if secondary_link_mode == 'raw':
                        secondary_for_mix = secondary_logits_pair
                        blend_weight = secondary_edge_weight
                    elif secondary_link_mode in {'calibrated', 'adaptive', 'low_margin_consensus'}:
                        primary_center = edge_logits_pair.mean(dim=1, keepdim=True)
                        primary_scale = (
                            edge_logits_pair.float()
                            .std(dim=1, keepdim=True, unbiased=False)
                            .clamp_min(1e-4)
                        )
                        secondary_center = secondary_logits_pair.mean(dim=1, keepdim=True)
                        secondary_scale = (
                            secondary_logits_pair.float()
                            .std(dim=1, keepdim=True, unbiased=False)
                            .clamp_min(1e-4)
                        )
                        secondary_scale_ratio = (primary_scale / secondary_scale).clamp(0.5, 2.0)
                        secondary_for_mix = (
                            secondary_logits_pair - secondary_center
                        ) * secondary_scale_ratio + primary_center
                        if secondary_link_mode == 'calibrated':
                            blend_weight = secondary_edge_weight
                        elif secondary_link_mode == 'adaptive':
                            if n_src >= 2:
                                primary_probs = torch.softmax(edge_logits_pair[0], dim=0)
                                secondary_probs = torch.softmax(secondary_for_mix[0], dim=0)
                                primary_top2 = torch.topk(primary_probs, k=2, dim=0)
                                secondary_top2 = torch.topk(secondary_probs, k=2, dim=0)
                                primary_margin = primary_top2.values[0] - primary_top2.values[1]
                                secondary_margin = (
                                    secondary_top2.values[0] - secondary_top2.values[1]
                                )
                                local_weight = (
                                    secondary_edge_weight + secondary_margin - primary_margin
                                ).clamp(0.15, 0.75)
                                same_parent = primary_top2.indices[0].eq(secondary_top2.indices[0])
                                local_weight = torch.where(
                                    same_parent,
                                    torch.maximum(
                                        local_weight,
                                        torch.full_like(local_weight, secondary_edge_weight),
                                    ),
                                    local_weight,
                                )
                                blend_weight = local_weight.view(1, 1, -1)
                            else:
                                blend_weight = secondary_edge_weight
                        else:
                            if n_src >= 2:
                                primary_probs = torch.softmax(edge_logits_pair[0], dim=0)
                                secondary_probs = torch.softmax(secondary_for_mix[0], dim=0)
                                primary_top2 = torch.topk(primary_probs, k=2, dim=0)
                                secondary_top2 = torch.topk(secondary_probs, k=2, dim=0)
                                primary_margin = primary_top2.values[0] - primary_top2.values[1]
                                same_parent = primary_top2.indices[0].eq(secondary_top2.indices[0])
                                uncertainty = (
                                    (secondary_low_margin_max - primary_margin)
                                    / secondary_low_margin_max
                                ).clamp(0.0, 1.0)
                                local_weight = active_secondary_edge_weight * uncertainty
                                local_weight = torch.where(
                                    same_parent,
                                    local_weight,
                                    torch.zeros_like(local_weight),
                                )
                                blend_weight = local_weight.view(1, 1, -1)
                            else:
                                blend_weight = 0.0
                    else:
                        raise ValueError(f'Unsupported secondary link mode: {secondary_link_mode}')

                    edge_logits_pair = (
                        1.0 - blend_weight
                    ) * edge_logits_pair + blend_weight * secondary_for_mix
                    if secondary_mix_temperature != 1.0:
                        mixed_center = edge_logits_pair.mean(dim=1, keepdim=True)
                        edge_logits_pair = (
                            mixed_center
                            + (edge_logits_pair - mixed_center) / secondary_mix_temperature
                        )

                raw = edge_logits_pair[0]
                if cfg.edge_activation == 'softmax':
                    probs = torch.softmax(raw, dim=0).cpu().numpy()
                else:
                    probs = torch.sigmoid(raw).cpu().numpy()

                _cand_src, _cand_tgt = np.nonzero(probs > cfg.threshold)
                _cand_prob = probs[_cand_src, _cand_tgt]
                _cand_order = np.lexsort((-_cand_tgt, -_cand_src, -_cand_prob))
                _cand_src = _cand_src[_cand_order]
                _cand_tgt = _cand_tgt[_cand_order]
                _cand_prob = _cand_prob[_cand_order]

                children_count: dict[int, int] = {}
                parents_count: dict[int, int] = {}

                for _cand_row in range(_cand_prob.shape[0]):
                    i = int(_cand_src[_cand_row])
                    j = int(_cand_tgt[_cand_row])
                    prob = _cand_prob[_cand_row]
                    n_ch = children_count.get(i, 0)
                    n_pa = parents_count.get(j, 0)
                    if cfg.max_children_per_node is not None and n_ch >= cfg.max_children_per_node:
                        continue
                    if cfg.max_parents_per_node is not None and n_pa >= cfg.max_parents_per_node:
                        continue

                    gi, gj = int(idx_src[i]), int(idx_tgt[j])
                    dist = float(
                        np.linalg.norm(
                            coords_so_far[gi, 1:].astype(np.float32)
                            - coords_so_far[gj, 1:].astype(np.float32)
                        )
                    )
                    all_edges.append((gi, gj, float(prob), dist))
                    children_count[i] = n_ch + 1
                    parents_count[j] = n_pa + 1

            del unet_out
            if secondary_unet_out is not None:
                del secondary_unet_out
            if division_unet_out is not None:
                del division_unet_out
            if division_det_logits is not None:
                del division_det_logits
            if primary_native_detection is not None:
                del primary_native_detection
            if secondary_native_detection is not None:
                del secondary_native_detection

    coords = np.concatenate(coord_lists) if coord_lists else np.empty((0, 4), dtype=np.float32)
    coords = coords.astype(np.float32)
    coords[:, 1:] *= ds_arr
    if division_model is not None and division_evidence_path is not None:
        native_counts = np.asarray(
            [
                len(division_coord_by_t.get(frame, np.empty((0, 4), np.float32)))
                for frame in range(T)
            ],
            np.int32,
        )
        native_offsets = np.concatenate([[0], np.cumsum(native_counts, dtype=np.int64)])
        native_chunks = [
            division_coord_by_t.get(frame, np.empty((0, 4), np.float32)) for frame in range(T)
        ]
        native_coords = (
            np.concatenate(native_chunks).astype(np.float32)
            if native_chunks
            else np.empty((0, 4), np.float32)
        )
        evidence_dtypes = {
            'source_id': np.int64,
            'target_id': np.int64,
            'probability': np.float32,
            'alternative_parent_probability': np.float32,
            'is_target_winner': np.uint8,
            'distance_um': np.float32,
            'source_target_rank': np.int16,
        }
        evidence_arrays = {}
        for evidence_key, evidence_chunks in division_evidence.items():
            evidence_arrays[evidence_key] = (
                np.concatenate(evidence_chunks).astype(evidence_dtypes[evidence_key], copy=False)
                if evidence_chunks
                else np.empty(0, evidence_dtypes[evidence_key])
            )
        division_evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_tmp = division_evidence_path.with_suffix('.npz.tmp')
        with evidence_tmp.open('wb') as evidence_handle:
            biohub_savez(
                evidence_handle,
                **evidence_arrays,
                native_node_coords=native_coords,
                native_frame_offsets=native_offsets,
                spacing_um=original_spacing.astype(np.float32),
            )
        evidence_tmp.replace(division_evidence_path)
    save_native_evidence(
        primary_native_state,
        primary_native_evidence_path,
        T,
        ds.scale,
        coords,
    ) if primary_native_state is not None else None
    save_native_evidence(
        secondary_native_state,
        secondary_native_evidence_path,
        T,
        ds.scale,
        coords,
    ) if secondary_native_state is not None else None
    return coords, all_edges


def _solve_ilp(graph: td.graph.InMemoryGraph, cfg: PredictConfig) -> td.graph.InMemoryGraph:
    return apply_ilp(graph, cfg)


def predict_movies(
    *,
    movie_ids: list[str],
    data_dir: Path,
    output_dir: Path,
    p1_weights: Path,
    p2_weights: Path | None,
    model_c_weights: Path | None,
    p1_evidence_dir: Path,
    p2_evidence_dir: Path,
    model_c_evidence_dir: Path,
    cfg: PredictConfig,
    unet_batch_size: int = 4,
    method: str = 'unet_transformer',
    helper_deadline_epoch: float = float('inf'),
    secondary_edge_weight: float = 0.15,
    secondary_detection_weight: float = 0.475,
    secondary_link_mode: str = 'low_margin_consensus',
    secondary_mix_temperature: float = 1.0,
    secondary_low_margin_max: float = 0.35,
    division_det_threshold: float = 0.99,
    division_pool_um: float = 3.0,
    native_evidence_det_threshold: float = 0.96875,
    native_evidence_pool_um: float = 5.0,
    division_radius_um: float = 20.0,
    division_topk: int = 16,
    division_min_probability: float = 0.01,
    native_evidence_radius_um: float = 20.0,
    native_evidence_topk: int = 16,
    native_evidence_min_probability: float = 0.01,
) -> Path:
    output_dir = output_dir / method / 'split_0'
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob('*.ready*'):
        stale.unlink()
    p1_evidence_dir.mkdir(parents=True, exist_ok=True)
    p2_evidence_dir.mkdir(parents=True, exist_ok=True)
    model_c_evidence_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detect_evidence._ACTIVE_DETECT_CFG = cfg
    if cfg.cudnn_benchmark and device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    model, window_size, downsample = load_model(p1_weights, device)

    secondary_model = None
    if p2_weights is not None:
        secondary_model, secondary_window, secondary_downsample = load_model(p2_weights, device)
        if secondary_window != window_size or secondary_downsample != downsample:
            raise ValueError('P1 and P2 models have incompatible inference grids')

    division_model = None
    if model_c_weights is not None:
        division_model, division_window, division_downsample = load_model(model_c_weights, device)
        if division_window != window_size or division_downsample != downsample:
            raise ValueError('Model C grid does not match P1')

    for name in tqdm(movie_ids, desc='Predicting', disable=not cfg.show_progress):
        ds_path = data_dir / name
        use_division = division_model is not None and time.time() < helper_deadline_epoch
        coords, edges = predict_video(
            model,
            ds_path,
            device,
            cfg=cfg,
            window_size=window_size,
            unet_batch_size=unet_batch_size,
            downsample=downsample,
            secondary_model=secondary_model,
            secondary_edge_weight=secondary_edge_weight,
            secondary_detection_weight=secondary_detection_weight,
            secondary_link_mode=secondary_link_mode,
            secondary_mix_temperature=secondary_mix_temperature,
            secondary_low_margin_max=secondary_low_margin_max,
            division_model=division_model if use_division else None,
            division_evidence_path=(model_c_evidence_dir / f'{name}.npz') if use_division else None,
            division_det_threshold=division_det_threshold,
            division_pool_um=division_pool_um,
            division_radius_um=division_radius_um,
            division_topk=division_topk,
            division_min_probability=division_min_probability,
            primary_native_evidence_path=(
                p1_evidence_dir / f'{name}.npz' if use_division else None
            ),
            secondary_native_evidence_path=(
                p2_evidence_dir / f'{name}.npz' if use_division else None
            ),
            native_evidence_det_threshold=native_evidence_det_threshold,
            native_evidence_pool_um=native_evidence_pool_um,
            native_evidence_radius_um=native_evidence_radius_um,
            native_evidence_topk=native_evidence_topk,
            native_evidence_min_probability=native_evidence_min_probability,
        )
        graph = build_graph(coords, edges)
        if use_division:
            edgegraft_append_graph_ids(p1_evidence_dir / f'{name}.npz', coords, graph)
            edgegraft_append_graph_ids(p2_evidence_dir / f'{name}.npz', coords, graph)
        graph = _solve_ilp(graph, cfg)
        save_graph(graph, output_dir / f'{name}.geff')
        ready_tmp = output_dir / f'{name}.ready.tmp'
        ready_tmp.write_text('complete\n')
        ready_tmp.replace(output_dir / f'{name}.ready')
    return output_dir
