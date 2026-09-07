import json
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
import torch
import zarr
from torch import nn

import biohub.modules.edgegraft.component as base

SPACING = np.asarray((1.625, 0.40625, 0.40625), np.float64)
HORIZON = 5
PAIR_FEATURES = [
    'p1_probability',
    'p2_probability',
    'p1_present',
    'p2_present',
    'p1_alternative',
    'p2_alternative',
    'p1_margin',
    'p2_margin',
    'p1_winner',
    'p2_winner',
    'p1_source_rank',
    'p2_source_rank',
    'p1_target_relative',
    'p2_target_relative',
    'model_probability_max',
    'model_probability_mean',
    'model_probability_absdiff',
    'distance_um',
    'target_union_candidates',
    'source_density_5um',
    'source_density_10um',
    'target_density_5um',
    'target_density_10um',
    'velocity_residual_um',
    'has_velocity',
    'is_raw_parent',
    'raw_edge_probability',
    'raw_parent_probability_gap',
    'ctx_p1_prev_valid',
    'ctx_p1_future_valid',
    'ctx_p1_path_min',
    'ctx_p1_path_mean',
    'ctx_p1_path_geomean',
    'ctx_p1_prev_current_residual',
    'ctx_p1_current_future_residual',
    'ctx_p1_prev_currentcosine',
    'ctx_p1_current_futurecosine',
    'ctx_p2_prev_valid',
    'ctx_p2_future_valid',
    'ctx_p2_path_min',
    'ctx_p2_path_mean',
    'ctx_p2_path_geomean',
    'ctx_p2_prev_current_residual',
    'ctx_p2_current_future_residual',
    'ctx_p2_prev_currentcosine',
    'ctx_p2_current_futurecosine',
    'ctx_prev_identity_agreement',
    'ctx_future_identity_agreement',
    'ctx_prev_probability_absdiff',
    'ctx_future_probability_absdiff',
    'geom_dz_um',
    'geom_dy_um',
    'geom_dx_um',
    'geom_distance_um_exact',
    'geom_source_z_norm',
    'geom_source_y_norm',
    'geom_source_x_norm',
    'geom_target_z_norm',
    'geom_target_y_norm',
    'geom_target_x_norm',
    'geom_incoming_valid',
    'geom_outgoing_valid',
    'geom_incoming_speed_um',
    'geom_incoming_candidatecosine',
    'geom_incoming_residual_um',
    'geom_outgoing_speed_um',
    'geom_outgoing_candidatecosine',
    'geom_outgoing_residual_um',
    'rankrel_p1_probability_to_max',
    'rankrel_p1_probability_rank',
    'rankrel_p2_probability_to_max',
    'rankrel_p2_probability_rank',
    'rankrel_model_probability_max_to_max',
    'rankrel_model_probability_max_rank',
    'rankrel_model_probability_mean_to_max',
    'rankrel_model_probability_mean_rank',
    'rankrel_raw_edge_probability_to_max',
    'rankrel_raw_edge_probability_rank',
    'rankrel_ctx_p1_path_geomean_to_max',
    'rankrel_ctx_p1_path_geomean_rank',
    'rankrel_ctx_p2_path_geomean_to_max',
    'rankrel_ctx_p2_path_geomean_rank',
    'rankrel_distance_um_from_min',
    'rankrel_distance_um_rank',
    'rankrel_velocity_residual_um_from_min',
    'rankrel_velocity_residual_um_rank',
    'rankrel_geom_incoming_residual_um_from_min',
    'rankrel_geom_incoming_residual_um_rank',
    'rankrel_geom_outgoing_residual_um_from_min',
    'rankrel_geom_outgoing_residual_um_rank',
]
RELATIVE_MAX = [
    'p1_probability',
    'p2_probability',
    'model_probability_max',
    'model_probability_mean',
    'raw_edge_probability',
    'ctx_p1_path_min',
    'ctx_p1_path_mean',
    'ctx_p1_path_geomean',
    'ctx_p2_path_min',
    'ctx_p2_path_mean',
    'ctx_p2_path_geomean',
]
RELATIVE_MIN = [
    'distance_um',
    'velocity_residual_um',
    'geom_distance_um_exact',
    'geom_incoming_residual_um',
    'geom_outgoing_residual_um',
]


class CropEncoder(nn.Module):
    def __init__(self, dim: int = 24):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(1, 12, 3, padding=1),
            nn.GroupNorm(3, 12),
            nn.SiLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(12, 24, 3, padding=1),
            nn.GroupNorm(6, 24),
            nn.SiLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(24, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d(1),
        )
        self.head = nn.Linear(32, dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(value).flatten(1))


class LinkRanker(nn.Module):
    def __init__(self, tab_dim: int):
        super().__init__()
        self.crop = CropEncoder(24)
        self.tab = nn.Sequential(nn.Linear(tab_dim, 48), nn.LayerNorm(48), nn.SiLU())
        self.score = nn.Sequential(
            nn.Linear(24 * 4 + 48, 96),
            nn.LayerNorm(96),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(96, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self, source: torch.Tensor, target: torch.Tensor, tab: torch.Tensor
    ) -> torch.Tensor:
        first = self.crop(source)
        second = self.crop(target)
        value = torch.cat(
            [
                first,
                second,
                torch.abs(first - second),
                first * second,
                self.tab(tab),
            ],
            dim=1,
        )
        return self.score(value).squeeze(1)


def _group_mean_max(
    values: torch.Tensor, groups: torch.Tensor, count: int
) -> tuple[torch.Tensor, torch.Tensor]:
    dim = values.shape[1]
    sums = values.new_zeros((count, dim))
    sums.index_add_(0, groups, values)
    counts = values.new_zeros((count, 1))
    counts.index_add_(0, groups, values.new_ones((len(values), 1)))
    means = sums / counts.clamp_min(1.0)
    maxima = values.new_full((count, dim), -torch.inf)
    maxima.scatter_reduce_(
        0,
        groups[:, None].expand(-1, dim),
        values,
        reduce='amax',
        include_self=True,
    )
    return means, maxima


class CandidateSetRanker(nn.Module):
    def __init__(self, tab_dim: int):
        super().__init__()
        self.crop = CropEncoder(24)
        self.tab = nn.Sequential(nn.Linear(tab_dim, 64), nn.LayerNorm(64), nn.SiLU())
        self.pair = nn.Sequential(
            nn.Linear(24 * 4 + 64, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.06),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
        )
        self.score = nn.Sequential(
            nn.Linear(128 * 5, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        tab: torch.Tensor,
        groups: torch.Tensor,
        count: int,
    ) -> torch.Tensor:
        first = self.crop(source)
        second = self.crop(target)
        pair = self.pair(
            torch.cat(
                [
                    first,
                    second,
                    torch.abs(first - second),
                    first * second,
                    self.tab(tab),
                ],
                dim=1,
            )
        )
        mean, maximum = _group_mean_max(pair, groups, count)
        context = torch.cat(
            [
                pair,
                mean[groups],
                maximum[groups],
                pair - mean[groups],
                pair - maximum[groups],
            ],
            dim=1,
        )
        return self.score(context).squeeze(1)


def _position(row: dict[str, Any]) -> np.ndarray:
    return np.asarray((row['z'], row['y'], row['x']), np.float64)


def edge_sets(edges: list[dict]) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    incoming: dict[int, set[int]] = defaultdict(set)
    outgoing: dict[int, set[int]] = defaultdict(set)
    for edge in edges:
        source = int(edge['source_id'])
        target = int(edge['target_id'])
        outgoing[source].add(target)
        incoming[target].add(source)
    return incoming, outgoing


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-8 else 0.0


def _augment_geometry(rows: list[dict], nodes: dict[int, dict], edges: list[dict]) -> None:
    if not rows or not nodes:
        return
    incoming, outgoing = edge_sets(edges)
    physical = {node: _position(value) * SPACING for node, value in nodes.items()}
    all_xyz = np.stack(list(physical.values()))
    lower = all_xyz.min(axis=0)
    span = np.maximum(all_xyz.max(axis=0) - lower, 1e-6)
    for row in rows:
        source = int(row['source'])
        target = int(row['target'])
        if source not in physical or target not in physical:
            continue
        source_xyz = physical[source]
        target_xyz = physical[target]
        displacement = target_xyz - source_xyz
        distance = float(np.linalg.norm(displacement))
        unit = displacement / max(distance, 1e-6)
        source_norm = (source_xyz - lower) / span
        target_norm = (target_xyz - lower) / span
        previous = next(iter(incoming[source])) if len(incoming[source]) == 1 else None
        following = next(iter(outgoing[target])) if len(outgoing[target]) == 1 else None
        in_vector = (
            source_xyz - physical[previous] if previous in physical else np.zeros(3, np.float64)
        )
        out_vector = (
            physical[following] - target_xyz if following in physical else np.zeros(3, np.float64)
        )
        values = row['features']
        values.update(
            {
                'geom_dz_um': displacement[0],
                'geom_dy_um': displacement[1],
                'geom_dx_um': displacement[2],
                'geom_unit_z': unit[0],
                'geom_unit_y': unit[1],
                'geom_unit_x': unit[2],
                'geom_distance_um_exact': distance,
                'geom_source_z_norm': source_norm[0],
                'geom_source_y_norm': source_norm[1],
                'geom_source_x_norm': source_norm[2],
                'geom_target_z_norm': target_norm[0],
                'geom_target_y_norm': target_norm[1],
                'geom_target_x_norm': target_norm[2],
                'geom_incoming_valid': float(previous in physical),
                'geom_outgoing_valid': float(following in physical),
            }
        )
        for prefix, vector in (('geom_incoming', in_vector), ('geom_outgoing', out_vector)):
            magnitude = float(np.linalg.norm(vector))
            residual = displacement - vector
            values.update(
                {
                    f'{prefix}_dz_um': vector[0],
                    f'{prefix}_dy_um': vector[1],
                    f'{prefix}_dx_um': vector[2],
                    f'{prefix}_speed_um': magnitude,
                    f'{prefix}_candidatecosine': cosine(displacement, vector),
                    f'{prefix}_residual_dz_um': residual[0],
                    f'{prefix}_residual_dy_um': residual[1],
                    f'{prefix}_residual_dx_um': residual[2],
                    f'{prefix}_residual_um': float(np.linalg.norm(residual)),
                }
            )


def _chains(nodes: dict[int, dict], edges: list[dict]) -> tuple[dict, dict]:
    incoming, outgoing = edge_sets(edges)
    history = {}
    future = {}
    for node in nodes:
        past = [node]
        current = node
        for _ in range(HORIZON):
            if len(incoming[current]) != 1:
                break
            current = next(iter(incoming[current]))
            past.append(current)
        history[node] = past
        ahead = [node]
        current = node
        for _ in range(HORIZON):
            if len(outgoing[current]) != 1:
                break
            current = next(iter(outgoing[current]))
            ahead.append(current)
        future[node] = ahead
    return history, future


def _linear_velocity(
    chain: list[int], position: dict[int, np.ndarray], backward: bool
) -> tuple[np.ndarray, float]:
    if len(chain) < 2:
        return np.zeros(3, np.float64), 0.0
    times = np.arange(len(chain), dtype=np.float64)
    if backward:
        times = -times
    values = np.stack([position[node] for node in chain])
    centered = times - times.mean()
    denominator = float(np.dot(centered, centered))
    slope = (centered[:, None] * (values - values.mean(axis=0))).sum(axis=0) / max(
        denominator, 1e-8
    )
    fitted = values.mean(axis=0) + centered[:, None] * slope
    residual = float(np.sqrt(np.mean(np.sum((values - fitted) ** 2, axis=1))))
    return slope, residual


def _augment_trajectory(rows: list[dict], nodes: dict[int, dict], edges: list[dict]) -> None:
    if not rows:
        return
    position = {node: _position(value) * SPACING for node, value in nodes.items()}
    history, future = _chains(nodes, edges)
    history_velocity = {}
    history_fit = {}
    future_velocity = {}
    future_fit = {}
    for node in position:
        history_velocity[node], history_fit[node] = _linear_velocity(history[node], position, True)
        future_velocity[node], future_fit[node] = _linear_velocity(future[node], position, False)
    for row in rows:
        source = int(row['source'])
        target = int(row['target'])
        if source not in position or target not in position:
            continue
        source_pos = position[source]
        target_pos = position[target]
        displacement = target_pos - source_pos
        source_velocity = history_velocity[source]
        target_velocity = future_velocity[target]
        values = row['features']
        values.update(
            {
                'traj_history_len': len(history[source]) - 1,
                'traj_future_len': len(future[target]) - 1,
                'traj_history_fit_residual': history_fit[source],
                'traj_future_fit_residual': future_fit[target],
                'traj_forward_linear_residual': float(
                    np.linalg.norm(target_pos - (source_pos + source_velocity))
                ),
                'traj_reverse_linear_residual': float(
                    np.linalg.norm(source_pos - (target_pos - target_velocity))
                ),
                'traj_velocity_agreement': float(np.linalg.norm(source_velocity - target_velocity)),
                'traj_velocitycosine': cosine(source_velocity, target_velocity),
                'traj_candidate_historycosine': cosine(displacement, source_velocity),
                'traj_candidate_futurecosine': cosine(displacement, target_velocity),
            }
        )
        acceleration = np.zeros(3, np.float64)
        if len(history[source]) >= 3:
            first = position[history[source][0]] - position[history[source][1]]
            previous = position[history[source][1]] - position[history[source][2]]
            acceleration = first - previous
        values['traj_acceleration_residual'] = float(
            np.linalg.norm(target_pos - (source_pos + source_velocity + 0.5 * acceleration))
        )
        for horizon in range(1, HORIZON + 1):
            if len(history[source]) > horizon:
                velocity = (source_pos - position[history[source][horizon]]) / horizon
                values[f'traj_forward_h{horizon}_valid'] = 1.0
                values[f'traj_forward_h{horizon}_residual'] = float(
                    np.linalg.norm(target_pos - (source_pos + velocity))
                )
                values[f'traj_forward_h{horizon}cosine'] = cosine(displacement, velocity)
            else:
                values[f'traj_forward_h{horizon}_valid'] = 0.0
                values[f'traj_forward_h{horizon}_residual'] = 0.0
                values[f'traj_forward_h{horizon}cosine'] = 0.0
            if len(future[target]) > horizon:
                velocity = (position[future[target][horizon]] - target_pos) / horizon
                values[f'traj_reverse_h{horizon}_valid'] = 1.0
                values[f'traj_reverse_h{horizon}_residual'] = float(
                    np.linalg.norm(source_pos - (target_pos - velocity))
                )
                values[f'traj_reverse_h{horizon}cosine'] = cosine(displacement, velocity)
            else:
                values[f'traj_reverse_h{horizon}_valid'] = 0.0
                values[f'traj_reverse_h{horizon}_residual'] = 0.0
                values[f'traj_reverse_h{horizon}cosine'] = 0.0


def add_relative(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    groups = result.groupby('target', sort=False)
    for column in RELATIVE_MAX:
        if column in result:
            result[f'rankrel_{column}_to_max'] = result[column] - groups[column].transform('max')
            result[f'rankrel_{column}_rank'] = groups[column].rank(
                method='average', ascending=False
            )
    for column in RELATIVE_MIN:
        if column in result:
            result[f'rankrel_{column}_from_min'] = result[column] - groups[column].transform('min')
            result[f'rankrel_{column}_rank'] = groups[column].rank(method='average', ascending=True)
    return result


def _rows_frame(rows: list[dict]) -> pd.DataFrame:
    records = []
    for row in rows:
        records.append(
            {
                'component': int(row['component']),
                'source': int(row['source']),
                'target': int(row['target']),
                'native_probability': float(row['native_probability']),
                **{name: float(value) for name, value in row['features'].items()},
            }
        )
    return pd.DataFrame(records)


def _extract_crops(
    zarr_path: Path, nodes: dict[int, dict], requested: np.ndarray
) -> dict[int, np.ndarray]:
    root: Any = zarr.open_group(str(zarr_path), mode='r')['0']
    rz, ryx = 3, 10
    result: dict[int, np.ndarray] = {}
    by_frame: dict[int, list[int]] = defaultdict(list)
    for node in map(int, requested):
        if node in nodes:
            by_frame[int(nodes[node]['t'])].append(node)
    for frame_index, node_ids in by_frame.items():
        frame = np.asarray(root[frame_index], np.float32)
        sample = frame[::2, ::4, ::4]
        low, high = np.percentile(sample, (1.0, 99.8))
        frame = np.clip((frame - low) / max(float(high - low), 1.0), 0.0, 1.0)
        padded = np.pad(frame, ((rz, rz), (ryx, ryx), (ryx, ryx)), mode='reflect')
        for node in node_ids:
            attrs = nodes[node]
            z = int(np.clip(round(float(attrs['z'])), 0, frame.shape[0] - 1))
            y = int(np.clip(round(float(attrs['y'])), 0, frame.shape[1] - 1))
            x = int(np.clip(round(float(attrs['x'])), 0, frame.shape[2] - 1))
            patch = padded[z : z + 2 * rz + 1, y : y + 2 * ryx + 1, x : x + 2 * ryx + 1]
            result[node] = np.rint(patch * 255.0).astype(np.uint8)
    return result


def _crop_tensor(ids: np.ndarray, crops: dict[int, np.ndarray]) -> torch.Tensor:
    values = np.zeros((len(ids), 7, 21, 21), np.uint8)
    for index, node in enumerate(map(int, ids)):
        value = crops.get(node)
        if value is not None:
            values[index] = value
    return torch.from_numpy(values.astype(np.float32) / 255.0).unsqueeze(1)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values.astype(np.float64, copy=False), -30.0, 30.0)
    return (1.0 / (1.0 + np.exp(-values))).astype(np.float32)


class EdgeGraftV15Runtime:
    def __init__(self, artifact_dir: str | Path):
        self.root = Path(artifact_dir)
        self._models: dict[tuple[str, int], Any] = {}
        self.ranker_report = json.loads((self.root / 'ranker_report.json').read_text())
        self.metric_report = json.loads((self.root / 'metric_report.json').read_text())
        self.metric_threshold = float(self.metric_report['frozen_median_threshold'])
        self._base_builder = base.EdgeGraftComponentRuntime.__new__(base.EdgeGraftComponentRuntime)

    def _load_crop(self, family: str, fold: int):
        key = (family, fold)
        if key in self._models:
            return self._models[key]
        checkpoint = torch.load(
            self.root / f'{family}_fold_{fold}.pt',
            map_location='cpu',
            weights_only=False,
        )
        model = (
            LinkRanker(len(checkpoint['features']))
            if family == 'v14'
            else CandidateSetRanker(len(checkpoint['features']))
        )
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        value = (
            model,
            list(checkpoint['features']),
            np.asarray(checkpoint['mean'], np.float32),
            np.asarray(checkpoint['std'], np.float32),
        )
        self._models[key] = value
        return value

    def _load_joblib(self, family: str, fold: int):
        key = (family, fold)
        if key not in self._models:
            self._models[key] = joblib.load(self.root / f'{family}_fold_{fold}.joblib')
        return self._models[key]

    @torch.no_grad()
    def _score_crop(
        self, family: str, fold: int, frame: pd.DataFrame, crops: dict[int, np.ndarray]
    ) -> np.ndarray:
        model, features, mean, std = self._load_crop(family, fold)
        output = np.empty(len(frame), np.float32)
        batch_targets = 256
        target_values = list(dict.fromkeys(map(int, frame.target)))
        for start in range(0, len(target_values), batch_targets):
            targets = set(target_values[start : start + batch_targets])
            index = np.flatnonzero(frame.target.isin(targets).to_numpy())
            part = frame.iloc[index].copy()
            codes, uniques = pd.factorize(part.target, sort=False)
            source = _crop_tensor(part.source.to_numpy(np.int64), crops)
            target = _crop_tensor(part.target.to_numpy(np.int64), crops)
            tab = (
                part.reindex(columns=features)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .to_numpy(np.float32)
            )
            tab = (tab - mean) / std
            if family == 'v14':
                score = model(source, target, torch.from_numpy(tab))
            else:
                score = model(
                    source,
                    target,
                    torch.from_numpy(tab),
                    torch.from_numpy(codes.astype(np.int64)),
                    len(uniques),
                )
            output[index] = score.float().cpu().numpy()
        return output

    def candidate_frame(
        self, raw_nodes, raw_edges, final_nodes, final_edges, p1_path, p2_path
    ) -> pd.DataFrame:
        rows = base.EdgeGraftComponentRuntime.candidate_rows(
            self._base_builder, raw_nodes, raw_edges, p1_path, p2_path
        )
        _augment_geometry(rows, raw_nodes, raw_edges)
        mapping = base.raw_to_final_map(raw_nodes, final_nodes)
        current = {(int(edge['source_id']), int(edge['target_id'])) for edge in final_edges}
        mapped = []
        for row in rows:
            source = mapping.get(int(row['source']))
            target = mapping.get(int(row['target']))
            if source is None or target is None:
                continue
            value = dict(row)
            value['source'] = int(source)
            value['target'] = int(target)
            value['features'] = dict(row['features'])
            value['features']['is_raw_parent'] = float((int(source), int(target)) in current)
            mapped.append(value)
        _augment_trajectory(mapped, final_nodes, final_edges)
        frame = _rows_frame(mapped)
        if frame.empty:
            return frame
        frame = frame.sort_values(
            ['model_probability_max', 'model_probability_mean', 'raw_edge_probability'],
            ascending=False,
        ).drop_duplicates(['source', 'target'], keep='first')
        return add_relative(frame.reset_index(drop=True))

    def select_decisions(
        self, stem: str, frame: pd.DataFrame, final_edges: list[dict], protected: set[int]
    ) -> pd.DataFrame:
        incoming, outgoing = edge_sets(final_edges)
        records = []
        for target, group in frame.groupby('target', sort=False):
            target = int(cast(Any, target))
            if len(incoming[target]) != 1:
                continue
            current_source = next(iter(incoming[target]))
            winner = group.loc[group.rank_score.idxmax()]
            source = int(winner.source)
            if source == current_source:
                continue
            values = np.sort(group.rank_score.to_numpy(np.float64))
            runner_up = float(values[-2]) if len(values) > 1 else 0.0
            baseline = group[group.source == current_source]
            current_present = not baseline.empty
            current_score = float(baseline.rank_score.max()) if current_present else np.nan
            source_target = next(iter(outgoing[source])) if len(outgoing[source]) == 1 else -1
            remove_count = 1 + int(source_target >= 0 and source_target != target)
            touched = {source, target, current_source}
            if source_target >= 0:
                touched.add(int(source_target))
            records.append(
                {
                    'dataset': stem,
                    'fold': int(zlib.crc32(stem.encode()) % 5),
                    'component': int(winner.component),
                    'source': source,
                    'target': target,
                    'current_source': current_source,
                    'source_current_target': int(source_target),
                    'topology': '2to1' if remove_count == 2 else '1to1',
                    'remove_count': remove_count,
                    'top_score': float(winner.rank_score),
                    'runner_up_score': runner_up,
                    'top_margin': float(winner.rank_score - runner_up),
                    'current_present': current_present,
                    'current_score': current_score,
                    'advantage': (
                        float(winner.rank_score - current_score) if current_present else np.nan
                    ),
                    'protected': bool(touched & protected),
                }
            )
        return pd.DataFrame(records)

    def rich_decisions(self, decision: pd.DataFrame, evidence: pd.DataFrame) -> pd.DataFrame:
        result = decision.copy()
        if result.empty:
            return result
        pair = evidence.sort_values(
            ['model_probability_max', 'model_probability_mean', 'raw_edge_probability'],
            ascending=False,
        ).drop_duplicates(['source', 'target'], keep='first')
        available = [name for name in PAIR_FEATURES if name in pair]
        for prefix, source_column in (('new', 'source'), ('old', 'current_source')):
            right = pair[['source', 'target', *available]].rename(
                columns={
                    'source': source_column,
                    **{name: f'{prefix}_{name}' for name in available},
                }
            )
            result = result.merge(
                right,
                on=[source_column, 'target'],
                how='left',
                validate='many_to_one',
            )
        result['current_score_filled'] = result.current_score.fillna(-1.0)
        result['advantage_filled'] = result.advantage.fillna(result.top_score + 1.0)
        result['topology_2to1'] = (result.topology == '2to1').astype(np.float32)
        result['embryo_6bba'] = result.dataset.str.startswith('6bba').astype(np.float32)
        for name in available:
            result[f'delta_{name}'] = pd.to_numeric(
                result[f'new_{name}'], errors='coerce'
            ) - pd.to_numeric(result[f'old_{name}'], errors='coerce')
        return result

    def apply(
        self,
        stem: str,
        zarr_path: str | Path,
        raw_nodes: dict[int, dict],
        raw_edges: list[dict],
        final_nodes: dict[int, dict],
        final_edges: list[dict],
        p1_path: str | Path,
        p2_path: str | Path,
    ):
        counters: Counter = Counter()
        fold = int(zlib.crc32(stem.encode('utf-8')) % 5)
        frame = self.candidate_frame(
            raw_nodes, raw_edges, final_nodes, final_edges, p1_path, p2_path
        )
        counters['candidate_rows'] = len(frame)
        counters['fold'] = fold
        if frame.empty:
            return final_edges, dict(counters)
        requested = np.unique(
            np.r_[frame.source.to_numpy(np.int64), frame.target.to_numpy(np.int64)]
        )
        crops = _extract_crops(Path(zarr_path), final_nodes, requested)
        frame['crop_oof_score'] = self._score_crop('v14', fold, frame, crops)
        groups = frame.groupby('target', sort=False).crop_oof_score
        frame['crop_oof_valid'] = 1.0
        frame['crop_oof_to_max'] = frame.crop_oof_score - groups.transform('max')
        frame['crop_oof_rank'] = groups.rank(method='average', ascending=False).astype(np.float32)
        frame['set_oof_score'] = self._score_crop('v15', fold, frame, crops)
        groups = frame.groupby('target', sort=False).set_oof_score
        frame['set_oof_valid'] = 1.0
        frame['set_oof_to_max'] = frame.set_oof_score - groups.transform('max')
        frame['set_oof_rank'] = groups.rank(method='average', ascending=False).astype(np.float32)

        rank_features = list(self.ranker_report['features'])
        rank_x = (
            frame.reindex(columns=rank_features)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(np.float32)
        )
        ranker = self._load_joblib('ranker', fold)
        frame['oof_score'] = ranker.predict_proba(rank_x)[:, 1]
        frame['rank_score'] = _sigmoid(frame.oof_score.to_numpy())
        frame = frame.sort_values('rank_score', ascending=False).drop_duplicates(
            ['source', 'target'], keep='first'
        )

        protected = base.protected_fork_nodes(final_edges)
        decision = self.select_decisions(stem, frame, final_edges, protected)
        counters['decisions'] = len(decision)
        if decision.empty:
            return final_edges, dict(counters)
        rich = self.rich_decisions(decision, frame)
        metric_features = list(self.metric_report['features'])
        metric_x = (
            rich.reindex(columns=metric_features)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(np.float32)
        )
        metric = self._load_joblib('metric', fold)
        rich['metric_score'] = metric.predict_proba(metric_x)[:, 1]
        selected = rich[
            (~rich.protected) & (rich.metric_score >= self.metric_threshold)
        ].sort_values(['metric_score', 'top_score', 'top_margin'], ascending=False)
        counters['selected'] = len(selected)
        chosen = []
        used_sources: set[int] = set()
        selected_rows: Any = selected.itertuples(index=False)
        for row in selected_rows:
            source = int(row.source)
            if source in used_sources:
                counters['source_conflict'] += 1
                continue
            used_sources.add(source)
            chosen.append(row)
        counters['applied'] = len(chosen)
        if not chosen:
            return final_edges, dict(counters)

        incoming, outgoing = edge_sets(final_edges)
        edge_by_pair = {
            (int(edge['source_id']), int(edge['target_id'])): dict(edge) for edge in final_edges
        }
        remove: set[tuple[int, int]] = set()
        add: set[tuple[int, int]] = set()
        frame_rows: Any = frame.itertuples(index=False)
        probability = {
            (int(row.source), int(row.target)): float(row.native_probability) for row in frame_rows
        }
        for row in chosen:
            source = int(row.source)
            target = int(row.target)
            current_source = int(row.current_source)
            source_target = int(row.source_current_target)
            if (current_source, target) not in edge_by_pair:
                raise RuntimeError('V15 stale target parent before atomic transaction')
            actual = next(iter(outgoing[source])) if len(outgoing[source]) == 1 else -1
            if len(outgoing[source]) > 1 or actual != source_target:
                raise RuntimeError('V15 stale source child before atomic transaction')
            remove.add((current_source, target))
            if source_target >= 0 and source_target != target:
                remove.add((source, source_target))
            add.add((source, target))
        for edge in remove:
            edge_by_pair.pop(edge, None)
        for source, target in add:
            edge_by_pair.setdefault(
                (source, target),
                {
                    'source_id': source,
                    'target_id': target,
                    'edge_prob': probability.get((source, target), 1.0),
                },
            )
        result = list(edge_by_pair.values())
        final_incoming, final_outgoing = edge_sets(result)
        if any(len(value) > 1 for value in final_incoming.values()):
            raise RuntimeError('V15 transaction produced in-degree > 1')
        if any(len(value) > 2 for value in final_outgoing.values()):
            raise RuntimeError('V15 transaction produced out-degree > 2')
        if base.protected_fork_nodes(result) != protected:
            raise RuntimeError('V15 changed protected fork topology')
        counters['removed'] = len(
            set(edge_by_pair)
            ^ {(int(edge['source_id']), int(edge['target_id'])) for edge in final_edges}
        )
        counters['removed_edges'] = len(remove)
        counters['added_edges'] = len(add)
        return result, dict(counters)
