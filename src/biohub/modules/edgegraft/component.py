from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from biohub.modules.edgegraft.features import (
    BASE_FEATURES,
    best_context,
    model_context,
)
from biohub.modules.edgegraft.geometry import (
    SPACING,
    conflict_pairs,
    density_maps,
    factorize,
    greedy_native_map,
    node_identity,
    node_position,
    protected_fork_nodes,
    raw_to_final_map,
)

CALIBRATED_THRESHOLD = 0.004891517842598334


def _segment_reduce(values, groups, count, reduce):
    if reduce == 'mean':
        output = torch.zeros((count, values.shape[1]), device=values.device, dtype=values.dtype)
        output.index_add_(0, groups, values)
        totals = torch.zeros(count, device=values.device, dtype=values.dtype)
        totals.index_add_(
            0,
            groups,
            torch.ones(len(groups), device=values.device, dtype=values.dtype),
        )
        return output / totals.clamp_min(1).unsqueeze(1)
    output = torch.full(
        (count, values.shape[1]),
        -torch.inf,
        device=values.device,
        dtype=values.dtype,
    )
    output.scatter_reduce_(
        0,
        groups[:, None].expand_as(values),
        values,
        reduce='amax',
        include_self=True,
    )
    return output


class ComponentEdgeNet(nn.Module):
    def __init__(self, tab_dim, hidden=96, rounds=3, emb_dim=24):
        super().__init__()
        image_dim = emb_dim * 4
        self.edge = nn.Sequential(
            nn.Linear(tab_dim + image_dim + 6, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(0.08),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.updates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden * 5, hidden * 2),
                    nn.LayerNorm(hidden * 2),
                    nn.SiLU(),
                    nn.Dropout(0.08),
                    nn.Linear(hidden * 2, hidden),
                )
                for _ in range(rounds)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(rounds)])
        self.out = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Linear(hidden // 2, 1)
        )

    def forward(self, tab, source_emb, target_emb, structural, source_group, target_group):
        image = torch.cat(
            [
                source_emb,
                target_emb,
                torch.abs(source_emb - target_emb),
                source_emb * target_emb,
            ],
            1,
        )
        hidden = self.edge(torch.cat([tab, image, structural], 1))
        source_count = int(source_group.max()) + 1
        target_count = int(target_group.max()) + 1
        for update, norm in zip(self.updates, self.norms):
            source_mean = _segment_reduce(hidden, source_group, source_count, 'mean')[source_group]
            source_max = _segment_reduce(hidden, source_group, source_count, 'max')[source_group]
            target_mean = _segment_reduce(hidden, target_group, target_count, 'mean')[target_group]
            target_max = _segment_reduce(hidden, target_group, target_count, 'max')[target_group]
            hidden = norm(
                hidden
                + update(torch.cat([hidden, source_mean, source_max, target_mean, target_max], 1))
            )
        return self.out(hidden).squeeze(1)


def read_evidence(path: str | Path, raw_nodes):
    with np.load(path, allow_pickle=False) as data:
        source = data['source_id'].astype(np.int64, copy=False)
        target = data['target_id'].astype(np.int64, copy=False)
        probability = data['probability'].astype(np.float32, copy=False)
        alternative = data['alternative_parent_probability'].astype(np.float32, copy=False)
        winner = data['is_target_winner'].astype(np.float32, copy=False)
        rank = data['source_target_rank'].astype(np.int16, copy=False)
        distance = data['distance_um'].astype(np.float32, copy=False)
        native = data['native_node_coords'].astype(np.float32, copy=False)
        offsets = data['native_frame_offsets'].astype(np.int64, copy=False)
        spacing = (
            data['spacing_um'].astype(np.float32, copy=False)
            if 'spacing_um' in data
            else SPACING.astype(np.float32)
        )
        if 'mapped_ab_node' in data:
            fused_mapping = data['mapped_ab_node'].astype(np.int64, copy=False)
            if 'fused_graph_node_id' in data:
                fused_graph_ids = data['fused_graph_node_id'].astype(np.int64, copy=False)
            elif 'ab_node_coords' in data:
                fused_coords = data['ab_node_coords'].astype(np.float32, copy=False)
                raw_by_identity = {
                    node_identity(row): int(node_id) for node_id, row in raw_nodes.items()
                }
                fused_graph_ids = np.asarray(
                    [
                        raw_by_identity.get(
                            (
                                int(round(float(coord[0]))),
                                round(float(coord[1]), 5),
                                round(float(coord[2]), 5),
                                round(float(coord[3]), 5),
                            ),
                            -1,
                        )
                        for coord in fused_coords
                    ],
                    np.int64,
                )
            else:
                fused_graph_ids = np.empty(0, np.int64)
            mapped = np.full(len(fused_mapping), -1, np.int64)
            usable = (fused_mapping >= 0) & (fused_mapping < len(fused_graph_ids))
            mapped[usable] = fused_graph_ids[fused_mapping[usable]]
        else:
            mapped = greedy_native_map(native, offsets, raw_nodes, spacing)
        valid = (source >= 0) & (target >= 0) & (source < len(mapped)) & (target < len(mapped))
        rows = []
        for index in np.flatnonzero(valid):
            mapped_source = int(mapped[source[index]])
            mapped_target = int(mapped[target[index]])
            if mapped_source < 0 or mapped_target < 0:
                continue
            rows.append(
                (
                    mapped_source,
                    mapped_target,
                    float(probability[index]),
                    float(alternative[index]),
                    float(winner[index]),
                    float(rank[index]),
                    float(distance[index]),
                )
            )
    deduped = {}
    for (
        source_id,
        target_id,
        probability_value,
        alternative_value,
        winner_value,
        rank_value,
        distance_value,
    ) in rows:
        key = (source_id, target_id)
        value = (
            probability_value,
            alternative_value,
            winner_value,
            rank_value,
            distance_value,
        )
        old = deduped.get(key)
        if old is None or value[0] > old[0]:
            deduped[key] = value
    context_rows = [
        (source_id, target_id, probability_value, alternative_value, winner_value, distance_value)
        for (
            source_id,
            target_id,
            probability_value,
            alternative_value,
            winner_value,
            _rank_value,
            distance_value,
        ) in rows
    ]
    return deduped, context_rows


class EdgeGraftComponentRuntime:
    def __init__(self, artifact_dir: str | Path, threshold: float = CALIBRATED_THRESHOLD):
        self.artifact_dir = Path(artifact_dir)
        checkpoint = torch.load(
            self.artifact_dir / 'component_gate_best.pt',
            map_location='cpu',
            weights_only=False,
        )
        self.features = list(checkpoint['features'])
        if self.features[: len(BASE_FEATURES)] != BASE_FEATURES or len(self.features) != 72:
            raise RuntimeError('Unexpected EdgeGRAFT checkpoint feature contract')
        self.mean = np.asarray(checkpoint['mean'], np.float32)
        self.std = np.asarray(checkpoint['std'], np.float32)
        self.model = ComponentEdgeNet(len(self.features))
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.threshold = float(threshold)

    def candidate_rows(self, raw_nodes, raw_edges, p1_path, p2_path):
        p1, p1_native = read_evidence(p1_path, raw_nodes)
        p2, p2_native = read_evidence(p2_path, raw_nodes)
        evidence = {'p1': p1, 'p2': p2}
        keep, component_of = conflict_pairs(evidence)
        candidates: dict[int, set[int]] = defaultdict(set)
        for source, target in keep:
            if source in raw_nodes and target in raw_nodes:
                candidates[target].add(source)
        raw_parent = {}
        raw_probability = {}
        predecessor = {}
        for edge in raw_edges:
            source = int(edge['source_id'])
            target = int(edge['target_id'])
            probability = float(edge.get('edge_prob') or 0.0)
            if probability > raw_probability.get(target, -np.inf):
                raw_parent[target] = source
                raw_probability[target] = probability
            predecessor[target] = source
        density = density_maps(raw_nodes)
        position = {node: node_position(row) for node, row in raw_nodes.items()}
        physical = {node: (value * SPACING).astype(np.float32) for node, value in position.items()}
        probability_max = {name: defaultdict(float) for name in evidence}
        for name, model in evidence.items():
            for (source, target), value in model.items():
                if (source, target) in keep:
                    probability_max[name][target] = max(probability_max[name][target], value[0])
        need_in = set(source for source, _target in keep)
        need_out = set(target for _source, target in keep)
        contexts = {
            'p1': best_context(p1_native, need_in, need_out),
            'p2': best_context(p2_native, need_in, need_out),
        }
        rows = []
        for target, sources in candidates.items():
            for source in sorted(sources):
                first = p1.get((source, target))
                second = p2.get((source, target))
                p1_probability = first[0] if first else 0.0
                p2_probability = second[0] if second else 0.0
                raw_edge_probability = raw_probability.get(target, 0.0)
                distance = float(np.linalg.norm((position[target] - position[source]) * SPACING))
                previous = predecessor.get(source)
                if previous in position:
                    predicted = position[source] + (position[source] - position[previous])
                    velocity_residual = float(
                        np.linalg.norm((position[target] - predicted) * SPACING)
                    )
                    has_velocity = 1.0
                else:
                    velocity_residual = distance
                    has_velocity = 0.0
                source_density = density.get(source, (0.0, 0.0))
                target_density = density.get(target, (0.0, 0.0))
                values = [
                    p1_probability,
                    p2_probability,
                    float(first is not None),
                    float(second is not None),
                    first[1] if first else 0.0,
                    second[1] if second else 0.0,
                    first[0] - first[1] if first else 0.0,
                    second[0] - second[1] if second else 0.0,
                    first[2] if first else 0.0,
                    second[2] if second else 0.0,
                    first[3] if first else 99.0,
                    second[3] if second else 99.0,
                    p1_probability - probability_max['p1'][target],
                    p2_probability - probability_max['p2'][target],
                    max(p1_probability, p2_probability),
                    (p1_probability + p2_probability) / 2.0,
                    abs(p1_probability - p2_probability),
                    distance,
                    float(len(sources)),
                    source_density[0],
                    source_density[1],
                    target_density[0],
                    target_density[1],
                    velocity_residual,
                    has_velocity,
                    float(raw_parent.get(target) == source),
                    raw_edge_probability if raw_parent.get(target) == source else 0.0,
                    max(p1_probability, p2_probability) - raw_edge_probability,
                ]
                features = dict(zip(BASE_FEATURES, values))
                p1_context, p1_identity = model_context(
                    'ctx_p1',
                    source,
                    target,
                    p1_probability,
                    physical,
                    *contexts['p1'],
                )
                p2_context, p2_identity = model_context(
                    'ctx_p2',
                    source,
                    target,
                    p2_probability,
                    physical,
                    *contexts['p2'],
                )
                features.update(p1_context)
                features.update(p2_context)
                features.update(
                    {
                        'ctx_prev_identity_agreement': float(
                            p1_identity[0] >= 0 and p1_identity[0] == p2_identity[0]
                        ),
                        'ctx_future_identity_agreement': float(
                            p1_identity[1] >= 0 and p1_identity[1] == p2_identity[1]
                        ),
                        'ctx_prev_probability_absdiff': abs(
                            p1_context['ctx_p1_prev_probability']
                            - p2_context['ctx_p2_prev_probability']
                        ),
                        'ctx_future_probability_absdiff': abs(
                            p1_context['ctx_p1_future_probability']
                            - p2_context['ctx_p2_future_probability']
                        ),
                    }
                )
                rows.append(
                    {
                        'component': int(component_of[(source, target)]),
                        'source': int(source),
                        'target': int(target),
                        'features': features,
                        'native_probability': max(p1_probability, p2_probability),
                    }
                )
        return rows

    @torch.no_grad()
    def _predict(self, rows):
        by_component: dict[int, list[dict]] = defaultdict(list)
        for row in rows:
            by_component[int(row['component'])].append(row)
        components = list(by_component)
        for start in range(0, len(components), 8):
            batch = [
                row
                for component in components[start : start + 8]
                for row in by_component[component]
            ]
            source_group = factorize([row['source'] for row in batch])
            target_group = factorize([row['target'] for row in batch])
            component_group = factorize([row['component'] for row in batch])
            source_degree = np.bincount(source_group)[source_group].astype(np.float32)
            target_degree = np.bincount(target_group)[target_group].astype(np.float32)
            component_edges = np.bincount(component_group)[component_group].astype(np.float32)
            component_sources = np.asarray(
                [
                    len({row['source'] for row in batch if row['component'] == value})
                    for value in [row['component'] for row in batch]
                ],
                np.float32,
            )
            component_targets = np.asarray(
                [
                    len({row['target'] for row in batch if row['component'] == value})
                    for value in [row['component'] for row in batch]
                ],
                np.float32,
            )
            structural = np.log1p(
                np.stack(
                    [
                        source_degree,
                        target_degree,
                        component_edges,
                        component_sources,
                        component_targets,
                        source_degree * target_degree,
                    ],
                    1,
                )
            )
            tabular = np.asarray(
                [[row['features'][name] for name in self.features] for row in batch], np.float32
            )
            tabular = (tabular - self.mean) / self.std
            zeros = torch.zeros((len(batch), 24), dtype=torch.float32)
            scores = (
                self.model(
                    torch.from_numpy(tabular),
                    zeros,
                    zeros,
                    torch.from_numpy(structural),
                    torch.from_numpy(source_group),
                    torch.from_numpy(target_group),
                )
                .cpu()
                .numpy()
            )
            for row, score in zip(batch, scores):
                row['score'] = float(score)
        return rows

    def apply(self, raw_nodes, raw_edges, final_nodes, final_edges, p1_path, p2_path):
        counters = Counter()
        rows = self.candidate_rows(raw_nodes, raw_edges, p1_path, p2_path)
        mapping = raw_to_final_map(raw_nodes, final_nodes)
        mapped = []
        seen = set()
        current_edges = {(int(edge['source_id']), int(edge['target_id'])) for edge in final_edges}
        for row in rows:
            source = mapping.get(int(row['source']))
            target = mapping.get(int(row['target']))
            if source is None or target is None:
                continue
            key = (int(row['component']), int(source), int(target))
            if key in seen:
                continue
            seen.add(key)
            result = dict(row)
            result['source'] = int(source)
            result['target'] = int(target)
            result['features'] = dict(row['features'])
            result['features']['is_raw_parent'] = float((int(source), int(target)) in current_edges)
            mapped.append(result)
        counters['raw_rows'] = len(rows)
        counters['mapped_rows'] = len(mapped)
        if not mapped:
            return final_edges, dict(counters)
        self._predict(mapped)

        incoming: dict[int, set[int]] = defaultdict(set)
        outgoing: dict[int, set[int]] = defaultdict(set)
        edge_by_pair = {}
        for edge in final_edges:
            source, target = int(edge['source_id']), int(edge['target_id'])
            incoming[target].add(source)
            outgoing[source].add(target)
            edge_by_pair[(source, target)] = dict(edge)
        protected = protected_fork_nodes(final_edges)
        by_component: dict[int, list[dict]] = defaultdict(list)
        for row in mapped:
            by_component[int(row['component'])].append(row)

        for group in by_component.values():
            counters['components'] += 1
            current_parent = {}
            for target in {int(row['target']) for row in group}:
                parents = incoming.get(target, set())
                if len(parents) != 1:
                    continue
                parent = next(iter(parents))
                if parent in protected or target in protected or len(outgoing.get(parent, ())) > 1:
                    continue
                current_parent[target] = parent
            represented = {(int(row['source']), int(row['target'])) for row in group}
            current_parent = {
                target: source
                for target, source in current_parent.items()
                if (source, target) in represented
            }
            if not current_parent:
                counters['ineligible'] += 1
                continue
            targets_set = set(current_parent)
            work = [row for row in group if int(row['target']) in targets_set]
            sources = sorted({int(row['source']) for row in work})
            targets = sorted({int(row['target']) for row in work})
            if not sources or len(sources) < len(targets):
                counters['ineligible'] += 1
                continue
            source_index = {value: index for index, value in enumerate(sources)}
            target_index = {value: index for index, value in enumerate(targets)}
            matrix = np.full((len(targets), len(sources)), -1e6, np.float64)
            probability_by_pair = {}
            for row in work:
                source, target = int(row['source']), int(row['target'])
                i, j = target_index[target], source_index[source]
                matrix[i, j] = max(matrix[i, j], float(row['score']))
                probability_by_pair[(source, target)] = max(
                    probability_by_pair.get((source, target), 0.0),
                    float(row['native_probability']),
                )
            base = {(source, target) for target, source in current_parent.items()}
            if len(base) != len(targets):
                counters['ineligible'] += 1
                continue
            base_values = []
            valid_base = True
            for source, target in base:
                column = source_index.get(source)
                if column is None or matrix[target_index[target], column] < -1e5:
                    valid_base = False
                    break
                base_values.append(matrix[target_index[target], column])
            if not valid_base:
                counters['ineligible'] += 1
                continue
            rows_index, columns_index = linear_sum_assignment(matrix, maximize=True)
            if len(rows_index) != len(targets) or np.any(matrix[rows_index, columns_index] < -1e5):
                counters['ineligible'] += 1
                continue
            proposed = {
                (int(sources[column]), int(targets[row]))
                for row, column in zip(rows_index, columns_index)
            }
            advantage = float(
                (matrix[rows_index, columns_index].sum() - np.sum(base_values))
                / max(len(targets), 1)
            )
            counters['eligible'] += 1
            if proposed == base or advantage < self.threshold:
                continue
            counters['selected'] += 1

            involved_sources = {source for source, _target in proposed | base}
            involved_targets = {target for _source, target in proposed | base}
            if any(len(outgoing.get(source, ())) > 1 for source in involved_sources):
                counters['source_fork'] += 1
                continue
            if any(
                any(len(outgoing.get(parent, ())) > 1 for parent in incoming.get(target, ()))
                for target in involved_targets
            ):
                counters['parent_fork'] += 1
                continue
            if any(outgoing.get(source, set()) - involved_targets for source in involved_sources):
                counters['outside_target'] += 1
                continue
            remove = set()
            add = set()
            for source, target in proposed:
                for parent in incoming.get(target, ()):
                    if parent != source:
                        remove.add((parent, target))
                for child in outgoing.get(source, ()):
                    if child != target:
                        remove.add((source, child))
                if (source, target) not in edge_by_pair:
                    add.add((source, target))
            for pair in remove:
                edge_by_pair.pop(pair, None)
                outgoing[pair[0]].discard(pair[1])
                incoming[pair[1]].discard(pair[0])
            for pair in add:
                edge_by_pair[pair] = {
                    'source_id': pair[0],
                    'target_id': pair[1],
                    'edge_prob': float(probability_by_pair.get(pair, 1.0)),
                }
                outgoing[pair[0]].add(pair[1])
                incoming[pair[1]].add(pair[0])
            if any(len(incoming[target]) > 1 for target in involved_targets):
                raise RuntimeError('EdgeGRAFT transaction produced in-degree > 1')
            if any(len(outgoing[source]) > 1 for source in involved_sources):
                raise RuntimeError('EdgeGRAFT transaction produced out-degree > 1')
            counters['applied'] += 1
            counters['removed'] += len(remove)
            counters['added'] += len(add)

        result = list(edge_by_pair.values())
        if protected_fork_nodes(result) != protected:
            raise RuntimeError('EdgeGRAFT changed protected fork topology')
        counters['threshold_million'] = int(round(self.threshold * 1_000_000))
        return result, dict(counters)
