from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from biohub.modules.live_v2.transactions import (
    FIRST_MIN_AGREEMENT,
    FIRST_SOURCE_FLOOR,
    RUNNER_MAX_WINNER_MARGIN,
    RUNNER_MIN_PAIR_VOTES,
    RUNNER_MIN_SCORE,
    canonical_pair,
    edge_maps,
    lineage_depth,
    native_best_pair_nodes,
    pair_rows,
)


class LiveV2GlobalBundleRuntime:
    def __init__(self, ug12_runtime) -> None:
        self.base = ug12_runtime
        self.primary = ug12_runtime.primary
        self.v2 = ug12_runtime.v2
        self.native_features = ug12_runtime.native_features
        self.ug2 = ug12_runtime.ug2
        self.ug2_threshold = float(ug12_runtime.threshold)
        pair_names = list(self.v2.features.PAIR_FEATURES)
        source_names = list(self.v2.features.SOURCE_FEATURES)
        self.pair_index = {name: pair_names.index(name) for name in pair_names}
        self.source_index = {name: source_names.index(name) for name in source_names}
        required_pair = {
            'parent_d1',
            'parent_d2',
            'sister_dist',
            'midpoint_offset',
            'daughter_angle_cos',
            'child_sum_to_parent',
            'line_valley_ratio',
        }
        required_source = {'now_elongation', 'next_elongation', 'elong_next_minus_now'}
        if not required_pair.issubset(self.pair_index):
            raise RuntimeError('Live V2 bundle pair-feature contract mismatch')
        if not required_source.issubset(self.source_index):
            raise RuntimeError('Live V2 bundle source-feature contract mismatch')

    def serving_evidence(
        self,
        dataset_path,
        model_c_evidence_path,
        p1_evidence_path,
        p2_evidence_path,
        nodes,
        original_edges,
        registration_shifts_um,
        spacing,
    ) -> dict[str, Any]:
        scored = self.v2.score(
            dataset_path,
            nodes,
            original_edges,
            registration_shifts_um,
            spacing,
            max_pairs_per_source=self.ug2.max_pairs,
        )
        source_ids = np.asarray(scored['source_ids'], np.int64)
        source_x = np.asarray(scored['_source_x'], np.float32)
        pair_x = np.asarray(scored['_pair_x'], np.float32)
        owner = np.asarray(scored['_pair_owner'], np.int32)
        pair_nodes = np.asarray(scored['pair_nodes'], np.int64)
        component_of = {int(key): int(value) for key, value in scored['_component_of'].items()}
        if not len(source_ids) or not len(pair_nodes):
            return {
                'source_ids': source_ids,
                'component_of': component_of,
                'ug2_candidates': [],
            }

        spacing_array = np.asarray(spacing, np.float64)
        native = {}
        for label, path in (
            ('model_c', model_c_evidence_path),
            ('p1', p1_evidence_path),
            ('p2', p2_evidence_path),
        ):
            path = Path(path)
            if not path.is_file():
                raise FileNotFoundError(f'Missing {label} native evidence: {path}')
            block = self.native_features(
                path,
                nodes,
                source_ids,
                owner,
                pair_nodes[:, 0],
                pair_nodes[:, 1],
                spacing_array,
            )
            if block.shape != (len(pair_nodes), 24):
                raise RuntimeError(f'{label} native feature mismatch: {block.shape}')
            native[label] = np.asarray(block, np.float32)

        ug1_input = np.concatenate([pair_x, native['model_c'], native['p1'], native['p2']], axis=1)
        source_score, deployed_best = self.primary.head.score(
            source_x, ug1_input, owner, pair_nodes
        )
        ug2_input = np.concatenate([pair_x, native['p1'], native['p2']], axis=1)
        ug2_probability, ug2_best = self.ug2.score(source_x, ug2_input, owner, pair_nodes)

        winner_by_tube: dict[int, int] = {}
        for row in np.flatnonzero(ug2_probability >= self.ug2_threshold):
            source = int(source_ids[row])
            tube = int(component_of.get(source, source))
            previous = winner_by_tube.get(tube)
            if previous is None or ug2_probability[row] > ug2_probability[previous]:
                winner_by_tube[tube] = int(row)
        ug2_candidates = sorted(
            (
                float(ug2_probability[row]),
                int(source_ids[row]),
                int(ug2_best[row, 0]),
                int(ug2_best[row, 1]),
                int(component_of.get(int(source_ids[row]), int(source_ids[row]))),
            )
            for row in winner_by_tube.values()
            if int(ug2_best[row, 0]) >= 0 and int(ug2_best[row, 1]) >= 0
        )
        ug2_candidates.reverse()

        v2_best_index = np.asarray(scored['best_pair'], np.int64)
        v2_best = np.full((len(source_ids), 2), -1, np.int64)
        valid_v2 = v2_best_index >= 0
        v2_best[valid_v2] = pair_nodes[v2_best_index[valid_v2]]
        native_best = {
            label: native_best_pair_nodes(owner, pair_nodes, block, len(source_ids))
            for label, block in native.items()
        }
        return {
            'source_ids': source_ids,
            'source_x': source_x,
            'pair_x': pair_x,
            'owner': owner,
            'pair_nodes': pair_nodes,
            'pair_rows': pair_rows(owner, len(source_ids)),
            'component_of': component_of,
            'source_score': np.asarray(source_score, np.float32),
            'deployed_best': deployed_best,
            'v2_best': v2_best,
            'v2_best_index': v2_best_index,
            'v2_raw': np.asarray(scored['source_score_raw'], np.float32),
            'model_c_x': native['model_c'],
            'model_c_best': native_best['model_c'],
            'p1_best': native_best['p1'],
            'p2_best': native_best['p2'],
            'ug2_candidates': ug2_candidates,
        }

    def _row_for_pair(self, evidence: dict[str, Any], source_row: int, pair) -> int:
        wanted = canonical_pair(pair)
        for row in evidence['pair_rows'][source_row]:
            if canonical_pair(evidence['pair_nodes'][int(row)]) == wanted:
                return int(row)
        return -1

    def candidate_bank(
        self,
        dataset_path,
        nodes,
        original_edges,
        evidence: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        stem = Path(dataset_path).stem
        if not stem.startswith('6bba'):
            return [], []
        outgoing, incoming, probability = edge_maps(original_edges)
        first: list[dict[str, Any]] = []
        below: list[dict[str, Any]] = []
        source_ids = evidence['source_ids']
        for source_row, source_value in enumerate(source_ids):
            source = int(source_value)
            children = outgoing.get(source, [])
            if len(children) != 1:
                continue
            current_child = int(children[0])
            a, b = canonical_pair(evidence['v2_best'][source_row])
            if a < 0 or b < 0 or a == b or current_child not in (a, b):
                continue
            pair_row = self._row_for_pair(evidence, source_row, (a, b))
            if pair_row < 0:
                continue

            expert_pairs = {
                'deployed': canonical_pair(evidence['deployed_best'][source_row]),
                'model_c': canonical_pair(evidence['model_c_best'][source_row]),
                'p1': canonical_pair(evidence['p1_best'][source_row]),
                'p2': canonical_pair(evidence['p2_best'][source_row]),
            }
            flags = {key: int(value == (a, b)) for key, value in expert_pairs.items()}
            native_votes = flags['model_c'] + flags['p1'] + flags['p2']
            all_votes = flags['deployed'] + native_votes
            px = evidence['pair_x'][pair_row]
            sx = evidence['source_x'][source_row]
            c_x = evidence['model_c_x'][pair_row]
            parent_a = float(px[self.pair_index['parent_d1']])
            parent_b = float(px[self.pair_index['parent_d2']])
            parent_min, parent_max = sorted((parent_a, parent_b))
            sister = float(px[self.pair_index['sister_dist']])
            midpoint = float(px[self.pair_index['midpoint_offset']])
            cosine = float(px[self.pair_index['daughter_angle_cos']])
            child_sum = float(px[self.pair_index['child_sum_to_parent']])
            line_valley = float(px[self.pair_index['line_valley_ratio']])
            now_elong = float(sx[self.source_index['now_elongation']])
            next_elong = float(sx[self.source_index['next_elongation']])
            elong_delta = float(sx[self.source_index['elong_next_minus_now']])
            score = float(evidence['source_score'][source_row])
            tube = int(evidence['component_of'].get(source, source))
            current_prob = float(probability.get((source, current_child), 0.0))
            second = b if current_child == a else a
            rivals = [value for value in incoming.get(second, []) if value != source]
            rival_values = [probability.get((value, second), 0.0) for value in rivals]
            rival_prob = max(rival_values) if rival_values else float('nan')
            source_depth = lineage_depth(incoming, source)
            daughter_depth = min(lineage_depth(outgoing, a), lineage_depth(outgoing, b))

            evidence_count = float(c_x[17])
            p50_count = float(c_x[19])
            agreement = sum(flags.values())
            common = {
                'source': source,
                'a': a,
                'b': b,
                'current_child': current_child,
                'source_tube': tube,
                'source_score': score,
                'parent_a': parent_a,
                'parent_b': parent_b,
                'agreement': agreement,
                'midpoint': midpoint,
                'model_c_max': float(c_x[21]),
            }
            if (
                agreement >= FIRST_MIN_AGREEMENT
                and score >= FIRST_SOURCE_FLOOR
                and cosine <= 0.0
                and p50_count >= 3.0
                and p50_count / max(evidence_count, 1.0) >= 0.30
                and now_elong >= 1.85
                and next_elong <= 2.25
            ):
                first.append({**common, 'family': 'opposition_morphology'})

            v2_anchor = (
                0.002 <= score <= 0.025
                and float(evidence['v2_raw'][source_row]) >= 0.50
                and parent_min <= 2.5
                and 6.5 <= parent_max <= 10.0
                and 7.0 <= sister <= 11.0
                and cosine <= -0.60
                and current_prob >= 0.65
                and source_depth >= 5
                and daughter_depth >= 5
            )
            v2_only = (
                0.03 <= score <= 0.15
                and flags['deployed'] == 1
                and native_votes == 0
                and parent_min <= 4.0
                and 9.5 <= parent_max <= 13.5
                and 10.5 <= sister <= 14.0
                and cosine <= -0.70
                and current_prob >= 0.65
                and (not np.isfinite(rival_prob) or rival_prob <= 0.30)
                and source_depth >= 5
                and daughter_depth >= 5
            )
            p2_contested = (
                0.0005 <= score <= 0.005
                and flags['p2'] == 1
                and flags['deployed'] + flags['model_c'] + flags['p1'] == 0
                and parent_min <= 3.5
                and 7.5 <= parent_max <= 11.0
                and 8.0 <= sister <= 12.0
                and cosine <= -0.40
                and current_prob >= 0.65
                and np.isfinite(rival_prob)
                and rival_prob >= 0.80
                and source_depth >= 5
                and daughter_depth >= 3
            )
            unanimous = (
                0.05 <= score <= 0.15
                and all_votes == 4
                and parent_min <= 2.0
                and 8.0 <= parent_max <= 12.0
                and 9.0 <= sister <= 13.5
                and cosine <= -0.40
                and current_prob >= 0.75
                and source_depth >= 4
                and daughter_depth >= 10
            )
            morphology = (
                next_elong <= 2.35
                and elong_delta <= 0.0
                and child_sum >= 1.80
                and line_valley >= 0.50
            )
            passed = []
            if morphology and v2_anchor:
                passed.append('v2_anchor')
            if morphology and v2_only and now_elong >= 2.30 and elong_delta <= -0.50:
                passed.append('v2_only_long')
            if morphology and p2_contested and c_x[21] >= 0.99 and c_x[2] == 1.0 and c_x[3] == 1.0:
                passed.append('p2_contested')
            if morphology and unanimous:
                passed.append('unanimous_low_count')
            if passed:
                below.append({**common, 'family': '+'.join(passed)})

        first.sort(key=lambda row: (-row['source_score'], -row['agreement'], row['midpoint']))
        below.sort(key=lambda row: (-row['source_score'], -row['model_c_max']))
        return self._one_per_tube(first), self._one_per_tube(below)

    @staticmethod
    def _one_per_tube(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result = []
        used = set()
        for row in rows:
            tube = int(row['source_tube'])
            if tube in used:
                continue
            used.add(tube)
            result.append(row)
        return result

    def apply_transactions(
        self,
        nodes,
        edges: list[dict[str, Any]],
        rows: list[dict[str, Any]],
        prefix: str,
        stats,
    ) -> list[dict[str, Any]]:
        work = [dict(edge) for edge in edges]
        outgoing, incoming, _ = edge_maps(work)
        initial_forks = {source for source, values in outgoing.items() if len(values) >= 2}
        protected_targets = {
            target for source in initial_forks for target in outgoing.get(source, [])
        }
        used_sources: set[int] = set()
        used_targets: set[int] = set()
        counters: Counter = Counter()
        counters['proposed'] = len(rows)
        for row in rows:
            source, a, b = int(row['source']), int(row['a']), int(row['b'])
            pair = {a, b}
            if source in used_sources or pair & used_targets:
                counters['conflict'] += 1
                continue
            if source not in nodes or a not in nodes or b not in nodes:
                counters['missing_node'] += 1
                continue
            current = outgoing.get(source, [])
            if source in initial_forks or len(current) != 1:
                counters['source_not_single'] += 1
                continue
            if set(current) != {int(row['current_child'])} or int(row['current_child']) not in pair:
                counters['substrate_mismatch'] += 1
                continue
            source_t = int(nodes[source]['t'])
            if int(nodes[a]['t']) != source_t + 1 or int(nodes[b]['t']) != source_t + 1:
                counters['nonadjacent'] += 1
                continue
            if pair & protected_targets:
                counters['protected_fork_target'] += 1
                continue
            displaced: list[tuple[int, int]] = []
            valid = True
            for child in pair:
                for parent in incoming.get(child, []):
                    if parent == source:
                        continue
                    if parent in initial_forks or len(outgoing.get(parent, [])) >= 2:
                        counters['protected_owner'] += 1
                        valid = False
                        break
                    displaced.append((parent, child))
                if not valid:
                    break
            if not valid:
                continue
            remove = {(source, child) for child in current if child not in pair} | set(displaced)
            work = [
                edge
                for edge in work
                if (int(edge['source_id']), int(edge['target_id'])) not in remove
            ]
            for parent, child in remove:
                if child in outgoing.get(parent, []):
                    outgoing[parent].remove(child)
                if parent in incoming.get(child, []):
                    incoming[child].remove(parent)
            counters['removed_edges'] += len(remove)
            present = {(int(e['source_id']), int(e['target_id'])) for e in work}
            distances = {a: float(row['parent_a']), b: float(row['parent_b'])}
            for child in (a, b):
                if (source, child) not in present:
                    work.append(
                        {
                            'source_id': source,
                            'target_id': child,
                            'edge_prob': float(row['source_score']),
                            'distance_um': distances[child],
                            'registered_distance_um': distances[child],
                            'learned_division': 1,
                            'live_v2_global_bundle': 1,
                            'live_v2_global_bundle_family': str(row['family']),
                        }
                    )
                    outgoing.setdefault(source, []).append(child)
                    incoming.setdefault(child, []).append(source)
                    counters['added_edges'] += 1
            if len(outgoing.get(source, [])) != 2:
                raise RuntimeError(f'{prefix} atomic transaction failed at source {source}')
            if len(incoming.get(a, [])) > 1 or len(incoming.get(b, [])) > 1:
                raise RuntimeError(f'{prefix} transaction produced duplicate parent')
            used_sources.add(source)
            used_targets.update(pair)
            counters['applied'] += 1
        final_out, _, _ = edge_maps(work)
        if initial_forks - {source for source, value in final_out.items() if len(value) >= 2}:
            raise RuntimeError(f'{prefix} removed a protected fork')
        for key, value in counters.items():
            stats[f'live_bundle_{prefix}_{key}'] = int(value)
        return work

    def runner_rows(self, nodes, edges, evidence: dict[str, Any]) -> list[dict[str, Any]]:
        outgoing, _, _ = edge_maps(edges)
        source_ids = evidence['source_ids']
        score = evidence['source_score']
        component = evidence['component_of']
        winners: dict[int, int] = {}
        for row in np.flatnonzero(score >= 0.40):
            source = int(source_ids[row])
            tube = int(component.get(source, source))
            previous = winners.get(tube)
            if previous is None or score[row] > score[previous]:
                winners[tube] = int(row)
        initial_forks = {source for source, values in outgoing.items() if len(values) >= 2}
        rows = []
        for row in np.flatnonzero(score >= RUNNER_MIN_SCORE):
            row = int(row)
            source = int(source_ids[row])
            tube = int(component.get(source, source))
            winner = winners.get(tube)
            if winner is None or winner == row:
                continue
            margin = float(score[winner] - score[row])
            if margin < 0.0 or margin > RUNNER_MAX_WINNER_MARGIN:
                continue
            votes = Counter(
                canonical_pair(values[row])
                for values in (
                    evidence['model_c_best'],
                    evidence['p1_best'],
                    evidence['p2_best'],
                    evidence['v2_best'],
                )
            )
            pair, count = votes.most_common(1)[0]
            if count < RUNNER_MIN_PAIR_VOTES or source in initial_forks:
                continue
            current = outgoing.get(source, [])
            if len(current) != 1 or current[0] not in pair:
                continue
            pair_row = self._row_for_pair(evidence, row, pair)
            if pair_row < 0:
                continue
            px = evidence['pair_x'][pair_row]
            rows.append(
                {
                    'source': source,
                    'a': int(pair[0]),
                    'b': int(pair[1]),
                    'current_child': int(current[0]),
                    'source_tube': tube,
                    'source_score': float(score[row]),
                    'parent_a': float(px[self.pair_index['parent_d1']]),
                    'parent_b': float(px[self.pair_index['parent_d2']]),
                    'family': 'high_confidence_tube_runner',
                    'winner_margin': margin,
                    'pair_votes': int(count),
                }
            )
        rows.sort(key=lambda value: -float(value['source_score']))
        return rows

    @staticmethod
    def validate_graph(nodes, edges) -> None:
        incoming: Counter = Counter()
        outgoing: Counter = Counter()
        for edge in edges:
            source, target = int(edge['source_id']), int(edge['target_id'])
            if source not in nodes or target not in nodes:
                raise RuntimeError('Live bundle produced a dangling edge')
            if int(nodes[target]['t']) != int(nodes[source]['t']) + 1:
                raise RuntimeError('Live bundle produced a non-adjacent edge')
            outgoing[source] += 1
            incoming[target] += 1
        if incoming and max(incoming.values()) > 1:
            raise RuntimeError('Live bundle produced in-degree > 1')
        if outgoing and max(outgoing.values()) > 2:
            raise RuntimeError('Live bundle produced out-degree > 2')

    def apply(
        self,
        dataset_path,
        model_c_evidence_path,
        p1_evidence_path,
        p2_evidence_path,
        nodes,
        edges,
        stats,
        registration_shifts_um=None,
        spacing=(1.625, 0.40625, 0.40625),
        **legacy_thresholds,
    ):
        original = [dict(edge) for edge in edges]
        try:
            evidence = self.serving_evidence(
                dataset_path,
                model_c_evidence_path,
                p1_evidence_path,
                p2_evidence_path,
                nodes,
                original,
                registration_shifts_um,
                spacing,
            )
        except Exception as error:
            stats['live_bundle_evidence_fallback'] = 1
            stats['live_bundle_evidence_error'] = f'{type(error).__name__}: {error}'
            return self.base.apply(
                dataset_path,
                model_c_evidence_path,
                p1_evidence_path,
                p2_evidence_path,
                nodes,
                original,
                stats,
                registration_shifts_um=registration_shifts_um,
                spacing=spacing,
                **legacy_thresholds,
            )

        ug1_edges = self.primary.apply(
            dataset_path,
            model_c_evidence_path,
            p1_evidence_path,
            p2_evidence_path,
            nodes,
            original,
            stats,
            registration_shifts_um=registration_shifts_um,
            spacing=spacing,
            **legacy_thresholds,
        )
        try:
            stats['unigraft2_threshold'] = self.ug2_threshold
            ug12_edges = self.base.merge(
                nodes,
                ug1_edges,
                evidence.get('ug2_candidates', []),
                evidence.get('component_of', {}),
                stats,
            )
        except Exception as error:
            stats['unigraft2_merge_fallback'] = 1
            stats['unigraft2_merge_error'] = f'{type(error).__name__}: {error}'
            ug12_edges = ug1_edges

        try:
            first, below = self.candidate_bank(dataset_path, nodes, original, evidence)
            work = self.apply_transactions(nodes, ug12_edges, first, 'opposition', stats)
            work = self.apply_transactions(nodes, work, below, 'below_gate', stats)
            runner = self.runner_rows(nodes, work, evidence)
            work = self.apply_transactions(nodes, work, runner, 'tube_runner', stats)
            self.validate_graph(nodes, work)
            stats['live_v2_global_bundle_enabled'] = 1
            stats['live_v2_global_bundle_no_extra_v2_pass'] = 1
            stats['live_v2_global_bundle_total_applied'] = int(
                stats.get('live_bundle_opposition_applied', 0)
                + stats.get('live_bundle_below_gate_applied', 0)
                + stats.get('live_bundle_tube_runner_applied', 0)
            )
            return work
        except Exception as error:
            stats['live_v2_global_bundle_fallback'] = 1
            stats['live_v2_global_bundle_error'] = f'{type(error).__name__}: {error}'
            return ug12_edges
