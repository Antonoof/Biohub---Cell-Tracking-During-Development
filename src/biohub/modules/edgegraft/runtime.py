import json
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np

import biohub.modules.edgegraft.component as base
import biohub.modules.edgegraft.ranker as shared

_SENTINEL_BASE = 10_000_000_000


def read_evidence_with_context_sentinels(path: str | Path, raw_nodes):
    with np.load(path, allow_pickle=False) as data:
        source = data['source_id'].astype(np.int64, copy=False)
        target = data['target_id'].astype(np.int64, copy=False)
        probability = data['probability'].astype(np.float32, copy=False)
        alternative = data['alternative_parent_probability'].astype(np.float32, copy=False)
        winner = data['is_target_winner'].astype(np.float32, copy=False)
        rank = data['source_target_rank'].astype(np.int16, copy=False)
        distance = data['distance_um'].astype(np.float32, copy=False)
        fused_mapping = data['mapped_ab_node'].astype(np.int64, copy=False)
        fused_coords = data['ab_node_coords'].astype(np.float32, copy=False)
        raw_by_identity = {
            base.node_identity(row): int(node_id) for node_id, row in raw_nodes.items()
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
                    _SENTINEL_BASE + index,
                )
                for index, coord in enumerate(fused_coords)
            ],
            np.int64,
        )
        mapped = np.full(len(fused_mapping), -1, np.int64)
        usable = (fused_mapping >= 0) & (fused_mapping < len(fused_graph_ids))
        mapped[usable] = fused_graph_ids[fused_mapping[usable]]
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
    for row in rows:
        source_id, target_id = row[0], row[1]
        value = row[2:]
        old = deduped.get((source_id, target_id))
        if old is None or value[0] > old[0]:
            deduped[(source_id, target_id)] = value
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


class EdgeGraftV3Runtime:
    def __init__(self, ranker_dir: str | Path, metric_dir: str | Path):
        self.ranker_dir = Path(ranker_dir)
        self.metric_dir = Path(metric_dir)
        self.ranker_report = json.loads((self.ranker_dir / 'report.json').read_text())
        self.metric_report = json.loads((self.metric_dir / 'report.json').read_text())
        self.threshold = float(self.metric_report['frozen_median_threshold'])
        self._base_builder = base.EdgeGraftComponentRuntime.__new__(base.EdgeGraftComponentRuntime)
        self._models: dict[tuple[str, int], Any] = {}

    def _model(self, family: str, fold: int):
        key = (family, fold)
        if key not in self._models:
            directory = self.ranker_dir if family == 'ranker' else self.metric_dir
            self._models[key] = joblib.load(directory / f'fold_{fold}.joblib')
        return self._models[key]

    def candidate_frame(self, raw_nodes, raw_edges, final_nodes, final_edges, p1_path, p2_path):
        original = base.read_evidence
        base.read_evidence = cast(Any, read_evidence_with_context_sentinels)
        try:
            frame = shared.EdgeGraftV15Runtime.candidate_frame(
                cast(Any, self), raw_nodes, raw_edges, final_nodes, final_edges, p1_path, p2_path
            )
            if not frame.empty:
                current = {(int(edge['source_id']), int(edge['target_id'])) for edge in final_edges}
                frame['is_current_parent'] = np.fromiter(
                    (
                        float((int(source), int(target)) in current)
                        for source, target in frame[['source', 'target']].itertuples(index=False)
                    ),
                    np.float32,
                    len(frame),
                )
        finally:
            base.read_evidence = original
        if not frame.empty:
            mapping = base.raw_to_final_map(raw_nodes, final_nodes)
            mapped_raw_edges = {
                (int(mapping[source]), int(mapping[target]))
                for edge in raw_edges
                for source, target in [(int(edge['source_id']), int(edge['target_id']))]
                if source in mapping and target in mapping
            }
            frame['is_raw_parent'] = np.fromiter(
                (
                    float((int(source), int(target)) in mapped_raw_edges)
                    for source, target in frame[['source', 'target']].itertuples(index=False)
                ),
                np.float32,
                len(frame),
            )
            geometry = [column for column in frame.columns if column.startswith('geom_')]
            for column in geometry:
                frame[column] = frame[column].astype(np.float32)
            frame = shared.add_relative(frame)
        return frame

    def select_decisions(self, stem, frame, final_edges, protected):
        return shared.EdgeGraftV15Runtime.select_decisions(
            cast(Any, self), stem, frame, final_edges, protected
        )

    def rich_decisions(self, decision, evidence):
        return shared.EdgeGraftV15Runtime.rich_decisions(cast(Any, self), decision, evidence)

    def apply(
        self,
        stem: str,
        raw_nodes: dict[int, dict],
        raw_edges: list[dict],
        final_nodes: dict[int, dict],
        final_edges: list[dict],
        p1_path: str | Path,
        p2_path: str | Path,
    ):
        counters: Counter = Counter()
        fold = int(zlib.crc32(stem.encode('utf-8')) % 5)
        counters['fold'] = fold
        frame = self.candidate_frame(
            raw_nodes, raw_edges, final_nodes, final_edges, p1_path, p2_path
        )
        counters['candidate_rows'] = len(frame)
        if frame.empty:
            return final_edges, dict(counters)

        rank_features = list(self.ranker_report['features'])
        rank_x = (
            frame.reindex(columns=rank_features)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(np.float32)
        )
        frame['rank_score'] = self._model('ranker', fold).predict_proba(rank_x)[:, 1]
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
        rich['metric_score'] = self._model('metric', fold).predict_proba(metric_x)[:, 1]
        selected = rich[(~rich.protected) & (rich.metric_score >= self.threshold)].sort_values(
            ['metric_score', 'top_score', 'top_margin'], ascending=False
        )
        counters['selected'] = len(selected)
        chosen = []
        used_sources: set[int] = set()
        for row in selected.itertuples(index=False):
            source = int(row.source)
            if source in used_sources:
                counters['source_conflict'] += 1
                continue
            used_sources.add(source)
            chosen.append(row)
        counters['applied'] = len(chosen)
        if not chosen:
            return final_edges, dict(counters)

        incoming, outgoing = shared.edge_sets(final_edges)
        edge_by_pair = {
            (int(edge['source_id']), int(edge['target_id'])): dict(edge) for edge in final_edges
        }
        remove: set[tuple[int, int]] = set()
        add: set[tuple[int, int]] = set()
        probability = {
            (int(row.source), int(row.target)): float(row.native_probability)
            for row in frame.itertuples(index=False)
        }
        for row in chosen:
            source = int(row.source)
            target = int(row.target)
            current_source = int(row.current_source)
            source_target = int(row.source_current_target)
            if (current_source, target) not in edge_by_pair:
                raise RuntimeError('V3 stale target parent before atomic transaction')
            actual = next(iter(outgoing[source])) if len(outgoing[source]) == 1 else -1
            if len(outgoing[source]) > 1 or actual != source_target:
                raise RuntimeError('V3 stale source child before atomic transaction')
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
        final_incoming, final_outgoing = shared.edge_sets(result)
        if any(len(value) > 1 for value in final_incoming.values()):
            raise RuntimeError('V3 transaction produced in-degree > 1')
        if any(len(value) > 2 for value in final_outgoing.values()):
            raise RuntimeError('V3 transaction produced out-degree > 2')
        if base.protected_fork_nodes(result) != protected:
            raise RuntimeError('V3 changed protected fork topology')
        counters['removed_edges'] = len(remove)
        counters['added_edges'] = len(add)
        return result, dict(counters)


__all__ = ['EdgeGraftV3Runtime']
