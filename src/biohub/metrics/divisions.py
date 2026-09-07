import warnings
from collections import deque
from typing import Any, NamedTuple

import polars as pl
import tracksdata as td
from scipy.sparse import SparseEfficiencyWarning
from tracksdata.metrics import DistanceMatching
from tracksdata.options import get_options, set_options


class DivisionCounts(NamedTuple):
    tp: int
    fn: int
    fp: int


def _reset_matching_attrs(graph: td.graph.BaseGraph) -> None:
    node_keys = graph.node_attr_keys()
    if td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID in node_keys:
        node_ids = graph.node_ids()
        if len(node_ids) > 0:
            reset: dict = {td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID: -1}
            if td.DEFAULT_ATTR_KEYS.MATCH_SCORE in node_keys:
                reset[td.DEFAULT_ATTR_KEYS.MATCH_SCORE] = 0.0
            graph.update_node_attrs(node_ids=node_ids, attrs=reset)
    if td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK in graph.edge_attr_keys():
        edge_ids = graph.edge_ids()
        if len(edge_ids) > 0:
            graph.update_edge_attrs(
                edge_ids=edge_ids,
                attrs={td.DEFAULT_ATTR_KEYS.MATCHED_EDGE_MASK: False},
            )


def extract_divisions(
    graph: td.graph.BaseGraph,
) -> dict[int, td.graph.BaseGraph]:
    divisions: dict[int, td.graph.BaseGraph] = {}
    for div_node in graph.dividing_nodes():
        parents = graph.predecessors(div_node)
        children = graph.successors(div_node)
        grandchildren = [gc for child in children for gc in graph.successors(child)]
        keep = [*parents, div_node, *children, *grandchildren]
        divisions[div_node] = graph.filter(node_ids=keep).subgraph()
    return divisions


def match_divisions(
    pred_graph: td.graph.BaseGraph,
    gt_graph: td.graph.BaseGraph,
    scale: tuple[float, ...] | None = None,
    max_distance: float = 7.0,
) -> dict[int, td.graph.BaseGraph]:
    matching = DistanceMatching(max_distance=max_distance, scale=scale)

    gt_divisions = extract_divisions(gt_graph)
    matched: dict[int, td.graph.BaseGraph] = {}

    prev_show_progress = get_options().show_progress
    set_options(show_progress=False)
    try:
        for div_node, gt_div in gt_divisions.items():
            pred_copy = pred_graph.copy()
            _reset_matching_attrs(pred_copy)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=SparseEfficiencyWarning)
                pred_copy.match(gt_div, matching=matching)
            matched[div_node] = pred_copy
    finally:
        set_options(show_progress=prev_show_progress)

    return matched


def match_full(
    pred_graph: td.graph.BaseGraph,
    gt_graph: td.graph.BaseGraph,
    scale: Any,
    max_distance: float,
) -> td.graph.BaseGraph:
    matching = DistanceMatching(max_distance=max_distance, scale=scale)

    pred_copy = pred_graph.copy()
    _reset_matching_attrs(pred_copy)

    prev_show_progress = get_options().show_progress
    set_options(show_progress=False)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=SparseEfficiencyWarning)
            pred_copy.match(gt_graph, matching=matching)
    finally:
        set_options(show_progress=prev_show_progress)

    return pred_copy


def matched_node_attrs(graph: td.graph.BaseGraph) -> pl.DataFrame:
    node_attrs = graph.node_attrs(
        attr_keys=[
            td.DEFAULT_ATTR_KEYS.NODE_ID,
            td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID,
            't',
        ],
    )
    return node_attrs.filter(
        pl.col(td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID).is_not_null()
        & (pl.col(td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID) != -1)
    )


def _has_stage_coverage(
    matched_attrs: pl.DataFrame,
    gt_div: td.graph.BaseGraph,
    divider_id: int,
) -> bool:
    if matched_attrs.is_empty():
        return False

    gt_time_counts = gt_div.node_attrs(attr_keys=['t']).group_by('t').agg(pl.len().alias('n'))
    one_node_times = set(gt_time_counts.filter(pl.col('n') == 1)['t'].to_list())
    if not one_node_times:
        return False

    children = gt_div.successors(divider_id)
    if len(children) < 2:
        return False

    def _descendants(seed: int) -> set[int]:
        out: set[int] = {seed}
        stack = [seed]
        while stack:
            for nxt in gt_div.successors(stack.pop()):
                if nxt not in out:
                    out.add(nxt)
                    stack.append(nxt)
        return out

    lineages = [_descendants(c) for c in children]

    matched_time_counts = matched_attrs.group_by('t').agg(pl.len().alias('n'))
    has_one = matched_time_counts.filter(pl.col('t').is_in(one_node_times)).height > 0
    if not has_one:
        return False

    matched_gt_ids = set(matched_attrs[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID].to_list())
    lineages_covered = sum(1 for lin in lineages if lin & matched_gt_ids)
    return lineages_covered >= 2


def _weakly_connected_components(
    graph: td.graph.BaseGraph,
    node_ids: list[int],
) -> list[tuple[set[int], set[int]]]:
    remaining = set(node_ids)
    components: list[tuple[set[int], set[int]]] = []
    while remaining:
        seed = next(iter(remaining))
        visited: set[int] = {seed}
        queue: deque[int] = deque([seed])
        component: set[int] = {seed}
        while queue:
            current = queue.popleft()
            neighbors = graph.successors(current) + graph.predecessors(current)
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    if neighbor in remaining:
                        component.add(neighbor)
        components.append((component, visited))
        remaining -= component
    return components


def _bipartite_max_matching(
    left: list[int],
    edges: dict[int, set[int]],
) -> dict[int, int]:
    match_r: dict[int, int] = {}
    match_l: dict[int, int] = {}

    def augment(u: int, seen: set[int]) -> bool:
        for v in edges.get(u, ()):
            if v in seen:
                continue
            seen.add(v)
            if v not in match_r or augment(match_r[v], seen):
                match_l[u] = v
                match_r[v] = u
                return True
        return False

    for u in left:
        augment(u, set())

    return match_l


def score_divisions(
    pred_graph: td.graph.BaseGraph,
    gt_graph: td.graph.BaseGraph,
    scale: tuple[float, ...] | None = None,
    max_distance: float = 7.0,
) -> dict[int, int]:
    matched = match_divisions(
        pred_graph,
        gt_graph,
        scale,
        max_distance,
    )
    gt_divisions = extract_divisions(gt_graph)
    pred_div_nodes = set(pred_graph.dividing_nodes())

    candidates: dict[int, set[int]] = {}
    for div_node, matched_pred in matched.items():
        matched_attrs = matched_node_attrs(matched_pred)
        node_ids = matched_attrs[td.DEFAULT_ATTR_KEYS.NODE_ID].to_list()
        components = _weakly_connected_components(matched_pred, node_ids)
        gt_div = gt_divisions[div_node]
        div_candidates: set[int] = set()
        for matched_subset, visited in components:
            comp_attrs = matched_attrs.filter(
                pl.col(td.DEFAULT_ATTR_KEYS.NODE_ID).is_in(list(matched_subset))
            )
            if _has_stage_coverage(comp_attrs, gt_div, div_node):
                div_candidates |= visited & pred_div_nodes
        candidates[div_node] = div_candidates

    pairing = _bipartite_max_matching(list(candidates), candidates)
    return {div: int(div in pairing) for div in candidates}


def count_matched_pred_divisions(
    pred_graph: td.graph.BaseGraph,
    gt_graph: td.graph.BaseGraph,
    scale: tuple[float, ...] | None = None,
    max_distance: float = 7.0,
) -> int:
    matched_pred = match_full(
        pred_graph,
        gt_graph,
        scale,
        max_distance,
    )

    node_attrs = matched_pred.node_attrs(
        attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID],
    )
    matched_nodes = node_attrs.filter(
        pl.col(td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID).is_not_null()
        & (pl.col(td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID) != -1)
    )

    count = 0
    for row in matched_nodes.iter_rows(named=True):
        pred_node = row[td.DEFAULT_ATTR_KEYS.NODE_ID]
        gt_node = row[td.DEFAULT_ATTR_KEYS.MATCHED_NODE_ID]
        if matched_pred.out_degree(pred_node) >= 2 and gt_graph.out_degree(gt_node) >= 1:
            count += 1
    return count


def evaluate_divisions(
    pred_graph: td.graph.BaseGraph,
    gt_graph: td.graph.BaseGraph,
    scale: tuple[float, ...] | None = None,
    max_distance: float = 7.0,
) -> DivisionCounts:
    scores = score_divisions(
        pred_graph,
        gt_graph,
        scale,
        max_distance,
    )
    tp = sum(scores.values())
    fn = len(scores) - tp
    matched_pred_divs = count_matched_pred_divisions(
        pred_graph,
        gt_graph,
        scale,
        max_distance,
    )
    fp = max(0, matched_pred_divs - tp)
    return DivisionCounts(tp=tp, fn=fn, fp=fp)
