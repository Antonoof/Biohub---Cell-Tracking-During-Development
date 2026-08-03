from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.spatial.distance import cdist

# Physical voxel size (z, y, x) in um: z is ~4x coarser than x/y, so distances
# have to be scaled before they can be compared against one isotropic radius.
SCALE_ZYX = (1.625, 0.40625, 0.40625)
MAX_MATCH_DIST_UM = 7.0
NODE_COUNT_PENALTY = 0.1
DIVISION_WEIGHT = 0.1

# Cost used for pairs beyond the matching radius: large enough that the
# assignment always prefers any admissible pair, finite so scipy still solves a
# dense rectangular problem.
_FORBIDDEN = 1e6


@dataclass
class TrackGraph:
    """A tracking graph: nodes with a timepoint and a centroid, plus edges."""

    ids: np.ndarray  # (N,) int node ids
    t: np.ndarray  # (N,) int timepoints
    pos: np.ndarray  # (N, 3) centroids in um (z, y, x)
    edges: np.ndarray  # (M, 2) int (source_id, target_id)

    index: dict[int, int] = field(init=False)
    succ: dict[int, tuple[int, ...]] = field(init=False)
    pred: dict[int, tuple[int, ...]] = field(init=False)
    by_t: dict[int, np.ndarray] = field(init=False)
    forks: set[int] = field(init=False)

    def __post_init__(self) -> None:
        self.ids = np.asarray(self.ids, dtype=np.int64)
        self.t = np.asarray(self.t, dtype=np.int64)
        self.pos = np.asarray(self.pos, dtype=np.float64).reshape(-1, 3)
        self.edges = np.asarray(self.edges, dtype=np.int64).reshape(-1, 2)

        self.index = {int(n): i for i, n in enumerate(self.ids)}

        succ: dict[int, list[int]] = {}
        pred: dict[int, list[int]] = {}
        for s, d in self.edges:
            succ.setdefault(int(s), []).append(int(d))
            pred.setdefault(int(d), []).append(int(s))
        self.succ = {k: tuple(v) for k, v in succ.items()}
        self.pred = {k: tuple(v) for k, v in pred.items()}
        # A duplicated edge is a data bug, not a division, so count distinct
        # children when deciding what is a fork.
        self.forks = {n for n, s in self.succ.items() if len(set(s)) >= 2}

        order = np.argsort(self.t, kind="stable")
        splits = np.flatnonzero(np.diff(self.t[order])) + 1
        self.by_t = {
            int(self.t[chunk[0]]): chunk
            for chunk in np.split(order, splits)
            if len(chunk)
        }

    def children(self, n: int) -> tuple[int, ...]:
        """Distinct successors, in a stable order."""
        return tuple(dict.fromkeys(self.succ.get(n, ())))

    def only_parent(self, n: int) -> int | None:
        """The sole predecessor of `n`, or None when it has zero or several."""
        parents = set(self.pred.get(n, ()))
        return parents.pop() if len(parents) == 1 else None


def build_graph(ids, t, z, y, x, edges) -> TrackGraph:
    """Build a graph from voxel-space columns, scaling centroids to um."""
    pos = np.stack(
        [np.asarray(z, dtype=np.float64) * SCALE_ZYX[0],
         np.asarray(y, dtype=np.float64) * SCALE_ZYX[1],
         np.asarray(x, dtype=np.float64) * SCALE_ZYX[2]],
        axis=1,
    )
    return TrackGraph(np.asarray(ids), np.asarray(t), pos, np.asarray(edges).reshape(-1, 2))


# --------------------------------------------------------------------------- #
# node matching
# --------------------------------------------------------------------------- #

def match_nodes(
    pred: TrackGraph,
    gt: TrackGraph,
    max_dist: float = MAX_MATCH_DIST_UM,
    gt_ids: Iterable[int] | None = None,
) -> dict[int, int]:
    """Map predicted node id -> GT node id, per timepoint.

    Each timepoint is solved as its own optimal assignment problem, so a
    predicted node pairs with at most one GT node and vice versa.  Only
    timepoints that carry GT nodes are considered.

    `gt_ids` restricts the GT side to a subset: the division metric re-matches
    predictions against each division window on its own, exactly as if that
    window were the whole ground truth.
    """
    if gt_ids is None:
        gt_by_t = gt.by_t
    else:
        grouped: dict[int, list[int]] = {}
        for n in gt_ids:
            i = gt.index[int(n)]
            grouped.setdefault(int(gt.t[i]), []).append(i)
        gt_by_t = {t: np.asarray(v, dtype=np.int64) for t, v in grouped.items()}

    matches: dict[int, int] = {}
    for t, gt_idx in gt_by_t.items():
        pred_idx = pred.by_t.get(int(t))
        if pred_idx is None or not len(gt_idx):
            continue

        cost = cdist(pred.pos[pred_idx], gt.pos[gt_idx])
        # Drop predictions that cannot match anything here, so the assignment
        # stays small even when a frame holds thousands of detections.
        keep = np.flatnonzero(cost.min(axis=1) <= max_dist)
        if not len(keep):
            continue
        pred_idx, cost = pred_idx[keep], cost[keep]

        cost = np.where(cost <= max_dist, cost, _FORBIDDEN)
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            if cost[r, c] < _FORBIDDEN:
                matches[int(pred.ids[pred_idx[r]])] = int(gt.ids[gt_idx[c]])
    return matches


# --------------------------------------------------------------------------- #
# edge Jaccard
# --------------------------------------------------------------------------- #

def _edge_metric(pred: TrackGraph, gt: TrackGraph, matches: dict[int, int]) -> dict:
    gt_edges = {(int(s), int(d)) for s, d in gt.edges}
    inverse = {g: p for p, g in matches.items()}  # injective, so this is safe

    tp = fp = 0
    covered: set[tuple[int, int]] = set()
    by_t: dict[int, list[int]] = {}  # t -> [fp, fn], for the "worst frames" list

    def blame(t: int, slot: int) -> None:
        by_t.setdefault(int(t), [0, 0])[slot] += 1

    for s, d in pred.edges:
        s, d = int(s), int(d)
        gs, gd = matches.get(s), matches.get(d)
        if gs is not None and gd is not None and (gs, gd) in gt_edges:
            tp += 1
            covered.add((gs, gd))
            continue
        # Not a TP: it is a FP only when the GT says something about it, i.e.
        # one endpoint matches a GT node that is already linked to a *different*
        # partner.  Edges into unannotated territory are ignored.
        into_taken = gd is not None and any(g != gs for g in gt.pred.get(gd, ()))
        out_taken = gs is not None and any(g != gd for g in gt.succ.get(gs, ()))
        if into_taken or out_taken:
            fp += 1
            blame(pred.t[pred.index[s]], 0)

    # Why each missed GT edge was missed: a detection that was never made is a
    # segmentation problem, a link to the wrong cell is a tracking problem, and
    # a track that simply stops is a gap-closing problem.
    missed = {"no_detection": 0, "wrong_link": 0, "broken_link": 0}
    for gs, gd in gt_edges - covered:
        ps, pd = inverse.get(gs), inverse.get(gd)
        if ps is None or pd is None:
            missed["no_detection"] += 1
        elif pred.succ.get(ps) or pred.pred.get(pd):
            # Both cells were detected and something was linked to them, just
            # not to each other (ps -> pd cannot exist or this would be a TP).
            missed["wrong_link"] += 1
        else:
            missed["broken_link"] += 1
        blame(gt.t[gt.index[gs]], 1)

    fn = len(gt_edges) - len(covered)
    denom = tp + fp + fn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "jaccard": tp / denom if denom else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "missed": missed,
        "errors_by_t": [
            {"t": t, "fp": counts[0], "fn": counts[1]} for t, counts in sorted(by_t.items())
        ],
    }


# --------------------------------------------------------------------------- #
# division Jaccard
# --------------------------------------------------------------------------- #

def _components(gt: TrackGraph) -> tuple[dict[int, int], set[int]]:
    """Weakly connected component id per GT node, plus the reliable ones.

    A component of a single isolated node says nothing about which lineage a
    prediction belongs to, so it is not counted as evidence.
    """
    parent = {int(n): int(n) for n in gt.ids}

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for s, d in gt.edges:
        ra, rb = find(int(s)), find(int(d))
        if ra != rb:
            parent[rb] = ra

    comp = {n: find(n) for n in parent}
    sizes: dict[int, int] = {}
    for c in comp.values():
        sizes[c] = sizes.get(c, 0) + 1
    return comp, {c for c, n in sizes.items() if n > 1}


def _division_windows(gt: TrackGraph):
    """Yield (parent, parent_side_ids, daughter_lineages, window_ids) per GT split.

    The window is grandparent -> dividing parent -> children -> grandchildren,
    which is what lets a predicted fork sit one timepoint either side of the GT
    split without any graph-wide reachability test.
    """
    for parent in sorted(gt.forks):
        children = gt.children(parent)
        lineages = [frozenset((c, *gt.children(c))) for c in children]
        parent_side = {parent, *gt.pred.get(parent, ())}
        window = set(parent_side)
        for lineage in lineages:
            window |= lineage
        yield parent, parent_side, lineages, window


def _max_matching(adjacency: dict[int, set[int]]) -> dict[int, int]:
    """Maximum-cardinality bipartite matching, right node -> left node.

    Hopcroft-Karp via scipy rather than a hand-rolled recursive Kuhn, so a
    densely annotated sample cannot blow the recursion limit.
    """
    lefts = sorted(adjacency)
    rights = sorted({r for members in adjacency.values() for r in members})
    if not lefts or not rights:
        return {}

    right_at = {r: i for i, r in enumerate(rights)}
    rows, cols = [], []
    for i, left in enumerate(lefts):
        for right in sorted(adjacency[left]):
            rows.append(i)
            cols.append(right_at[right])

    graph = csr_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)),
        shape=(len(lefts), len(rights)),
    )
    matched = maximum_bipartite_matching(graph, perm_type="column")
    return {rights[c]: lefts[i] for i, c in enumerate(matched) if c >= 0}


def _branch_components(
    pred: TrackGraph,
    fork: int,
    matches: dict[int, int],
    comp: dict[int, int],
    reliable: set[int],
) -> dict[int, int | None]:
    """Which GT lineage each branch of `fork` appears to live in.

    A matched direct child settles it.  Otherwise an unambiguous matched
    grandchild may stand in; grandchildren pointing at several components
    cancel each other out and the branch supplies no evidence.
    """
    out: dict[int, int | None] = {}
    for child in pred.children(fork):
        g = matches.get(child)
        if g is not None:
            # Direct-child evidence takes precedence, so a wrong grandchild
            # downstream cannot invalidate a correctly matched child.
            c = comp.get(g)
            out[child] = c if c in reliable else None
            continue
        seen = {
            comp[matches[gc]]
            for gc in pred.children(child)
            if gc in matches and comp[matches[gc]] in reliable
        }
        out[child] = seen.pop() if len(seen) == 1 else None
    return out


def _components_conflict(branches: dict[int, int | None]) -> bool:
    """True when two branches of one fork point at different GT lineages."""
    return len({c for c in branches.values() if c is not None}) >= 2


def _branches_merge(pred: TrackGraph, fork: int) -> bool:
    """True when the fork's local branches are not two independent paths."""
    children = pred.children(fork)
    if any(pred.only_parent(c) != fork for c in children):
        return True
    seen: set[int] = set()
    for child in children:
        grandchildren = set(pred.children(child))
        if grandchildren & seen:
            return True
        seen |= grandchildren
    return False


def _local_nodes(pred: TrackGraph, fork: int) -> list[int]:
    children = pred.children(fork)
    return [fork, *children, *(gc for c in children for gc in pred.children(c))]


def _fork_recovers(
    pred: TrackGraph,
    fork: int,
    local: dict[int, int],
    lineages: list[frozenset[int]],
    branches: dict[int, int | None],
) -> bool:
    """Whether `fork` is a valid local reconstruction of one GT division.

    `local` is the match against this division's window only, so "downstream of
    the fork" is decided structurally: the supporting nodes must be a direct
    child of the fork or that child's own children.
    """
    if _components_conflict(branches):
        return False

    support: dict[int, set[int]] = {}
    for child in pred.children(fork):
        if pred.only_parent(child) != fork:
            continue  # a shared child is not a distinct daughter path
        candidates = [child]
        candidates += [gc for gc in pred.children(child) if pred.only_parent(gc) == child]
        found = {
            i
            for n in candidates
            if (g := local.get(n)) is not None
            for i, lineage in enumerate(lineages)
            if g in lineage
        }
        if found:
            support[child] = found

    # Both GT daughter lineages have to land on *different* predicted branches.
    return len(_max_matching(support)) >= 2


def _division_metric(pred: TrackGraph, gt: TrackGraph, matches: dict[int, int]) -> dict:
    comp, reliable = _components(gt)

    gt_divisions = sorted(gt.forks)
    recoverable: dict[int, set[int]] = {}
    anchored: set[int] = set()
    # Why each GT division did or did not come back, so a division score of 0
    # can be read as "the tracker never forks here" vs "it forks in the wrong
    # place" without re-running the metric by hand.
    diagnosis: dict[int, dict] = {}

    for parent, parent_side, lineages, window in _division_windows(gt):
        local = match_nodes(pred, gt, gt_ids=window)
        anchors = {n for n, g in local.items() if g in parent_side}
        # A fork may sit on a matched parent-side node or one step after it.
        forks = {n for n in anchors if n in pred.forks}
        forks |= {s for a in anchors for s in pred.children(a) if s in pred.forks}
        anchored |= forks

        ok = {
            f
            for f in forks
            if _fork_recovers(
                pred, f, local, lineages,
                _branch_components(pred, f, matches, comp, reliable),
            )
        }
        if ok:
            recoverable[parent] = ok

        i = gt.index[parent]
        diagnosis[parent] = {
            "parent": int(parent),
            "t": int(gt.t[i]),
            # Back to voxels, which is what the viewer draws in.
            "z": float(gt.pos[i, 0] / SCALE_ZYX[0]),
            "y": float(gt.pos[i, 1] / SCALE_ZYX[1]),
            "x": float(gt.pos[i, 2] / SCALE_ZYX[2]),
            "n_daughters_detected": sum(
                1 for lineage in lineages if any(g in lineage for g in local.values())
            ),
            "n_candidate_forks": len(forks),
            "anchored": bool(anchors),
        }

    paired = _max_matching(recoverable)  # fork -> GT division
    tp = len(paired)
    fn = len(gt_divisions) - tp

    hit = set(paired.values())
    for parent, info in diagnosis.items():
        if parent in hit:
            info["status"], info["reason"] = "recovered", "matched a predicted fork"
        elif parent in recoverable:
            info["status"] = "missed"
            info["reason"] = "its only valid fork was paired with another division"
        elif not info["anchored"]:
            info["status"] = "missed"
            info["reason"] = "the dividing cell itself was never detected"
        elif not info["n_candidate_forks"]:
            info["status"] = "missed"
            info["reason"] = (
                "the cell was detected but the prediction never forks here"
                f" ({info['n_daughters_detected']}/2 daughters detected)"
            )
        else:
            info["status"] = "missed"
            info["reason"] = "a fork was predicted here but its two branches do not follow the GT daughters"

    false_positives: set[int] = set()
    for fork in sorted(pred.forks):
        if fork in paired:
            continue
        g = matches.get(fork)
        if g is not None and gt.succ.get(g):
            false_positives.add(fork)  # the GT annotates this lineage and does not fork here
            continue
        if fork in anchored:
            false_positives.add(fork)  # a candidate that failed topology or lost the pairing
            continue
        branches = _branch_components(pred, fork, matches, comp, reliable)
        if _components_conflict(branches):
            false_positives.add(fork)  # the two branches belong to different GT lineages
            continue
        if _branches_merge(pred, fork) and any(n in matches for n in _local_nodes(pred, fork)):
            false_positives.add(fork)
    # Anything else is a fork in unannotated territory, which the sparse GT
    # cannot judge either way, so it is ignored.

    fp = len(false_positives)
    denom = tp + fp + fn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "jaccard": tp / denom if denom else None,
        "n_gt_divisions": len(gt_divisions),
        "n_pred_forks": len(pred.forks),
        "divisions": [diagnosis[p] for p in gt_divisions],
    }


def _node_count_penalty(n_pred_nodes: int, n_true_nodes: int | None) -> float:
    """1 - a * max(0, T_pred - T_true) / T_true, the over-prediction factor.

    Only over-prediction is penalised. Detecting *fewer* nodes than the
    organisers' estimate is not a bonus: without the max() a sample that
    under-detects scores above its own edge Jaccard, which inflated the local
    score by up to 4% against the leaderboard.
    """
    if not n_true_nodes:
        return 1.0
    over = max(0, n_pred_nodes - n_true_nodes)
    return max(0.0, 1.0 - NODE_COUNT_PENALTY * over / n_true_nodes)


# --------------------------------------------------------------------------- #
# public entry points
# --------------------------------------------------------------------------- #

def evaluate(pred: TrackGraph, gt: TrackGraph, estimated_true_nodes: int | None = None) -> dict:
    """Score one sample: edge Jaccard, division Jaccard and the combined score."""
    matches = match_nodes(pred, gt)
    edge = _edge_metric(pred, gt, matches)
    division = _division_metric(pred, gt, matches)

    n_pred_nodes = int(len(pred.ids))
    penalty = _node_count_penalty(n_pred_nodes, estimated_true_nodes)
    adjusted = max(0.0, (edge["jaccard"] or 0.0) * penalty)

    n_gt_nodes = int(len(gt.ids))
    return {
        "edge": edge,
        "division": division,
        "n_pred_nodes": n_pred_nodes,
        "n_true_nodes": estimated_true_nodes,
        "n_matched_nodes": len(matches),
        "n_gt_nodes": n_gt_nodes,
        # How many annotated cells were detected at all: the ceiling every edge
        # the tracker could possibly get right sits under.
        "detection_recall": len(matches) / n_gt_nodes if n_gt_nodes else None,
        "node_penalty": penalty,
        "adjusted_edge_jaccard": adjusted,
        "score": adjusted + DIVISION_WEIGHT * (division["jaccard"] or 0.0),
        # Frames worth actually looking at, worst first.
        "worst_frames": sorted(
            edge["errors_by_t"], key=lambda e: (-(e["fp"] + e["fn"]), e["t"])
        )[:6],
    }


def aggregate(results: dict[str, dict]) -> dict:
    """Combine per-sample results the way the leaderboard does.

    The edge term is the per-sample adjusted Jaccard weight-averaged by sample
    size (TP+FP+FN); the division term is micro-averaged, i.e. one Jaccard over
    the summed counts, so a sample with no divisions cannot skew it.
    """
    weights = [r["edge"]["tp"] + r["edge"]["fp"] + r["edge"]["fn"] for r in results.values()]
    total_w = sum(weights)
    edge = (
        sum(r["adjusted_edge_jaccard"] * w for r, w in zip(results.values(), weights)) / total_w
        if total_w
        else None
    )

    div = {
        k: sum(r["division"][k] for r in results.values())
        for k in ("tp", "fp", "fn", "n_gt_divisions", "n_pred_forks")
    }
    denom = div["tp"] + div["fp"] + div["fn"]
    div["jaccard"] = div["tp"] / denom if denom else None

    # Pooled edge counts, for context only: the reported edge term stays the
    # weighted mean above, which is not the Jaccard of these sums.
    totals = {k: sum(r["edge"][k] for r in results.values()) for k in ("tp", "fp", "fn")}

    return {
        "n_samples": len(results),
        "adjusted_edge_jaccard": edge,
        "edge_totals": totals,
        "division": div,
        "score": (edge or 0.0) + DIVISION_WEIGHT * (div["jaccard"] or 0.0),
    }
