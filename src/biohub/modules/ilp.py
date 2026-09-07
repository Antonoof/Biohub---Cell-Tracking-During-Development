import contextlib
import os
from typing import cast

import tracksdata as td


@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def apply_ilp(graph: td.graph.InMemoryGraph, cfg) -> td.graph.InMemoryGraph:
    if not cfg.use_ilp or graph.num_edges() == 0:
        return graph
    solver = td.solvers.ILPSolver(
        edge_weight=cfg.ilp_edge_weight * td.EdgeAttr('edge_prob'),
        appearance_weight=cfg.ilp_appearance_weight,
        disappearance_weight=cfg.ilp_disappearance_weight,
        division_weight=cfg.ilp_division_weight,
    )
    with suppress_output():
        return cast(td.graph.InMemoryGraph, solver.solve(graph))
