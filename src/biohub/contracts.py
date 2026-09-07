from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from biohub.constants import VOXEL_SCALE_ZYX
from biohub.data.coordinates import voxels_to_um


class EvaluationLevel(StrEnum):
    LEGACY_PARITY = 'legacy_parity'
    CONDITIONAL_HEAD_OOF = 'conditional_head_oof'
    STRICT_NESTED = 'strict_nested'


class SupervisionMode(StrEnum):
    KNOWN_ONLY = 'known_only'
    SPARSE_GT = 'sparse_gt'
    PSEUDO = 'pseudo'


@dataclass(frozen=True)
class MovieRecord:
    movie_id: str
    embryo: str
    source_group: str
    split_role: str
    zarr_path: Path | None
    geff_path: Path | None
    image_shape_tzyx: tuple[int, int, int, int] | None
    estimated_number_of_nodes: float | None
    t_max: float | None
    voxel_scale_um: tuple[float, float, float] = VOXEL_SCALE_ZYX
    quantiles: dict[str, float] = field(default_factory=dict)

    @property
    def has_image(self) -> bool:
        return self.zarr_path is not None and self.zarr_path.exists()

    @property
    def has_tracks(self) -> bool:
        return self.geff_path is not None and self.geff_path.exists()


@dataclass
class GraphState:
    movie_id: str
    node_ids: NDArray[np.int64]
    t: NDArray[np.int64]
    z: NDArray[np.float64]
    y: NDArray[np.float64]
    x: NDArray[np.float64]
    source_ids: NDArray[np.int64]
    target_ids: NDArray[np.int64]
    node_attrs: dict[str, NDArray[Any]] = field(default_factory=dict)
    edge_attrs: dict[str, NDArray[Any]] = field(default_factory=dict)
    estimated_number_of_nodes: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.node_ids = np.asarray(self.node_ids, dtype=np.int64)
        self.t = np.asarray(self.t, dtype=np.int64)
        self.z = np.asarray(self.z, dtype=np.float64)
        self.y = np.asarray(self.y, dtype=np.float64)
        self.x = np.asarray(self.x, dtype=np.float64)
        self.source_ids = np.asarray(self.source_ids, dtype=np.int64)
        self.target_ids = np.asarray(self.target_ids, dtype=np.int64)

    @property
    def n_nodes(self) -> int:
        return int(self.node_ids.size)

    @property
    def n_edges(self) -> int:
        return int(self.source_ids.size)

    def node_xyz_um(
        self, scale: tuple[float, float, float] = VOXEL_SCALE_ZYX
    ) -> NDArray[np.float64]:
        return voxels_to_um(self.z, self.y, self.x, scale)

    def copy(self) -> 'GraphState':
        return GraphState(
            movie_id=self.movie_id,
            node_ids=self.node_ids.copy(),
            t=self.t.copy(),
            z=self.z.copy(),
            y=self.y.copy(),
            x=self.x.copy(),
            source_ids=self.source_ids.copy(),
            target_ids=self.target_ids.copy(),
            node_attrs={key: value.copy() for key, value in self.node_attrs.items()},
            edge_attrs={key: value.copy() for key, value in self.edge_attrs.items()},
            estimated_number_of_nodes=self.estimated_number_of_nodes,
            extra=dict(self.extra),
        )


@dataclass(frozen=True)
class TopologyIssue:
    code: str
    message: str


@dataclass
class MetricRow:
    movie_id: str
    edge_tp: float
    edge_fp: float
    edge_fn: float
    division_tp: float
    division_fp: float
    division_fn: float
    num_pred_nodes: float
    node_recall: float
    total_node_ratio: float
    edge_jaccard: float
    adj_edge_jaccard: float
    estimated_number_of_nodes: float
    status: str = 'ok'
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            'movie_id': self.movie_id,
            'edge_tp': self.edge_tp,
            'edge_fp': self.edge_fp,
            'edge_fn': self.edge_fn,
            'division_tp': self.division_tp,
            'division_fp': self.division_fp,
            'division_fn': self.division_fn,
            'num_pred_nodes': self.num_pred_nodes,
            'node_recall': self.node_recall,
            'total_node_ratio': self.total_node_ratio,
            'edge_jaccard': self.edge_jaccard,
            'adj_edge_jaccard': self.adj_edge_jaccard,
            'estimated_number_of_nodes': self.estimated_number_of_nodes,
            'status': self.status,
            'error': self.error,
        }


@dataclass(frozen=True)
class CompletenessReport:
    expected_movies: tuple[str, ...]
    scored_movies: tuple[str, ...]
    missing_movies: tuple[str, ...]
    failed_movies: tuple[str, ...]
    missing_node_estimates: tuple[str, ...]
    complete: bool
    evaluation_level: EvaluationLevel
    require_complete: bool


@dataclass(frozen=True)
class ArtifactRef:
    kind: str
    identifier: str
    path: Path | None = None
    sha256: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
