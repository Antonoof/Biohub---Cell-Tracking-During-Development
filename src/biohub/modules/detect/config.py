from dataclasses import dataclass
from pathlib import Path


@dataclass
class PredictConfig:
    det_threshold: float = 0.5
    det_tta: bool = True
    subvoxel_refinement: bool = True
    amp_fp16: bool = False
    pool_kernel_um: float = 3.0
    edge_activation: str = 'softmax'
    threshold: float = 0.5

    use_ilp: bool = False
    ilp_edge_weight: float = -1.0
    ilp_appearance_weight: float = 0.1
    ilp_disappearance_weight: float = 0.1
    ilp_division_weight: float = 1.0

    max_parents_per_node: int | None = None
    max_children_per_node: int | None = None
    working_dir: Path | None = None
    show_progress: bool = True
    gpu_shard: str = 'single'
    uncompressed_evidence: bool = False
    uncompressed_evidence_max_bytes: int = 0
    dual_seed_min_candidate_retention: float = 0.9
    bidirectional_edge_weight: float = 0.0
    retention_guard_secondary_edge_weight: float | None = None
    cudnn_benchmark: bool = False

    def __post_init__(self) -> None:
        if not self.use_ilp:
            if self.max_parents_per_node is None:
                self.max_parents_per_node = 1
            if self.max_children_per_node is None:
                self.max_children_per_node = 2
