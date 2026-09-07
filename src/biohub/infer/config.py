from pathlib import Path
from typing import Any

import yaml
from pydantic import Field

from biohub.config import FrozenModel, resolve_path
from biohub.paths import PROJECT_ROOT


class GraphConfig(FrozenModel):
    edge_max_um: float = 14.0
    enforce_next_frame: bool = True
    single_parent_repair: bool = True
    prune_isolated: bool = True
    motion_relink: bool = True
    relink_tight_um: float = 6.0
    relink_relaxed_um: float = 10.0
    relink_velocity_weight: float = 0.5
    relink_frame_registration: bool = False
    relink_registration_weight: float = 1.0
    relink_velocity_weight_axes: tuple[float, float, float] = (0.0, 0.45, 0.47)
    relink_learned_bonus: float = 1.0
    relink_max_match_cost: float = 0.0
    relink_cost_cap_includes_learned: bool = False
    relink_motion_steps: int = 1
    relink_motion_weight: float = 0.0
    relink_corrector_uses_one_step: bool = True
    relink_joint_assignment: bool = False
    relink_tight_bonus_um: float = 0.0
    relink_orphan_prior: bool = False
    relink_orphan_base_um: float = 7.5
    relink_orphan_scale: float = 1.0
    relink_orphan_floor: float = 0.5
    relink_raw_distance_weight: float = 0.05
    relink_max_frame_nodes: int = 4000
    motion_feature_z_extent_um: float = 102.375
    gap_close: bool = True
    gap_close_um: float = 5.8
    gap_density_adaptive: bool = True
    gap_density_reference_um: float = 6.5
    gap_density_gain: float = 0.04
    gap_density_max_step_delta_um: float = 0.125
    gap_density_neighbors: int = 3
    gap_reuse_existing: bool = True
    gap_reuse_um: float = 4.0
    gap_max_added_frac: float = 0.03
    gap_max_added_abs: int = 2000
    gap_refine_synthetic: bool = True
    gap_refine_win_z: int = 1
    gap_refine_win_yx: int = 3
    gap_refine_max_shift_um: float = 3.2
    ownership_endpoint_graft: bool = False
    ownership_endpoint_agree_um: float = 3.0
    ownership_endpoint_existing_um: float = 3.0
    ownership_endpoint_parent_max_um: float = 14.0
    ownership_endpoint_sister_max_um: float = 8.5
    ownership_endpoint_min_prob: float = 0.03
    deepcenter_gap_veto: bool = False
    deepcenter_gap_threshold: float = 0.25
    deepcenter_gap_confirm_min_span_um: float = 8.5
    deepcenter_score_win_z: int = 1
    deepcenter_score_win_yx: int = 2
    deepcenter_score_cache_max_frames: int = 8
    deepcenter_device: str = 'cpu'
    safe_divisions: bool = True
    safe_div_max_um: float = 9.0
    safe_div_sister_max_um: float = 14.0
    safe_div_existing_child_max_um: float = 10.0
    safe_div_sister_score_weight: float = 0.15
    safe_div_sister_symmetry_tau: float = 0.0
    safe_div_require_mutual_nn: bool = False
    safe_div_require_divergence: bool = False
    safe_div_diverge_um: float = 0.0
    deepcenter_safe_div_veto: bool = False
    deepcenter_safe_div_threshold: float = 0.12
    safe_div_frame_frac_cap: float = 0.0076
    safe_div_global_frac_cap: float = 0.00375
    filter_short_tracks: bool = True
    min_track_len: int = 6
    keep_division_components: bool = True
    boundary_track_rescue: bool = False
    boundary_track_min_len: int = 2
    linefit_smooth: bool = True
    linefit_weight: float = 0.8
    linefit_window: int = 3
    node_budget: bool = False
    node_budget_ratio: float = 1.0
    node_budget_max_drop_frac: float = 0.0
    node_budget_dense_min_detected: int = 0
    node_budget_dense_ratio: float = 1.0
    node_budget_dense_max_drop_frac: float = 0.0
    frame_cache_max_frames: int = 8


class DivisionConfig(FrozenModel):
    combined_decoder: bool = True
    model_c_threshold: float = 0.96
    gbm_threshold: float = 0.65
    gbm_rescue_delta: float = 0.15
    gbm_steal_delta: float = 0.25


class ModelBundleConfig(FrozenModel):
    source_cardinality_enabled: bool = True
    unigraft_p1p2_enabled: bool = True
    live_v2_global_bundle_enabled: bool = True
    ownership_full_population_enabled: bool = True
    edgegraft_enabled: bool = True
    candidategraft_enabled: bool = True
    motion_corrector_enabled: bool = True
    motion_corrector_strength: float = 1.0
    deepcenter_enabled: bool = True
    deepcenter_expected_epoch: int = 2


class DetectionConfig(FrozenModel):
    threshold: float = 0.96875
    unet_batch_size: int = 4
    subvoxel_refinement: bool = True
    det_tta: bool = True
    edge_activation: str = 'softmax'
    pool_kernel_um: float = 3.0
    native_evidence_pool_um: float = 5.0
    division_det_threshold: float = 0.99
    division_pool_um: float = 3.0
    division_radius_um: float = 20.0
    division_topk: int = 16
    division_min_probability: float = 0.01
    native_evidence_radius_um: float = 20.0
    native_evidence_topk: int = 16
    native_evidence_min_probability: float = 0.01


class AssociationConfig(FrozenModel):
    secondary_edge_weight: float = 0.15
    secondary_detection_weight: float = 0.475
    secondary_link_mode: str = 'low_margin_consensus'
    secondary_mix_temperature: float = 1.0
    secondary_low_margin_max: float = 0.35
    dual_seed_edge_threshold: float = 0.48
    minimum_candidate_retention: float = 0.9
    guarded_secondary_edge_weight: float = 0.0
    bidirectional_edge_weight: float = 0.3


class IlpConfig(FrozenModel):
    use: bool = True
    edge_weight: float = -1.0
    appearance: float = 0.0
    disappearance: float = 1.5
    division: float = 1.0


class SpeedConfig(FrozenModel):
    vectorized_candidates: bool = True
    fast_geff_reader: bool = True
    fast_shard_writer: bool = True
    skip_redundant_raw_copies: bool = True
    lazy_baseline: bool = True
    longest_first: bool = True
    stage_timing: bool = True
    uncompressed_evidence: bool = True
    uncompressed_evidence_max_mb: float = 512.0
    cudnn_benchmark: bool = True
    amp_fp16: bool = False
    thread_boost: bool = True
    thread_pool_total: int = 0
    function_profile: str = 'smallest'
    runtime_accel: bool = True


class RuntimeConfig(FrozenModel):
    gpu_workers: int = 1
    cpu_workers: int = 0
    cpu_workers_while_predicting: int = 0
    hard_limit_seconds: int = 42600
    finalize_reserve_seconds: int = 600
    min_upgrade_seconds: int = 60
    emergency_shard: bool = True
    slice: str = ''


class TrackingPathsConfig(FrozenModel):
    test_dir: Path
    shard_dir: Path
    working_dir: Path
    model_c_evidence_dir: Path
    p1_evidence_dir: Path
    p2_evidence_dir: Path
    submission: Path | None = None
    run_stats: Path | None = None


class BundlePathsConfig(FrozenModel):
    bundle_dir: Path
    p1_checkpoint: Path
    p2_checkpoint: Path
    model_c_dir: Path
    model_c_checkpoint: Path
    deepcenter_checkpoint: Path
    motion_checkpoint: Path
    source_cardinality_dir: Path
    unigraft_p1p2_dir: Path
    live_v2_bundle_dir: Path
    ownership_dir: Path
    edgegraft_dir: Path
    candidategraft_dir: Path


class TrackingConfig(FrozenModel):
    experiment_tag: str = 'biohub_baseline'
    voxel_scale_um: tuple[float, float, float]
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    association: AssociationConfig = Field(default_factory=AssociationConfig)
    ilp: IlpConfig = Field(default_factory=IlpConfig)
    graph: GraphConfig
    division: DivisionConfig
    models: ModelBundleConfig
    speed: SpeedConfig = Field(default_factory=SpeedConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    paths: TrackingPathsConfig
    bundle: BundlePathsConfig


def _read_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text())
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f'Config {path} must be a mapping, got {type(loaded)}')
    return loaded


def resolve_bundle_paths(
    bundle_dir: Path,
    overrides: dict[str, Path] | None = None,
) -> dict[str, Path]:
    root = bundle_dir.resolve()
    paths = {
        'bundle_dir': root,
        'p1_checkpoint': root
        / 'p1'
        / 'weights'
        / 'unet_transformer'
        / 'split_0'
        / 'edge_predictor_best.pth',
        'p2_checkpoint': root
        / 'p2'
        / 'weights'
        / 'unet_transformer'
        / 'split_0'
        / 'edge_predictor_best.pth',
        'model_c_dir': root / 'model_c',
        'model_c_checkpoint': root / 'model_c' / 'weights' / 'edge_predictor_best.pth',
        'deepcenter_checkpoint': root / 'deepcenter' / 'best.pt',
        'motion_checkpoint': root / 'motion_corrector' / 'motion_corrector_best.pt',
        'source_cardinality_dir': root / 'source_cardinality',
        'unigraft_p1p2_dir': root / 'unigraft_p1p2',
        'live_v2_bundle_dir': root / 'live_v2_bundle',
        'ownership_dir': root / 'ownership',
        'edgegraft_dir': root / 'edgegraft',
        'candidategraft_dir': root / 'candidategraft',
    }
    if overrides:
        for key, value in overrides.items():
            paths[key] = Path(value).resolve()
    return paths


def build_tracking_paths(
    work_dir: Path,
    test_dir: Path,
    *,
    submission: Path | None = None,
) -> dict[str, Path]:
    work_dir = work_dir.resolve()
    test_dir = test_dir.resolve()
    return {
        'test_dir': test_dir,
        'shard_dir': work_dir / 'submission_shards',
        'working_dir': work_dir,
        'model_c_evidence_dir': work_dir / 'evidence_model_c',
        'p1_evidence_dir': work_dir / 'evidence_p1',
        'p2_evidence_dir': work_dir / 'evidence_p2',
        'submission': submission or (work_dir / 'submission.csv'),
        'run_stats': work_dir / 'run_stats.csv',
    }


TRACKING_KEYS = (
    'experiment_tag',
    'voxel_scale_um',
    'detection',
    'association',
    'ilp',
    'graph',
    'division',
    'models',
    'speed',
    'runtime',
)


def load_tracking_config(
    *,
    pipeline: str = 'baseline',
    bundle_dir: Path,
    test_dir: Path,
    work_dir: Path,
    submission: Path | None = None,
    config_path: Path | None = None,
    bundle_paths: dict[str, Path] | None = None,
) -> TrackingConfig:
    if config_path is not None:
        path = Path(config_path)
    elif str(pipeline).endswith(('.yaml', '.yml')):
        path = Path(pipeline)
    else:
        path = PROJECT_ROOT / 'configs' / 'infer.yaml'
    if not path.is_file():
        raise FileNotFoundError(f'Missing tracking config: {path}')
    raw = _read_yaml(path)
    payload = {key: raw[key] for key in TRACKING_KEYS if key in raw}
    payload['voxel_scale_um'] = tuple(float(v) for v in payload['voxel_scale_um'])
    payload['paths'] = build_tracking_paths(work_dir, test_dir, submission=submission)
    payload['bundle'] = resolve_bundle_paths(resolve_path(bundle_dir), bundle_paths)
    return TrackingConfig.model_validate(payload)
