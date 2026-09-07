import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import threadpoolctl
import torch

from biohub.infer.config import TrackingConfig
from biohub.models.deepcenter import DeepCenterUNet3D
from biohub.models.motion import MOTION_FEATURES, MotionResidual
from biohub.modules.candidategraft import CandidateGraftDirectRuntime
from biohub.modules.cardinality import SourceCardinalityGraphRuntime
from biohub.modules.division_gbm import DivisionGBMRuntime
from biohub.modules.edgegraft import EdgeGraftV3Runtime
from biohub.modules.graph.pipeline import emergency_shard, filter_output_graph, process_graph
from biohub.modules.live_v2 import LiveV2GlobalBundleRuntime
from biohub.modules.model_c import CombinedModelCPrimaryRuntime
from biohub.modules.ownership import FullPopulationOwnershipRuntime
from biohub.modules.unigraft import IndependentP1P2LateEnsembleRuntime

BASE_STATS = {
    'raw_edges': 0,
    'dropped_nonconsecutive_edges': 0,
    'dropped_long_edges': 0,
    'dropped_multi_parent_edges': 0,
    'motion_relink_edges': 0,
    'motion_relink_tight_edges': 0,
    'motion_relink_relaxed_edges': 0,
    'motion_relink_frames': 0,
    'motion_relink_joint_frames': 0,
    'motion_relink_replaced_raw_edges': 0,
    'motion_relink_fallback_raw': 0,
    'motion_relink_skipped_large_frame': 0,
    'relink_cost_cap_rejected': 0,
    'relink_cost_cap_emptied_frames': 0,
    'relink_orphan_declined': 0,
    'relink_registered_frames': 0,
    'relink_registration_shift_um_sum': 0.0,
    'motion_corrector_candidates': 0,
    'motion_corrector_abs_residual_sum': 0.0,
    'gap_candidates': 0,
    'gap_pairs_selected': 0,
    'gap_reused_existing': 0,
    'gap_inserted_synthetic': 0,
    'gap_added_nodes': 0,
    'gap_added_edges': 0,
    'gap_skipped_node_cap': 0,
    'gap_density_candidates_expanded': 0,
    'gap_density_candidates_restricted': 0,
    'gap_density_selected_outside_base': 0,
    'gap_refined_synthetic': 0,
    'gap_refine_failed': 0,
    'gap_refine_rejected_shift': 0,
    'deepcenter_gap_checked': 0,
    'deepcenter_gap_accepted': 0,
    'deepcenter_gap_rejected': 0,
    'deepcenter_gap_missing': 0,
    'deepcenter_gap_bypassed_short_span': 0,
    'deepcenter_gap_bypassed_observed_node': 0,
    'safe_division_geometric_candidates': 0,
    'safe_division_mutual_nn_rejected': 0,
    'safe_division_divergence_rejected': 0,
    'safe_division_symmetry_rejected': 0,
    'deepcenter_safe_div_checked': 0,
    'deepcenter_safe_div_accepted': 0,
    'deepcenter_safe_div_rejected': 0,
    'deepcenter_safe_div_missing': 0,
    'public_wide_safe_div_pre_ug_added': 0,
    'public_wide_safe_div_post_ug_retained': 0,
    'public_wide_safe_div_post_ug_removed': 0,
    'public_wide_safe_div_final_retained': 0,
    'public_wide_safe_div_final_removed': 0,
    'safe_division_candidates': 0,
    'safe_divisions_added': 0,
    'safe_division_skipped_cap': 0,
    'pruned_isolated_nodes': 0,
    'short_track_components_removed': 0,
    'short_track_nodes_removed': 0,
    'short_track_edges_removed': 0,
    'boundary_track_rescued_components': 0,
    'boundary_track_rescued_nodes': 0,
    'linefit_smoothed_nodes': 0,
    'linefit_skipped_nodes': 0,
    'node_budget_excess': 0,
    'node_budget_dropped_nodes': 0,
    'node_budget_dropped_edges': 0,
    'node_budget_dense_applied': 0,
    'node_budget_ratio_used': 0.0,
    'node_budget_detected_nodes': 0,
    'ownership_endpoint_consensus': 0,
    'ownership_endpoint_already_present': 0,
    'ownership_endpoint_no_source': 0,
    'ownership_endpoint_inserted': 0,
    'ownership_endpoint_kept': 0,
    'ownership_endpoint_retracted': 0,
    'ownership_endpoint_sources_considered': 0,
    'ownership_endpoint_missing_evidence': 0,
}


class GraphUpgrade:
    STAGE_ORDER = [
        'graph_read',
        'edge_prefilter',
        'motion_relink',
        'single_parent',
        'gap_close',
        'division',
        'edgegraft',
        'prune',
        'component_filters',
        'candidategraft',
        'shard_write',
    ]

    def __init__(self, config: TrackingConfig) -> None:
        self.config = config
        self.division_runtime = None
        self.edgegraft_runtime = None
        self.candidategraft_runtime = None
        self.motion_corrector = None
        self.deepcenter_bundle = None
        self._deepcenter_load_attempted = False
        self._thread_limiter = None
        self._thread_budget = 0
        self._thread_retune_hook = None
        self._bind_thresholds()

    def _bind_thresholds(self) -> None:
        graph = self.config.graph
        division = self.config.division
        models = self.config.models
        speed = self.config.speed
        paths = self.config.paths
        bundle = self.config.bundle

        self.voxel_scale_um = tuple(float(v) for v in self.config.voxel_scale_um)
        self.scale = np.asarray(self.voxel_scale_um, dtype=np.float64)
        self.submission_columns = [
            'dataset',
            'row_type',
            'node_id',
            't',
            'z',
            'y',
            'x',
            'source_id',
            'target_id',
        ]
        self.csv_columns = ['id', *self.submission_columns]
        self.base_stats = BASE_STATS

        self.test_dir = paths.test_dir
        self.shard_dir = paths.shard_dir
        self.working_dir = paths.working_dir
        self.model_c_evidence_dir = paths.model_c_evidence_dir
        self.p1_evidence_dir = paths.p1_evidence_dir
        self.p2_evidence_dir = paths.p2_evidence_dir

        self.edge_max_um = float(graph.edge_max_um)
        self.enforce_next_frame = bool(graph.enforce_next_frame)
        self.single_parent_repair = bool(graph.single_parent_repair)
        self.prune_isolated = bool(graph.prune_isolated)
        self.motion_relink = bool(graph.motion_relink)
        self.relink_tight_um = float(graph.relink_tight_um)
        self.relink_relaxed_um = float(graph.relink_relaxed_um)
        self.relink_velocity_weight = float(graph.relink_velocity_weight)
        self.relink_velocity_axes = np.asarray(
            graph.relink_velocity_weight_axes or [self.relink_velocity_weight] * 3,
            dtype=float,
        )
        self.relink_frame_registration = bool(graph.relink_frame_registration)
        self.relink_registration_weight = float(graph.relink_registration_weight)
        self.relink_learned_bonus = float(graph.relink_learned_bonus)
        self.relink_max_match_cost = float(graph.relink_max_match_cost)
        self.relink_cap_includes_learned = bool(graph.relink_cost_cap_includes_learned)
        self.relink_motion_steps = int(graph.relink_motion_steps)
        self.relink_motion_weight = float(graph.relink_motion_weight)
        self.relink_corrector_one_step = bool(graph.relink_corrector_uses_one_step)
        self.relink_joint_assignment = bool(graph.relink_joint_assignment)
        self.relink_tight_bonus_um = float(graph.relink_tight_bonus_um)
        self.relink_orphan_prior = bool(graph.relink_orphan_prior)
        self.relink_orphan_base_um = float(graph.relink_orphan_base_um)
        self.relink_orphan_scale = float(graph.relink_orphan_scale)
        self.relink_orphan_floor = float(graph.relink_orphan_floor)
        self.relink_raw_weight = float(graph.relink_raw_distance_weight)
        self.relink_max_frame_nodes = int(graph.relink_max_frame_nodes)
        self.z_extent_um = float(graph.motion_feature_z_extent_um)

        self.gap_close = bool(graph.gap_close)
        self.gap_close_um = float(graph.gap_close_um)
        self.gap_density_adaptive = bool(graph.gap_density_adaptive)
        self.gap_density_reference_um = float(graph.gap_density_reference_um)
        self.gap_density_gain = float(graph.gap_density_gain)
        self.gap_density_max_step_delta_um = float(graph.gap_density_max_step_delta_um)
        self.gap_density_neighbors = int(graph.gap_density_neighbors)
        self.gap_reuse_existing = bool(graph.gap_reuse_existing)
        self.gap_reuse_um = float(graph.gap_reuse_um)
        self.gap_max_added_frac = float(graph.gap_max_added_frac)
        self.gap_max_added_abs = int(graph.gap_max_added_abs)
        self.gap_refine_synthetic = bool(graph.gap_refine_synthetic)
        self.gap_refine_win_z = int(graph.gap_refine_win_z)
        self.gap_refine_win_yx = int(graph.gap_refine_win_yx)
        self.gap_refine_max_shift_um = float(graph.gap_refine_max_shift_um)

        self.deepcenter_gap_veto = bool(graph.deepcenter_gap_veto)
        self.deepcenter_gap_threshold = float(graph.deepcenter_gap_threshold)
        self.deepcenter_gap_confirm_min_span_um = float(graph.deepcenter_gap_confirm_min_span_um)
        self.deepcenter_score_win_z = int(graph.deepcenter_score_win_z)
        self.deepcenter_score_win_yx = int(graph.deepcenter_score_win_yx)
        self.deepcenter_score_cache_max_frames = int(graph.deepcenter_score_cache_max_frames)
        self.deepcenter_device = str(graph.deepcenter_device)
        self.deepcenter_model_cfg = {
            'path': str(bundle.deepcenter_checkpoint),
            'expected_epoch': int(models.deepcenter_expected_epoch),
        }

        self.ownership_endpoint_graft = bool(graph.ownership_endpoint_graft)
        self.ownership_endpoint_agree_um = float(graph.ownership_endpoint_agree_um)
        self.ownership_endpoint_existing_um = float(graph.ownership_endpoint_existing_um)
        self.ownership_endpoint_parent_max_um = float(graph.ownership_endpoint_parent_max_um)
        self.ownership_endpoint_sister_max_um = float(graph.ownership_endpoint_sister_max_um)
        self.ownership_endpoint_min_prob = float(graph.ownership_endpoint_min_prob)

        self.safe_divisions = bool(graph.safe_divisions)
        self.safe_div_max_um = float(graph.safe_div_max_um)
        self.safe_div_sister_max_um = float(graph.safe_div_sister_max_um)
        self.safe_div_existing_child_max_um = float(graph.safe_div_existing_child_max_um)
        self.safe_div_frame_frac_cap = float(graph.safe_div_frame_frac_cap)
        self.safe_div_global_frac_cap = float(graph.safe_div_global_frac_cap)
        self.safe_div_sister_score_weight = float(graph.safe_div_sister_score_weight)
        self.safe_div_sister_symmetry_tau = float(graph.safe_div_sister_symmetry_tau)
        self.safe_div_require_mutual_nn = bool(graph.safe_div_require_mutual_nn)
        self.safe_div_require_divergence = bool(graph.safe_div_require_divergence)
        self.safe_div_diverge_um = float(graph.safe_div_diverge_um)
        self.deepcenter_safe_div_veto = bool(graph.deepcenter_safe_div_veto)
        self.deepcenter_safe_div_threshold = float(graph.deepcenter_safe_div_threshold)

        self.filter_short_tracks = bool(graph.filter_short_tracks)
        self.min_track_len = int(graph.min_track_len)
        self.keep_division_components = bool(graph.keep_division_components)
        self.boundary_track_rescue = bool(graph.boundary_track_rescue)
        self.boundary_track_min_len = int(graph.boundary_track_min_len)
        self.linefit_smooth = bool(graph.linefit_smooth)
        self.linefit_weight = float(graph.linefit_weight)
        self.linefit_window = int(graph.linefit_window)
        self.frame_cache_max = int(graph.frame_cache_max_frames)
        self.node_budget = bool(graph.node_budget)
        self.node_budget_ratio = float(graph.node_budget_ratio)
        self.node_budget_max_drop_frac = float(graph.node_budget_max_drop_frac)
        self.node_budget_dense_min_detected = int(graph.node_budget_dense_min_detected)
        self.node_budget_dense_ratio = float(graph.node_budget_dense_ratio)
        self.node_budget_dense_max_drop_frac = float(graph.node_budget_dense_max_drop_frac)

        self.combined_decoder = bool(division.combined_decoder)
        self.model_c_threshold = float(division.model_c_threshold)
        self.gbm_threshold = float(division.gbm_threshold)
        self.gbm_rescue_delta = float(division.gbm_rescue_delta)
        self.gbm_steal_delta = float(division.gbm_steal_delta)

        self.model_c_dir = bundle.model_c_dir
        self.source_cardinality_dir = bundle.source_cardinality_dir
        self.source_cardinality_enabled = bool(models.source_cardinality_enabled)
        self.unigraft_p1p2_dir = bundle.unigraft_p1p2_dir
        self.unigraft_p1p2_enabled = bool(models.unigraft_p1p2_enabled)
        self.live_v2_bundle_dir = bundle.live_v2_bundle_dir
        self.live_v2_bundle_enabled = bool(models.live_v2_global_bundle_enabled)
        self.ownership_full_dir = bundle.ownership_dir
        self.ownership_full_enabled = bool(models.ownership_full_population_enabled)
        self.edgegraft_dir = bundle.edgegraft_dir
        self.edgegraft_enabled = bool(models.edgegraft_enabled)
        self.candidategraft_dir = bundle.candidategraft_dir
        self.candidategraft_enabled = bool(models.candidategraft_enabled)
        self.motion_corrector_enabled = bool(models.motion_corrector_enabled)
        self.motion_corrector_path = bundle.motion_checkpoint
        self.motion_corrector_strength = float(models.motion_corrector_strength)

        self.fast_geff_reader = bool(speed.fast_geff_reader)
        self.fast_shard_writer = bool(speed.fast_shard_writer)
        self.skip_redundant_raw_copies = bool(speed.skip_redundant_raw_copies)
        self.stage_timing = bool(speed.stage_timing)

        self.shard_dir.mkdir(parents=True, exist_ok=True)
        if models.motion_corrector_enabled:
            self.motion_corrector = self.load_motion_corrector()

    def load_motion_corrector(self):
        if not self.motion_corrector_enabled:
            return None
        checkpoint = torch.load(self.motion_corrector_path, map_location='cpu', weights_only=False)
        if list(checkpoint['features']) != list(MOTION_FEATURES):
            raise ValueError(f'Unexpected motion-corrector features: {checkpoint["features"]}')
        model = MotionResidual(len(MOTION_FEATURES))
        model.load_state_dict(checkpoint['model'])
        model.eval()
        return {
            'model': model,
            'mean': torch.as_tensor(checkpoint['mean'], dtype=torch.float32),
            'std': torch.as_tensor(checkpoint['std'], dtype=torch.float32),
            'residual_scale': float(checkpoint['residual_scale']),
        }

    def load_deepcenter(self):
        if self._deepcenter_load_attempted:
            return self.deepcenter_bundle
        self._deepcenter_load_attempted = True
        if not (self.deepcenter_gap_veto or self.deepcenter_safe_div_veto):
            return None
        checkpoint_path = Path(str(self.deepcenter_model_cfg['path']))
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f'DeepCenter checkpoint is missing: {checkpoint_path}')
        device = torch.device(self.deepcenter_device)
        if device.type == 'cpu':
            torch.set_num_threads(1)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        expected = int(self.deepcenter_model_cfg['expected_epoch'])
        if int(checkpoint.get('epoch', -1)) != expected:
            raise RuntimeError(
                f'DeepCenter epoch mismatch: {checkpoint.get("epoch")} != {expected}'
            )
        cfg = SimpleNamespace(**checkpoint.get('config', {}))
        model = DeepCenterUNet3D(base_channels=int(getattr(cfg, 'base_channels', 24)))
        model.load_state_dict(checkpoint['model_state'])
        model.to(device).eval()
        self.deepcenter_bundle = {'model': model, 'cfg': cfg, 'device': device}
        return self.deepcenter_bundle

    def init_edgegraft_runtime(self):
        if self.edgegraft_runtime is not None or not self.edgegraft_enabled:
            return self.edgegraft_runtime
        self.edgegraft_runtime = EdgeGraftV3Runtime(
            self.edgegraft_dir / 'ranker',
            self.edgegraft_dir / 'metric',
        )
        return self.edgegraft_runtime

    def init_candidategraft_runtime(self):
        if self.candidategraft_runtime is not None or not self.candidategraft_enabled:
            return self.candidategraft_runtime
        self.candidategraft_runtime = CandidateGraftDirectRuntime(
            self.candidategraft_dir,
            self.edgegraft_dir,
        )
        return self.candidategraft_runtime

    def init_cardinality_runtime(self):
        if self.division_runtime is not None or not self.combined_decoder:
            return self.division_runtime
        gbm = DivisionGBMRuntime(self.model_c_dir)
        legacy = CombinedModelCPrimaryRuntime(
            gbm, self.model_c_dir, threshold=self.model_c_threshold
        )
        if not self.source_cardinality_enabled:
            self.division_runtime = legacy
            return self.division_runtime
        self.division_runtime = SourceCardinalityGraphRuntime(
            gbm,
            self.source_cardinality_dir,
            self.model_c_dir,
            legacy_fallback_runtime=legacy,
        )
        return self.division_runtime

    def init_unigraft_runtime(self):
        runtime = self.init_cardinality_runtime()
        if runtime is None or not self.unigraft_p1p2_enabled:
            return runtime
        if isinstance(
            runtime,
            (
                IndependentP1P2LateEnsembleRuntime,
                LiveV2GlobalBundleRuntime,
                FullPopulationOwnershipRuntime,
            ),
        ):
            return runtime
        head_path = self.unigraft_p1p2_dir / 'p1p2_only_source_cardinality_head.pt'
        if not head_path.is_file():
            raise FileNotFoundError(f'Incomplete independent UG2 dataset: head={head_path}')
        self.division_runtime = IndependentP1P2LateEnsembleRuntime(
            runtime,
            self.unigraft_p1p2_dir,
        )
        return self.division_runtime

    def init_live_v2_runtime(self):
        runtime = self.init_unigraft_runtime()
        if runtime is None or not self.live_v2_bundle_enabled:
            return runtime
        if isinstance(runtime, (LiveV2GlobalBundleRuntime, FullPopulationOwnershipRuntime)):
            return runtime
        self.division_runtime = LiveV2GlobalBundleRuntime(runtime)
        return self.division_runtime

    def init_ownership_runtime(self):
        runtime = self.init_live_v2_runtime()
        if runtime is None or not self.ownership_full_enabled:
            return runtime
        if isinstance(runtime, FullPopulationOwnershipRuntime):
            return runtime
        self.division_runtime = FullPopulationOwnershipRuntime(
            runtime,
            self.ownership_full_dir,
        )
        return self.division_runtime

    def init_division_runtime(self):
        return self.init_ownership_runtime()

    def set_thread_budget(self, threads: int) -> int:
        threads = max(1, int(threads))
        if threads == self._thread_budget:
            return threads
        try:
            self._thread_limiter = threadpoolctl.threadpool_limits(limits=threads)
        except Exception:
            self._thread_limiter = None
        try:
            torch.set_num_threads(threads)
        except Exception:
            pass
        for variable in (
            'OMP_NUM_THREADS',
            'MKL_NUM_THREADS',
            'OPENBLAS_NUM_THREADS',
            'NUMEXPR_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS',
            'BLIS_NUM_THREADS',
        ):
            os.environ[variable] = str(threads)
        self._thread_budget = threads
        return threads

    def set_thread_retune_hook(self, hook) -> None:
        self._thread_retune_hook = hook

    def retune_threads(self) -> None:
        hook = self._thread_retune_hook
        if hook is None:
            return
        try:
            hook()
        except Exception:
            pass

    def mark_stage(self, stats, name: str, started: float) -> float:
        now = time.monotonic()
        if self.stage_timing:
            stats[f't_{name}'] = stats.get(f't_{name}', 0.0) + (now - started)
        self.retune_threads()
        return now

    def stage_summary(self, stats: dict) -> str:
        parts = []
        for name in self.STAGE_ORDER:
            seconds = float(stats.get(f't_{name}', 0.0) or 0.0)
            if seconds >= 1.0:
                parts.append(f'{name}={seconds / 60.0:.1f}m')
        return ' '.join(parts) if parts else '(all stages under 1s)'

    def filter_output_graph(
        self,
        nodes_by_id,
        raw_edges,
        dataset=None,
        division_mode='combined',
        raw_node_attrs=None,
        raw_edge_attrs=None,
    ):
        return filter_output_graph(
            self,
            nodes_by_id,
            raw_edges,
            dataset=dataset,
            division_mode=division_mode,
            raw_node_attrs=raw_node_attrs,
            raw_edge_attrs=raw_edge_attrs,
        )

    def process_graph(self, geff_path: Path, division_mode: str) -> dict:
        return process_graph(self, geff_path, division_mode)

    def emergency_shard(self, geff_path: Path) -> dict:
        return emergency_shard(self, geff_path)
