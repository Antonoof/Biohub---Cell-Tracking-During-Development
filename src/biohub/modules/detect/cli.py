import argparse
import json
from pathlib import Path

from biohub.modules.detect.config import PredictConfig
from biohub.modules.detect.predict import predict_movies


def predict_from_job(job: dict) -> Path:
    environment = job.get('environment') or {}
    cfg = PredictConfig(
        det_threshold=float(job['det_threshold']),
        det_tta=bool(job.get('det_tta', True)),
        edge_activation=str(job.get('edge_activation', 'softmax')),
        subvoxel_refinement=bool(job.get('subvoxel_refinement', True)),
        amp_fp16=bool(job.get('amp_fp16', False)),
        pool_kernel_um=float(job.get('pool_kernel_um', 3.0)),
        use_ilp=bool(job['use_ilp']),
        ilp_edge_weight=float(job['ilp_edge_weight']),
        ilp_appearance_weight=float(job['ilp_appearance_weight']),
        ilp_disappearance_weight=float(job['ilp_disappearance_weight']),
        ilp_division_weight=float(job['ilp_division_weight']),
        working_dir=Path(job['working_dir']),
        gpu_shard=str(job.get('gpu_shard', 'single')),
        show_progress=bool(job.get('show_progress', True)),
        uncompressed_evidence=environment.get('BIOHUB_UNCOMPRESSED_EVIDENCE', '0') == '1'
        or bool(job.get('uncompressed_evidence', False)),
        uncompressed_evidence_max_bytes=int(
            job.get(
                'uncompressed_evidence_max_bytes',
                environment.get('BIOHUB_UNCOMPRESSED_EVIDENCE_MAX_BYTES', 0) or 0,
            )
        ),
        dual_seed_min_candidate_retention=float(
            job.get(
                'dual_seed_min_candidate_retention',
                environment.get('BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION', 0.9),
            )
        ),
        bidirectional_edge_weight=float(
            job.get(
                'bidirectional_edge_weight',
                environment.get('BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT', 0),
            )
        ),
        retention_guard_secondary_edge_weight=float(
            job.get(
                'retention_guard_secondary_edge_weight',
                environment.get(
                    'BIOHUB_RETENTION_GUARD_SECONDARY_EDGE_WEIGHT',
                    job.get('secondary_edge_weight', 0.15),
                ),
            )
        ),
        cudnn_benchmark=bool(job.get('cudnn_benchmark', False))
        or environment.get('BIOHUB_CUDNN_BENCHMARK', '0') == '1',
    )
    return predict_movies(
        movie_ids=list(job['movie_ids']),
        data_dir=Path(job['data_dir']),
        output_dir=Path(job['output_dir']),
        p1_weights=Path(job['p1_weights']),
        p2_weights=Path(job['p2_weights']) if job.get('p2_weights') else None,
        model_c_weights=Path(job['model_c_weights']) if job.get('model_c_weights') else None,
        p1_evidence_dir=Path(job['p1_evidence_dir']),
        p2_evidence_dir=Path(job['p2_evidence_dir']),
        model_c_evidence_dir=Path(job['model_c_evidence_dir']),
        cfg=cfg,
        unet_batch_size=int(job.get('unet_batch_size', 4)),
        method=str(job.get('method', 'unet_transformer')),
        helper_deadline_epoch=float(job.get('helper_deadline_epoch', 'inf')),
        secondary_edge_weight=float(job.get('secondary_edge_weight', 0.15)),
        secondary_detection_weight=float(job.get('secondary_detection_weight', 0.475)),
        secondary_link_mode=str(job.get('secondary_link_mode', 'low_margin_consensus')),
        secondary_mix_temperature=float(job.get('secondary_mix_temperature', 1.0)),
        secondary_low_margin_max=float(job.get('secondary_low_margin_max', 0.35)),
        division_det_threshold=float(job.get('division_det_threshold', 0.99)),
        division_pool_um=float(job.get('division_pool_um', 3.0)),
        native_evidence_det_threshold=float(job.get('native_evidence_det_threshold', 0.96875)),
        native_evidence_pool_um=float(job.get('native_evidence_pool_um', 5.0)),
        division_radius_um=float(job.get('division_radius_um', 20.0)),
        division_topk=int(job.get('division_topk', 16)),
        division_min_probability=float(job.get('division_min_probability', 0.01)),
        native_evidence_radius_um=float(job.get('native_evidence_radius_um', 20.0)),
        native_evidence_topk=int(job.get('native_evidence_topk', 16)),
        native_evidence_min_probability=float(job.get('native_evidence_min_probability', 0.01)),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Biohub detector inference')
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args(argv)
    predict_from_job(json.loads(args.job.read_text()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
