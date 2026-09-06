#!/usr/bin/env python3
"""Export Model-C evidence on its native detections and map it onto A+B nodes.

Model C was trained and validated using its own 0.99/3um detection path.  The
first division sidecar instead forced C to score the A+B 0.99/5um detections,
which is a train/serve mismatch.  This exporter restores C's native detection
path, retains compact edge-competition evidence, and computes a conservative
one-to-one nearest mapping onto the frozen A+B proposal nodes.  The mapping
allows a later event decoder to alter only A+B graph edges; C never injects
nodes into the submission.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import zarr
from scipy.spatial import cKDTree
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--repo", type=Path, default=Path("/home/tweak/bio_track_repo"))
    p.add_argument("--data-dir", type=Path, default=Path("/home/tweak/bio/train"))
    p.add_argument(
        "--ab-proposals",
        type=Path,
        default=Path("/home/tweak/bio/ab_proposals_export/biohub_ab_proposals"),
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/home/tweak/bio/models/unet_transformer_division_balanced/"
            "split_0/best_division_pair_epoch067.pth"
        ),
    )
    p.add_argument(
        "--split",
        type=Path,
        default=Path("/home/tweak/bio/division_balanced_175_20_split.json"),
    )
    p.add_argument("--subset", choices=("train", "held", "practice", "all"), default="held")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("/home/tweak/bio/model_c_native_division_evidence_held20"),
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--predictor-script",
        type=Path,
        default=None,
        help=(
            "Optional predictor module providing the checkpoint's exact "
            "load_model() and extract_pos_features() implementations. Use "
            "this for architecture variants such as the teammate MedNeXt "
            "model; unset preserves the original repository loader."
        ),
    )
    p.add_argument(
        "--model-label",
        default="Model C",
        help="Human-readable checkpoint label used in logs and the summary.",
    )
    p.add_argument("--det-threshold", type=float, default=0.99)
    p.add_argument("--pool-kernel-um", type=float, default=3.0)
    p.add_argument(
        "--no-det-tta",
        action="store_true",
        help="Disable the four-view detection TTA used for epoch-67 validation.",
    )
    p.add_argument("--radius-um", type=float, default=20.0)
    p.add_argument("--map-max-um", type=float, default=6.0)
    p.add_argument("--topk-per-source", type=int, default=16)
    p.add_argument("--min-probability", type=float, default=0.01)
    p.add_argument("--slice", default="")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument(
        "--save-latents",
        action="store_true",
        help=(
            "Store Model-C's 32-D node embeddings.  Two arrays are written: "
            "native_forward_embedding for a node used as the source in (t,t+1), "
            "and native_backward_embedding for a node used as the target in "
            "(t-1,t).  This preserves the exact adjacent-frame context needed "
            "by an atomic division decoder without storing dense feature maps."
        ),
    )
    return p.parse_args()


def selected_names(path: Path, subset: str, expression: str, smoke: bool) -> list[str]:
    split = json.loads(path.read_text())
    if subset == "all":
        names = sorted(set(split["train"] + split["held"] + split["practice"]))
    else:
        names = list(split[subset])
    names = [str(x).removesuffix(".zarr") for x in names]
    if expression:
        parts = [int(x) if x else None for x in expression.split(":")]
        names = names[slice(*parts)]
    return names[:1] if smoke else names


def top_other_parent(probability: torch.Tensor):
    if probability.shape[0] == 1:
        best = probability[0]
        row = torch.zeros_like(best, dtype=torch.long)
        return best, row, torch.zeros_like(best)
    values, rows = torch.topk(probability, 2, dim=0, sorted=True)
    return values[0], rows[0], values[1]


def greedy_one_to_one_map(
    native_coords: np.ndarray,
    native_offsets: np.ndarray,
    ab_coords: np.ndarray,
    ab_offsets: np.ndarray,
    spacing: np.ndarray,
    max_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    mapped = np.full(len(native_coords), -1, np.int64)
    distance = np.full(len(native_coords), np.inf, np.float32)
    frames = min(len(native_offsets), len(ab_offsets)) - 1
    for t in range(frames):
        n0, n1 = int(native_offsets[t]), int(native_offsets[t + 1])
        a0, a1 = int(ab_offsets[t]), int(ab_offsets[t + 1])
        if n1 <= n0 or a1 <= a0:
            continue
        native_um = native_coords[n0:n1, 1:].astype(np.float32) * spacing
        ab_um = ab_coords[a0:a1, 1:].astype(np.float32) * spacing
        k = min(4, len(ab_um))
        query_distance, query_index = cKDTree(ab_um).query(native_um, k=k)
        if k == 1:
            query_distance = query_distance[:, None]
            query_index = query_index[:, None]
        options = []
        for native_row in range(len(native_um)):
            for rank in range(k):
                d = float(query_distance[native_row, rank])
                if d <= max_um:
                    options.append((d, native_row, int(query_index[native_row, rank])))
        used_native: set[int] = set()
        used_ab: set[int] = set()
        for d, native_row, ab_row in sorted(options):
            if native_row in used_native or ab_row in used_ab:
                continue
            used_native.add(native_row)
            used_ab.add(ab_row)
            mapped[n0 + native_row] = a0 + ab_row
            distance[n0 + native_row] = d
    return mapped, distance


@torch.no_grad()
def export_video(stem, args, base, extract_pos_features, open_dataset, model, downsample):
    started = time.time()
    with np.load(args.ab_proposals / f"{stem}.npz", allow_pickle=False) as z:
        ab_coords = z["coords"].astype(np.int32)
        ab_offsets = z["frame_offsets"].astype(np.int64)
        ab_downsample = tuple(int(x) for x in z["downsample"])
        spacing = z["voxel_scale_um"].astype(np.float32)
    if tuple(downsample) != ab_downsample:
        raise ValueError(f"{stem}: C downsample {downsample} != A+B {ab_downsample}")

    dataset_path = args.data_dir / f"{stem}.zarr"
    if not dataset_path.exists():
        dataset_path = args.data_dir / stem
    ds = open_dataset(dataset_path, normalize=False, load_image=False, downsample=downsample)
    arr = zarr.open_group(str(ds.zarr_path), mode="r")["0"]
    q_low = float(ds.quantiles["0.001"])
    q_high = float(ds.quantiles["0.999"])
    target_shape = list(ds.image_shape[1:])
    frames = int(ds.image_shape[0])
    voxel_size = tuple(float(s) * float(d) for s, d in zip(ds.scale, downsample))
    pool_kernel = base.pool_kernel_from_um(args.pool_kernel_um, voxel_size)
    device = next(model.parameters()).device
    ds_tensor = torch.as_tensor(downsample, dtype=torch.float32, device=device)

    coords_by_t: dict[int, np.ndarray] = {}
    forward_embedding_by_t: dict[int, np.ndarray] = {}
    backward_embedding_by_t: dict[int, np.ndarray] = {}
    offsets: dict[int, tuple[int, int]] = {}
    node_count = 0
    evidence = {key: [] for key in (
        "source_id", "target_id", "probability", "alternative_parent_probability",
        "is_target_winner", "distance_um", "source_target_rank",
    )}

    for t in tqdm(
        range(frames - 1),
        desc=f"  native {args.model_label} {stem}",
        leave=False,
    ):
        image = torch.stack(
            [base._load_frame(arr, t + dt, target_shape, downsample) for dt in (0, 1)]
        )
        image = ((image - q_low) / (q_high - q_low + 1e-6)).clamp(0.0).unsqueeze(0)
        image_device = image.to(device, non_blocking=True)
        features, detection = model.encode(image_device)
        if not args.no_det_tta:
            for dims in [(-1,), (-2,), (-2, -1)]:
                _flipped_features, flipped_detection = model.encode(image_device.flip(dims))
                for local_frame in range(2):
                    detection[local_frame] = (
                        detection[local_frame]
                        + flipped_detection[local_frame].flip(dims)
                    )
            detection = [value / 4.0 for value in detection]
        for local_frame, absolute_frame in enumerate((t, t + 1)):
            if absolute_frame in coords_by_t:
                continue
            coord = base._detect_cells_pooled(
                detection[local_frame][0],
                absolute_frame,
                args.det_threshold,
                pool_kernel,
            ).astype(np.int32)
            coord[:, 1:] *= np.asarray(downsample, np.int32)
            coords_by_t[absolute_frame] = coord
            offsets[absolute_frame] = (node_count, node_count + len(coord))
            node_count += len(coord)

        source_raw_all = coords_by_t[t]
        target_raw_all = coords_by_t[t + 1]
        if not len(source_raw_all) or not len(target_raw_all):
            continue
        source_raw = source_raw_all[:, 1:]
        target_raw = target_raw_all[:, 1:]
        source_ds = (
            source_raw.astype(np.float32) / np.asarray(downsample, np.float32)
        ).astype(np.float32)
        target_ds = (
            target_raw.astype(np.float32) / np.asarray(downsample, np.float32)
        ).astype(np.float32)
        n_source, n_target = len(source_ds), len(target_ds)
        src = torch.from_numpy(source_ds).unsqueeze(0).to(device)
        tgt = torch.from_numpy(target_ds).unsqueeze(0).to(device)
        source_rel = np.column_stack([np.zeros(n_source, np.float32), source_ds])
        target_rel = np.column_stack([np.ones(n_target, np.float32), target_ds])
        window_shape = (2,) + tuple(int(x) for x in ds.image_shape[1:])
        pos_src = torch.from_numpy(
            extract_pos_features(source_rel, window_shape).astype(np.float32)
        ).unsqueeze(0).to(device)
        pos_tgt = torch.from_numpy(
            extract_pos_features(target_rel, window_shape).astype(np.float32)
        ).unsqueeze(0).to(device)
        mask_src = torch.ones(1, n_source, dtype=torch.bool, device=device)
        mask_tgt = torch.ones(1, n_target, dtype=torch.bool, device=device)
        feat_src = model._index_features(features[:, 0], src, mask_src)
        feat_tgt = model._index_features(features[:, 1], tgt, mask_tgt)
        if args.save_latents:
            # Each transition gets the two representations produced together
            # by Model C on the real adjacent-frame window.  A frame therefore
            # has a forward/source view and a backward/target view rather than
            # an averaged lifetime embedding that would blur mitotic timing.
            forward_embedding_by_t[t] = (
                feat_src[0].float().cpu().to(torch.float16).numpy()
            )
            backward_embedding_by_t[t + 1] = (
                feat_tgt[0].float().cpu().to(torch.float16).numpy()
            )
        logits = model.predict_edges(
            feat_src, feat_tgt, src * ds_tensor, tgt * ds_tensor,
            pos_src, pos_tgt, mask_src, mask_tgt,
        )[0]
        probability = torch.softmax(logits, dim=0).float()
        src_um = torch.as_tensor(source_raw * spacing, device=device).float()
        tgt_um = torch.as_tensor(target_raw * spacing, device=device).float()
        distance = torch.cdist(src_um, tgt_um)
        eligible = distance <= args.radius_um
        masked = probability.masked_fill(~eligible, -1.0)
        k = min(args.topk_per_source, n_target)
        values, targets = torch.topk(masked, k, dim=1, sorted=True)
        keep = values >= args.min_probability
        best_prob, best_row, second_prob = top_other_parent(probability)
        source_rows = torch.arange(n_source, device=device).unsqueeze(1).expand_as(targets)
        picked_source = source_rows[keep]
        picked_target = targets[keep]
        target_best = best_row[picked_target]
        alt = torch.where(
            target_best == picked_source,
            second_prob[picked_target],
            best_prob[picked_target],
        )
        s0 = offsets[t][0]
        s1 = offsets[t + 1][0]
        evidence["source_id"].append((picked_source + s0).cpu().numpy().astype(np.int64))
        evidence["target_id"].append((picked_target + s1).cpu().numpy().astype(np.int64))
        evidence["probability"].append(values[keep].cpu().numpy().astype(np.float32))
        evidence["alternative_parent_probability"].append(
            alt.cpu().numpy().astype(np.float32)
        )
        evidence["is_target_winner"].append(
            (target_best == picked_source).cpu().numpy().astype(np.uint8)
        )
        evidence["distance_um"].append(
            distance[picked_source, picked_target].cpu().numpy().astype(np.float32)
        )
        ranks = torch.arange(k, device=device).unsqueeze(0).expand_as(targets)
        evidence["source_target_rank"].append(ranks[keep].cpu().numpy().astype(np.int16))
        del image, features, detection, logits, probability, distance

    native_counts = np.asarray([len(coords_by_t[t]) for t in range(frames)], np.int32)
    native_offsets = np.concatenate([[0], np.cumsum(native_counts, dtype=np.int64)])
    native_coords = np.concatenate([coords_by_t[t] for t in range(frames)]).astype(np.int16)
    latent_arrays = {}
    if args.save_latents:
        latent_dim = int(model.unet_out_channels)
        zero = lambda count: np.zeros((count, latent_dim), dtype=np.float16)
        latent_arrays = {
            "native_forward_embedding": np.concatenate([
                forward_embedding_by_t.get(t, zero(len(coords_by_t[t])))
                for t in range(frames)
            ]).astype(np.float16, copy=False),
            "native_backward_embedding": np.concatenate([
                backward_embedding_by_t.get(t, zero(len(coords_by_t[t])))
                for t in range(frames)
            ]).astype(np.float16, copy=False),
        }
    mapped_ab, map_distance = greedy_one_to_one_map(
        native_coords, native_offsets, ab_coords, ab_offsets, spacing, args.map_max_um
    )

    arrays = {}
    dtypes = {
        "source_id": np.int64, "target_id": np.int64, "probability": np.float32,
        "alternative_parent_probability": np.float32, "is_target_winner": np.uint8,
        "distance_um": np.float32, "source_target_rank": np.int16,
    }
    for key, chunks in evidence.items():
        arrays[key] = (
            np.concatenate(chunks).astype(dtypes[key], copy=False)
            if chunks else np.empty(0, dtypes[key])
        )
    output = args.output / f"{stem}.npz"
    temporary = output.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            **arrays,
            native_node_coords=native_coords,
            native_frame_offsets=native_offsets,
            mapped_ab_node=mapped_ab,
            map_distance_um=map_distance,
            ab_node_coords=ab_coords.astype(np.int16),
            ab_frame_offsets=ab_offsets,
            spacing_um=spacing,
            **latent_arrays,
        )
    temporary.replace(output)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "dataset": stem,
        "native_nodes": len(native_coords),
        "ab_nodes": len(ab_coords),
        "mapped_nodes": int((mapped_ab >= 0).sum()),
        "evidence_rows": len(arrays["source_id"]),
        "winner_rows": int(arrays["is_target_winner"].sum()),
        "latents": bool(args.save_latents),
        "seconds": time.time() - started,
        "bytes": output.stat().st_size,
    }


def main() -> None:
    args = parse_args()
    for entry in (str(args.repo / "scripts"), str(args.repo / "src")):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    if args.predictor_script is None:
        import predict_unet_transformer as base  # noqa: PLC0415
    else:
        if not args.predictor_script.exists():
            raise FileNotFoundError(args.predictor_script)
        spec = importlib.util.spec_from_file_location(
            "biohub_exact_evidence_predictor",
            args.predictor_script,
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(
                f"Unable to load predictor module: {args.predictor_script}"
            )
        base = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = base
        spec.loader.exec_module(base)
    from biohub_tracking.io import open_dataset  # noqa: PLC0415
    if hasattr(base, "extract_pos_features"):
        extract_pos_features = base.extract_pos_features
    else:
        from train_unet_transformer import extract_pos_features  # noqa: PLC0415

    names = selected_names(args.split, args.subset, args.slice, args.smoke)
    if args.output.exists() and not args.resume:
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, window, downsample = base.load_model(args.checkpoint, device)
    if window != 2:
        raise ValueError(f"Expected Model C window=2, got {window}")
    model.eval()
    print(
        f"Native {args.model_label} evidence: subset={args.subset} "
        f"videos={len(names)} "
        f"device={device} det={args.det_threshold} pool={args.pool_kernel_um}um",
        flush=True,
    )

    rows: dict[str, dict] = {}
    manifest = args.output / "manifest.csv"
    if args.resume and manifest.exists():
        with manifest.open(newline="") as handle:
            rows = {row["dataset"]: row for row in csv.DictReader(handle)}
    for index, stem in enumerate(names, 1):
        if args.resume and (args.output / f"{stem}.npz").exists():
            print(f"[{index:3d}/{len(names)}] SKIP {stem}", flush=True)
            continue
        row = export_video(
            stem, args, base, extract_pos_features, open_dataset, model, tuple(downsample)
        )
        rows[stem] = row
        fields = list(row)
        with manifest.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows[name] for name in sorted(rows))
        print(
            f"[{index:3d}/{len(names)}] {stem}: native={row['native_nodes']:,} "
            f"mapped={row['mapped_nodes']:,} rows={row['evidence_rows']:,} "
            f"time={row['seconds']/60:.1f}m",
            flush=True,
        )
    summary = {
        "version": "model-c-parity-checkpoint-native-detection-competition-evidence-v2",
        "model_label": args.model_label,
        "checkpoint": str(args.checkpoint),
        "subset": args.subset,
        "videos": len(rows),
        "det_threshold": args.det_threshold,
        "pool_kernel_um": args.pool_kernel_um,
        "det_tta": not args.no_det_tta,
        "radius_um": args.radius_um,
        "map_max_um": args.map_max_um,
        "latents": bool(args.save_latents),
        "latent_dim": int(model.unet_out_channels) if args.save_latents else 0,
        "latent_context": (
            "forward=(t,t+1) source; backward=(t-1,t) target"
            if args.save_latents else None
        ),
        "bytes": sum(int(row["bytes"]) for row in rows.values()),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
