#!/usr/bin/env python3
"""Export raw shared A+B detections for proposal-aware association training.

No edge prediction, ILP, or graph post-processing is run. Each video becomes a
compressed NPZ containing raw-voxel coordinates, fused/member detection
confidence, frame offsets, and downsample-identical transitions.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import shutil
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import zarr
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import predict_unet_transformer as base
from biohub_tracking.io import open_dataset


@dataclass
class Member:
    name: str
    path: Path
    weight: float
    device: torch.device
    model: object
    window_size: int
    downsample: tuple[int, ...]


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--models-json", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/biohub_ab_proposals"))
    p.add_argument("--det-threshold", type=float, default=0.99)
    p.add_argument("--pool-kernel-um", type=float, default=5.0)
    p.add_argument("--slice", default="", help="Python slice, for example :10 or 50:100")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--zip", action="store_true")
    return p.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_members(spec_path: Path) -> list[Member]:
    specs = json.loads(spec_path.read_text())
    members = []
    for spec in specs:
        if not spec.get("enabled", True) or float(spec.get("weight", 0)) <= 0:
            continue
        path = Path(spec["path"])
        if not path.exists():
            raise FileNotFoundError(f"{spec['name']} checkpoint missing: {path}")
        requested = str(spec.get("device", "cuda:0"))
        device = torch.device(requested if torch.cuda.is_available() else "cpu")
        model, window, downsample = base.load_model(path, device)
        members.append(Member(str(spec["name"]), path, float(spec["weight"]),
                              device, model, int(window), tuple(downsample)))
        print(f"Loaded {spec['name']} weight={spec['weight']} device={device}: {path}", flush=True)
    if not members:
        raise ValueError("No enabled detector members")
    reference = (members[0].window_size, members[0].downsample)
    for m in members[1:]:
        if (m.window_size, m.downsample) != reference:
            raise ValueError(f"Detector configs differ: {members[0].name}={reference}, "
                             f"{m.name}={(m.window_size, m.downsample)}")
    return members


def weighted_average(values, weights, device):
    total = float(sum(weights))
    out = values[0].to(device, non_blocking=True) * (weights[0] / total)
    for value, weight in zip(values[1:], weights[1:]):
        out = out + value.to(device, non_blocking=True) * (weight / total)
    return out


def sample_probs(logits: torch.Tensor, coords: np.ndarray) -> np.ndarray:
    if len(coords) == 0:
        return np.empty(0, dtype=np.float32)
    # encode() returns a batch dimension and a singleton detection channel;
    # reduce to the spatial Z,Y,X logit volume before advanced indexing.
    volume = logits
    while volume.ndim > 3:
        volume = volume[0]
    if volume.ndim != 3:
        raise ValueError(f"Unexpected detection-logit shape: {tuple(logits.shape)}")
    index = torch.as_tensor(coords[:, 1:].astype(np.int64), device=logits.device)
    probs = torch.sigmoid(volume[index[:, 0], index[:, 1], index[:, 2]])
    return probs.detach().float().cpu().numpy()


@torch.no_grad()
def export_video(members: list[Member], path: Path, output_path: Path,
                 det_threshold: float, pool_kernel_um: float) -> dict:
    started = time.time()
    primary = members[0].device
    weights = [m.weight for m in members]
    W = members[0].window_size
    downsample = members[0].downsample

    ds = open_dataset(path, normalize=False, load_image=False, downsample=downsample)
    if "0.001" not in ds.quantiles or "0.999" not in ds.quantiles:
        raise ValueError(f"Missing 0.001/0.999 quantiles: {path}")
    zarr_arr = zarr.open_group(str(ds.zarr_path), mode="r")["0"]
    T = int(ds.image_shape[0])
    target_shape = list(ds.image_shape[1:])
    q_low, q_high = float(ds.quantiles["0.001"]), float(ds.quantiles["0.999"])
    voxel_size = tuple(s * d for s, d in zip(ds.scale, downsample))
    pool_kernel = base.pool_kernel_from_um(pool_kernel_um, voxel_size)

    # One cached strided read per frame. This both avoids duplicate I/O across
    # overlapping model windows and identifies frozen transitions.
    frames = [base._load_frame(zarr_arr, t, target_shape, downsample) for t in range(T)]
    frozen_sources = np.asarray(
        [t for t in range(T - 1) if torch.equal(frames[t], frames[t + 1])],
        dtype=np.int16,
    )

    stride = max(W - 1, 1)
    starts = list(range(0, T - W + 1, stride))
    if not starts or starts[-1] + W < T:
        last = max(T - W, 0)
        if not starts or last != starts[-1]:
            starts.append(last)

    seen = set()
    coords_by_t: dict[int, np.ndarray] = {}
    fused_prob_by_t: dict[int, np.ndarray] = {}
    member_prob_by_t: dict[int, np.ndarray] = {}
    for ws in starts:
        frame_indices = list(range(ws, ws + W))
        imgs_cpu = torch.stack([frames[t] for t in frame_indices])
        imgs_cpu = ((imgs_cpu - q_low) / (q_high - q_low + 1e-6)).clamp(0.0).unsqueeze(0)
        member_dets = []
        for member in members:
            imgs = imgs_cpu.to(member.device, non_blocking=True)
            _features, det = member.model.encode(imgs)
            member_dets.append(det)
        for fi, t in enumerate(frame_indices):
            if t in seen:
                continue
            fused_logits = weighted_average(
                [member_dets[k][fi] for k in range(len(members))], weights, primary,
            )
            coords = base._detect_cells_pooled(
                fused_logits[0], t, det_threshold, pool_kernel,
            )
            coords_by_t[t] = coords
            fused_prob_by_t[t] = sample_probs(fused_logits, coords)
            member_prob_by_t[t] = np.stack([
                sample_probs(member_dets[k][fi], coords)
                for k in range(len(members))
            ], axis=1) if len(coords) else np.empty((0, len(members)), np.float32)
            seen.add(t)
        del member_dets, imgs_cpu

    if len(seen) != T:
        raise RuntimeError(f"{path.stem}: detected {len(seen)}/{T} frames")
    counts = np.asarray([len(coords_by_t[t]) for t in range(T)], dtype=np.int32)
    offsets = np.concatenate([[0], np.cumsum(counts, dtype=np.int64)])
    coords = np.concatenate([coords_by_t[t] for t in range(T)]).astype(np.int16)
    coords[:, 1:] *= np.asarray(downsample, dtype=np.int16)
    fused_prob = np.concatenate([fused_prob_by_t[t] for t in range(T)]).astype(np.float32)
    member_prob = np.concatenate([member_prob_by_t[t] for t in range(T)]).astype(np.float32)
    if not (len(coords) == len(fused_prob) == len(member_prob)):
        raise AssertionError("Proposal arrays have inconsistent lengths")

    np.savez_compressed(
        output_path,
        coords=coords,
        fused_det_prob=fused_prob,
        member_det_prob=member_prob,
        frame_counts=counts,
        frame_offsets=offsets,
        frozen_sources=frozen_sources,
        image_shape=np.asarray((T, *zarr_arr.shape[1:]), dtype=np.int32),
        voxel_scale_um=np.asarray(ds.scale, dtype=np.float32),
        downsample=np.asarray(downsample, dtype=np.int16),
        det_threshold=np.asarray(det_threshold, dtype=np.float32),
        pool_kernel_um=np.asarray(pool_kernel_um, dtype=np.float32),
    )
    del frames, coords_by_t, fused_prob_by_t, member_prob_by_t
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "dataset": path.stem,
        "frames": T,
        "nodes": len(coords),
        "mean_nodes_per_frame": float(counts.mean()),
        "min_nodes_per_frame": int(counts.min()),
        "max_nodes_per_frame": int(counts.max()),
        "frozen_transitions": len(frozen_sources),
        "seconds": time.time() - started,
        "file": output_path.name,
        "bytes": output_path.stat().st_size,
    }


def main():
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    members = load_members(args.models_json)
    stems = sorted(p.name[:-5] for p in args.data_dir.glob("*.zarr"))
    if args.slice:
        parts = [int(x) if x else None for x in args.slice.split(":")]
        stems = stems[slice(*parts)]
    if not stems:
        raise FileNotFoundError(f"No .zarr videos selected under {args.data_dir}")
    if args.output_dir.exists() and not args.resume:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    manifest_path = args.output_dir / "manifest.csv"
    for i, stem in enumerate(stems, 1):
        out = args.output_dir / f"{stem}.npz"
        if args.resume and out.exists():
            print(f"[{i:3d}/{len(stems)}] SKIP {stem}", flush=True)
            continue
        row = export_video(
            members, args.data_dir / stem, out,
            args.det_threshold, args.pool_kernel_um,
        )
        rows.append(row)
        print(f"[{i:3d}/{len(stems)}] {stem}: nodes={row['nodes']:,} "
              f"frozen={row['frozen_transitions']} time={row['seconds']/60:.1f}m", flush=True)
        # Incremental manifest survives a late-session interruption.
        existing = []
        if manifest_path.exists():
            with manifest_path.open(newline="") as f:
                existing = list(csv.DictReader(f))
        by_name = {str(r["dataset"]): r for r in existing}
        by_name[stem] = row
        fields = list(row)
        with manifest_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(by_name[k] for k in sorted(by_name))

    model_info = [{
        "name": m.name, "path": str(m.path), "weight": m.weight,
        "device": str(m.device), "sha256": sha256(m.path),
        "window_size": m.window_size, "downsample": list(m.downsample),
    } for m in members]
    manifest = {
        "format": "biohub_ab_proposals_v1",
        "data_dir": str(args.data_dir),
        "selected_videos": len(stems),
        "det_threshold": args.det_threshold,
        "pool_kernel_um": args.pool_kernel_um,
        "models": model_info,
        "notes": "Raw A+B shared detections before edge prediction, ILP, or post-processing.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if args.zip:
        zip_path = args.output_dir.parent / "biohub_ab_proposals.zip"
        print(f"Creating {zip_path} ...", flush=True)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=3, allowZip64=True) as zf:
            for path in sorted(args.output_dir.iterdir()):
                zf.write(path, arcname=f"biohub_ab_proposals/{path.name}")
        print(f"Archive: {zip_path} ({zip_path.stat().st_size/1e6:.1f} MB)", flush=True)
    print(f"Done: {len(stems)} videos -> {args.output_dir}")


if __name__ == "__main__":
    main()
