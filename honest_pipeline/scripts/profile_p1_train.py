#!/usr/bin/env python3
"""torch.profiler on a few real P1 train steps (node2).

Uses the same batch keys / loss path as train_epoch in train_unet_transformer.py.
Writes chrome trace + tables under honest_pipeline/runs/_profile/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, schedule
from torch.utils.data import DataLoader


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--wait", type=int, default=2)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--active", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--split", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--gradient-checkpointing", action="store_true", default=False)
    p.add_argument("--out-dir", type=str, default="")
    args = p.parse_args()

    root = Path("/data/projects/ryzhichkin/biohub")
    william = root / "william-duckworth-reproducible-training-pipeline"
    scripts = william / "helpers/01_p1_p2_base/shared_repo/scripts"
    src = william / "helpers/01_p1_p2_base/shared_repo/src"
    sys.path.insert(0, str(src))
    sys.path.insert(0, str(scripts))

    out_dir = Path(args.out_dir) if args.out_dir else (root / "honest_pipeline/runs/_profile")
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("BIOHUB_WEIGHTS_DIR", str(out_dir / "weights"))
    Path(os.environ["BIOHUB_WEIGHTS_DIR"]).mkdir(parents=True, exist_ok=True)

    from train_unet_transformer import (  # noqa: E402
        _POS_EMBED_DIM,
        DEFAULT_AUGMENTATIONS,
        FrameWindowDataset,
        UNetNodeTransformer,
        build_matched_edge_targets,
        compute_batch_loss,
        compute_detection_loss,
        detect_and_match,
        load_dataset_windows,
    )
    from biohub_tracking.models.temporal_unet import TemporalUNet3D  # noqa: E402
    from tqdm import tqdm

    data_dir = root / "kaggle/input/competitions/biohub-cell-tracking-during-development/train"
    folds = json.loads(
        (root / "honest_pipeline/splits/dataset_splits_gkf5_train175.json").read_text()
    )
    fold_data = folds[args.split]
    train_files = [data_dir / name for name in fold_data["train"]]
    print(f"Profiling fold {args.split}: {len(train_files)} train movies", flush=True)

    downsample = (1, 4, 4)
    train_video_data = []
    for f in tqdm(train_files, desc="train"):
        video_meta, windows = load_dataset_windows(f, window_size=2, downsample=downsample)
        train_video_data.append((video_meta, windows))
    max_nodes = max(max(w.node_counts) for _, ws in train_video_data for w in ws)
    print(f"max_nodes={max_nodes}", flush=True)
    train_ds = FrameWindowDataset(
        train_video_data, max_nodes=max_nodes, augmentations=DEFAULT_AUGMENTATIONS
    )
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=4 if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )

    device = torch.device("cuda")
    # Match trainer: Flash OK if TemporalAttention chunks ≤32k voxels.
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    print(
        f"SDPA flash={torch.backends.cuda.flash_sdp_enabled()} "
        f"mem={torch.backends.cuda.mem_efficient_sdp_enabled()} "
        f"math={torch.backends.cuda.math_sdp_enabled()} "
        f"grad_ckpt={args.gradient_checkpointing}",
        flush=True,
    )

    unet = TemporalUNet3D(
        in_channels=1,
        out_channels=32,
        layers=(32, 64, 128),
        gradient_checkpointing=args.gradient_checkpointing,
    )
    model = UNetNodeTransformer(
        unet=unet,
        unet_out_channels=32,
        pos_feat_dim=4 * _POS_EMBED_DIM,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()

    det_neg_weight = 0.01
    det_loss_weight = 10.0
    pool_kernel_um = 5.0
    total_steps = args.wait + args.warmup + args.active
    it = iter(loader)

    def one_step() -> float:
        nonlocal it
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        imgs = batch["imgs"].to(device, dtype=torch.float32, non_blocking=True)
        coords = batch["coords"].to(device, non_blocking=True)
        masks = batch["masks"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        image_shape = tuple(batch["image_shape"][0].tolist())
        voxel_size = tuple(batch["voxel_size"][0].tolist())
        ds_scale = batch["downsample"][0].to(device)

        B, W = imgs.shape[:2]
        unet_out, det_logits = model.encode(imgs)
        det_loss = sum(
            compute_detection_loss(det_logits[i], coords[:, i], masks[:, i], det_neg_weight)
            for i in range(W)
        ) / W

        frame_det = []
        for i in range(W):
            det_c, det_p, det_m, matches = detect_and_match(
                det_logits[i],
                coords[:, i],
                masks[:, i],
                image_shape,
                voxel_size=voxel_size,
                pool_kernel_um=pool_kernel_um,
                max_detections=256,
                frame_index=i,
                window_size=W,
            )
            unet_feat = model._index_features(unet_out[:, i], det_c, det_m)
            frame_det.append((det_c, det_p, det_m, matches, unet_feat))

        block_losses = []
        for i in range(W - 1):
            ns = frame_det[i][0].shape[1]
            nt = frame_det[i + 1][0].shape[1]
            pair_target = build_matched_edge_targets(
                frame_det[i][3], frame_det[i + 1][3], targets[:, i], ns, nt
            )
            edge_logits = model.predict_edges(
                frame_det[i][4],
                frame_det[i + 1][4],
                frame_det[i][0] * ds_scale,
                frame_det[i + 1][0] * ds_scale,
                frame_det[i][1],
                frame_det[i + 1][1],
                frame_det[i][2],
                frame_det[i + 1][2],
            )
            block_losses.append(
                compute_batch_loss(
                    edge_logits, pair_target, frame_det[i][2], frame_det[i + 1][2]
                )
            )
        edge_loss = sum(block_losses) / len(block_losses)
        loss = edge_loss + det_loss_weight * det_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        return float(loss.detach())

    def trace_handler(prof: profile) -> None:
        chrome = out_dir / "p1_trace.json"
        prof.export_chrome_trace(str(chrome))
        print(f"Wrote chrome trace: {chrome}", flush=True)

        cuda_tbl = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=40)
        cpu_tbl = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=25)
        shape_tbl = prof.key_averages(group_by_input_shape=True).table(
            sort_by="self_cuda_time_total", row_limit=30
        )
        stack_tbl = prof.key_averages(group_by_stack_n=10).table(
            sort_by="self_cuda_time_total", row_limit=40
        )

        (out_dir / "p1_cuda_ops.txt").write_text(cuda_tbl)
        (out_dir / "p1_cpu_ops.txt").write_text(cpu_tbl)
        (out_dir / "p1_shapes.txt").write_text(shape_tbl)
        (out_dir / "p1_stacks.txt").write_text(stack_tbl)

        print("\n=== TOP CUDA ops (self time) ===", flush=True)
        print(cuda_tbl, flush=True)
        print("\n=== TOP CPU ops (self time) ===", flush=True)
        print(cpu_tbl, flush=True)

    sched = schedule(wait=args.wait, warmup=args.warmup, active=args.active, repeat=1)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=sched,
        on_trace_ready=trace_handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for step in range(total_steps):
            loss = one_step()
            torch.cuda.synchronize()
            prof.step()
            print(
                f"step {step}/{total_steps} loss={loss:.4f} "
                f"peak_mem={torch.cuda.max_memory_allocated() / 1024**3:.1f}GB",
                flush=True,
            )

    print(f"DONE peak_mem_gb={torch.cuda.max_memory_allocated() / 1024**3:.2f}", flush=True)
    print(f"artifacts: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
