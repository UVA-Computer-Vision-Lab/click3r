# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# Modified by the Click3R authors: adapted from dust3r/training.py.
#
# --------------------------------------------------------
# Click3R training with symmetric correspondence prompts
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import signal
import sys
import time
import math
from collections import defaultdict
from pathlib import Path
from typing import Sized

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

# for visualization of prompted points
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from click3r.model import Click3R, inf  # noqa: F401, needed when loading the model
from dust3r.datasets import get_data_loader  # noqa
from dust3r.losses import *  # noqa: F403  # names used in configurable loss expressions
from click3r.correspondence_encoder import CorrespondenceEncoder
from click3r.inference import loss_of_one_batch
from click3r.geometry import (  # noqa: F401
    build_dataset,
    _to_hom,
    _ensure_true_shape,
    _get_K_and_w2c,
    _project_points,
    _valid_mask,
    _sample_num_pairs,
)

import dust3r.utils.path_to_croco  # noqa: F401
import croco.utils.misc as misc  # noqa
from croco.utils.misc import NativeScalerWithGradNormCount as NativeScaler  # noqa


def get_args_parser():
    parser = argparse.ArgumentParser("Click3R training", add_help=False)
    # model and criterion
    parser.add_argument(
        "--model",
        default="Click3R(patch_embed_cls='ManyAR_PatchEmbed')",
        type=str,
        help="string containing the model to build",
    )
    parser.add_argument(
        "--pretrained", default=None, help="path of a starting checkpoint"
    )
    parser.add_argument(
        "--train_criterion",
        default="ConfLoss(Regr3D(L21, norm_mode='avg_dis'), alpha=0.2)",
        type=str,
        help="train criterion",
    )
    parser.add_argument(
        "--test_criterion", default=None, type=str, help="test criterion"
    )

    # dataset
    parser.add_argument("--train_dataset", required=True, type=str, help="training set")
    parser.add_argument(
        "--test_dataset", default="[None]", type=str, help="testing set"
    )

    # training
    parser.add_argument("--seed", default=0, type=int, help="Random seed")
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )
    parser.add_argument(
        "--epochs",
        default=800,
        type=int,
        help="Maximum number of epochs for the scheduler",
    )
    parser.add_argument(
        "--max_train_steps",
        default=0,
        type=int,
        help="Limit batches per epoch for smoke tests; 0 runs the full epoch",
    )

    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="LR",
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=1.5e-4,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.0,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=40, metavar="N", help="epochs to warmup LR"
    )

    parser.add_argument(
        "--amp",
        type=int,
        default=0,
        choices=[0, 1],
        help="Use Automatic Mixed Precision for pretraining",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="Use torch.compile for model optimization (requires PyTorch 2.0+)",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode: 'default' (balanced, no CUDA graphs), 'reduce-overhead' (uses CUDA graphs), 'max-autotune' (slower compile)",
    )
    parser.add_argument(
        "--disable_cudnn_benchmark",
        action="store_true",
        default=False,
        help="set cudnn.benchmark = False",
    )
    # others
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    parser.add_argument(
        "--eval_freq", type=int, default=1, help="Test loss evaluation frequency"
    )
    parser.add_argument(
        "--save_freq",
        default=1,
        type=int,
        help="Save checkpoint-last.pth every this many epochs",
    )
    parser.add_argument(
        "--keep_freq",
        default=20,
        type=int,
        help="Keep a numbered checkpoint every this many epochs",
    )
    parser.add_argument(
        "--print_freq",
        default=20,
        type=int,
        help="Print training progress every this many iterations",
    )

    # Correspondence prompt generation
    parser.add_argument(
        "--enable_prompts",
        action="store_true",
        help="Condition both decoder streams on correspondence prompts.",
    )
    parser.add_argument(
        "--image_position_mode",
        type=str,
        default="external",
        choices=["external", "internal", "zeros"],
        help="Image positional encoding: 'external' uses DUSt3R token-grid coordinates.",
    )
    parser.add_argument(
        "--evaluation_prompt_count",
        type=int,
        default=8,
        help="Prompt count for the helper's fixed-count mode. Training and validation use the 0-12 distribution.",
    )
    parser.add_argument(
        "--prompt_probability",
        type=float,
        default=1.0,
        help="Probability to apply prompting on a batch (during training).",
    )
    parser.add_argument(
        "--flip_swapped_image_ids",
        action="store_true",
        default=False,
        help="Flip prompt image IDs on swapped pairs when symmetrizing. The released "
        "checkpoint was trained without the flip; evaluation and the demo use it.",
    )
    parser.add_argument(
        "--prompt_token_offset",
        type=float,
        default=0.0,
        help="Token offset in prompt coordinate encoding during training and "
        "validation. The released checkpoint was trained with 0.0, the default that "
        "evaluation and the demo use.",
    )
    parser.add_argument(
        "--overlap_matching",
        action="store_true",
        default=False,
        help="Use overlapping pointmap matching for GT prompts (filters projected matches using world-space distance). If False, uses projection-based approach.",
    )
    # wandb (optional; logged alongside TensorBoard)
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging."
    )
    parser.add_argument(
        "--wandb_project", type=str, default="click3r", help="W&B project name."
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default=None, help="W&B run name."
    )
    parser.add_argument(
        "--wandb_tags",
        type=str,
        nargs="+",
        default=None,
        help="W&B tags (space-separated).",
    )
    parser.add_argument(
        "--wandb_run_id",
        type=str,
        default=None,
        help="W&B run ID to resume from (optional).",
    )

    # output dir
    parser.add_argument(
        "--output_dir",
        default="./output/",
        type=str,
        help="path where to save the output",
    )
    return parser


def train(args):
    if torch.cuda.is_available():
        misc.init_distributed_mode(args)
    else:
        if int(os.environ.get("WORLD_SIZE", 1)) != 1:
            raise RuntimeError(
                "Distributed training requires CUDA; use --gpus 1 for a CPU smoke run."
            )
        args.distributed = False
        args.rank = 0
        args.world_size = 1
    global_rank = misc.get_rank()

    print("output_dir: " + args.output_dir)
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # auto resume
    last_ckpt_fname = os.path.join(args.output_dir, "checkpoint-last.pth")
    args.resume = last_ckpt_fname if os.path.isfile(last_ckpt_fname) else None

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # fix the seed
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = not args.disable_cudnn_benchmark

    # training dataset and loader
    print("Building train dataset {:s}".format(args.train_dataset))
    data_loader_train = build_dataset(
        args.train_dataset, args.batch_size, args.num_workers, test=False
    )
    print("Building test dataset {:s}".format(args.test_dataset))
    data_loader_test = {
        dataset.split("(")[0]: build_dataset(
            dataset, args.batch_size, args.num_workers, test=True
        )
        for dataset in args.test_dataset.split("+")
    }

    # model
    print("Loading model: {:s}".format(args.model))
    model = eval(args.model)
    for module in model.modules():
        if isinstance(module, CorrespondenceEncoder):
            module.token_offset = args.prompt_token_offset
    print(f">> Creating train criterion = {args.train_criterion}")
    train_criterion = eval(args.train_criterion).to(device)
    print(f">> Creating test criterion = {args.test_criterion or args.train_criterion}")
    test_criterion = eval(args.test_criterion or args.train_criterion).to(device)

    model.to(device)
    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    # --- Symmetric adapters: runtime switches ---
    if hasattr(model_without_ddp, "prompts_enabled"):
        model_without_ddp.prompts_enabled = bool(args.enable_prompts)

    def _cfg_adapter(m):
        if m is None:
            return
        if hasattr(m, "image_position_mode"):
            m.image_position_mode = args.image_position_mode

    if hasattr(model_without_ddp, "correspondence_adapter1"):
        _cfg_adapter(model_without_ddp.correspondence_adapter1)
    if hasattr(model_without_ddp, "correspondence_adapter2"):
        _cfg_adapter(model_without_ddp.correspondence_adapter2)

    if args.pretrained and not args.resume:
        print("Loading pretrained: ", args.pretrained)
        ckpt = torch.load(args.pretrained, map_location=device, weights_only=False)
        print(model.load_state_dict(ckpt["model"], strict=False))
        del ckpt  # free mem

        # Print parameter statistics after loading pretrained model
        print("\n" + "=" * 80)
        print("Parameter Statistics:")
        print("=" * 80)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        print(f"Total parameters:     {total_params:>15,} (100.00%)")
        print(
            f"Trainable parameters: {trainable_params:>15,} ({100 * trainable_params / total_params:>5.2f}%)"
        )
        print(
            f"Frozen parameters:    {frozen_params:>15,} ({100 * frozen_params / total_params:>5.2f}%)"
        )

        # Print which modules are trainable
        trainable_modules = set()
        frozen_modules = set()
        for name, param in model.named_parameters():
            module_name = name.split(".")[0]
            if param.requires_grad:
                trainable_modules.add(module_name)
            else:
                frozen_modules.add(module_name)

        print(f"\nTrainable modules: {', '.join(sorted(trainable_modules))}")
        print(f"Frozen modules:    {', '.join(sorted(frozen_modules))}")
        print("=" * 80 + "\n")

    # Keep the original model for optimizer state and portable checkpoints.
    # Compile the forward pass before DDP wrapping.
    if args.compile:
        print(f">> Applying torch.compile with mode={args.compile_mode}")
        torch_version = tuple(int(x) for x in torch.__version__.split(".")[:2])
        if torch_version < (2, 0):
            raise RuntimeError(
                f"torch.compile requires PyTorch 2.0+, found {torch.__version__}"
            )

        model = torch.compile(model, mode=args.compile_mode)
        print(f"✓ Model compiled successfully with mode={args.compile_mode}")
        print("Note: First few iterations will be slower due to compilation")
    else:
        print("torch.compile disabled (use --compile to enable)")

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        # Empty prompts bypass the adapters, so used parameters can change by batch.
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            find_unused_parameters=True,
            static_graph=False,
        )

    # following timm: set wd as 0 for bias and norm layers
    param_groups = misc.get_parameter_groups(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    if global_rank == 0 and args.output_dir is not None:
        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    # optional W&B init
    wandb_run = None
    if misc.is_main_process() and args.wandb:
        try:
            import wandb

            # Resume wandb run if run_id is provided
            if args.wandb_run_id:
                wandb_run = wandb.init(
                    project=args.wandb_project,
                    id=args.wandb_run_id,
                    resume="allow",
                    name=args.wandb_run_name,
                    tags=args.wandb_tags if args.wandb_tags else [],
                    config=vars(args),
                )
                print(f"[W&B] Resuming run with ID: {args.wandb_run_id}")
            else:
                wandb_run = wandb.init(
                    project=args.wandb_project,
                    name=args.wandb_run_name,
                    tags=args.wandb_tags if args.wandb_tags else [],
                    config=vars(args),
                )
        except Exception as e:
            print(f"[W&B] init failed: {e}")

    def write_log_stats(epoch, train_stats, test_stats):
        if misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            log_stats = dict(
                epoch=epoch, **{f"train_{k}": v for k, v in train_stats.items()}
            )
            for test_key in test_stats:
                log_stats.update(
                    {test_key + "_" + k: v for k, v in test_stats[test_key].items()}
                )
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    def save_model(epoch, fname, best_so_far):
        misc.save_model(
            args=args,
            model_without_ddp=model_without_ddp,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            epoch=epoch,
            fname=fname,
            best_so_far=best_so_far,
        )

    best_so_far = misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )
    if best_so_far is None:
        best_so_far = float("inf")

    # --- SIGTERM handler for clean shutdown on SLURM timeout ---
    _sigterm_received = False

    def _sigterm_handler(signum, frame):
        nonlocal _sigterm_received
        _sigterm_received = True
        print(
            f"\n[SIGTERM] Received signal {signum}, will save checkpoint and exit after current iteration."
        )

    signal.signal(signal.SIGTERM, _sigterm_handler)

    # --- wandb step offset for resume ---
    # checkpoint-last.pth stores the last completed epoch, so a resumed run replays
    # steps that the interrupted run may already have logged to wandb. Shift all
    # steps past the last logged one.
    _wandb_step_offset = 0
    if args.resume and wandb_run is not None:
        try:
            # Steps are int(epoch_f * 1000), so the resumed run starts at
            # step start_epoch * 1000.
            wandb_last_step = wandb_run.step if hasattr(wandb_run, "step") else 0
            expected_start_step = args.start_epoch * 1000
            if wandb_last_step > expected_start_step:
                _wandb_step_offset = wandb_last_step - expected_start_step + 1
                print(
                    f"[W&B] Step offset = {_wandb_step_offset} (wandb last step={wandb_last_step}, expected start={expected_start_step})"
                )
        except Exception as e:
            print(f"[W&B] Could not compute step offset: {e}")

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    train_stats = test_stats = {}

    for epoch in range(args.start_epoch, args.epochs + 1):
        # Save immediately the last checkpoint
        if epoch > args.start_epoch:
            if args.save_freq and epoch % args.save_freq == 0 or epoch == args.epochs:
                save_model(epoch - 1, "last", best_so_far)

        # Test
        new_best = False
        if epoch > 0 and args.eval_freq > 0 and epoch % args.eval_freq == 0:
            test_stats = {}
            for test_name, testset in data_loader_test.items():
                # Test without prompts
                stats_no_prompts = test_one_epoch(
                    model,
                    test_criterion,
                    testset,
                    device,
                    epoch,
                    log_writer=log_writer,
                    args=args,
                    prefix=f"{test_name}_no_prompts",
                    use_prompts=False,
                )
                test_stats[f"{test_name}_no_prompts"] = stats_no_prompts

                # Test WITH prompts (same random sampling strategy as training)
                stats_with_prompts = test_one_epoch(
                    model,
                    test_criterion,
                    testset,
                    device,
                    epoch,
                    log_writer=log_writer,
                    args=args,
                    prefix=f"{test_name}_with_prompts",
                    use_prompts=True,
                )
                test_stats[f"{test_name}_with_prompts"] = stats_with_prompts

                # Select the best checkpoint by the no-prompt validation loss
                if stats_no_prompts["loss_med"] < best_so_far:
                    best_so_far = stats_no_prompts["loss_med"]
                    new_best = True

            # Log test stats to wandb
            if wandb_run is not None and misc.is_main_process():
                try:
                    wandb_payload = {}
                    for test_name, stats in test_stats.items():
                        for stat_name, stat_val in stats.items():
                            wandb_payload[f"{test_name}_{stat_name}"] = stat_val
                    if wandb_payload:
                        wandb_run.log(
                            wandb_payload, step=epoch * 1000 + _wandb_step_offset
                        )
                except Exception as e:
                    print(f"[W&B] logging test stats failed: {e}")

        # Save logs & checkpoints
        write_log_stats(epoch, train_stats, test_stats)
        if epoch > args.start_epoch:
            if args.keep_freq and epoch % args.keep_freq == 0:
                save_model(epoch - 1, str(epoch), best_so_far)
            if new_best:
                save_model(epoch - 1, "best", best_so_far)
        if epoch >= args.epochs:
            break

        # Train
        train_stats = train_one_epoch(
            model,
            train_criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            log_writer=log_writer,
            args=args,
            wandb_run=wandb_run,
            wandb_step_offset=_wandb_step_offset,
        )

        # --- Handle SIGTERM (SLURM timeout) ---
        if _sigterm_received:
            print(f"[SIGTERM] Saving checkpoint at epoch {epoch} before exit...")
            save_model(epoch, "last", best_so_far)
            if wandb_run is not None:
                try:
                    import wandb

                    wandb_run.finish()
                    print("[SIGTERM] wandb run finished cleanly.")
                except Exception as e:
                    print(f"[SIGTERM] wandb finish failed: {e}")
            print(
                f"[SIGTERM] Exiting. Resume from checkpoint-last.pth (epoch {epoch})."
            )
            sys.exit(0)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))

    # Clean wandb finish on normal completion
    if wandb_run is not None:
        try:
            import wandb

            wandb_run.finish()
            print("[W&B] Run finished cleanly.")
        except Exception as e:
            print(f"[W&B] finish failed: {e}")

    save_final_model(args, args.epochs, model_without_ddp, best_so_far=best_so_far)


def save_final_model(args, epoch, model_without_ddp, best_so_far=None):
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / "checkpoint-final.pth"
    to_save = {
        "args": args,
        "model": model_without_ddp
        if isinstance(model_without_ddp, dict)
        else model_without_ddp.cpu().state_dict(),
        "epoch": epoch,
    }
    if best_so_far is not None:
        to_save["best_so_far"] = best_so_far
    print(f">> Saving model to {checkpoint_path} ...")
    misc.save_on_master(to_save, checkpoint_path)


def _attach_prompts_from_gt(batch, args, eval_mode=False):
    """
    Attach GT-based prompts to batch.

    Args:
        batch: Tuple of (view1, view2) dicts
        args: Arguments containing enable_prompts, evaluation_prompt_count, etc.
        eval_mode: If True, use args.evaluation_prompt_count prompts per pair and
                   ignore args.prompt_probability. If False (training and
                   validation), sample the count with _sample_num_pairs.
    """
    # Check if prompting is enabled
    if not args.enable_prompts or (eval_mode and args.evaluation_prompt_count <= 0):
        batch[0].pop("sparse_prompts", None)
        batch[1].pop("sparse_prompts", None)
        return batch

    v1, v2 = batch[0], batch[1]
    dev = v1["img"].device
    B = v1["img"].shape[0]
    shape1 = _ensure_true_shape(v1)  # (B,2) (H,W)
    shape2 = _ensure_true_shape(v2)

    # Maximum number of pairs (for tensor allocation)
    MAX_P = 12

    # GT world points of view 1 (view 2 as fallback): (B,H,W,3) or (B,M,3)
    pts3d = v1.get("pts3d", None)
    if pts3d is None:
        pts3d = v2.get("pts3d", None)

    # Flatten dense pts3d (B, H, W, 3) -> (B, M, 3) if necessary
    if pts3d is not None and pts3d.dim() == 4:
        B_pts, H, W, C = pts3d.shape
        pts3d = pts3d.reshape(B_pts, H * W, C)

    K1, w2c1 = _get_K_and_w2c(v1)
    K2, w2c2 = _get_K_and_w2c(v2)
    if (
        (pts3d is None)
        or (K1 is None)
        or (K2 is None)
        or (w2c1 is None)
        or (w2c2 is None)
    ):
        # No GT/cameras → no prompts at all for this batch
        batch[0].pop("sparse_prompts", None)
        batch[1].pop("sparse_prompts", None)
        return batch

    uv1, z1 = _project_points(K1, w2c1, pts3d)  # (B,M,2), (B,M)
    uv2, z2 = _project_points(K2, w2c2, pts3d)
    vis = _valid_mask(uv1, z1, shape1, 0) & _valid_mask(uv2, z2, shape2, 0)  # (B,M)

    coords1_list, coords2_list, pair_ids_list, vmask_list = [], [], [], []
    num_pairs_per_sample = []  # Track for logging

    for b in range(B):
        # Sample number of pairs for this specific image pair
        if eval_mode:
            # Use a fixed count; coordinate sampling still uses the current RNG.
            P_sample = args.evaluation_prompt_count
        else:
            # Random count (training and validation)
            P_sample = _sample_num_pairs()
        num_pairs_per_sample.append(P_sample)

        # Apply prompt probability (only in training mode; eval mode always uses prompts)
        if not eval_mode and torch.rand(1).item() > float(args.prompt_probability):
            P_sample = 0

        idx = torch.nonzero(vis[b], as_tuple=False).squeeze(-1)

        if idx.numel() == 0 or P_sample == 0:
            # No valid GT pairs or no prompts sampled → mark all as invalid
            coords1_list.append(torch.zeros(MAX_P, 2, device=dev))
            coords2_list.append(torch.zeros(MAX_P, 2, device=dev))
            pair_ids_list.append(torch.zeros(MAX_P, dtype=torch.long, device=dev))
            vmask = torch.zeros(MAX_P, dtype=torch.bool, device=dev)
            vmask_list.append(vmask)
            continue

        # Sample up to P_sample valid pairs
        actual_pairs = min(P_sample, idx.numel())
        sel = idx[torch.randperm(idx.numel(), device=dev)[:actual_pairs]]

        # Pad to MAX_P for uniform tensor size
        if sel.numel() < MAX_P:
            if sel.numel() > 0:
                # Repeat existing selections to fill
                pad = sel[
                    torch.randint(0, sel.numel(), (MAX_P - sel.numel(),), device=dev)
                ]
                sel_padded = torch.cat([sel, pad], dim=0)
            else:
                # No valid points, use zeros
                sel_padded = torch.zeros(MAX_P, dtype=torch.long, device=dev)
        else:
            sel_padded = sel[:MAX_P]

        c1 = uv1[b, sel_padded]
        c2 = uv2[b, sel_padded]
        coords1_list.append(c1)
        coords2_list.append(c2)
        pair_ids_list.append(torch.arange(MAX_P, device=dev, dtype=torch.long))

        # Valid mask: only first actual_pairs are valid
        vmask = torch.zeros(MAX_P, dtype=torch.bool, device=dev)
        vmask[:actual_pairs] = True
        vmask_list.append(vmask)

    coords1 = torch.stack(coords1_list, dim=0)  # (B,MAX_P,2)
    coords2 = torch.stack(coords2_list, dim=0)  # (B,MAX_P,2)
    pair_ids = torch.stack(pair_ids_list, dim=0)  # (B,MAX_P)
    valid_mask = torch.stack(vmask_list, dim=0)  # (B,MAX_P) bool

    v1["sparse_prompts"] = {
        "coords": coords1,
        "image_ids": torch.zeros(B, MAX_P, device=dev, dtype=torch.long),  # 0 for view1
        "pair_ids": pair_ids,
        "shape": shape1,
        "valid_mask": valid_mask,
    }
    v2["sparse_prompts"] = {
        "coords": coords2,
        "image_ids": torch.ones(B, MAX_P, device=dev, dtype=torch.long),  # 1 for view2
        "pair_ids": pair_ids,
        "shape": shape2,
        "valid_mask": valid_mask,  # share the same per-sample mask
    }

    # Per-pair prompt counts; train_one_epoch and test_one_epoch pop them for logging
    v1["num_pairs_sampled"] = num_pairs_per_sample
    v2["num_pairs_sampled"] = num_pairs_per_sample

    return batch


def _attach_prompts_from_gt_overlap(batch, args, eval_mode=False):
    """
    Attach GT correspondences by matching the two world-space pointmaps.

    Project valid view-1 points into view 2, sample its pointmap at the rounded
    pixel coordinates, and retain matches within 0.05 world-space units.
    This filters occluded points whose sampled surface is farther away.

    Args:
        batch: Tuple of (view1, view2) dicts
        args: Arguments containing enable_prompts, evaluation_prompt_count, etc.
        eval_mode: If True, use args.evaluation_prompt_count prompts per pair and
                   ignore args.prompt_probability. If False (training and
                   validation), sample the count with _sample_num_pairs.
    """
    # Check if prompting is enabled
    if not args.enable_prompts or (eval_mode and args.evaluation_prompt_count <= 0):
        batch[0].pop("sparse_prompts", None)
        batch[1].pop("sparse_prompts", None)
        return batch

    v1, v2 = batch[0], batch[1]
    dev = v1["img"].device
    B = v1["img"].shape[0]
    shape1 = _ensure_true_shape(v1)  # (B,2) (H,W)
    shape2 = _ensure_true_shape(v2)

    # Maximum number of pairs (for tensor allocation)
    MAX_P = 12

    # Get pts3d from BOTH views (in world coordinates)
    pts3d_v1 = v1.get("pts3d", None)  # (B, H, W, 3) world coords visible from view1
    pts3d_v2 = v2.get("pts3d", None)  # (B, H, W, 3) world coords visible from view2
    valid_mask_v1 = v1.get("valid_mask", None)
    valid_mask_v2 = v2.get("valid_mask", None)

    # Intrinsics and world-to-camera poses
    K1, w2c1 = _get_K_and_w2c(v1)
    K2, w2c2 = _get_K_and_w2c(v2)

    # Check if we have all required data
    if (
        pts3d_v1 is None
        or pts3d_v2 is None
        or K1 is None
        or K2 is None
        or w2c1 is None
        or w2c2 is None
    ):
        batch[0].pop("sparse_prompts", None)
        batch[1].pop("sparse_prompts", None)
        return batch

    # Handle valid masks
    if valid_mask_v1 is None:
        if pts3d_v1.dim() == 4:
            valid_mask_v1 = torch.ones(pts3d_v1.shape[:3], dtype=torch.bool, device=dev)
        else:
            valid_mask_v1 = torch.ones(pts3d_v1.shape[:2], dtype=torch.bool, device=dev)
    elif isinstance(valid_mask_v1, np.ndarray):
        valid_mask_v1 = torch.from_numpy(valid_mask_v1).to(dev)

    if valid_mask_v2 is None:
        if pts3d_v2.dim() == 4:
            valid_mask_v2 = torch.ones(pts3d_v2.shape[:3], dtype=torch.bool, device=dev)
        else:
            valid_mask_v2 = torch.ones(pts3d_v2.shape[:2], dtype=torch.bool, device=dev)
    elif isinstance(valid_mask_v2, np.ndarray):
        valid_mask_v2 = torch.from_numpy(valid_mask_v2).to(dev)

    coords1_list, coords2_list, pair_ids_list, vmask_list = [], [], [], []
    num_pairs_per_sample = []

    for b in range(B):
        # Sample number of pairs for this specific image pair
        if eval_mode:
            P_sample = args.evaluation_prompt_count
        else:
            P_sample = _sample_num_pairs()
        num_pairs_per_sample.append(P_sample)

        # Apply prompt probability (only in training mode)
        if not eval_mode and torch.rand(1).item() > float(args.prompt_probability):
            P_sample = 0

        if P_sample == 0:
            coords1_list.append(torch.zeros(MAX_P, 2, device=dev))
            coords2_list.append(torch.zeros(MAX_P, 2, device=dev))
            pair_ids_list.append(torch.zeros(MAX_P, dtype=torch.long, device=dev))
            vmask_list.append(torch.zeros(MAX_P, dtype=torch.bool, device=dev))
            continue

        # Get pts3d for this batch item
        if pts3d_v1.dim() == 4:
            p3d_v1 = pts3d_v1[b]  # (H1, W1, 3) - world coords
            p3d_v2 = pts3d_v2[b]  # (H2, W2, 3) - world coords
            vmask1 = valid_mask_v1[b]  # (H1, W1)
            vmask2 = valid_mask_v2[b]  # (H2, W2)
            H1, W1, _ = p3d_v1.shape
            H2, W2, _ = p3d_v2.shape
        else:
            # Already flattened
            p3d_v1 = pts3d_v1[b]  # (M1, 3)
            p3d_v2 = pts3d_v2[b]  # (M2, 3)
            vmask1 = valid_mask_v1[b]  # (M1,)
            vmask2 = valid_mask_v2[b]  # (M2,)
            # Infer H, W from shape
            H1, W1 = shape1[b].tolist()
            H2, W2 = shape2[b].tolist()
            p3d_v1 = p3d_v1.reshape(H1, W1, 3)
            p3d_v2 = p3d_v2.reshape(H2, W2, 3)
            vmask1 = vmask1.reshape(H1, W1)
            vmask2 = vmask2.reshape(H2, W2)

        # Flatten view1's pts3d to (M, 3) for projection
        p3d_v1_flat = p3d_v1.reshape(-1, 3)  # (H1*W1, 3)
        vmask1_flat = vmask1.reshape(-1)  # (H1*W1,)

        # Project view-1 world points into view 2.
        pts3d_batch = p3d_v1_flat.unsqueeze(0)  # (1, M, 3)
        uv2, z2_proj_batch = _project_points(
            K2[b : b + 1], w2c2[b : b + 1], pts3d_batch
        )  # (1, M, 2), (1, M)
        uv2 = uv2[0]  # (M, 2)
        z2_proj = z2_proj_batch[0]  # (M,)

        # Step 1: Filter by valid mask and projection bounds (vectorized)
        valid_v1 = vmask1_flat  # (H1*W1,)
        valid_proj = (
            (z2_proj > 0)
            & (uv2[:, 0] >= 0)
            & (uv2[:, 0] < W2 - 1)
            & (uv2[:, 1] >= 0)
            & (uv2[:, 1] < H2 - 1)
        )
        valid_mask_combined = valid_v1 & valid_proj  # (H1*W1,)

        # Get indices of potentially valid points
        valid_indices = torch.where(valid_mask_combined)[0]

        if len(valid_indices) == 0:
            coords1_list.append(torch.zeros(MAX_P, 2, device=dev))
            coords2_list.append(torch.zeros(MAX_P, 2, device=dev))
            pair_ids_list.append(torch.zeros(MAX_P, dtype=torch.long, device=dev))
            vmask_list.append(torch.zeros(MAX_P, dtype=torch.bool, device=dev))
            continue

        # Step 2: Get coordinates for valid points
        y1_coords = valid_indices // W1
        x1_coords = valid_indices % W1
        x2_float = uv2[valid_indices, 0]
        y2_float = uv2[valid_indices, 1]
        x2_coords = torch.round(x2_float).long()
        y2_coords = torch.round(y2_float).long()
        # Step 3: Check 3D distance (vectorized)
        # Compare world-space points at the projected pixel locations.
        pts_v1 = p3d_v1[y1_coords, x1_coords]  # (N, 3)
        pts_v2 = p3d_v2[y2_coords, x2_coords]  # (N, 3)
        dist_3d = torch.norm(pts_v1 - pts_v2, dim=1)  # (N,)
        valid_3d = dist_3d <= 0.05  # 0.05 world units (5 cm in metric scenes)

        # Final valid matches
        x1_coords = x1_coords[valid_3d]
        y1_coords = y1_coords[valid_3d]
        x2_coords = x2_coords[valid_3d]
        y2_coords = y2_coords[valid_3d]

        num_matches = len(x1_coords)
        if num_matches == 0:
            coords1_list.append(torch.zeros(MAX_P, 2, device=dev))
            coords2_list.append(torch.zeros(MAX_P, 2, device=dev))
            pair_ids_list.append(torch.zeros(MAX_P, dtype=torch.long, device=dev))
            vmask_list.append(torch.zeros(MAX_P, dtype=torch.bool, device=dev))
            continue

        coords_v1 = torch.stack(
            [x1_coords.float(), y1_coords.float()], dim=1
        )  # (num_matches, 2)
        coords_v2 = torch.stack(
            [x2_coords.float(), y2_coords.float()], dim=1
        )  # (num_matches, 2)

        # Random subset of the matches
        actual_pairs = min(P_sample, num_matches)
        if num_matches > actual_pairs:
            rand_idx = torch.randperm(num_matches, device=dev)[:actual_pairs]
            coords_v1 = coords_v1[rand_idx]
            coords_v2 = coords_v2[rand_idx]

        # Pad to MAX_P for uniform tensor size
        if actual_pairs < MAX_P:
            # Repeat existing selections to fill
            if actual_pairs > 0:
                repeat_idx = torch.randint(
                    0, actual_pairs, (MAX_P - actual_pairs,), device=dev
                )
                coords_v1_padded = torch.cat([coords_v1, coords_v1[repeat_idx]], dim=0)
                coords_v2_padded = torch.cat([coords_v2, coords_v2[repeat_idx]], dim=0)
            else:
                coords_v1_padded = torch.zeros(MAX_P, 2, device=dev)
                coords_v2_padded = torch.zeros(MAX_P, 2, device=dev)
        else:
            coords_v1_padded = coords_v1[:MAX_P]
            coords_v2_padded = coords_v2[:MAX_P]
            actual_pairs = MAX_P

        coords1_list.append(coords_v1_padded)
        coords2_list.append(coords_v2_padded)
        pair_ids_list.append(torch.arange(MAX_P, device=dev, dtype=torch.long))

        # Valid mask: only first actual_pairs are valid
        vmask = torch.zeros(MAX_P, dtype=torch.bool, device=dev)
        vmask[:actual_pairs] = True
        vmask_list.append(vmask)

    coords1 = torch.stack(coords1_list, dim=0)  # (B, MAX_P, 2)
    coords2 = torch.stack(coords2_list, dim=0)  # (B, MAX_P, 2)
    pair_ids = torch.stack(pair_ids_list, dim=0)  # (B, MAX_P)
    valid_mask = torch.stack(vmask_list, dim=0)  # (B, MAX_P) bool

    v1["sparse_prompts"] = {
        "coords": coords1,
        "image_ids": torch.zeros(B, MAX_P, device=dev, dtype=torch.long),
        "pair_ids": pair_ids,
        "shape": shape1,
        "valid_mask": valid_mask,
    }
    v2["sparse_prompts"] = {
        "coords": coords2,
        "image_ids": torch.ones(B, MAX_P, device=dev, dtype=torch.long),
        "pair_ids": pair_ids,
        "shape": shape2,
        "valid_mask": valid_mask,
    }

    # Per-pair prompt counts; train_one_epoch and test_one_epoch pop them for logging
    v1["num_pairs_sampled"] = num_pairs_per_sample
    v2["num_pairs_sampled"] = num_pairs_per_sample

    return batch


def visualize_prompts(batch, save_dir, epoch, iter_step, max_samples=2):
    """
    Visualize GT-projected prompts overlaid on image pairs.

    Args:
        batch: Tuple of (view1_dict, view2_dict) with "sparse_prompts" attached
        save_dir: Directory to save visualization images
        epoch: Current epoch number
        iter_step: Current iteration number
        max_samples: Max number of samples from batch to visualize
    """
    import os

    os.makedirs(save_dir, exist_ok=True)

    v1, v2 = batch[0], batch[1]

    # Check if prompts exist
    if "sparse_prompts" not in v1 or "sparse_prompts" not in v2:
        print(f"[Vis] No prompts found in batch (epoch={epoch}, iter={iter_step})")
        return

    # Get batch data
    imgs1 = v1["img"]  # (B, 3, H, W)
    imgs2 = v2["img"]
    prompts1 = v1["sparse_prompts"]
    prompts2 = v2["sparse_prompts"]

    coords1 = prompts1["coords"].cpu()  # (B, P, 2) - (x, y) in pixels
    coords2 = prompts2["coords"].cpu()
    pair_ids = prompts1["pair_ids"].cpu()  # (B, P)
    valid_mask = prompts1["valid_mask"].cpu()  # (B, P)

    B, P, _ = coords1.shape
    num_viz = min(B, max_samples)

    # Color map for different pair IDs
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for b in range(num_viz):
        # (3,H,W) tensors to (H,W,3) arrays
        img1_np = imgs1[b].cpu().permute(1, 2, 0).numpy()
        img2_np = imgs2[b].cpu().permute(1, 2, 0).numpy()

        # Clip to [0, 1] for imshow
        img1_np = np.clip(img1_np, 0, 1)
        img2_np = np.clip(img2_np, 0, 1)

        # Create figure with both views side by side
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].imshow(img1_np)
        axes[1].imshow(img2_np)

        axes[0].set_title(f"View 1 (Epoch {epoch}, Iter {iter_step}, Sample {b})")
        axes[1].set_title("View 2")

        # Plot correspondence pairs
        num_valid = valid_mask[b].sum().item()
        for p in range(P):
            if not valid_mask[b, p]:
                continue  # Skip invalid prompts

            x1, y1 = coords1[b, p].tolist()
            x2, y2 = coords2[b, p].tolist()
            pair_id = pair_ids[b, p].item()
            color = colors[pair_id % 10]

            # Draw points on both views
            axes[0].plot(
                x1,
                y1,
                "o",
                color=color,
                markersize=8,
                markeredgecolor="white",
                markeredgewidth=1.5,
            )
            axes[1].plot(
                x2,
                y2,
                "o",
                color=color,
                markersize=8,
                markeredgecolor="white",
                markeredgewidth=1.5,
            )

            # Add pair ID label
            if p < 8:  # Only label first few to avoid clutter
                axes[0].text(
                    x1 + 5,
                    y1 - 5,
                    str(pair_id),
                    color="white",
                    fontsize=10,
                    bbox=dict(boxstyle="round", facecolor=color, alpha=0.7),
                )
                axes[1].text(
                    x2 + 5,
                    y2 - 5,
                    str(pair_id),
                    color="white",
                    fontsize=10,
                    bbox=dict(boxstyle="round", facecolor=color, alpha=0.7),
                )

        axes[0].axis("off")
        axes[1].axis("off")

        # Add info text
        fig.text(
            0.5,
            0.02,
            f"Valid prompts: {num_valid}/{P}  |  Colors indicate correspondence pairs",
            ha="center",
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        save_path = os.path.join(
            save_dir, f"prompts_e{epoch:03d}_i{iter_step:04d}_s{b}.png"
        )
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        if b == 0:  # Print once
            print(f"[Vis] Saved prompt visualization to {save_path}")


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Sized,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    args,
    log_writer=None,
    wandb_run=None,
    wandb_step_offset: int = 0,
):
    assert torch.backends.cuda.matmul.allow_tf32 is True

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    accum_iter = args.accum_iter

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, "sampler") and hasattr(data_loader.sampler, "set_epoch"):
        data_loader.sampler.set_epoch(epoch)
    optimizer.zero_grad()

    for data_iter_step, batch in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        if args.max_train_steps and data_iter_step >= args.max_train_steps:
            break
        epoch_f = epoch + data_iter_step / len(data_loader)

        if data_iter_step % accum_iter == 0:
            misc.adjust_learning_rate(optimizer, epoch_f, args)

        # GT correspondence prompts with a random count per pair
        num_pairs_stats = {}
        if args.enable_prompts:
            if args.overlap_matching:
                batch = _attach_prompts_from_gt_overlap(batch, args)
            else:
                batch = _attach_prompts_from_gt(batch, args)
            # Extract num_pairs_sampled for logging (can't be moved to device as it's a list)
            num_pairs_list = batch[0].pop("num_pairs_sampled", None)
            batch[1].pop("num_pairs_sampled", None)

            # Compute statistics for logging
            if num_pairs_list is not None and len(num_pairs_list) > 0:
                num_pairs_stats = {
                    "num_pairs_mean": float(np.mean(num_pairs_list)),
                    "num_pairs_min": float(np.min(num_pairs_list)),
                    "num_pairs_max": float(np.max(num_pairs_list)),
                }

        loss_tuple = loss_of_one_batch(
            batch,
            model,
            criterion,
            device,
            symmetrize_batch=True,
            use_amp=bool(args.amp),
            ret="loss",
            flip_image_ids=args.flip_swapped_image_ids,
        )
        loss, loss_details = loss_tuple
        loss_value = float(loss)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value), force=True)
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        del loss
        del batch

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(epoch=epoch_f)
        metric_logger.update(lr=lr)
        metric_logger.update(loss=loss_value, **loss_details)

        # Add num_pairs statistics to the console log
        if num_pairs_stats:
            metric_logger.update(**num_pairs_stats)

        if (data_iter_step + 1) % accum_iter == 0 and (
            (data_iter_step + 1) % (accum_iter * args.print_freq)
        ) == 0:
            loss_value_reduce = misc.all_reduce_mean(
                loss_value
            )  # MUST BE EXECUTED BY ALL NODES
            if log_writer is not None:
                epoch_1000x = int(epoch_f * 1000)
                log_writer.add_scalar("train_loss", loss_value_reduce, epoch_1000x)
                log_writer.add_scalar("train_lr", lr, epoch_1000x)
                log_writer.add_scalar("train_iter", epoch_1000x, epoch_1000x)
                for name, val in loss_details.items():
                    log_writer.add_scalar("train_" + name, val, epoch_1000x)
                # Log num_pairs statistics to TensorBoard
                for name, val in num_pairs_stats.items():
                    log_writer.add_scalar("train_" + name, val, epoch_1000x)
            if wandb_run is not None and misc.is_main_process():
                try:
                    payload = {"train_loss": loss_value_reduce, "train_lr": lr}
                    for name, val in loss_details.items():
                        payload["train_" + name] = val
                    # Log num_pairs statistics to wandb
                    for name, val in num_pairs_stats.items():
                        payload["train_" + name] = val
                    wandb_run.log(payload, step=int(epoch_f * 1000) + wandb_step_offset)
                except Exception:
                    pass

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def test_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Sized,
    device: torch.device,
    epoch: int,
    args,
    log_writer=None,
    prefix="test",
    use_prompts=False,
):
    model.eval()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.meters = defaultdict(lambda: misc.SmoothedValue(window_size=9**9))
    header = "{} Test Epoch: [{}]".format(prefix, epoch)

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    if hasattr(data_loader, "dataset") and hasattr(data_loader.dataset, "set_epoch"):
        data_loader.dataset.set_epoch(epoch)
    if hasattr(data_loader, "sampler") and hasattr(data_loader.sampler, "set_epoch"):
        data_loader.sampler.set_epoch(epoch)

    # Track num_pairs across all batches for statistics
    all_num_pairs = []

    for _, batch in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        # Optionally attach prompts using same strategy as training (random sampling)
        num_pairs_list = None
        if use_prompts and args.enable_prompts:
            if args.overlap_matching:
                batch = _attach_prompts_from_gt_overlap(batch, args)
            else:
                batch = _attach_prompts_from_gt(batch, args)
            # Extract num_pairs_sampled for logging
            num_pairs_list = batch[0].pop("num_pairs_sampled", None)
            batch[1].pop("num_pairs_sampled", None)

            # Accumulate for overall statistics
            if num_pairs_list is not None:
                all_num_pairs.extend(num_pairs_list)

        loss_tuple = loss_of_one_batch(
            batch,
            model,
            criterion,
            device,
            symmetrize_batch=True,
            use_amp=bool(args.amp),
            ret="loss",
            flip_image_ids=args.flip_swapped_image_ids,
        )
        loss_value, loss_details = loss_tuple
        metric_logger.update(loss=float(loss_value), **loss_details)

    # Compute num_pairs statistics if prompts were used
    if use_prompts and len(all_num_pairs) > 0:
        num_pairs_stats = {
            "num_pairs_mean": float(np.mean(all_num_pairs)),
            "num_pairs_min": float(np.min(all_num_pairs)),
            "num_pairs_max": float(np.max(all_num_pairs)),
        }
        # Add to metric logger for console output
        for k, v in num_pairs_stats.items():
            metric_logger.meters[k].update(v)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    aggs = [("avg", "global_avg"), ("med", "median")]
    results = {
        f"{k}_{tag}": getattr(meter, attr)
        for k, meter in metric_logger.meters.items()
        for tag, attr in aggs
    }

    if log_writer is not None:
        for name, val in results.items():
            log_writer.add_scalar(prefix + "_" + name, val, 1000 * epoch)

    return results


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    train(args)
