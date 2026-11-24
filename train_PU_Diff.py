"""Train PU-Diff from the supplied paper, Sections 3 and 4.2.

Iterations 1-1000: labeled supervision only.
Subsequent iterations: Lsup + lambda1*Lunsup + lambda2*Llat + lambda3*Lrect.
"""

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from dataloaders.dataset import (TrainingDataset, flip_coordinates,
                                 training_partition, weak_targets_to_strong)
from networks.spuq import SPUQ
from networks.uadr import UADR
from networks.unet import UNet
from utils.evaluation import evaluate
from utils.losses import segmentation_loss
from utils.runtime import (CHECKPOINT_VERSION, PROJECT_ROOT, atomic_torch_save,
                           load_checkpoint, resolve_device, seed_everything, write_json)


DATASETS = {
    "acdc": {"folder": "ACDC", "num_classes": 4, "img_channels": 1, "labeled_num": 7, "data_format": "h5"},
    "mscmrseg": {"folder": "MS-CMRSeg", "num_classes": 4, "img_channels": 1, "labeled_num": 3, "data_format": "h5"},
    "prostate": {"folder": "MSD-Prostate", "num_classes": 3, "img_channels": 2, "labeled_num": 3, "data_format": "h5"},
    "kvasir": {"folder": "Kvasir-SEG", "num_classes": 2, "img_channels": 3, "labeled_num": 80, "data_format": "images"},
}


def build_parser():
    parser = argparse.ArgumentParser(description="PU-Diff semi-supervised 2D segmentation")
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="acdc")
    parser.add_argument("--root_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--data_format", choices=("h5", "images"), default=None)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--img_channels", type=int, default=None)
    parser.add_argument("--labeled_num", type=int, default=None, help="number of labeled patients/cases")
    parser.add_argument("--labeled_list", type=str, default=None, help="explicit labeled case/patient list")
    parser.add_argument("--patch_size", nargs=2, type=int, default=[256, 256])
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--labeled_bs", type=int, default=6)
    parser.add_argument("--max_iterations", type=int, default=30000)
    parser.add_argument("--warmup_iterations", type=int, default=1000)
    parser.add_argument("--base_lr", type=float, default=0.001)
    parser.add_argument("--diffusion_lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--conf_thresh", type=float, default=0.8)
    parser.add_argument("--ema_decay", type=float, default=0.95)
    parser.add_argument("--diffusion_steps", type=int, default=50)
    parser.add_argument("--latent_size", type=int, default=64)
    parser.add_argument("--lambda_unsup", type=float, default=1.0)
    parser.add_argument("--lambda_lat", type=float, default=1.0)
    parser.add_argument("--lambda_rect", type=float, default=1.0)
    parser.add_argument("--max_rotation", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--inference_batch_size", type=int, default=4)
    parser.add_argument("--val_interval", type=int, default=None,
                        help="0 disables training-time validation; never reads test.list")
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--resume", type=str, default=None)
    return parser


def parse_training_args(argv=None):
    parser = build_parser()
    initial = parser.parse_args(argv)
    if initial.resume:
        checkpoint = load_checkpoint(initial.resume)
        parser.set_defaults(**{key: value for key, value in checkpoint["config"].items()
                               if key in vars(initial)})
    args = parser.parse_args(argv)
    preset = DATASETS[args.dataset]
    if args.val_interval is None:
        args.val_interval = 0 if args.dataset == "prostate" else 200
    for key in ("data_format", "num_classes", "img_channels", "labeled_num"):
        if getattr(args, key) is None:
            setattr(args, key, preset[key])
    if args.root_path is None:
        args.root_path = str(PROJECT_ROOT / "datasets" / preset["folder"])
    if args.output_dir is None:
        suffix = "custom" if args.labeled_list else str(args.labeled_num)
        args.output_dir = str(PROJECT_ROOT / "outputs" / "PU-Diff" / f"{preset['folder']}_{suffix}_labeled")
    for key in ("root_path", "output_dir", "labeled_list", "resume"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, str(Path(value).expanduser().resolve()))
    return args


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2 ** 32)
    random.seed(seed)
    np.random.seed(seed)


def batch_stream(dataset, indices, batch_size, workers, pin_memory, generator):
    subset = Subset(dataset, indices)
    # Replacement sampling keeps all batches full even with a very small labeled set.
    num_samples = max(batch_size, ((len(subset) + batch_size - 1) // batch_size) * batch_size)
    loader = DataLoader(subset, batch_size=batch_size,
                        sampler=RandomSampler(subset, replacement=True, num_samples=num_samples,
                                              generator=generator),
                        num_workers=workers, worker_init_fn=seed_worker, pin_memory=pin_memory,
                        generator=generator, persistent_workers=workers > 0)
    while True:
        yield from loader


def export_segmentation(path, model, config):
    # This file has everything inference needs; SPUQ and UADR are excluded.
    atomic_torch_save({"format_version": CHECKPOINT_VERSION,
                       "model": model.state_dict(), "config": config}, path)


def save_checkpoint(path, iteration, best_dice, model, spuq, uadr, optimizer,
                    diffusion_optimizer, config, partition, stream_generators):
    atomic_torch_save({
        "format_version": CHECKPOINT_VERSION, "partition": partition,
        "stream_states": {key: generator.get_state() for key, generator in stream_generators.items()},
        "iteration": iteration, "best_dice": best_dice, "config": config,
        "model": model.state_dict(), "spuq": spuq.state_dict(), "uadr": uadr.state_dict(),
        "optimizer": optimizer.state_dict(), "diffusion_optimizer": diffusion_optimizer.state_dict(),
        "random_state": random.getstate(), "numpy_state": np.random.get_state(),
        "torch_state": torch.get_rng_state(),
        "cuda_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, path)


def restore_checkpoint(path, model, spuq, uadr, optimizer, diffusion_optimizer, config,
                       device, partition, stream_generators):
    checkpoint = load_checkpoint(path)
    required = ("iteration", "spuq", "uadr", "optimizer", "diffusion_optimizer", "random_state",
                "numpy_state", "torch_state", "cuda_state", "best_dice")
    if any(key not in checkpoint for key in required):
        raise ValueError("--resume needs a full training checkpoint, not exported U-Net weights")
    # Refuse to silently resume a different objective, dataset or architecture.
    flexible = {"resume", "output_dir", "device", "num_workers", "max_iterations",
                "val_interval", "save_interval", "root_path", "labeled_list", "inference_batch_size"}
    for key, value in checkpoint["config"].items():
        if key not in flexible and config.get(key) != value:
            raise ValueError(f"Resume configuration differs: {key}={config.get(key)!r}, saved={value!r}")
    previous_partition = checkpoint.get("partition")
    if previous_partition is None:
        previous_partition = json.loads((Path(path).parent / "partition.json").read_text(encoding="utf-8"))
    if previous_partition != partition:
        raise ValueError("Dataset ordering or labeled partition changed since the checkpoint")
    model.load_state_dict(checkpoint["model"])
    spuq.load_state_dict(checkpoint["spuq"])
    uadr.load_state_dict(checkpoint["uadr"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    diffusion_optimizer.load_state_dict(checkpoint["diffusion_optimizer"])
    for opt in (optimizer, diffusion_optimizer):
        for state in opt.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and key != "step":
                    state[key] = value.to(device)
    random.setstate(checkpoint["random_state"])
    np.random.set_state(checkpoint["numpy_state"])
    torch.set_rng_state(checkpoint["torch_state"])
    if checkpoint["cuda_state"] is not None and torch.cuda.is_available():
        for index, state in enumerate(checkpoint["cuda_state"][:torch.cuda.device_count()]):
            torch.cuda.set_rng_state(state, index)
    for key, state in checkpoint.get("stream_states", {}).items():
        if key in stream_generators:
            stream_generators[key].set_state(state)
    return checkpoint["iteration"], checkpoint["best_dice"]


def train(args):
    if not 0 < args.labeled_bs < args.batch_size:
        raise ValueError("labeled_bs must be between 0 and batch_size")
    if args.max_iterations <= 0 or args.warmup_iterations < 0:
        raise ValueError("Invalid iteration counts")
    if args.num_classes < 2 or args.num_classes > 65536 or not 0 < args.conf_thresh <= 1 or not 0 <= args.ema_decay < 1:
        raise ValueError("Invalid class count, confidence threshold or EMA momentum")
    if args.diffusion_steps < 2 or args.latent_size < 8 or args.latent_size % 8:
        raise ValueError("Diffusion needs at least two steps; latent_size must be divisible by 8")
    if min(args.patch_size) < 32 or args.val_interval < 0 or args.save_interval <= 0:
        raise ValueError("Invalid patch size or saving/validation interval")
    if args.data_format == "images" and (args.img_channels not in (1, 3) or args.num_classes != 2):
        raise ValueError("The images/masks layout supports binary grayscale/RGB segmentation")
    if min(args.lambda_unsup, args.lambda_lat, args.lambda_rect) < 0:
        raise ValueError("Loss weights must be nonnegative")
    if args.base_lr <= 0 or args.diffusion_lr <= 0 or args.weight_decay < 0:
        raise ValueError("Learning rates must be positive; weight_decay must be nonnegative")
    if args.num_workers < 0 or args.img_channels < 1 or not 0 <= args.max_rotation <= 180:
        raise ValueError("Invalid worker/channel count or rotation range")
    if args.inference_batch_size <= 0:
        raise ValueError("inference_batch_size must be positive")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    output = Path(args.output_dir)
    if not args.resume and (output / "config.json").exists():
        raise FileExistsError(f"Training output exists: {output}. Use --resume or a new --output_dir")
    validation_list = Path(args.root_path) / "val.list"
    if args.val_interval and (not validation_list.exists() or not validation_list.read_text(encoding="utf-8-sig").strip()):
        raise ValueError("Training validation requires a nonempty val.list; use --val_interval 0 otherwise")
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[
        logging.FileHandler(output / "train.log", encoding="utf-8"), logging.StreamHandler()], force=True)
    config = vars(args).copy()
    names, labeled_indices, unlabeled_indices = training_partition(
        args.root_path, args.data_format, args.labeled_num, args.labeled_list)
    dataset = TrainingDataset(args.root_path, names, labeled_indices, args.img_channels,
                              args.data_format, args.patch_size, args.max_rotation, args.num_classes)
    # Persist the actual partition, including ordering, for publication/reproduction.
    partition = {"labeled": [names[i] for i in labeled_indices],
                 "unlabeled": [names[i] for i in unlabeled_indices]}

    model = UNet(args.img_channels, args.num_classes).to(device)
    spuq = SPUQ(args.num_classes, model.feature_channels, args.conf_thresh, args.ema_decay).to(device)
    uadr = UADR(args.img_channels, args.num_classes, args.diffusion_steps, args.latent_size).to(device)
    # Sec. 4.2 explicitly specifies SGD for the main network.
    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9,
                                weight_decay=args.weight_decay)
    # The paper does not specify the auxiliary optimizer; this choice is documented.
    diffusion_optimizer = torch.optim.Adam(uadr.parameters(), lr=args.diffusion_lr)
    start, best_dice = 0, -1.0
    stream_generators = {name: torch.Generator().manual_seed(args.seed + offset)
                         for offset, name in enumerate(("warmup", "labeled", "unlabeled"), start=1)}
    if args.resume:
        start, best_dice = restore_checkpoint(args.resume, model, spuq, uadr, optimizer,
                                               diffusion_optimizer, config, device,
                                               partition, stream_generators)
    if start >= args.max_iterations:
        raise ValueError(f"Checkpoint is already at iteration {start}; increase --max_iterations to continue")
    write_json(output / "partition.json", partition)
    write_json(output / "config.json", config)
    warmup_batches = batch_stream(dataset, labeled_indices, args.batch_size,
                                  args.num_workers, device.type == "cuda", stream_generators["warmup"])
    labeled_batches = batch_stream(dataset, labeled_indices, args.labeled_bs,
                                   args.num_workers, device.type == "cuda", stream_generators["labeled"])
    unlabeled_batches = batch_stream(dataset, unlabeled_indices, args.batch_size - args.labeled_bs,
                                     args.num_workers, device.type == "cuda", stream_generators["unlabeled"])
    writer = SummaryWriter(str(output / "tensorboard"), purge_step=start + 1 if args.resume else None)
    model.train()
    uadr.train()
    logging.info("PU-Diff: %d labeled slices/images, %d unlabeled; config=%s",
                 len(labeled_indices), len(unlabeled_indices), config)
    try:
        for iteration in trange(start + 1, args.max_iterations + 1, desc="PU-Diff"):
            supervised_only = iteration <= args.warmup_iterations
            if iteration == args.warmup_iterations + 1:
                warmup_batches.close()
            labeled = next(warmup_batches if supervised_only else labeled_batches)
            labeled_image = labeled["image"].to(device, non_blocking=True)
            labeled_target = labeled["label"].to(device, non_blocking=True)
            labeled_valid = labeled["valid_mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            diffusion_optimizer.zero_grad(set_to_none=True)
            loss_sup = segmentation_loss(model(labeled_image), labeled_target, labeled_valid)
            loss_unsup = loss_sup.new_zeros(())
            loss_lat = loss_sup.new_zeros(())
            loss_rect = loss_sup.new_zeros(())

            if not supervised_only:
                unlabeled = next(unlabeled_batches)
                raw = unlabeled["image"].to(device, non_blocking=True)
                weak = unlabeled["image_weak"].to(device, non_blocking=True)
                strong = unlabeled["image_strong"].to(device, non_blocking=True)
                weak_logits, features = model(weak, return_features=True)
                strong_logits = model(strong)
                raw_valid = unlabeled["valid_mask"].to(device, non_blocking=True)
                weak_valid = flip_coordinates(raw_valid, unlabeled["horizontal"], unlabeled["vertical"])
                quantified = spuq(features, weak_logits.detach().softmax(dim=1), weak_valid)
                strong_target, strong_valid = weak_targets_to_strong(
                    quantified["labels"], unlabeled["angle"], weak_valid)
                # Eq. (13): strong student prediction learns from the aligned weak target.
                loss_unsup = segmentation_loss(strong_logits, strong_target, strong_valid)

                # Invert weak flips so UADR receives the unaugmented Xu and aligned maps.
                def to_raw(tensor):
                    return flip_coordinates(tensor, unlabeled["horizontal"], unlabeled["vertical"])

                uncertainty = to_raw(quantified["uncertainty"])
                low_labels = to_raw(quantified["low_labels"])
                loss_lat = uadr(raw, uncertainty, low_labels)
                # Sampling has no graph; Eq. (15) updates only the main segmentation net.
                corrected_raw = uadr.rectify(raw, uncertainty, low_labels,
                                             to_raw(quantified["high_mask"]),
                                             to_raw(quantified["labels"]))
                corrected_weak = to_raw(corrected_raw)  # flip is its own inverse
                loss_rect = segmentation_loss(weak_logits, corrected_weak, weak_valid)

            total = (loss_sup + args.lambda_unsup * loss_unsup +
                     args.lambda_lat * loss_lat + args.lambda_rect * loss_rect)
            if not torch.isfinite(total).item():
                raise FloatingPointError(f"Non-finite total loss at iteration {iteration}")
            total.backward()
            optimizer.step()
            if not supervised_only:
                diffusion_optimizer.step()
            terms = {"total": total.item(), "sup": loss_sup.item(), "unsup": loss_unsup.item(),
                     "lat": loss_lat.item(), "rect": loss_rect.item()}
            for name, value in terms.items():
                writer.add_scalar(f"loss/{name}", value, iteration)
            if iteration % 20 == 0 or iteration == 1:
                logging.info("iteration=%d warmup=%s losses=%s", iteration, supervised_only, terms)

            if args.val_interval and iteration % args.val_interval == 0:
                metrics = evaluate(model, args.root_path, "val", args.num_classes,
                                   args.img_channels, device, args.patch_size, args.data_format,
                                   batch_size=args.inference_batch_size)
                write_json(output / "validation_last.json", {"iteration": iteration, **metrics})
                dice = metrics["mean"][0]
                for name, value in zip(metrics["metric_order"], metrics["mean"]):
                    writer.add_scalar(f"validation/{name}", value, iteration)
                logging.info("validation iteration=%d mean=%s units=%s",
                             iteration, metrics["mean"], metrics["distance_unit"])
                if dice > best_dice:
                    best_dice = dice
                    export_segmentation(output / "unet_best.pth", model, config)
                    write_json(output / "validation_best.json", {"iteration": iteration, **metrics})

            if iteration % args.save_interval == 0 or iteration == args.max_iterations:
                save_checkpoint(output / "checkpoint_last.pth", iteration, best_dice, model,
                                spuq, uadr, optimizer, diffusion_optimizer, config, partition, stream_generators)
                export_segmentation(output / "unet_last.pth", model, config)
    finally:
        for stream in (warmup_batches, labeled_batches, unlabeled_batches):
            stream.close()
        writer.close()


if __name__ == "__main__":
    train(parse_training_args())
