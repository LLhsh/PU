"""Shared runtime, serialization and checkpoint helpers; no training side effects."""

import json
import math
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_VERSION = 2


def resolve_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Install a CUDA-enabled PyTorch or use --device cpu.")
        index = torch.cuda.current_device() if device.index is None else device.index
        if not 0 <= index < torch.cuda.device_count():
            raise ValueError(f"CUDA device index {index} is unavailable")
        torch.cuda.set_device(index)
        device = torch.device("cuda", index)
    elif device.type != "cpu":
        raise ValueError("Supported devices are auto, cpu, cuda and cuda:N")
    return device


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, filename = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    os.close(descriptor)
    temporary = Path(filename)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(path):
    # Full training checkpoints contain Python/NumPy RNG states, not just tensors.
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint or "config" not in checkpoint:
        raise ValueError("Expected a PU-Diff checkpoint containing model and config")
    version = checkpoint.get("format_version", 1)
    if version > CHECKPOINT_VERSION:
        raise ValueError(f"Checkpoint format {version} is newer than supported format {CHECKPOINT_VERSION}")
    return checkpoint


def load_segmenter(path, device):
    from networks.unet import UNet

    checkpoint = load_checkpoint(path)
    config = checkpoint["config"]
    required = ("img_channels", "num_classes", "patch_size", "data_format")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Checkpoint configuration is missing: {missing}")
    model = UNet(config["img_channels"], config["num_classes"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, config
