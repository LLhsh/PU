"""Shared HDF5/image IO, normalization and shape contracts."""

import re
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


def read_list(path):
    path = Path(path)
    with path.open(encoding="utf-8-sig") as stream:
        values = [line.strip() for line in stream if line.strip() and not line.lstrip().startswith("#")]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate entries in {path}")
    if not values:
        raise ValueError(f"Empty split list: {path}")
    return values


def case_id(name):
    name = Path(name).name
    for suffix in (".nii.gz", ".hdf5", ".h5", ".nii", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"):
        if name.lower().endswith(suffix):
            return name[:-len(suffix)]
    return name


def case_group(name):
    name = case_id(name).split("_slice_")[0]
    match = re.match(r"patient\d+", name, flags=re.IGNORECASE)
    return match.group(0).lower() if match else re.sub(r"_frame\d+$", "", name)


def minmax_normalize(image):
    image = np.asarray(image, dtype=np.float32)
    if image.size == 0 or not np.isfinite(image).all():
        raise ValueError("Image must be non-empty and contain only finite values")
    minimum, maximum = float(image.min()), float(image.max())
    if minimum >= 0 and maximum <= 1:
        return image
    return (image - minimum) / max(maximum - minimum, 1e-8)


def as_chw(image, channels):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2 and channels == 1:
        image = image[None]
    elif image.ndim == 3 and image.shape[0] == channels:
        pass
    elif image.ndim == 3 and image.shape[-1] == channels:
        image = image.transpose(2, 0, 1)
    else:
        raise ValueError(f"Expected a 2D image with {channels} channels, got {image.shape}")
    return np.ascontiguousarray(image)


def as_volume(image, channels):
    image = np.asarray(image, dtype=np.float32)
    if channels == 1 and image.ndim == 3:
        return image[:, None]
    if image.ndim == 4 and image.shape[1] == channels:
        return image
    if image.ndim == 4 and image.shape[-1] == channels:
        return image.transpose(0, 3, 1, 2)
    raise ValueError(f"Expected [D,H,W], [D,C,H,W] or [D,H,W,C], got {image.shape}")


def validate_label(label, spatial_shape, num_classes, source):
    label = np.asarray(label)
    if tuple(label.shape) != tuple(spatial_shape):
        raise ValueError(f"Image/label shape mismatch in {source}: {spatial_shape} vs {label.shape}")
    if not np.isfinite(label).all() or not np.equal(label, np.floor(label)).all():
        raise ValueError(f"Labels must contain finite integer class indices: {source}")
    if label.size == 0 or label.min() < 0 or label.max() >= num_classes:
        raise ValueError(f"Labels must be in [0,{num_classes - 1}]: {source}")
    return label.astype(np.int64)


def h5_path(root, name, slices=False):
    directory = Path(root) / "data"
    filenames = [str(name)] if str(name).lower().endswith((".h5", ".hdf5")) else [f"{name}.h5", f"{name}.hdf5"]
    directories = [directory / "slices", directory] if slices else [directory]
    candidates = [parent / filename for parent in directories for filename in filenames]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"HDF5 file not found; expected one of: {candidates}")


def image_path(root, name):
    path = Path(root) / "images" / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def mask_path(root, name):
    directory = Path(root) / "masks"
    # Prefer lossless PNG masks, while accepting original same-name JPEG layouts.
    candidates = [directory / f"{case_id(name)}{suffix}" for suffix in (".png", ".tif", ".tiff", ".bmp")]
    candidates.append(directory / Path(name).name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No mask found for {name} under {directory}")


def read_image(path, channels):
    if channels not in (1, 3):
        raise ValueError("Image files support one or three channels; use HDF5 for multi-sequence MRI")
    with Image.open(path) as source:
        image = np.asarray(source.convert("RGB" if channels == 3 else "L"))
    return as_chw(image, channels) / 255.0


def read_binary_mask(path):
    with Image.open(path) as source:
        # Handle both binary 0/1 and 0/255 masks; tolerate JPEG compression near zero.
        mask = np.asarray(source.convert("L"))
    threshold = 0 if mask.max() <= 1 else 127
    return (mask > threshold).astype(np.int64)


def read_h5(path, include_label=True):
    with h5py.File(path, "r") as source:
        image = source["image"][:]
        label = source["label"][:] if include_label else None
        metadata = {}
        for key in ("spacing", "origin", "direction"):
            if key in source:
                metadata[key] = np.asarray(source[key][:])
            elif key in source.attrs:
                metadata[key] = np.asarray(source.attrs[key])
        metadata["is_volume"] = bool(source.attrs["is_volume"]) if "is_volume" in source.attrs else None
    return minmax_normalize(image), label, metadata
