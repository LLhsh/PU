"""Data loading and spatially aligned weak/strong views for PU-Diff."""

import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from dataloaders.io import (as_chw, case_group, case_id, h5_path, image_path,
                            mask_path, read_binary_mask, read_h5, read_image,
                            read_list, validate_label)


def training_partition(root, data_format, labeled_num, labeled_list=None):
    root = Path(root)
    names = read_list(root / ("train_slices.list" if data_format == "h5" else "train.list"))
    if len({case_id(name) for name in names}) != len(names):
        raise ValueError("Training entries must have unique case/slice identifiers")
    source = read_list(root / "train.list") if (root / "train.list").exists() else names
    train_groups = {case_group(name) for name in source}
    if not {case_group(name) for name in names}.issubset(train_groups):
        raise ValueError("train_slices.list contains cases outside train.list")
    if data_format == "h5" and (root / "train.list").exists():
        listed_cases = {case_id(name) for name in source}
        slice_parents = {case_id(name).split("_slice_")[0] for name in names}
        if not slice_parents.issubset(listed_cases):
            raise ValueError("Training slices reference volume names absent from train.list")
    seen_groups = set(train_groups)
    for split in ("val", "test"):
        path = root / f"{split}.list"
        if path.is_file() and path.read_text(encoding="utf-8-sig").strip():
            current_groups = {case_group(name) for name in read_list(path)}
            overlap = seen_groups & current_groups
            if overlap:
                raise ValueError(f"Patients/cases in {split} overlap a preceding split: {sorted(overlap)}")
            seen_groups.update(current_groups)
    if labeled_list:
        selected = {case_id(name) for name in read_list(labeled_list)}
        available = {case_id(name) for name in names} | {case_id(name).split("_slice_")[0] for name in names}
        available |= {case_group(name) for name in names}
        unknown = selected - available
        if unknown:
            raise ValueError(f"Labeled entries do not belong to training data: {sorted(unknown)}")
        labeled = [i for i, name in enumerate(names)
                   if case_id(name) in selected or case_id(name).split("_slice_")[0] in selected
                   or case_group(name) in selected]
    else:
        # Use train.list ordering, rather than an assumed number of slices per patient.
        groups = list(dict.fromkeys(case_group(name) for name in source))
        if not 0 < labeled_num < len(groups):
            raise ValueError("labeled_num must leave both labeled and unlabeled training cases")
        selected = set(groups[:labeled_num])
        labeled = [i for i, name in enumerate(names) if case_group(name) in selected]
    chosen = set(labeled)
    unlabeled = [i for i in range(len(names)) if i not in chosen]
    if not labeled or not unlabeled:
        raise ValueError("The training partition needs labeled and unlabeled samples")
    return names, labeled, unlabeled


def flip_coordinates(tensor, horizontal, vertical):
    """Apply/invert the weak flips, for [B,C,H,W] or [B,H,W] tensors."""
    shape = (tensor.shape[0],) + (1,) * (tensor.ndim - 1)
    horizontal = torch.as_tensor(horizontal, dtype=torch.bool, device=tensor.device).reshape(shape)
    vertical = torch.as_tensor(vertical, dtype=torch.bool, device=tensor.device).reshape(shape)
    tensor = torch.where(horizontal, tensor.flip(-1), tensor)
    return torch.where(vertical, tensor.flip(-2), tensor)


def weak_targets_to_strong(labels, angles, weak_valid=None):
    """Transport weak targets to strong rotation coordinates and mask new padding."""
    targets, masks = [], []
    if weak_valid is None:
        weak_valid = torch.ones_like(labels, dtype=torch.bool)
    for label, angle, valid in zip(labels, angles, weak_valid):
        targets.append(TF.rotate(label[None].float(), float(angle),
                                 interpolation=InterpolationMode.NEAREST)[0].long())
        masks.append(TF.rotate(valid[None].float(), float(angle),
                               interpolation=InterpolationMode.NEAREST)[0].bool())
    return torch.stack(targets), torch.stack(masks)


class WeakStrongAugment:
    def __init__(self, patch_size=(256, 256), max_rotation=180.0):
        self.patch_size = tuple(patch_size)
        self.max_rotation = max_rotation
        self.jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                             saturation=0.4, hue=0.1)

    def __call__(self, image, label, generate_views=True):
        image = torch.from_numpy(image.copy()).float()
        label = torch.from_numpy(label.copy()).long()
        height, width = image.shape[-2:]
        valid = torch.ones((height, width), dtype=torch.bool)
        padding = (0, max(0, self.patch_size[1] - width),
                   0, max(0, self.patch_size[0] - height))
        image, label = F.pad(image, padding), F.pad(label, padding)
        valid = F.pad(valid, padding, value=False)
        top = random.randint(0, image.shape[-2] - self.patch_size[0])
        left = random.randint(0, image.shape[-1] - self.patch_size[1])
        image = image[:, top:top + self.patch_size[0], left:left + self.patch_size[1]]
        label = label[top:top + self.patch_size[0], left:left + self.patch_size[1]]
        valid = valid[top:top + self.patch_size[0], left:left + self.patch_size[1]]
        if not generate_views:
            return {"image": image, "label": label, "valid_mask": valid}
        horizontal, vertical = random.random() < 0.5, random.random() < 0.5
        weak = image.clone()
        if horizontal:
            weak = weak.flip(-1)
        if vertical:
            weak = weak.flip(-2)
        # Strong geometric perturbation is explicitly recorded for target alignment.
        angle = random.uniform(-self.max_rotation, self.max_rotation)
        strong = TF.rotate(weak, angle, interpolation=InterpolationMode.BILINEAR)
        if strong.shape[0] in (1, 3):
            strong = self.jitter(strong)
        else:
            # For multi-sequence MRI, apply brightness/contrast per scalar channel.
            strong = torch.cat([self.jitter(channel[None]) for channel in strong], dim=0)
        if random.random() < 0.5:
            strong = TF.gaussian_blur(strong, kernel_size=5, sigma=random.uniform(0.1, 2.0))
        return {"image": image, "label": label, "valid_mask": valid, "image_weak": weak,
                "image_strong": strong.clamp(0, 1), "horizontal": horizontal,
                "vertical": vertical, "angle": angle}


class TrainingDataset(Dataset):
    def __init__(self, root, names, labeled_indices, image_channels=1,
                 data_format="h5", patch_size=(256, 256), max_rotation=180.0, num_classes=4):
        self.root = Path(root)
        self.names = names
        self.labeled = set(labeled_indices)
        self.image_channels = image_channels
        self.data_format = data_format
        self.num_classes = num_classes
        self.augment = WeakStrongAugment(patch_size, max_rotation)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        if self.data_format == "h5":
            # Unlabeled HDF5 files need not contain a label dataset at all.
            image, label, _ = read_h5(h5_path(self.root, name, slices=True), index in self.labeled)
            image = as_chw(image, self.image_channels)
        else:
            image = read_image(image_path(self.root, name), self.image_channels)
            label = read_binary_mask(mask_path(self.root, name)) if index in self.labeled else None
        if label is None:
            label = np.zeros(image.shape[-2:], dtype=np.int64)
        else:
            label = validate_label(label, image.shape[-2:], self.num_classes, name)
        sample = self.augment(image.astype(np.float32), label, generate_views=index not in self.labeled)
        sample["case"] = name
        return sample
