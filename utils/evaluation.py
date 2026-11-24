"""Foreground DSC, IoU, HD95 and ASD using segmentation-only inference."""

from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from dataloaders.io import (case_id, h5_path, image_path, mask_path, read_binary_mask,
                            read_h5, read_image, read_list, validate_label)
from utils.inference import predict_array, save_prediction


def binary_metrics(prediction, target, spacing):
    prediction, target = prediction.astype(bool), target.astype(bool)
    p, t = prediction.sum(), target.sum()
    if not p and not t:
        return np.array((1.0, 1.0, 0.0, 0.0))
    if not p or not t:
        return np.array((0.0, 0.0, np.inf, np.inf))
    intersection = np.logical_and(prediction, target).sum()
    dice, iou = 2 * intersection / (p + t), intersection / (p + t - intersection)
    structure = ndimage.generate_binary_structure(prediction.ndim, 1)
    pred_surface = prediction ^ ndimage.binary_erosion(prediction, structure=structure)
    true_surface = target ^ ndimage.binary_erosion(target, structure=structure)
    forward = ndimage.distance_transform_edt(~true_surface, sampling=spacing)[pred_surface]
    backward = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)[true_surface]
    hd95 = np.percentile(np.concatenate((forward, backward)), 95)
    return np.array((dice, iou, hd95, forward.mean()))


@torch.no_grad()
def evaluate(net, root, split, num_classes, image_channels, device,
             patch_size=(256, 256), data_format="h5", output_dir=None,
             batch_size=4, overwrite=False):
    cases = read_list(Path(root) / f"{split}.list")
    if len({case_id(case) for case in cases}) != len(cases):
        raise ValueError("Evaluation cases must have unique output identifiers")
    scores, per_case, units = [], {}, {}
    for case in cases:
        if data_format == "h5":
            image, label, metadata = read_h5(h5_path(root, case), include_label=True)
        else:
            image = read_image(image_path(root, case), image_channels)
            label = read_binary_mask(mask_path(root, case))
            metadata = {}
        if label.ndim not in (2, 3):
            raise ValueError(f"Expected a 2D mask or 3D volume for {case}, got {label.shape}")
        prediction = predict_array(net, image, image_channels, device, patch_size,
                                   is_volume=label.ndim == 3, batch_size=batch_size)
        label = validate_label(label, prediction.shape, num_classes, case)
        spacing = metadata.get("spacing")
        if spacing is None:
            spacing = np.ones(label.ndim)
            units[case] = "voxel/pixel"
        else:
            spacing = np.asarray(spacing, dtype=np.float64)
            if spacing.shape != (label.ndim,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
                raise ValueError(f"Invalid spacing for {case}: {spacing}")
            units[case] = "mm"
        case_scores = np.stack([binary_metrics(prediction == cls, label == cls, spacing)
                                for cls in range(1, num_classes)])
        scores.append(case_scores)
        per_case[case] = case_scores.tolist()
        if output_dir:
            save_prediction(prediction, Path(output_dir) / f"{case_id(case)}_prediction",
                             metadata, overwrite=overwrite)
    distance_units = set(units.values())
    # Averaging mm and unit-voxel distances produces an uninterpretable result.
    result = np.stack(scores).mean(axis=0)
    if len(distance_units) > 1:
        result[:, 2:] = np.nan
    return {"per_class": result.tolist(), "mean": result.mean(axis=0).tolist(),
            "per_case": per_case, "metric_order": ["DSC", "IoU", "HD95", "ASD"],
            "foreground_classes": list(range(1, num_classes)),
            "distance_unit": next(iter(distance_units)) if len(distance_units) == 1 else "mixed; distance mean undefined",
            "per_case_distance_unit": units}
