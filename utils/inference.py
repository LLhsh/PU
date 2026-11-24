"""Label-free prediction shared by evaluation and the standalone predict entry."""

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from dataloaders.io import as_chw, as_volume, minmax_normalize, read_h5, read_image


@torch.no_grad()
def predict_array(net, image, image_channels, device, patch_size=(256, 256),
                  is_volume=False, batch_size=4):
    if batch_size <= 0:
        raise ValueError("Inference batch size must be positive")
    if len(patch_size) != 2 or min(patch_size) < 16:
        raise ValueError("Inference patch_size must contain two dimensions of at least 16 pixels")
    sections = as_volume(image, image_channels) if is_volume else as_chw(image, image_channels)[None]
    if sections.shape[0] == 0:
        raise ValueError("Cannot predict an empty volume")
    original_shape = sections.shape[-2:]
    predictions = []
    was_training = net.training
    net.eval()
    try:
        for start in range(0, sections.shape[0], batch_size):
            tensor = torch.from_numpy(np.ascontiguousarray(sections[start:start + batch_size])).float().to(device)
            tensor = F.interpolate(tensor, size=patch_size, mode="bilinear", align_corners=False)
            logits = net(tensor)
            logits = F.interpolate(logits, size=original_shape, mode="bilinear", align_corners=False)
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    finally:
        net.train(was_training)
    prediction = np.concatenate(predictions, axis=0)
    return prediction if is_volume else prediction[0]


def read_prediction_input(path, image_channels):
    path = Path(path)
    metadata = {}
    if path.suffix.lower() in (".h5", ".hdf5"):
        image, _, metadata = read_h5(path, include_label=False)
        is_volume = metadata.get("is_volume")
        if is_volume is None:
            is_volume = image.ndim == (3 if image_channels == 1 else 4)
        return image, bool(is_volume), metadata
    if path.name.lower().endswith((".nii", ".nii.gz")):
        import SimpleITK as sitk

        itk_image = sitk.ReadImage(str(path))
        image = sitk.GetArrayFromImage(itk_image).astype(np.float32)
        if image_channels > 1:
            if image.ndim == 4 and image.shape[-1] == image_channels:
                image = image.transpose(0, 3, 1, 2)
            elif image.ndim == 4 and image.shape[0] == image_channels:
                image = image.transpose(1, 0, 2, 3)
            else:
                raise ValueError(f"Unsupported multi-channel NIfTI layout: {image.shape}")
        dimensions = itk_image.GetDimension()
        direction = np.asarray(itk_image.GetDirection()).reshape(dimensions, dimensions)[:3, :3]
        metadata.update(spacing=np.asarray(itk_image.GetSpacing()[:3])[::-1],
                        origin=np.asarray(itk_image.GetOrigin()[:3]), direction=direction,
                        source_format="nifti")
        return minmax_normalize(image), True, metadata
    return read_image(path, image_channels), False, metadata


def save_prediction(prediction, prefix, metadata=None, overwrite=False):
    """Write lossless indices. NIfTI geometry is preserved when it is available."""
    from PIL import Image

    prefix = Path(prefix)
    metadata = metadata or {}
    outputs = [Path(str(prefix) + ".npy")]
    if prediction.ndim == 2:
        outputs.append(Path(str(prefix) + ".png"))
    elif all(key in metadata for key in ("spacing", "origin", "direction")):
        spacing = np.asarray(metadata["spacing"], dtype=np.float64)
        origin = np.asarray(metadata["origin"], dtype=np.float64)
        direction = np.asarray(metadata["direction"], dtype=np.float64)
        if (prediction.ndim != 3 or spacing.shape != (3,) or origin.shape != (3,)
                or direction.size != 9 or not np.isfinite(spacing).all()
                or not np.isfinite(origin).all() or not np.isfinite(direction).all()
                or np.any(spacing <= 0)):
            raise ValueError("Expected valid 3D spacing/origin/direction to export a NIfTI prediction")
        outputs.append(Path(str(prefix) + ".nii.gz"))
    if not overwrite:
        existing = [str(path) for path in outputs if path.exists()]
        if existing:
            raise FileExistsError(f"Prediction files already exist: {existing}; choose a new output or --overwrite")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    dtype = np.uint8 if prediction.max() <= 255 else np.uint16
    prediction = prediction.astype(dtype)
    np.save(outputs[0], prediction)
    if prediction.ndim == 2:
        Image.fromarray(prediction).save(outputs[1])
    elif len(outputs) > 1:
        import SimpleITK as sitk

        image = sitk.GetImageFromArray(prediction)
        image.SetSpacing(tuple(float(value) for value in metadata["spacing"][::-1]))
        image.SetOrigin(tuple(float(value) for value in metadata["origin"]))
        image.SetDirection(tuple(float(value) for value in np.asarray(metadata["direction"]).ravel()))
        sitk.WriteImage(image, str(outputs[1]))
    return [str(path) for path in outputs]
