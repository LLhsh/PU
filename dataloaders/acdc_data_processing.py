"""Convert NIfTI images/labels into the HDF5 layout used by PU-Diff.

This command only prepares data. It does not create or randomize dataset splits.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import SimpleITK as sitk


def nifti_stem(path):
    return path.name[:-7] if path.name.lower().endswith(".nii.gz") else path.stem


def main():
    parser = argparse.ArgumentParser(description="Convert segmentation NIfTI volumes to HDF5")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--label_dir", default=None, help="defaults to image_dir; supports colocated ACDC labels")
    parser.add_argument("--output_dir", required=True, help="dataset root containing split lists")
    parser.add_argument("--label_suffix", default="_gt")
    parser.add_argument("--img_channels", type=int, default=1)
    parser.add_argument("--recursive", action="store_true", help="read nested patient directories")
    parser.add_argument("--allow_unlabeled", action="store_true", help="write image-only HDF5 when a label is absent")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.img_channels <= 0:
        raise ValueError("img_channels must be positive")
    image_root = Path(args.image_dir).expanduser().resolve()
    label_root = Path(args.label_dir).expanduser().resolve() if args.label_dir else image_root
    output = Path(args.output_dir) / "data"
    (output / "slices").mkdir(parents=True, exist_ok=True)
    candidates = image_root.rglob("*") if args.recursive else image_root.iterdir()
    images = sorted(p for p in candidates if p.is_file() and p.name.lower().endswith((".nii", ".nii.gz"))
                    and not nifti_stem(p).endswith("_4d")
                    and not (args.label_suffix and nifti_stem(p).endswith(args.label_suffix)))
    if not images:
        raise ValueError(f"No NIfTI images found under {image_root}")
    if len({nifti_stem(path) for path in images}) != len(images):
        raise ValueError("Duplicate volume names across input directories")
    for path in images:
        case = nifti_stem(path)
        relative_parent = path.parent.relative_to(image_root)
        candidates = [parent / f"{case}{args.label_suffix}{ext}"
                      for parent in (label_root / relative_parent, label_root)
                      for ext in (".nii.gz", ".nii")]
        label_path = next((candidate for candidate in candidates if candidate.exists()), None)
        if label_path is not None and label_path.resolve() == path.resolve():
            raise ValueError("The image and label paths are identical; provide a separate --label_dir")
        if label_path is None and not args.allow_unlabeled:
            raise FileNotFoundError(f"Missing label for {case}")
        image_itk = sitk.ReadImage(str(path))
        label_itk = sitk.ReadImage(str(label_path)) if label_path is not None else None
        image = sitk.GetArrayFromImage(image_itk).astype(np.float32)
        label = sitk.GetArrayFromImage(label_itk) if label_itk is not None else None
        if args.img_channels > 1:
            if image.ndim == 4 and image.shape[-1] == args.img_channels:
                image = image.transpose(0, 3, 1, 2)  # vector image: D,H,W,C -> D,C,H,W
            elif image.ndim == 4 and image.shape[0] == args.img_channels:
                image = image.transpose(1, 0, 2, 3)  # scalar 4D: C,D,H,W -> D,C,H,W
            else:
                raise ValueError(f"Unsupported multi-channel volume layout: {image.shape}")
        spatial_shape = image.shape if args.img_channels == 1 else (image.shape[0], *image.shape[-2:])
        if len(spatial_shape) != 3 or (label is not None and tuple(spatial_shape) != label.shape):
            raise ValueError(f"Image/label shape mismatch for {case}: {image.shape}, {None if label is None else label.shape}")
        if not np.isfinite(image).all():
            raise ValueError(f"Non-finite image values for {case}")
        if label is not None:
            if not np.isfinite(label).all() or np.any(label < 0) or not np.equal(label, np.floor(label)).all():
                raise ValueError(f"Invalid class indices for {case}")
            label = label.astype(np.int64)
        image_spacing = np.asarray(image_itk.GetSpacing()[:3])
        if label_itk is not None and not np.allclose(image_spacing, label_itk.GetSpacing()[:3]):
            raise ValueError(f"Image/label spacing mismatch for {case}")
        if label_itk is not None and not np.allclose(image_itk.GetOrigin()[:3], label_itk.GetOrigin()[:3]):
            raise ValueError(f"Image/label origin mismatch for {case}")
        image_direction = np.asarray(image_itk.GetDirection()).reshape(image_itk.GetDimension(), -1)
        if label_itk is not None:
            label_direction = np.asarray(label_itk.GetDirection()).reshape(label_itk.GetDimension(), -1)
            if not np.allclose(image_direction[:3, :3], label_direction[:3, :3]):
                raise ValueError(f"Image/label direction mismatch for {case}")
        destinations = [output / f"{case}.h5"]
        destinations += [output / "slices" / f"{case}_slice_{index}.h5" for index in range(image.shape[0])]
        if not args.overwrite and any(destination.exists() for destination in destinations):
            raise FileExistsError(f"HDF5 output already exists for {case}; use --overwrite to replace it")
        image = (image - image.min()) / max(float(image.max() - image.min()), 1e-8)
        with h5py.File(output / f"{case}.h5", "w") as destination:
            destination.create_dataset("image", data=image, compression="gzip")
            if label is not None:
                destination.create_dataset("label", data=label, compression="gzip")
            destination.create_dataset("spacing", data=image_spacing[::-1])  # array order: z,y,x
            destination.create_dataset("origin", data=image_itk.GetOrigin()[:3])
            destination.create_dataset("direction", data=image_direction[:3, :3])
            destination.attrs["is_volume"] = True
        for index, section in enumerate(image):
            with h5py.File(output / "slices" / f"{case}_slice_{index}.h5", "w") as destination:
                destination.create_dataset("image", data=section, compression="gzip")
                if label is not None:
                    destination.create_dataset("label", data=label[index], compression="gzip")
                destination.attrs["is_volume"] = False
        print(f"Converted {case}: {image.shape[0]} slices, labeled={label is not None}")


if __name__ == "__main__":
    main()
