"""Predict images/HDF5/NIfTI without any ground-truth labels or split lists."""

import argparse
from pathlib import Path

from dataloaders.io import case_id
from utils.inference import predict_array, read_prediction_input, save_prediction
from utils.runtime import load_segmenter, resolve_device, write_json


def main():
    parser = argparse.ArgumentParser(description="PU-Diff label-free U-Net prediction")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--input", required=True, help="image, HDF5, NIfTI or a directory of these files")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch_size", type=int, default=4, help="slices per inference batch")
    parser.add_argument("--layout", choices=("auto", "slice", "volume"), default="auto",
                        help="override ambiguous legacy HDF5 dimensions")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    source = Path(args.input).expanduser().resolve()
    if source.is_dir():
        extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".h5", ".hdf5", ".nii", ".nii.gz")
        inputs = sorted(path for path in source.iterdir() if path.is_file() and path.name.lower().endswith(extensions))
    elif source.is_file():
        inputs = [source]
    else:
        raise FileNotFoundError(source)
    if not inputs:
        raise ValueError("No supported input files found")
    if len({case_id(path) for path in inputs}) != len(inputs):
        raise ValueError("Inputs have colliding output names; predict them into separate directories")
    device = resolve_device(args.device)
    net, config = load_segmenter(args.ckpt, device)
    records = []
    for path in inputs:
        image, is_volume, metadata = read_prediction_input(path, config["img_channels"])
        if args.layout != "auto":
            is_volume = args.layout == "volume"
        prediction = predict_array(net, image, config["img_channels"], device,
                                   config["patch_size"], is_volume, args.batch_size)
        files = save_prediction(prediction, Path(args.output_dir) / f"{case_id(path)}_prediction",
                                 metadata, overwrite=args.overwrite)
        records.append({"input": str(path), "shape": list(prediction.shape), "outputs": files})
        print(f"Predicted {path.name}: shape={prediction.shape}")
    write_json(Path(args.output_dir) / "predictions.json",
               {"checkpoint": str(Path(args.ckpt).resolve()), "device": str(device), "cases": records})


if __name__ == "__main__":
    main()
