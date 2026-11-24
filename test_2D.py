"""Evaluate the exported main U-Net; no SPUQ or UADR is instantiated."""

import argparse
import json
from pathlib import Path

from utils.evaluation import evaluate
from utils.runtime import PROJECT_ROOT, json_safe, load_segmenter, resolve_device, write_json


def main():
    parser = argparse.ArgumentParser(description="PU-Diff: segmentation-only evaluation")
    parser.add_argument("--root_path", default=None, help="defaults to the dataset root saved in the checkpoint")
    parser.add_argument("--ckpt", required=True, help="unet_best.pth, unet_last.pth or a full checkpoint")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    device = resolve_device(args.device)
    net, config = load_segmenter(args.ckpt, device)
    root = args.root_path or config.get("root_path", str(PROJECT_ROOT / "datasets" / "ACDC"))
    output = Path(args.output_dir) if args.output_dir else Path(args.ckpt).resolve().parent / f"{args.split}_predictions"
    result = evaluate(net, root, args.split, config["num_classes"],
                      config["img_channels"], device, config["patch_size"],
                      config["data_format"], output, args.batch_size, args.overwrite)
    result["checkpoint"] = str(Path(args.ckpt).resolve())
    result["split"] = args.split
    encoded = json.dumps(json_safe(result), ensure_ascii=False, indent=2, allow_nan=False)
    write_json(output / f"metrics_{args.split}.json", result)
    print(encoded)


if __name__ == "__main__":
    main()
