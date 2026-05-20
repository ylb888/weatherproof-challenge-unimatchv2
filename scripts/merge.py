import argparse
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


WEATHERPROOF_PALETTE = np.array(
    [
        [0, 0, 0],
        [180, 120, 120],
        [120, 120, 180],
        [128, 64, 128],
        [70, 130, 180],
        [112, 112, 112],
        [107, 142, 35],
        [152, 251, 152],
        [230, 230, 230],
        [34, 139, 34],
    ],
    dtype=np.uint8,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge WeatherProof prediction masks by pixel-wise weighted voting."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="prediction directories, or their mask subdirectories",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="optional weights, one per input directory; defaults to equal weights",
    )
    parser.add_argument(
        "--output",
        default="output/weatherproof/merged_predictions",
        help="output prediction directory",
    )
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="only save label masks, not colored masks",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="raise an error if any input misses a mask from the first input",
    )
    return parser.parse_args()


def mask_root(path):
    path = Path(path)
    if (path / "mask").is_dir():
        return path / "mask"
    return path


def list_masks(path):
    root = mask_root(path)
    masks = {
        mask.relative_to(root): mask
        for mask in root.rglob("*.png")
        if not mask.name.endswith("_color.png")
    }
    if not masks:
        raise FileNotFoundError(f"No mask png files found under {root}")
    return root, masks


def normalize_weights(weights, n_inputs):
    if weights is None:
        weights = [1.0] * n_inputs
    if len(weights) != n_inputs:
        raise ValueError(f"Expected {n_inputs} weights, got {len(weights)}")
    weights = np.array(weights, dtype=np.float32)
    if np.any(weights < 0):
        raise ValueError("Weights must be non-negative")
    if weights.sum() <= 0:
        raise ValueError("At least one weight must be positive")
    return weights


def rgb_to_labels(rgb, num_classes):
    rgb = rgb[..., :3].astype(np.uint8)
    labels = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    for cls in range(num_classes):
        color = WEATHERPROOF_PALETTE[cls]
        labels[np.all(rgb == color, axis=-1)] = cls
    if np.any(labels == 255):
        unknown = rgb[labels == 255][0]
        raise ValueError(f"Unknown RGB color in mask: {unknown.tolist()}")
    return labels


def load_mask(path, num_classes):
    image = Image.open(path)
    arr = np.array(image, dtype=np.uint8)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        return rgb_to_labels(arr, num_classes)
    raise ValueError(f"Unsupported mask format for {path}: mode={image.mode}, shape={arr.shape}")


def merge_one(rel_path, mask_sets, weights, num_classes, strict):
    preds = []
    used_weights = []
    missing = []

    for idx, masks in enumerate(mask_sets):
        path = masks.get(rel_path)
        if path is None:
            missing.append(idx)
            continue
        preds.append(load_mask(path, num_classes))
        used_weights.append(weights[idx])

    if missing and strict:
        raise FileNotFoundError(f"{rel_path} missing in input indices {missing}")
    if not preds:
        raise FileNotFoundError(f"{rel_path} missing in all candidate inputs")

    shape = preds[0].shape
    for pred in preds[1:]:
        if pred.shape != shape:
            raise ValueError(f"Shape mismatch for {rel_path}: {[p.shape for p in preds]}")

    scores = np.zeros((num_classes,) + shape, dtype=np.float32)
    for pred, weight in zip(preds, used_weights):
        valid = pred < num_classes
        for cls in range(num_classes):
            scores[cls] += weight * ((pred == cls) & valid)

    return scores.argmax(axis=0).astype(np.uint8)


def save_prediction(pred, rel_path, output_root, save_color):
    mask_path = output_root / "mask" / rel_path
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pred).save(mask_path)

    if save_color:
        color_path = output_root / "color" / f"{rel_path.with_suffix('')}_color.png"
        color_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(WEATHERPROOF_PALETTE[pred]).save(color_path)


def main():
    args = parse_args()
    roots_and_masks = [list_masks(path) for path in args.inputs]
    roots = [item[0] for item in roots_and_masks]
    mask_sets = [item[1] for item in roots_and_masks]
    weights = normalize_weights(args.weights, len(mask_sets))
    output_root = Path(args.output)

    rel_paths = sorted(mask_sets[0])
    progress = rel_paths
    if tqdm is not None:
        progress = tqdm(rel_paths, desc="Merging masks", unit="mask")

    merged = 0
    for rel_path in progress:
        pred = merge_one(rel_path, mask_sets, weights, args.num_classes, args.strict)
        save_prediction(pred, rel_path, output_root, save_color=not args.no_color)
        merged += 1

    print("Inputs:")
    for root, weight in zip(roots, weights):
        print(f"  weight={weight:g}  {root}")
    print(f"Merged masks: {merged}")
    print(f"Output: {output_root}")


if __name__ == "__main__":
    main()


# python scripts/merge.py \
#   --inputs \
#     /root/data1/ylb/project/CodeX/UniMatch-V2/output/weatherproof/test_predictions_0.79/mask \
#     /root/data1/ylb/project/CodeX/UniMatch-V2/output/weatherproof/test_predictions/mask \
#     /root/data1/ylb/project/CodeX/UniMatch-V2/output/weatherproof/test_reference_ade_base \
#   --weights 0.4 0.4 0.2 \
#   --output /root/data1/ylb/project/CodeX/UniMatch-V2/output/weatherproof/merged_0.4_0.4_0.2
