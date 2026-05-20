import argparse
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare WeatherProof prediction masks against a reference prediction set."
    )
    parser.add_argument(
        "--reference",
        default="output/weatherproof/test_predictions",
        help="reference prediction directory, or its mask subdirectory",
    )
    parser.add_argument(
        "--candidates",
        nargs="+",
        default=[
            "output/weatherproof/test_predictions_0.79",
            # "output/weatherproof/test_predictions_5",
            # "output/weatherproof/test_predictions_10",
            # "output/weatherproof/test_predictions_20",
            # "output/weatherproof/test_predictions_30",
            # "output/weatherproof/test_predictions_45",
            # "output/weatherproof/test_predictions_55",
            # "output/weatherproof/test_predictions_60",
        ],
        help="candidate prediction directories, or their mask subdirectories",
    )
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--ignore-index", type=int, default=255)
    return parser.parse_args()


def mask_root(path):
    path = Path(path)
    if (path / "mask").is_dir():
        return path / "mask"
    return path


def list_masks(root):
    root = mask_root(root)
    masks = {
        path.relative_to(root): path
        for path in root.rglob("*.png")
        if not path.name.endswith("_color.png")
    }
    return root, masks


def update_confusion(confusion, pred, target, num_classes, ignore_index):
    valid = target != ignore_index
    valid &= target >= 0
    valid &= target < num_classes
    valid &= pred >= 0
    valid &= pred < num_classes

    encoded = num_classes * target[valid].astype(np.int64) + pred[valid].astype(np.int64)
    confusion += np.bincount(encoded, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def compute_iou(confusion):
    intersection = np.diag(confusion).astype(np.float64)
    union = confusion.sum(axis=1) + confusion.sum(axis=0) - intersection
    iou = np.full_like(intersection, np.nan, dtype=np.float64)
    valid = union > 0
    iou[valid] = intersection[valid] / union[valid]
    miou = np.nanmean(iou)
    return iou, miou


def compare_candidate(reference_masks, candidate_root, candidate_masks, num_classes, ignore_index):
    common = sorted(set(reference_masks) & set(candidate_masks))
    missing_in_candidate = len(reference_masks) - len(common)
    extra_in_candidate = len(candidate_masks) - len(common)
    if not common:
        raise RuntimeError(f"No common mask files found for {candidate_root}")

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    progress = common
    if tqdm is not None:
        progress = tqdm(common, desc=f"Comparing {candidate_root.parent.name}", unit="mask")

    for rel_path in progress:
        ref = np.array(Image.open(reference_masks[rel_path]))
        cand = np.array(Image.open(candidate_masks[rel_path]))
        if ref.shape != cand.shape:
            raise ValueError(
                f"Shape mismatch for {rel_path}: reference {ref.shape}, candidate {cand.shape}"
            )
        update_confusion(confusion, cand, ref, num_classes, ignore_index)

    iou, miou = compute_iou(confusion)
    return {
        "candidate": str(candidate_root),
        "count": len(common),
        "missing": missing_in_candidate,
        "extra": extra_in_candidate,
        "iou": iou,
        "miou": miou,
    }


def main():
    args = parse_args()
    reference_root, reference_masks = list_masks(args.reference)
    if not reference_masks:
        raise FileNotFoundError(f"No reference masks found under {reference_root}")

    results = []
    for candidate in args.candidates:
        candidate_root, candidate_masks = list_masks(candidate)
        if not candidate_masks:
            raise FileNotFoundError(f"No candidate masks found under {candidate_root}")
        results.append(
            compare_candidate(
                reference_masks,
                candidate_root,
                candidate_masks,
                args.num_classes,
                args.ignore_index,
            )
        )

    results.sort(key=lambda item: item["miou"], reverse=True)

    print(f"Reference: {reference_root}")
    print("")
    for result in results:
        iou_text = " ".join(
            f"{idx}:{value * 100:.2f}" if not np.isnan(value) else f"{idx}:nan"
            for idx, value in enumerate(result["iou"])
        )
        print(
            f"{result['candidate']}: mIoU={result['miou'] * 100:.4f}, "
            f"matched={result['count']}, missing={result['missing']}, extra={result['extra']}"
        )
        print(f"  class IoU: {iou_text}")

    best = results[0]
    print("")
    print(f"Closest to reference by mIoU: {best['candidate']} ({best['miou'] * 100:.4f})")


if __name__ == "__main__":
    main()
