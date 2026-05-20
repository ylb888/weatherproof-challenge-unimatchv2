import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
import yaml

from model.semseg.dpt import DPT

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


WEATHERPROOF_PALETTE = np.array(
    [
        [0, 0, 0],        # background
        [180, 120, 120],  # building
        [120, 120, 180],  # structure
        [128, 64, 128],   # road
        [70, 130, 180],   # sky
        [112, 112, 112],  # stone
        [107, 142, 35],   # terrain-grass
        [152, 251, 152],  # terrain-other
        [230, 230, 230],  # terrain-snow
        [34, 139, 34],    # tree
    ],
    dtype=np.uint8,
)


def disable_xformers_attention():
    # Some environments can import xFormers but cannot run its CUDA kernels.
    # For single-image inference, PyTorch attention is slower but reliable.
    from model.backbone.dinov2_layers import attention

    attention.XFORMERS_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(description="Inference on WeatherProof test_input images.")
    parser.add_argument("--config", default="configs/weatherproof.yaml", type=str)
    parser.add_argument("--checkpoint", default="exp/weatherproof/unimatch_v2/dinov2_base/best_ema.pth", type=str)
    parser.add_argument("--input", default="data/test_input", type=str)
    parser.add_argument("--id-path", default=None, type=str)
    parser.add_argument("--output", default="exp/weatherproof/test_predictions", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--checkpoint-key", default="model_ema", choices=["model", "model_ema"])
    parser.add_argument("--resize-multiple", default=14, type=int)
    parser.add_argument("--save-confidence", action="store_true")
    parser.add_argument("--max-images", default=None, type=int)
    parser.add_argument("--image-pattern", default="*_degraded.png", type=str)
    return parser.parse_args()


def build_model(cfg):
    model_configs = {
        "small": {"encoder_size": "small", "features": 64, "out_channels": [48, 96, 192, 384]},
        "base": {"encoder_size": "base", "features": 128, "out_channels": [96, 192, 384, 768]},
        "large": {"encoder_size": "large", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "giant": {"encoder_size": "giant", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }
    encoder_name = cfg["backbone"].split("_")[-1]
    return DPT(**{**model_configs[encoder_name], "nclass": cfg["nclass"]})


def load_checkpoint(model, checkpoint_path, checkpoint_key, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint[checkpoint_key] if isinstance(checkpoint, dict) and checkpoint_key in checkpoint else checkpoint

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned_state_dict[key] = value

    model.load_state_dict(cleaned_state_dict, strict=True)


def find_images(input_dir, image_pattern, id_path=None):
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    if id_path:
        with open(id_path, "r") as f:
            ids = [line.strip() for line in f if line.strip()]
        image_paths = []
        for sample_id in ids:
            image_rel = sample_id if Path(sample_id).suffix else sample_id + "_degraded.png"
            image_path = Path(input_dir) / image_rel
            if image_path.suffix.lower() in suffixes:
                image_paths.append(image_path)
        return image_paths

    if image_pattern:
        return sorted(path for path in Path(input_dir).rglob(image_pattern) if path.suffix.lower() in suffixes)
    return sorted(path for path in Path(input_dir).rglob("*") if path.suffix.lower() in suffixes)


def resize_to_multiple(image_tensor, multiple):
    if multiple <= 1:
        return image_tensor, image_tensor.shape[-2:]

    ori_h, ori_w = image_tensor.shape[-2:]
    new_h = int(ori_h / multiple + 0.5) * multiple
    new_w = int(ori_w / multiple + 0.5) * multiple
    new_h = max(new_h, multiple)
    new_w = max(new_w, multiple)

    if (new_h, new_w) == (ori_h, ori_w):
        return image_tensor, (ori_h, ori_w)

    return F.interpolate(image_tensor, (new_h, new_w), mode="bilinear", align_corners=True), (ori_h, ori_w)


def save_prediction(pred, conf, image_path, input_root, output_root, save_confidence):
    rel_path = image_path.relative_to(input_root)
    if rel_path.name.endswith("_degraded.png"):
        mask_rel_path = rel_path.with_name(rel_path.name.replace("_degraded.png", "_gt-intern.png"))
    else:
        mask_rel_path = rel_path.with_name(f"{rel_path.stem}_gt-intern.png")
    stem = mask_rel_path.with_suffix("")

    mask_path = output_root / "mask" / mask_rel_path
    color_path = output_root / "color" / f"{stem}_color.png"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    color_path.parent.mkdir(parents=True, exist_ok=True)

    pred_np = pred.astype(np.uint8)
    Image.fromarray(pred_np).save(mask_path)
    Image.fromarray(WEATHERPROOF_PALETTE[pred_np]).save(color_path)

    if save_confidence:
        conf_path = output_root / "confidence" / f"{stem}_conf.png"
        conf_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray((conf * 255).clip(0, 255).astype(np.uint8)).save(conf_path)


def main():
    args = parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    disable_xformers_attention()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    input_root = Path(args.input)
    output_root = Path(args.output)

    model = build_model(cfg)
    load_checkpoint(model, args.checkpoint, args.checkpoint_key, device)
    model.to(device)
    model.eval()

    normalize = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    image_paths = find_images(input_root, args.image_pattern, args.id_path)
    if args.max_images is not None:
        image_paths = image_paths[:args.max_images]
    if not image_paths:
        raise FileNotFoundError(f"No images found under {input_root}")

    progress = tqdm(image_paths, desc="Testing", unit="img") if tqdm is not None else image_paths

    with torch.no_grad():
        for idx, image_path in enumerate(progress, start=1):
            image = Image.open(image_path).convert("RGB")
            image_tensor = normalize(image).unsqueeze(0).to(device)
            image_tensor, ori_size = resize_to_multiple(image_tensor, args.resize_multiple)

            logits = model(image_tensor)
            if logits.shape[-2:] != ori_size:
                logits = F.interpolate(logits, ori_size, mode="bilinear", align_corners=True)

            prob = logits.softmax(dim=1)
            conf, pred = prob.max(dim=1)

            save_prediction(
                pred.squeeze(0).cpu().numpy(),
                conf.squeeze(0).cpu().numpy(),
                image_path,
                input_root,
                output_root,
                args.save_confidence,
            )

            if tqdm is None and (idx % 50 == 0 or idx == len(image_paths)):
                print(f"[{idx}/{len(image_paths)}] processed {image_path}")

    print(f"Done. Predictions saved to {output_root}")


if __name__ == "__main__":
    main()
