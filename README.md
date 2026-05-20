# weatherproof-challenge-unimatchv2

WeatherProof semantic segmentation training and inference code based on UniMatch V2 with a DINOv2 backbone.

This repository is arranged for the WeatherProof Challenge setting:

- labeled training uses clean images from `train_scenes`
- unlabeled training uses degraded images from `train_scenes`
- inference uses degraded images from the provided input directory, usually `data/test_input`

## Data Layout

Place the WeatherProof data under `data/`:

```text
data/
├── train_scenes/
│   └── <scene>/
│       ├── 000_gt-clean.png
│       ├── 000_degraded.png
│       └── 000_gt-intern.png
└── test_input/
    └── <scene>/
        └── 000_degraded.png
```

Split files should contain sample ids such as:

```text
1018_0_0_2022_10_16/000
1018_0_0_2022_10_16/001
```

Prepare WeatherProof split files under:

```text
splits/weatherproof/
├── labeled.txt
├── unlabeled.txt
└── val.txt
```

Each line can be either:

```text
<scene>/<id>
```

or:

```text
<scene>/<id>_degraded.png
```

## Dataset Rules

The WeatherProof loader uses these paths:

```text
labeled train_l:
  image = data/train_scenes/<scene>/<id>_gt-clean.png
  mask  = data/train_scenes/<scene>/<id>_gt-intern.png

unlabeled train_u:
  image = data/train_scenes/<scene>/<id>_degraded.png
  mask  = none

validation:
  image = data/train_scenes/<scene>/<id>_degraded.png
  mask  = data/train_scenes/<scene>/<id>_gt-intern.png

inference:
  image = images under the path passed to --input, usually data/test_input
```

The task uses 10 classes:

```text
0 background
1 building
2 structure
3 road
4 sky
5 stone
6 terrain-vegetation
7 terrain-other
8 terrain-snow
9 tree
```

## Pretrained Encoder

Download the DINOv2 encoder weight and place it under `pretrained/`.

For DINOv2-Base, use:

```text
pretrained/dinov2_vitb14_pretrain.pth
```

## Training

Default WeatherProof semi-supervised training:

```bash
sh scripts/train_weatherproof.sh <GPU_NUM> <PORT>
```

Example:

```bash
sh scripts/train_weatherproof.sh 4 29500
```

The script uses:

```text
config: configs/weatherproof.yaml
labeled split: splits/weatherproof/labeled.txt
unlabeled split: splits/weatherproof/unlabeled.txt
save path: exp/weatherproof/unimatch_v2/dinov2_base_train_all
```

You can override paths with environment variables:

```bash
LABELED_ID_PATH=splits/weatherproof/labeled.txt \
UNLABELED_ID_PATH=splits/weatherproof/unlabeled.txt \
REFERENCE_PRED_DIR=output/weatherproof/test_predictions \
REFERENCE_INPUT_DIR=data/test_input \
sh scripts/train_weatherproof.sh 4 29500
```

Supervised-only training:

```bash
sh scripts/train_weatherproof_supervised.sh <GPU_NUM> <PORT>
```

## Inference

Run inference on `data/test_input`:

```bash
sh scripts/test_weatherproof.sh \
  configs/weatherproof.yaml \
  exp/weatherproof/unimatch_v2/dinov2_base/best_ema.pth \
  data/test_input \
  output/weatherproof/test_predictions
```

By default, the script scans the input directory directly. To restrict inference to an id list, pass it as the sixth argument:

```bash
sh scripts/test_weatherproof.sh \
  configs/weatherproof.yaml \
  exp/weatherproof/unimatch_v2/dinov2_base/best_ema.pth \
  data/test_input \
  output/weatherproof/test_predictions \
  cuda \
  splits/weatherproof/unlabeled.txt
```

Prediction outputs are saved as:

```text
output/weatherproof/test_predictions/
├── mask/
│   └── <scene>/<id>_gt-intern.png
└── color/
    └── <scene>/<id>_gt-intern_color.png
```

## Acknowledgement

This project is based on UniMatch V2:

```bibtex
@article{unimatchv2,
  title={UniMatch V2: Pushing the Limit of Semi-Supervised Semantic Segmentation},
  author={Yang, Lihe and Zhao, Zhen and Zhao, Hengshuang},
  journal={TPAMI},
  year={2025}
}
```
