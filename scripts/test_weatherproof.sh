#!/bin/bash

config=${1:-configs/weatherproof.yaml}
checkpoint=${2:-exp/weatherproof/unimatch_v2/dinov2_base/best_ema.pth}
input_dir=${3:-data/test_input}
output_dir=${4:-output/weatherproof/test_predictions}
device=${5:-cuda}
id_path=${6:-}

# Avoid accidentally loading torch libraries from another conda environment.
unset PYTHONPATH
unset LD_LIBRARY_PATH

mkdir -p "$output_dir"

ARGS=(
    --config "$config"
    --checkpoint "$checkpoint"
    --checkpoint-key model_ema
    --input "$input_dir"
    --output "$output_dir"
    --device "$device"
)

if [ -n "$id_path" ]; then
    ARGS+=(--id-path "$id_path")
fi

python test_weatherproof.py "${ARGS[@]}" 2>&1 | tee "$output_dir/test.log"


# sh scripts/test_weatherproof.sh \
#   configs/weatherproof.yaml \
