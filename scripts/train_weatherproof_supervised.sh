#!/bin/bash

method='supervised'
exp='dinov2_base'

config=configs/weatherproof.yaml
labeled_id_path=${LABELED_ID_PATH:-splits/weatherproof/labeled.txt}
val_id_path=${VAL_ID_PATH:-splits/weatherproof/val.txt}
save_path=exp/weatherproof/$method/$exp
save_interval=${3:-0}

mkdir -p $save_path

python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    $method.py \
    --config=$config --labeled-id-path $labeled_id_path \
    --val-id-path $val_id_path \
    --save-interval $save_interval \
    --save-path $save_path --port $2 2>&1 | tee $save_path/out.log
