#!/bin/bash

method='unimatch_v2'
exp='dinov2_base_train_all'

config=configs/weatherproof.yaml
labeled_id_path=${LABELED_ID_PATH:-splits/weatherproof/labeled.txt}
unlabeled_id_path=${UNLABELED_ID_PATH:-splits/weatherproof/unlabeled.txt}
reference_pred_dir=${REFERENCE_PRED_DIR:-output/weatherproof/test_predictions}
reference_input_dir=${REFERENCE_INPUT_DIR:-data/test_input}
save_path=exp/weatherproof/$method/$exp
save_interval=${3:-0}

mkdir -p $save_path

python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    $method.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --no-val \
    --reference-pred-dir $reference_pred_dir \
    --reference-input-dir $reference_input_dir \
    --reference-id-path $unlabeled_id_path \
    --save-interval $save_interval \
    --save-path $save_path --port $2 2>&1 | tee $save_path/out.log
