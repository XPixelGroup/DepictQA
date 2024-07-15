#!/bin/bash
src_dir=../../src/
export PYTHONPATH=$src_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$1

python $src_dir/infer.py \
    --meta_path BAPPS/metas/test_refA_detail_mbapps_50.json \
    --dataset_name mbapps_refA_detail \
    --task_name quality_single_A \
    --batch_size 16 \
