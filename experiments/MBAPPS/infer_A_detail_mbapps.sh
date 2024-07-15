#!/bin/bash
src_dir=../../src/
export PYTHONPATH=$src_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$1

python $src_dir/infer.py \
    --meta_path BAPPS/metas/test_A_detail_mbapps_50.json \
    --dataset_name mbapps_A_detail \
    --task_name quality_single_A_noref \
    --batch_size 16 \
