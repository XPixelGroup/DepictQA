#!/bin/bash
src_dir=../../src/
export PYTHONPATH=$src_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$1

python $src_dir/infer.py \
    --meta_path BAPPS/metas/test_AB_single_bapps_9.4k_s64.json \
    --dataset_name bapps_AB \
    --task_name quality_compare_noref \
    --batch_size 16 \
