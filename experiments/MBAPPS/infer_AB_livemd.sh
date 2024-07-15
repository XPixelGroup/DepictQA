#!/bin/bash
src_dir=../../src/
export PYTHONPATH=$src_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$1

python $src_dir/infer.py \
    --meta_path LIVEMD/metas/test_AB_single_livemd_2k.json \
    --dataset_name livemd_AB \
    --task_name quality_compare_noref \
    --batch_size 16 \
