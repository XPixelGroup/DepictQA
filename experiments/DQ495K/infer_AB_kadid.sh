#!/bin/bash
src_dir=../../src/
export PYTHONPATH=$src_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$1

python $src_dir/infer.py \
    --meta_path KADID10K/metas/test_AB_single_kadid_7.5k.json \
    --dataset_name kadid_AB \
    --task_name quality_compare_noref \
    --batch_size 16 \
