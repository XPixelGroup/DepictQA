#!/bin/bash
src_dir=../../src/
export PYTHONPATH=$src_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$1

python $src_dir/infer.py \
    --meta_path TID2013/metas/test_AB_single_tid2013_2.5k.json \
    --dataset_name tid_AB \
    --task_name quality_compare_noref \
    --batch_size 16 \
