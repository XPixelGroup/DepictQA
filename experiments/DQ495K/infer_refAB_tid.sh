#!/bin/bash
src_dir=../../src/
export PYTHONPATH=$src_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$1

python $src_dir/infer.py \
    --meta_path TID2013/metas/test_refAB_single_tid2013_2.5k.json \
    --dataset_name tid_refAB \
    --task_name quality_compare \
    --batch_size 16 \
