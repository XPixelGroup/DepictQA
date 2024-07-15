#!/bin/bash
src_dir=../../src/
export PYTHONPATH=$src_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$1

python $src_dir/infer.py \
    --meta_path PIPAL/metas/test_refAB_single_pipal_17.5k.json \
    --dataset_name pipal_refAB \
    --task_name quality_compare \
    --batch_size 1 \
