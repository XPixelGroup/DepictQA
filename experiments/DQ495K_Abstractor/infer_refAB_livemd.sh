#!/bin/bash
src_dir=../../src/
export PYTHONPATH=$src_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$1

python $src_dir/infer.py \
    --meta_path LIVEMD/metas/test_refAB_single_livemd_2k.json \
    --dataset_name livemd_refAB \
    --task_name quality_compare \
    --batch_size 16 \
