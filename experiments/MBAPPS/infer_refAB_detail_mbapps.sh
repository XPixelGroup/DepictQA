#!/bin/bash
src_dir=../../src/
export PYTHONPATH=$src_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$1

python $src_dir/infer.py \
    --meta_path BAPPS/metas/test_refAB_detail_mbapps_200.json \
    --dataset_name mbapps_refAB_detail \
    --task_name quality_compare \
    --batch_size 16 \
