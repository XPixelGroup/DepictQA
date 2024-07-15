#!/bin/bash
src_dir=../../src/
export PYTHONPATH=$src_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$1

python $src_dir/infer.py \
    --meta_path BAPPS/metas/test_AB_detail_mbapps_200.json \
    --dataset_name mbapps_AB_detail \
    --task_name quality_compare_noref \
    --batch_size 16 \
