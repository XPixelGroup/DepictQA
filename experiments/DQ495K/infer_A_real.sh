#!/bin/bash
src_dir=../../src/
export PYTHONPATH=$src_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$1

python $src_dir/infer.py \
    --meta_path web_download/metas/test_A_web.json \
    --dataset_name A_web_detail \
    --task_name quality_single_A_noref \
    --batch_size 1 \
    # different images have different sizes, thus batch_size is 1
