#!/bin/bash
src_dir=../../src/
export PYTHONPATH=$src_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$1

python $src_dir/infer.py \
    --meta_path KADIS700K/A_md_detail/metas/test_A_md_detail_100.json \
    --dataset_name kadis_A_md_detail \
    --task_name quality_single_A_noref \
    --batch_size 16 \
