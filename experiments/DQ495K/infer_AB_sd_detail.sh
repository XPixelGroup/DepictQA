#!/bin/bash
src_dir=../../src/
export PYTHONPATH=$src_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$1

python $src_dir/infer.py \
    --meta_path KADIS700K/AB_sd_detail/metas/test_AB_sd_detail_200.json \
    --dataset_name kadis_AB_sd_detail \
    --task_name quality_compare_noref \
    --batch_size 16 \
