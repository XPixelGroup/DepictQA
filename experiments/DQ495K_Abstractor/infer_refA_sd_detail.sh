#!/bin/bash
src_dir=../../src/
export PYTHONPATH=$src_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$1

python $src_dir/infer.py \
    --meta_path KADIS700K_s384/refA_sd_detail/metas/test_refA_sd_detail_200.json \
    --dataset_name kadis_refA_sd_detail \
    --task_name quality_single_A \
    --batch_size 16 \
