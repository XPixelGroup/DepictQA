#!/bin/bash
src_dir=../../src/
export PYTHONPATH=$src_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$1

python $src_dir/infer.py \
    --meta_path KADIS700K_s384/refAB_sd_detail/metas/test_refAB_sd_detail_200.json \
    --dataset_name kadis_refAB_sd_detail \
    --task_name quality_compare \
    --batch_size 16 \
