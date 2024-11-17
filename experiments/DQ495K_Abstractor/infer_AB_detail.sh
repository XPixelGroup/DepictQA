#!/bin/bash
src_dir=../../src/
export PYTHONPATH=$src_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$1

python $src_dir/infer.py \
    --meta_path KADIS700K_s384/AB_sd_detail/metas/test_AB_sd_detail_200.json \
                KADIS700K_s384/AB_md_detail/metas/test_AB_md_detail_100.json \
    --dataset_name kadis_AB_sd_detail \
                   kadis_AB_md_detail \
    --task_name quality_compare_noref \
    --batch_size 16 \
