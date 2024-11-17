#!/bin/bash
src_dir=../../src/
export PYTHONPATH=$src_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$1

python $src_dir/infer.py \
    --meta_path KADIS700K_s384/A_sd_brief/metas/test_A_sd_single_4k_exclude.json \
                KADIS700K_s384/A_md_brief/metas/test_A_md_single_3k_exclude.json \
    --dataset_name kadis_A_sd_brief \
                   kadis_A_md_brief \
    --task_name quality_single_A_noref \
    --batch_size 16 \
