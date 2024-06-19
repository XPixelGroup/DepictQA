#!/bin/bash
src_dir=../../src/
export PYTHONPATH=$src_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$1

python $src_dir/infer.py \
    --meta_path KADIS700K/refA_md_brief/metas/test_refA_md_single_5k.json \
    --dataset_name kadis_refA_md_brief \
    --task_name quality_single_A \
    --batch_size 16 \
