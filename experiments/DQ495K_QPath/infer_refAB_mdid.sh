#!/bin/bash
src_dir=../../src/
export PYTHONPATH=$src_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$1

python $src_dir/infer.py \
    --meta_path MDID2013/metas/test_refAB_single_mdid_2k.json \
    --dataset_name mdid_refAB \
    --task_name quality_compare \
    --batch_size 16 \
