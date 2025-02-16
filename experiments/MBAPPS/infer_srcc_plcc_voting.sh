#!/bin/bash
src_dir=../../src/
export PYTHONPATH=$src_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$1

python $src_dir/infer.py \
    --meta_path KADID10K/metas_srcc_plcc_voting/test_refAB_single_kadid_124k.json \
                CSIQ/metas_srcc_plcc_voting/test_refAB_single_csiq_12k.json \
                TID2013/metas_srcc_plcc_voting/test_refAB_single_tid2013_179k.json \
    --dataset_name kadid_srcc_plcc_voting \
                   csiq_srcc_plcc_voting \
                   tid_srcc_plcc_voting \
    --task_name quality_compare \
    --batch_size 16 \
