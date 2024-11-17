#!/bin/bash
src_dir=../../src/
export PYTHONPATH=$src_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$1

python $src_dir/infer.py \
    --meta_path BAPPS/metas/test_refAB_single_bapps_9.4k_s64.json \
                KADID10K/metas/test_refAB_single_kadid_7.5k.json \
                PIPAL/metas/test_refAB_single_pipal_17.5k.json \
                TID2013/metas/test_refAB_single_tid2013_2.5k.json \
                LIVEMD/metas/test_refAB_single_livemd_2k.json \
                MDID2013/metas/test_refAB_single_mdid_2k.json \
    --dataset_name bapps_refAB \
                   kadid_refAB \
                   pipal_refAB \
                   tid_refAB \
                   livemd_refAB \
                   mdid_refAB \
    --task_name quality_compare \
    --batch_size 16 \
