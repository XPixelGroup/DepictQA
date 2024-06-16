#!/bin/bash
export PYTHONPATH=../../src/:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$1

python -m serve.depictqa_worker
