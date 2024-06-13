#!/bin/bash
export PYTHONPATH=../../src/:$PYTHONPATH
python -m serve.depictqa_worker
