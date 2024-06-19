#!/bin/bash
src_dir=../../src/
export PYTHONPATH=$src_dir:$PYTHONPATH

deepspeed --include localhost:$1 --master_addr 127.0.0.1 --master_port 28500 $src_dir/train.py
