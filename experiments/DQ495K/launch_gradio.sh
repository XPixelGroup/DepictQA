#!/bin/bash
export PYTHONPATH=../../src/:$PYTHONPATH
python -m serve.gradio_web_server
