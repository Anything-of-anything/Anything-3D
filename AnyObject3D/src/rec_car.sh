#! /bin/bash
# CUDA_HOME=/path/to/cuda-11.3
# PATH=$CUDA_HOME/bin:$PATH
# LD_LIBRARY_PATH=$CUDA_HOME/lib64
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=$PWD/3DFuse:$PYTHONPATH
python ./main.py --image car --point_coords 1000 1000 --seed 302651 
