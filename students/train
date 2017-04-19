#!/bin/bash
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --gres=gpu:1
#SBATCH --time=9:00:00
export THEANO_FLAGS="device=gpu"
source ~/.myworkrc
source activate py3
source ~/.cudarc_new
python --version
nvcc --version
dev=$(nvidia-smi |grep "0%" -B1|awk '{print $2}'|head -n 1)
echo $dev
CUDA_VISIBLE_DEVICES=$dev python cli.py train $* --budget-secs=28800
