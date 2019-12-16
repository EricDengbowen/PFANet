#!/usr/bin/bash

#SBATCH -J Saliency_withalpha0.8
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem 8G
#SBATCH -p general
source /usr2/share/gpu.sbatch
python train.py

