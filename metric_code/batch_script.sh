#!/bin/bash
#SBATCH --gres=gpu:1  
#SBATCH --job-name='m1'
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH -c 1      # cores requested
#SBATCH --mem=35G  # memory in Mb
#SBATCH -o outlogs/m1 # send stdout to outfile
#SBATCH -e errlogs/m1  # send stderr to errfile
#SBATCH -t 96:00:00  # time requested in hour:minute:second

source ~/.bashrc
conda activate /n/fs/percepaudio/models/research/audioset/env1/

python main.py --summary_folder m35 --type scratch --learning_rate 1e-4 --layers 14 --loss_layers 14 --batch_size 16

