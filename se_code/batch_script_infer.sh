#!/bin/bash
#SBATCH --gres=gpu:1  
#SBATCH --job-name='m1_1571167563.85'
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH -c 1      # cores requested
#SBATCH --mem=20G  # memory in Mb
#SBATCH -o outlogs/m1_1571167563.85_infer  # send stdout to outfile
#SBATCH -e errlogs/m1_1571167563.85_infer  # send stderr to errfile
#SBATCH -t 00:20:00  # time requested in hour:minute:second
#module load python

source ~/.bashrc
conda activate /n/fs/percepaudio/models/research/audioset/env1/

python se_infer.py --model_name m52_1571167563.85 --model_folder /n/fs/percepaudio/DeepFeaturesSpeech/SpeechDenoisingWithDeepFeatureLosses/m52 --time_from_epoch_inSEC 1571167563.85

##
#SUMMARY OF MODELS RUN for training