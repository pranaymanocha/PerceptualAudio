#!/bin/bash
#SBATCH --gres=gpu:1  
#SBATCH --job-name='SE_m1'
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH -c 1      # cores requested
#SBATCH --mem=20G  # memory in Mb
#SBATCH -o outlogs/m1  # send stdout to outfile
#SBATCH -e errlogs/m1 # send stderr to errfile
#SBATCH -t 96:00:00  # time requested in hour:minute:second
#module load python

source ~/.bashrc
conda activate /n/fs/percepaudio/models/research/audioset/env1/

python se_train.py --out_folder m1 --model_folder ../code/summaries/m1 --learning_rate 1e-3

##
#SUMMARY OF MODELS RUN for training
#parser.add_argument('--loss_layers', help='number of layers in the model', default=14, type=int)
#parser.add_argument('--out_folder', help='summary folder name', default='m_example')
#parser.add_argument('--loss_norm', help='loss norm - NM,SBN,None', default='SBN')
#parser.add_argument('--channels_increase', help='doubling channels after how many layers - 1,2,3,4,5,6', default=5, type=int)
#parser.add_argument('--model_folder', help='path of the loss model parameters saved location (without the / at the end )', default='')
#parser.add_argument('--learning_rate', help='learning rate', default=1e-4,type=float)
#parser.add_argument('--feature_loss_layers', help='number of feature loss layers used', default=14,type=int)
#parser.add_argument('--train_from_checkpoint', help='train_from_checkpoint',default=0,type=int)
#parser.add_argument('--time_from_epoch_inSEC', help='time_from_EPOCH_insec',default='')
#parser.add_argument('--kernel_size', help='kernel convolution size',default=3,type=int)