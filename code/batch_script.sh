#!/bin/bash
#SBATCH --gres=gpu:1  
#SBATCH --job-name='m17'
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH -c 1      # cores requested
#SBATCH --mem=25G  # memory in Mb
#SBATCH -o outlogs/m17 # send stdout to outfile
#SBATCH -e errlogs/m17  # send stderr to errfile
#SBATCH -t 96:00:00  # time requested in hour:minute:second
#module load python

source ~/.bashrc
conda activate /n/fs/percepaudio/models/research/audioset/env1/

python main3_scratch.py --summary_folder m17 --type scratch --learning_rate 1e-4

###SUMMARY
#main.py
#m1 linear 1e-4
#m2 finetune 1e-4
#m3 scratch 1e-4
#m4 finetune 1e-2
#m5 scratch 1e-2

#main1.py
#m6 finetune 1e-4
#m7 scratch  1e-4

#m8: main1 1e-6
#m9: main1 1e-6

#m10: main1 1e-5 finetune
#m11: main1 1e-5 scratch

##main2.py
#m12: linear 1e-6 dropout(no,no)
#m13: main2 finetune 1e-5 dropout(no,yes)
#m14: main2 scratch 1e-5 dropout(yes,yes)

#m15: main finetune 1e-5 dropout(yes,yes)

#m16: main.py scratch 1e-3
#m17: main3_scratch.py 1e-4 (batch version)

##main.py
#dropout(no,no)
#dropout(yes,yes)
#dropout(yes,yes)

##main1.py
#dropout(no,no)
#dropout(no,no)
#dropout(no,no)

#main2.py
#dropout(no,no)
#dropout(no,yes)
#dropout(yes,yes)

#parser.add_argument('--layers', help='number of layers in the model', default=14, type=int)
#parser.add_argument('--learning_rate', help='learning rate', default=1e-5, type=float)
#parser.add_argument('--summary_folder', help='summary folder name', default='m_example')
#parser.add_argument('--optimiser', help='choose optimiser - gd/adam', default='adam')
#parser.add_argument('--loss_norm', help='loss norm - NM,SBN,None', default='SBN')
#parser.add_argument('--channels_increase', help='doubling channels after how many layers - 1,2,3,4,5,6', default=5, type=int)
#parser.add_argument('--loss_layers', help='loss to be taken for the first how many layers', default=14, type=int)
#parser.add_argument('--filter_size', help='filter size for the convolutions', default=3, type=int)
#parser.add_argument('--train_from_checkpoint', help='train_from_checkpoint', default=0, type=int)
#parser.add_argument('--epochs', help='number of training epochs', default=2000, type=int)
#parser.add_argument('--type', help='lin/fin/scratch', default='scratch', type=float)
#parser.add_argument('--pretrained_model_path', help='Model Path for the pretrained model', default='../pre-model')