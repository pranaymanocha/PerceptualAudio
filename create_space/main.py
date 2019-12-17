import numpy
import numpy as np
import random
import os
import argparse
from helper import *
import csv
import tqdm

'''
directory structure:
a) DAPS: "/n/fs/percepaudio/www/mturk_hosts/website_perturbation/snr_resize_folder/" -> choose one speaker and one random file in that
b) Librispeech: "/n/fs/percepaudio/www/mturk_hosts/website_perturbation/snr_resize_folder/" -> choose one audio folder (denotes speaker) and then randomly choose one audio file

1) Run this file - this creates the audio files
2) audio normalise all audio files - use ffmpeg-normalize
Ready!
'''

def argument_parser():
    """
    Get an argument parser for a training script.
    """
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--saved_file', help='name of the php file to be saved', default='example.php', type=str)
    parser.add_argument('--dataset_used', help='dataset to be used', default='librispeech', type=str)    
    parser.add_argument('--numberofpairs', help='Number of lines in the file', default=10, type=int)
    parser.add_argument('--min_difference', help='Number of lines in the file', default=10, type=int)
    parser.add_argument('--folder', help='Folder to be made where files saved', default='example')
    parser.add_argument('--EQ_bands', help='EQ bands (2 frequency levels)', default='5e2_1e3')
    
    return parser

args = argument_parser().parse_args()

#generate the php file
a=ChooseRandomPoint(args.dataset_used,args.numberofpairs,args.min_difference)
a1=SplitList(a)
a2=CreatePHPFile(a1,args.saved_file)

if args.dataset_used=='librispeech':
    path='/n/fs/percepaudio/librispeech/LibriSpeech/resized_files_2/'
elif args.dataset_used=='daps':
    path='/n/fs/percepaudio/www/mturk_hosts/website_perturbation/snr_resize_folder/'

path_new=args.folder

if os.path.isdir(path_new)==False:
    os.mkdir(path_new)

#function_map(args)

#create clips according to the php file
if args.dataset_used=='librispeech':
    librispeech(args,path,path_new)
elif args.dataset_used=='daps':
    daps(args,path,path_new)