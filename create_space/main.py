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
    parser.add_argument('--min_difference', help='Difference in the direction moving the fastest', default=50, type=int)
    parser.add_argument('--folder', help='Folder to be made where files saved', default='example')
    parser.add_argument('--EQ_bands', help='EQ bands (2 frequency levels)', default='5e2_1e3')
    parser.add_argument('--process', help='0 (just create php), 1(just create files), 2(do both)', default=2, type=int)
    parser.add_argument('--job', help='0 (usual create_space), 1 (triplets for finetuning - 3 points in a random direction), 2 (triplets in general)', default=0, type=int)
    return parser

args = argument_parser().parse_args()


if args.job==0:
    
    if args.process==0 or args.process==2:
        #generate the php file
        a=ChooseRandomPoint(args.dataset_used,args.numberofpairs,args.min_difference)
        a1=SplitList(a)
        a2=CreatePHPFile(a1,args.saved_file)

    if args.process==1 or args.process==2:

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
            
elif args.job==1:
    if args.process==0 or args.process==2:
        a=create_triplets_finetune(args)
        CreatePHPFile_finetune(a,args.saved_file)
    
    if args.process==1 or args.process==2:

        if args.dataset_used=='librispeech':
            path='/n/fs/percepaudio/librispeech/LibriSpeech/resized_files_2/'
        elif args.dataset_used=='daps':
            path='/n/fs/percepaudio/www/mturk_hosts/website_perturbation/snr_resize_folder/'

        path_new=args.folder

        if os.path.isdir(path_new)==False:
            os.mkdir(path_new)
        
        #create clips according to the php file
        if args.dataset_used=='librispeech':
            librispeech(args,path,path_new)
        elif args.dataset_used=='daps':
            daps(args,path,path_new)
else:
    print('not yet implemented')