import numpy
import numpy as np
import random
import os
import argparse
from helper import *
from network_model import *
import csv
import tqdm
import numpy as np
import numpy
import os
from tqdm import tqdm

import numpy as np

from tqdm import tqdm
from scipy.io import wavfile
import os, csv
import tensorflow as tf
import pickle

'''
File meant to be used for evaluation:
1) MOS Evaluation on 3 datasets: voco,fftnet,bwe
2) Triplet evaluation on two sets: 1) FFTnet and My Own Space
Ready!
'''

def argument_parser():
    """
    Get an argument parser for a training script.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--triplet_or_MOS', help='Evaluate on Triplets OR MOS', default='triplet', type=str)
    parser.add_argument('--dataset_used', help='Triplets(My or FFTnet) OR MOS(voco,fftnet,bwe)', default='fftnet', type=str)
    parser.add_argument('--path_load_model', help='Path for Loading the Model', default='../metric_code/summaries/m23', type=str)
    parser.add_argument('--pickle_file_name', help='Name of the pickle filename at the end', default='m_example', type=str)
    parser.add_argument('--split_triplet', help='Decimal Split for Triplet Considering Point', default=0.70, type=float)
    parser.add_argument('--map_score', help='Get MAP Scores Across a Held Out Set', default=0, type=int)
    
    return parser

args = argument_parser().parse_args()

dataset,list_methods=load_full_data_list(A_triplet_mymetric,A_triplet_fftnet,args)
dataset=load_full_data(dataset,list_methods,args)

SE_LAYERS = 13 # NUMBER OF INTERNAL LAYERS
SE_CHANNELS = 64 # NUMBER OF FEATURE CHANNELS PER LAYER
SE_LOSS_LAYERS = 14 # NUMBER OF FEATURE LOSS LAYERS
SE_NORM = "NM" # TYPE OF LAYER NORMALIZATION (NM, SBN or None)
SE_LOSS_TYPE = "FL" # TYPE OF TRAINING LOSS (L1, L2 or FL)

# FEATURE LOSS NETWORK
LOSS_LAYERS = 14 # NUMBER OF INTERNAL LAYERS
LOSS_BASE_CHANNELS = 32 # NUMBER OF FEATURE CHANNELS PER LAYER IN FIRT LAYER
LOSS_BLK_CHANNELS = 5 # NUMBER OF LAYERS BETWEEN CHANNEL NUMBER UPDATES
LOSS_NORM =  'SBN' # TYPE OF LAYER NORMALIZATION (NM, SBN or None)

SET_WEIGHT_EPOCH = 10 # NUMBER OF EPOCHS BEFORE FEATURE LOSS BALANCE
FILTER_SIZE = 3

def model_run(args,noise1='pagal'):
    
    modfolder=args.path_load_model
    
    tf.reset_default_graph()
    
    with tf.variable_scope(tf.get_variable_scope()):
        input=tf.placeholder(tf.float32,shape=[None,None,None,1])
        clean=tf.placeholder(tf.float32,shape=[None,None,None,1])
        keep_prob = tf.placeholder_with_default(1.0, shape=())
        dist_all,dist = featureloss(clean, input, keep_prob, loss_layers=SE_LOSS_LAYERS, n_layers=LOSS_LAYERS, norm_type=LOSS_NORM,base_channels=LOSS_BASE_CHANNELS, blk_channels=LOSS_BLK_CHANNELS,ksz=3)
    
    distance_overall=[]
    
    with tf.Session() as sess:
    
        try:
            loss_saver = tf.train.Saver([var for var in tf.trainable_variables()])
            loss_saver.restore(sess, "%s/loss_model.ckpt" % modfolder)
            print('Loaded Pretrained Weights')
        except:
            
            loss_saver = tf.train.Saver([var for var in tf.trainable_variables()])
            loss_saver.restore(sess, "%s/my_test_model" % modfolder)
            print('Loaded Pretrained Weights')
        
        if args.triplet_or_MOS=='triplet':
            for j in tqdm(range(len(dataset['anchor_inname']))):
                ref,wav_in,wav_out=load_full_data_prepare_triplets(dataset,'sets',j)
                dist1= sess.run([dist],feed_dict={input:ref, clean:wav_in})
                dist2= sess.run([dist],feed_dict={input:ref, clean:wav_out})
                distance_overall.append([dist1,dist2])
                
        elif args.triplet_or_MOS=='MOS':
            for j in tqdm(range(len(dataset[noise1]['inname']))):
                wav_in,wav_out=load_full_data_prepare_MOS(dataset,noise1,j,list_methods[-1])
                dist1= sess.run([dist],feed_dict={input:wav_out, clean:wav_in})
                distance_overall.append(dist1[0])
    
    return distance_overall

distance1=[]
if args.triplet_or_MOS=='triplet':
    
    distance_overall=model_run(args)
    distance1.extend(distance_overall)
    
elif args.triplet_or_MOS=='MOS':
    for method in list_methods[:-1]:
        distance_overall=[]
        distance_overall=model_run(args,method)
        distance1.append(distance_overall)

with open(str(args.triplet_or_MOS)+'_'+str(args.dataset_used)+'_'+str(args.pickle_file_name)+'.p','wb') as f:
    pickle.dump(distance1, f)

pickle_file=str(args.triplet_or_MOS)+'_'+str(args.dataset_used)+'_'+str(args.pickle_file_name)+'.p'
score=get_score(args,pickle_file,A_triplet_mymetric,A_triplet_fftnet,dataset)

print("F1/Correlation Score {}".format(score))

if args.map_score==1:
    score_map_final=score_map(args)
    print("MAP - linear/reverb/compression/combined {}".format(score_map_final))