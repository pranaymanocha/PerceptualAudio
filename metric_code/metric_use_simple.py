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
import numpy as np
import argparse
from network_model import *
from helper import *
import argparse
import librosa

def argument_parser():
    """
    Get an argument parser for a training script.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--saved_name', help='name of the file saved as a pickle file',default='m_example',type=str)
    parser.add_argument('--modfolder', help='path of the saved model to be used for inference', default='../pre-model/scratch_loss',type=str)
    parser.add_argument('--type', help='pretrained/linear/finetune/scratch', default='scratch',type=str)
    parser.add_argument('--e0', help='path of file1', default='../sample_audio/ref.wav',type=str)
    parser.add_argument('--e1', help='path of file2', default='../sample_audio/2.wav',type=str)
    return parser

args = argument_parser().parse_args()

#sample loading the audio files in a dictionary. '4.wav is perceptually farther than 2.wav from ref.wav.'


def load_full_data_list(args): # check change path names
    
    dataset={}
    print("Loading Files....")
    
    dataset['all']={}
    dataset['all']['inname']=[]
    dataset['all']['outname']=[]
    
    dataset['all']['inname'].append(args.e0)
    dataset['all']['outname'].append(args.e1)
                        
    return dataset 


def load_full_data(dataset):
    
    dataset['all']['inaudio']  = [None]*len(dataset['all']['inname'])
    dataset['all']['outaudio']  = [None]*len(dataset['all']['outname'])

    for id in tqdm(range(len(dataset['all']['inname']))):

        if dataset['all']['inaudio'][id] is None:
            
            inputData, sr = librosa.load(dataset['all']['inname'][id],sr=22050)
            outputData, sr = librosa.load(dataset['all']['outname'][id],sr=22050)
            
            ## convert to 16 bit floating point
            inputData = np.round(inputData.astype(np.float)*32768)
            outputData = np.round(outputData.astype(np.float)*32768)
            
            inputData_wav  = np.reshape(inputData, [-1, 1])
            outputData_wav  = np.reshape(outputData, [-1, 1])

            shape_wav = np.shape(inputData_wav)
            shape_wav1 = np.shape(outputData_wav)

            inputData_wav = np.reshape(inputData_wav, [1, 1,shape_wav[0], shape_wav[1]])
            outputData_wav = np.reshape(outputData_wav, [1, 1,shape_wav1[0], shape_wav1[1]])

            inputData_wav  = np.float32(inputData_wav)
            outputData_wav  = np.float32(outputData_wav)

            dataset['all']['inaudio'][id]  = inputData_wav
            dataset['all']['outaudio'][id]  = outputData_wav
              
    return dataset

######### Data loading
dataset=load_full_data_list(args)
dataset=load_full_data(dataset)

######### Parameters of the model 

#MAKE SURE THAT YOU UPDATE THESE PARAMETERS IF YOU MAKE ANY CHANGES TO THE MODEL.

#################################

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
SAVE_EPOCHS = 10 # NUMBER OF EPOCHS BETWEEN MODEL SAVES
FILTER_SIZE=3

def load_full_data(dataset,sets,id_value):
    
    noisy=dataset[sets]['inaudio'][id_value]
    clean=dataset['all']['outaudio'][id_value]
    
    clean=np.reshape(clean,[clean.shape[2]])
    noisy=np.reshape(noisy,[noisy.shape[2]])
    
    shape1=clean.shape[0]
    shape2=noisy.shape[0]
    
    if shape1>shape2:
        difference=shape1-shape2
        a=(np.zeros(difference))
        noisy=np.append(a,noisy,axis=0)
    elif shape1<shape2:
        difference=shape2-shape1
        a=(np.zeros(difference))
        clean=np.append(a,clean,axis=0)
   
    clean=np.reshape(clean,[1,1,clean.shape[0],1])
    noisy=np.reshape(noisy,[1,1,noisy.shape[0],1])
    
    return [clean,noisy]


def model_run():
    
    modfolder= args.modfolder
    
    tf.reset_default_graph()
    
    with tf.variable_scope(tf.get_variable_scope()):
        
        input1_wav=tf.placeholder(tf.float32,shape=[None, None, None,1])

        clean1_wav=tf.placeholder(tf.float32,shape=[None, None, None,1])
        
        keep_prob = tf.placeholder_with_default(1.0, shape=())
        
        if args.type!='pretrained':
            
            others,loss_sum = featureloss(input1_wav,clean1_wav,keep_prob,loss_layers=SE_LOSS_LAYERS,n_layers=LOSS_LAYERS, norm_type=LOSS_NORM, base_channels=LOSS_BASE_CHANNELS,blk_channels=LOSS_BLK_CHANNELS,ksz=FILTER_SIZE)
        
        else:
            
            others,loss_sum = featureloss_pretrained(input1_wav,clean1_wav,keep_prob,loss_layers=SE_LOSS_LAYERS,n_layers=LOSS_LAYERS, norm_type=LOSS_NORM, base_channels=LOSS_BASE_CHANNELS,blk_channels=LOSS_BLK_CHANNELS,ksz=FILTER_SIZE) 
        
        distance = loss_sum
    
    distance_overall=[]
    
    with tf.Session() as sess:

        #sess.run(tf.global_variables_initializer())
        loss_saver = tf.train.Saver([var for var in tf.trainable_variables()])
        
        if args.type=='pretrained':
            loss_saver.restore(sess, "%s/loss_model.ckpt" % modfolder)
        else:
            loss_saver.restore(sess, "%s/my_test_model" % modfolder)
                
        for j in tqdm(range(len(dataset['all']['inname']))):
            wav_in,wav_out=load_full_data(dataset,'all',j)
            dist= sess.run([distance],feed_dict={input1_wav:wav_out, clean1_wav:wav_in})
            distance_overall.append(dist)
    
    return [distance_overall]

distance=[]
distance_overall=model_run()
print("Distance between the files is {}".format(distance_overall[0][0][0]))
distance.append(distance_overall)
with open('saved_distances/'+str(args.saved_name)+'.p', 'wb') as f:
    pickle.dump(distance, f)