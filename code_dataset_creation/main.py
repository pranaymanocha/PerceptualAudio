import numpy as np

from tqdm import tqdm
from scipy.io import wavfile
import os, csv
import tensorflow as tf
import pickle

from helper import *
from network_model import *
from dataloader import *
from MAP_eval import *

import numpy as np

import argparse

def argument_parser():
    """
    Get an argument parser for a training script.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--layers', help='number of layers in the model', default=14, type=int)
    parser.add_argument('--learning_rate', help='learning rate', default=1e-3, type=float)
    parser.add_argument('--summary_folder', help='summary folder name', default='m21')
    parser.add_argument('--optimiser', help='choose optimiser - gd/adam', default='adam')
    parser.add_argument('--loss_norm', help='loss norm - NM,SBN,None', default='SBN')
    parser.add_argument('--channels_increase', help='doubling channels after how many layers - 1,2,3,4,5,6', default=5, type=int)
    parser.add_argument('--loss_layers', help='loss to be taken for the first how many layers', default=6, type=int)
    parser.add_argument('--filter_size', help='filter size for the convolutions', default=3, type=int)
    parser.add_argument('--train_from_checkpoint', help='train_from_checkpoint', default=0, type=int)
    parser.add_argument('--epochs', help='number of training epochs', default=2000, type=int)
    parser.add_argument('--model_input', help='waveform/logmelspec/downsampled', default='waveform')
    parser.add_argument('--training_data', help='linear/linear+combination/linear+comb+MIT', default='waveform')
    return parser

args = argument_parser().parse_args()

print(args.layers)
print(args.learning_rate)
print(args.summary_folder)
print(args.optimiser)
print(args.loss_norm)
print(args.channels_increase)
print(args.loss_layers)
print(args.filter_size)
print(args.train_from_checkpoint)
print(args.epochs)
print(args.model_input)

dataset=load_full_data_list()
dataset=split_trainAndtest(dataset)


if args.model_input=='waveform':
    dataset_train=loadall_audio_train_waveform(dataset)
    dataset_test=loadall_audio_test_waveform(dataset)
elif args.model_input=='logmelspec':
    dataset_train=loadall_audio_train_spectrogram(dataset)
    dataset_test=loadall_audio_train_spectrogram(dataset)
elif args.model_input=='downsampled':
    dataset_train=loadall_audio_train_both(dataset)
    dataset_test=loadall_audio_train_both(dataset)

SE_LAYERS = 13 # NUMBER OF INTERNAL LAYERS
SE_CHANNELS = 64 # NUMBER OF FEATURE CHANNELS PER LAYER
SE_LOSS_LAYERS = args.loss_layers # NUMBER OF FEATURE LOSS LAYERS
SE_NORM = "NM" # TYPE OF LAYER NORMALIZATION (NM, SBN or None)
SE_LOSS_TYPE = "FL" # TYPE OF TRAINING LOSS (L1, L2 or FL)

# FEATURE LOSS NETWORK
LOSS_LAYERS = args.layers # NUMBER OF INTERNAL LAYERS
LOSS_BASE_CHANNELS = 32 # NUMBER OF FEATURE CHANNELS PER LAYER IN FIRT LAYER
LOSS_BLK_CHANNELS = args.channels_increase # NUMBER OF LAYERS BETWEEN CHANNEL NUMBER UPDATES
LOSS_NORM = args.loss_norm # TYPE OF LAYER NORMALIZATION (NM, SBN or None)

FILTER_SIZE = args.filter_size
epoches=args.epochs

if args.model_input=='waveform':
    
    with tf.variable_scope(tf.get_variable_scope()):
        input1_wav=tf.placeholder(tf.float32,shape=[None, None, None,1])

        loss_weights = tf.placeholder(tf.float32, shape=[SE_LOSS_LAYERS])

        clean1_wav=tf.placeholder(tf.float32,shape=[None, None, None,1])

        enhanced = featureloss_waveform(input1_wav,clean1_wav,loss_weights,loss_layers=SE_LOSS_LAYERS,n_layers=LOSS_LAYERS, norm_type=LOSS_NORM, base_channels=LOSS_BASE_CHANNELS,blk_channels=LOSS_BLK_CHANNELS,ksz=FILTER_SIZE) 

        enhanced1=tf.reshape(enhanced,[-1,len(enhanced)])
        weights = tf.Variable(tf.random_normal([len(enhanced),1]),
                          name="weights",trainable=True)
        weights2=tf.reshape(weights,[-1])
        weights1=tf.nn.softmax(weights2)
        weights3=tf.reshape(weights1,[-1,1])
        distance = tf.matmul(enhanced1, weights3)
        dist_sigmoid=tf.nn.sigmoid(distance)
        dense3=tf.layers.dense(distance,16,activation=tf.nn.relu)
        drop_out = tf.nn.dropout(dense3, 0.50)  # DROP-OUT here
        dense4=tf.layers.dense(drop_out,6,activation=tf.nn.relu)
        drop_out_1 = tf.nn.dropout(dense4, 0.50)  # DROP-OUT here
        dense2=tf.layers.dense(drop_out_1,2,None)
        label_task= tf.placeholder(tf.float32,shape=[None,2])
        net1 = tf.nn.softmax_cross_entropy_with_logits(labels=label_task,logits=dense2)
        loss_1=tf.reduce_mean(net1)
        if args.optimiser=='adam':
            opt_task = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss_1,var_list=[var for var in tf.trainable_variables()])
        elif args.optimiser=='gd':
            opt_task = tf.train.GradientDescentOptimizer(learning_rate=args.learning_rate).minimize(loss_1,var_list=[var for var in tf.trainable_variables()])

            
elif args.model_input=='logmelspec':
    with tf.variable_scope(tf.get_variable_scope()):
        input1_spec=tf.placeholder(tf.float32,shape=[None, None, None,1])

        loss_weights = tf.placeholder(tf.float32, shape=[SE_LOSS_LAYERS])

        clean1_spec=tf.placeholder(tf.float32,shape=[None, None, None,1])

        enhanced = featureloss_spectrogram(input1_spec,clean1_spec,loss_weights,loss_layers=SE_LOSS_LAYERS,n_layers=LOSS_LAYERS, norm_type=LOSS_NORM, base_channels=LOSS_BASE_CHANNELS,blk_channels=LOSS_BLK_CHANNELS) 

        enhanced1=tf.reshape(enhanced,[-1,len(enhanced)])
        weights = tf.Variable(tf.random_normal([len(enhanced),1]),
                          name="weights",trainable=True)
        weights2=tf.reshape(weights,[-1])
        weights1=tf.nn.softmax(weights2)
        weights3=tf.reshape(weights1,[-1,1])
        distance = tf.matmul(enhanced1, weights3)
        dist_sigmoid=tf.nn.sigmoid(distance)
        dense3=tf.layers.dense(distance,16,activation=tf.nn.relu)
        drop_out = tf.nn.dropout(dense3, 0.50)  # DROP-OUT here
        dense4=tf.layers.dense(drop_out,6,activation=tf.nn.relu)
        drop_out_1 = tf.nn.dropout(dense4, 0.50)  # DROP-OUT here
        dense2=tf.layers.dense(drop_out_1,2,None)
        label_task= tf.placeholder(tf.float32,shape=[None,2])
        net1 = tf.nn.softmax_cross_entropy_with_logits(labels=label_task,logits=dense2)
        loss_1=tf.reduce_mean(net1)
        if args.optimiser=='adam':
            opt_task = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss_1,var_list=[var for var in tf.trainable_variables()])
        elif args.optimiser=='gd':
            opt_task = tf.train.GradientDescentOptimizer(learning_rate=args.learning_rate).minimize(loss_1,var_list=[var for var in tf.trainable_variables()])

            
elif args.model_input=='downsampled':
    with tf.variable_scope(tf.get_variable_scope()):
        input1_spec=tf.placeholder(tf.float32,shape=[None, None, None,1])
        input1_wav=tf.placeholder(tf.float32,shape=[None, None, None,1])
        wav2spec=waveform2spec_1(input1_wav,n_layers=4,kernel=2,reuse=False)
        reshaped_wav=tf.reshape(wav2spec[-1],[1,248,64,1])
        reshaped_spec=tf.reshape(input1_spec,[1,248,64,1])
        spec_waveform=tf.concat([reshaped_wav, reshaped_spec], -1)
        reshaped_waveform=tf.reshape(spec_waveform,[1,248,64,2])

        loss_weights = tf.placeholder(tf.float32, shape=[SE_LOSS_LAYERS])

        clean1_spec=tf.placeholder(tf.float32,shape=[None, None, None,1])
        clean1_wav=tf.placeholder(tf.float32,shape=[None, None, None,1])
        wav2spec_1=waveform2spec_2(clean1_wav,n_layers=4,kernel=2,reuse=False)
        reshaped_wav_1=tf.reshape(wav2spec_1[-1],[1,248,64,1])
        reshaped_spec_1=tf.reshape(clean1_spec,[1,248,64,1])
        spec_waveform_1=tf.concat([reshaped_wav_1, reshaped_spec_1], -1)
        reshaped_waveform_1=tf.reshape(spec_waveform_1,[1,248,64,2])

        enhanced = featureloss_spectrogram(reshaped_waveform,reshaped_waveform_1,loss_weights,loss_layers=SE_LOSS_LAYERS,n_layers=LOSS_LAYERS, norm_type=LOSS_NORM, base_channels=LOSS_BASE_CHANNELS,blk_channels=LOSS_BLK_CHANNELS) 

        enhanced1=tf.reshape(enhanced,[-1,len(enhanced)])
        weights = tf.Variable(tf.random_normal([len(enhanced),1]),
                          name="weights",trainable=True)
        weights2=tf.reshape(weights,[-1])
        weights1=tf.nn.softmax(weights2)
        weights3=tf.reshape(weights1,[-1,1])
        distance = tf.matmul(enhanced1, weights3)
        dist_sigmoid=tf.nn.sigmoid(distance)
        dense3=tf.layers.dense(distance,16,activation=tf.nn.relu)
        drop_out = tf.nn.dropout(dense3, 0.50)  # DROP-OUT here
        dense4=tf.layers.dense(drop_out,6,activation=tf.nn.relu)
        drop_out_1 = tf.nn.dropout(dense4, 0.50)  # DROP-OUT here
        dense2=tf.layers.dense(drop_out_1,2,None)
        label_task= tf.placeholder(tf.float32,shape=[None,2])
        net1 = tf.nn.softmax_cross_entropy_with_logits(labels=label_task,logits=dense2)
        loss_1=tf.reduce_mean(net1)
        if args.optimiser=='adam':
            opt_task = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss_1,var_list=[var for var in tf.trainable_variables()])
        elif args.optimiser=='gd':
            opt_task = tf.train.GradientDescentOptimizer(learning_rate=args.learning_rate).minimize(loss_1,var_list=[var for var in tf.trainable_variables()])
        

with tf.name_scope('performance'):
    
    tf_loss_ph_train = tf.placeholder(tf.float32,shape=None,name='loss_summary_train')
    tf_loss_summary_train = tf.summary.scalar('loss_train', tf_loss_ph_train)
 
    tf_loss_ph_test = tf.placeholder(tf.float32,shape=None,name='loss_summary_test')
    tf_loss_summary_test = tf.summary.scalar('loss_test', tf_loss_ph_test)
    
    tf_loss_ph_map_linear = tf.placeholder(tf.float32,shape=None,name='loss_summary_map_linear')
    tf_loss_summary_map_linear = tf.summary.scalar('loss_map_linear', tf_loss_ph_map_linear)
    
    tf_loss_ph_map_reverb = tf.placeholder(tf.float32,shape=None,name='loss_summary_map_reverb')
    tf_loss_summary_map_reverb = tf.summary.scalar('loss_map_reverb', tf_loss_ph_map_reverb)
    
    tf_loss_ph_map_mp3 = tf.placeholder(tf.float32,shape=None,name='loss_summary_map_mp3')
    tf_loss_summary_map_mp3 = tf.summary.scalar('loss_map_mp3', tf_loss_ph_map_mp3)
    
    tf_loss_ph_map_combined = tf.placeholder(tf.float32,shape=None,name='loss_summary_map_combined')
    tf_loss_summary_map_combined = tf.summary.scalar('loss_map_combined', tf_loss_ph_map_combined)

performance_summaries_train = tf.summary.merge([tf_loss_summary_train])
performance_summaries_test = tf.summary.merge([tf_loss_summary_test])
performance_summaries_map_linear = tf.summary.merge([tf_loss_summary_map_linear])
performance_summaries_map_reverb = tf.summary.merge([tf_loss_summary_map_reverb])
performance_summaries_map_mp3 = tf.summary.merge([tf_loss_summary_map_mp3])
performance_summaries_map_combined = tf.summary.merge([tf_loss_summary_map_combined])


with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    outfolder = args.summary_folder
    saver = tf.train.Saver(max_to_keep=10)
    
    if args.train_from_checkpoint==0:
        os.mkdir(os.path.join('summaries',outfolder))
    elif args.train_from_checkpoint==1:
        path=os.path.join('summaries',outfolder)
        saver.restore(sess, "%s/my_test_model" % path)
        print('Loaded Checkpoint')
    
    summ_writer = tf.summary.FileWriter(os.path.join('summaries',outfolder), sess.graph)  
    
    for epoch in range(epoches):
        loss_epoch=[]
        
        batches=len(dataset_train['train']['inname'])
        n_batches = batches // 1
        
        for j in tqdm(range(batches)):
            
            if args.model_input=='waveform':
                wav_in,wav_out,labels=load_full_data_waveform(dataset_train,'train',j)
            elif args.model_input=='logmelspec':
                spec_in,spec_out,labels=load_full_data_spectrogram(dataset_train,'train',j)
            elif args.model_input=='downsampled':
                spec_in,spec_out,wav_in,wav_out,labels=load_full_data_both(dataset_train,'train',j)
            y=np.zeros((labels.shape[0],2))
            for i in range(labels.shape[0]):
                if float(labels[i])==0:
                    y[i]+=[1,0]
                elif float(labels[i])==1:
                    y[i]+=[0,1]
            loss_ones=np.ones([SE_LOSS_LAYERS])
            if args.model_input=='waveform':
                _,dist,loss_train= sess.run([opt_task,distance,loss_1],feed_dict={input1_wav:wav_in, clean1_wav:wav_out, loss_weights:loss_ones,label_task:y})
            elif args.model_input=='logmelspec':
                _,dist,loss_train= sess.run([opt_task,distance,loss_1],feed_dict={input1_spec:spec_in, clean1_spec:spec_out, loss_weights:loss_ones,label_task:y})
            elif args.model_input=='downsampled':
                 _,dist,loss_train= sess.run([opt_task,distance,loss_1],feed_dict={input1_spec:spec_in, clean1_spec:spec_out, input1_wav:wav_in, clean1_wav:wav_out,loss_weights:loss_ones,label_task:y})
            loss_epoch.append(loss_train)
                    
        if epoch%10==0:
            
            loss_epoch_test=[]
            batches=len(dataset_test['test']['inname'])
            n_batches = batches // 1
            for j in tqdm(range(batches)):
                
                if args.model_input=='waveform':
                    wav_in,wav_out,labels=load_full_data_waveform(dataset_test,'test',j)
                elif args.model_input=='logmelspec':
                    spec_in,spec_out,labels=load_full_data_spectrogram(dataset_test,'test',j)
                elif args.model_input=='downsampled':
                    spec_in,spec_out,wav_in,wav_out,labels=load_full_data_both(dataset_test,'test',j)
                y=np.zeros((labels.shape[0],2))
                for i in range(labels.shape[0]):
                    if float(labels[i])==0:
                        y[i]+=[1,0]
                    elif float(labels[i])==1:
                        y[i]+=[0,1]
                loss_ones=np.ones([SE_LOSS_LAYERS])
                if args.model_input=='waveform':
                    dist,loss_train= sess.run([distance,loss_1],feed_dict={input1_wav:wav_in, clean1_wav:wav_out, loss_weights:loss_ones,label_task:y})
                elif args.model_input=='logmelspec':
                    dist,loss_train= sess.run([distance,loss_1],feed_dict={input1_spec:spec_in, clean1_spec:spec_out, loss_weights:loss_ones,label_task:y})
                elif args.model_input=='downsampled':
                     dist,loss_train= sess.run([distance,loss_1],feed_dict={input1_spec:spec_in, clean1_spec:spec_out, input1_wav:wav_in, clean1_wav:wav_out,loss_weights:loss_ones,label_task:y})
                loss_epoch_test.append(loss_train)
                
                
            [ap0,auc0]=scores_map('linear',args.model_input)
            [ap1,auc1]=scores_map('reverb',args.model_input)
            [ap2,auc2]=scores_map('mp3',args.model_input)
            [ap3,auc3]=scores_map('combined',args.model_input)

            
            summ_map_linear = sess.run(performance_summaries_map_linear, feed_dict={tf_loss_ph_map_linear:ap0})
            summ_writer.add_summary(summ_map_linear, epoch)

            summ_map_reverb = sess.run(performance_summaries_map_reverb, feed_dict={tf_loss_ph_map_reverb:ap1})
            summ_writer.add_summary(summ_map_reverb, epoch)

            summ_map_mp3 = sess.run(performance_summaries_map_mp3, feed_dict={tf_loss_ph_map_mp3:ap2})
            summ_writer.add_summary(summ_map_mp3, epoch)
            
            summ_map_combined = sess.run(performance_summaries_map_combined, feed_dict={tf_loss_ph_map_combined:ap3})
            summ_writer.add_summary(summ_map_combined, epoch)
             
            summ_test = sess.run(performance_summaries_test, feed_dict={tf_loss_ph_test:sum(loss_epoch_test) / len(loss_epoch_test)})
            summ_writer.add_summary(summ_test, epoch)

        summ = sess.run(performance_summaries_train, feed_dict={tf_loss_ph_train: sum(loss_epoch) / len(loss_epoch)})
        summ_writer.add_summary(summ, epoch)

        print("Epoch {} Train Loss {}".format(epoch,sum(loss_epoch) / len(loss_epoch)))
        
        if epoch%20==0:
            saver.save(sess, os.path.join('summaries',outfolder,'my_test_model'))