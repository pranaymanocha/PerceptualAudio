from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from pip_pasm.network_model import *
from pip_pasm.helper import *
import os
import inspect


class PASM():
    def __init__(self, modfolder='pre-model/scratch_loss', type='scratch'):
        
        
        ## Training Parameters
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
        self.modfolder = modfolder 
        self.type = type
        
        modfolder= os.path.abspath(os.path.join(inspect.getfile(self.__init__), '..', 'pre-model/'+self.type+'_loss'))
        tf.reset_default_graph()

        with tf.variable_scope(tf.get_variable_scope()):

            input1_wav=tf.placeholder(tf.float32,shape=[None, None, None,1])
            self.input1_wav = input1_wav
            clean1_wav=tf.placeholder(tf.float32,shape=[None, None, None,1])
            self.clean1_wav = clean1_wav
            keep_prob = tf.placeholder_with_default(1.0, shape=())

            if self.type!='pretrained':
                
                others,loss_sum = featureloss(input1_wav,clean1_wav,keep_prob,loss_layers=SE_LOSS_LAYERS,n_layers=LOSS_LAYERS, norm_type=LOSS_NORM, base_channels=LOSS_BASE_CHANNELS,blk_channels=LOSS_BLK_CHANNELS,ksz=FILTER_SIZE)

            else:
                
                others,loss_sum = featureloss_pretrained(input1_wav,clean1_wav,keep_prob,loss_layers=SE_LOSS_LAYERS,n_layers=LOSS_LAYERS, norm_type=LOSS_NORM, base_channels=LOSS_BASE_CHANNELS,blk_channels=LOSS_BLK_CHANNELS,ksz=FILTER_SIZE) 
            
            res=tf.reduce_sum(others,0)
            distance=res
            self.distance = distance
            #distance = loss_sum
        
        sess = tf.compat.v1.Session()
        #with tf.Session() as sess:
        loss_saver = tf.train.Saver([var for var in tf.trainable_variables()])

        if self.type=='pretrained':
            loss_saver.restore(sess, "%s/loss_model.ckpt" % modfolder)
        else:
            loss_saver.restore(sess, "%s/my_test_model" % modfolder)
        self.sess=sess
    
    def forward(self, wav_in=1, wav_out=1):
        
        dist= self.sess.run([self.distance],feed_dict={self.input1_wav:wav_out, self.clean1_wav:wav_in})
        
        return dist

## TO DO: write this class to train on the JND Data    
class Train_PASM():
    def __init__(self, modfolder='../pre-model/scratch_loss', type='scratch'):
        
        ## Training Parameters
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
        self.modfolder = modfolder 
        self.type = type
        
        modfolder= self.modfolder

        tf.reset_default_graph()

        with tf.variable_scope(tf.get_variable_scope()):

            input1_wav=tf.placeholder(tf.float32,shape=[None, None, None,1])
            self.input1_wav = input1_wav
            clean1_wav=tf.placeholder(tf.float32,shape=[None, None, None,1])
            self.clean1_wav = clean1_wav
            keep_prob = tf.placeholder_with_default(1.0, shape=())
            
            others,loss_sum = featureloss_batch(input1_wav,clean1_wav,keep_prob,loss_layers=SE_LOSS_LAYERS,n_layers=LOSS_LAYERS, norm_type=LOSS_NORM, base_channels=LOSS_BASE_CHANNELS,blk_channels=LOSS_BLK_CHANNELS,ksz=FILTER_SIZE) 

            res=tf.reduce_sum(others,0)
            distance=res
            
            dist_sigmoid=tf.nn.sigmoid(distance)
            dist_sigmoid_1=tf.reshape(dist_sigmoid,[-1,1,1])

            if args.type=='linear':

                dense3=tf.layers.dense(dist_sigmoid_1,16,activation=tf.nn.relu)
                dense4=tf.layers.dense(dense3,6,activation=tf.nn.relu)
                dense2=tf.layers.dense(dense4,2,None)
                label_task= tf.placeholder(tf.float32,shape=[None,2])
                net1 = tf.nn.softmax_cross_entropy_with_logits(labels=label_task,logits=dense2)
                loss_1=tf.reduce_mean(net1)
                if args.optimiser=='adam':
                    opt_task = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss_1,var_list=[var for var in tf.trainable_variables() if not var.name.startswith("loss_conv")])
                elif args.optimiser=='gd':
                    opt_task = tf.train.GradientDescentOptimizer(learning_rate=args.learning_rate).minimize(loss_1,var_list=[var for var in tf.trainable_variables() if not var.name.startswith("loss_conv")])

            else:
                
                dense3=tf.layers.dense(dist_sigmoid_1,16,activation=tf.nn.relu)
                dense4=tf.layers.dense(dense3,6,activation=tf.nn.relu)
                dense2=tf.layers.dense(dense4,2,None)
                label_task= tf.placeholder(tf.float32,shape=[None,2])
                net1 = tf.nn.softmax_cross_entropy_with_logits(labels=label_task,logits=dense2)
                loss_1=tf.reduce_mean(net1)
                if args.optimiser=='adam':
                    opt_task = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss_1,var_list=[var for var in tf.trainable_variables()])
                elif args.optimiser=='gd':
                    opt_task = tf.train.GradientDescentOptimizer(learning_rate=args.learning_rate).minimize(loss_1,var_list=[var for var in tf.trainable_variables()])
            #distance = loss_sum
        
        sess = tf.compat.v1.Session()
        #with tf.Session() as sess:
        loss_saver = tf.train.Saver([var for var in tf.trainable_variables()])

        if self.type=='pretrained':
            loss_saver.restore(sess, "%s/loss_model.ckpt" % modfolder)
        else:
            loss_saver.restore(sess, "%s/my_test_model" % modfolder)
        self.sess=sess
    
    def forward(self, wav_in=1, wav_out=1):
        
        dist= self.sess.run([self.distance],feed_dict={self.input1_wav:wav_out, self.clean1_wav:wav_in})
        
        return dist