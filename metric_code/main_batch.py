import numpy as np

from tqdm import tqdm
from scipy.io import wavfile
import os, csv
import tensorflow as tf
import pickle

from helper import *
#from network_model import *
from dataloader import *
#from MAP_eval import *

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import numpy as np

import argparse


def load_full_data_list(datafolder='../'): #check change path names

    #sets=['train','val']
    dataset={}
    dataset['all']={}
    
    print('Loading files..')
    dataset['all']['inname'] = []
    dataset['all']['outname'] = []
    dataset['all']['label']=[]
    
    
    print("Prefetching the Combined")
    #data_path='prefetch_audio_new_mp3_new_morebandwidth'
    list_path='../'
    file = open(os.path.join(datafolder,'dataset_train_combined_all_shuffled.txt'), 'r')
    for line in file:
        split_line=line.split('\t')
        dataset['all']['inname'].append("%s"%(os.path.join(list_path,split_line[0])))
        dataset['all']['outname'].append("%s"%(os.path.join(list_path,split_line[1])))
        dataset['all']['label'].append(split_line[2][:-1])
     
    print("Prefetching the Reverb")  
    list_path='../'
    file = open(os.path.join(list_path,'dataset_train_shuffled_reverbBatch.txt'), 'r')
    for line in file:
        split_line=line.split('\t')
        dataset['all']['inname'].append("%s"%(os.path.join(list_path,split_line[0])))
        dataset['all']['outname'].append("%s"%(os.path.join(list_path,split_line[1])))
        dataset['all']['label'].append(split_line[2][:-1])
    
    print("Prefetching the Linear Noises")
    noises=['applause','blue_noise','brown_noise','crickets','pink_noise','reverb_noise','siren','violet_noise','water_drops','white_noise','mp3']
    for noise in noises:
        file = open(os.path.join(datafolder,'dataset_train.txt'), 'r')
        for line in file: 
            split_line=line.split('\t')
            if split_line[3][:-1].strip()==noise:
                dataset['all']['inname'].append("%s_list/%s"%(datafolder+noise,split_line[0]))
                dataset['all']['outname'].append("%s_list/%s"%(datafolder+noise,split_line[1]))
                dataset['all']['label'].append(split_line[2])
                          
    print("Prefetching the EQ")
    list_path='../'
    file = open(os.path.join(list_path,'dataset_shuffled_eqBatch.txt'), 'r')
    for line in file:
        split_line=line.split('\t')
        dataset['all']['inname'].append("%s"%(os.path.join(list_path,split_line[0])))
        dataset['all']['outname'].append("%s"%(os.path.join(list_path,split_line[1])))
        dataset['all']['label'].append(split_line[2][:-1])
        
    return dataset


def lossnet(input, keep_prob,n_layers=14, training=True, reuse=False, norm_type="SBN",
               ksz=3, base_channels=32, blk_channels=5):
    layers = []

    if norm_type == "NM": # ADAPTIVE BATCH NORM
        norm_fn = nm
    elif norm_type == "SBN": # BATCH NORM
        norm_fn = slim.batch_norm
    else: # NO LAYER NORMALIZATION
        norm_fn = None

    for id in range(n_layers):

        n_channels = base_channels * (2 ** (id // blk_channels)) # UPDATE CHANNEL COUNT

        if id == 0:
            net = slim.conv2d(input, n_channels, [1, ksz], activation_fn=lrelu, normalizer_fn=norm_fn, stride=[1, 2],
                              scope='loss_conv_%d' % id, padding='SAME', reuse=reuse)
            net = slim.dropout(net, keep_prob, scope='Dropout_%d' %id)
            layers.append(net)
        elif id < n_layers - 1:
            net = slim.conv2d(layers[-1], n_channels, [1, ksz], activation_fn=lrelu, normalizer_fn=norm_fn,
                              stride=[1, 2], scope='loss_conv_%d' % id, padding='SAME', reuse=reuse)
            net = slim.dropout(net, keep_prob, scope='Dropout_%d' %id)
            layers.append(net)
        else:
            net = slim.conv2d(layers[-1], n_channels, [1, ksz], activation_fn=lrelu, normalizer_fn=norm_fn,
                              scope='loss_conv_%d' % id, padding='SAME', reuse=reuse)
            layers.append(net)

    return layers

def l1_loss_batch(target):
    return tf.reduce_mean(tf.abs(target),axis=[1,2,3])

def featureloss_train(target, current, keep_prob,loss_layers, n_layers=14, norm_type="SBN", base_channels=32, blk_channels=5,ksz=3):

    feat_current = lossnet(current, reuse=False, n_layers=n_layers, norm_type=norm_type,
                         base_channels=base_channels, blk_channels=blk_channels,training=False,ksz=ksz,keep_prob=keep_prob)

    feat_target = lossnet(target, reuse=True, n_layers=n_layers, norm_type=norm_type,
                         base_channels=base_channels, blk_channels=blk_channels,training=False,ksz=ksz,keep_prob=keep_prob)
    
    loss_vec = []
    
    channels = np.asarray([base_channels * (2 ** (id // blk_channels)) for id in range(n_layers)])
    
    for id in range(loss_layers):
        a=feat_current[id]-feat_target[id]
        weights = tf.Variable(tf.random_normal([channels[id]]),
                      name="weights_%d" %id, trainable=True)
        a1=tf.transpose(a, [0, 1, 3, 2])
        #a1=tf.reshape(a,[t,1,channels[id],-1])
        result=tf.multiply(a1, weights[:,tf.newaxis])
        loss_result=l1_loss_batch(result)
        loss_vec.append(loss_result)
    
    return loss_vec,loss_vec

def argument_parser():
    """
    Get an argument parser for a training script.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--layers', help='number of layers in the model', default=14, type=int)
    parser.add_argument('--learning_rate', help='learning rate', default=1e-4, type=float)
    parser.add_argument('--summary_folder', help='summary folder name', default='m_example')
    parser.add_argument('--optimiser', help='choose optimiser - gd/adam', default='adam')
    parser.add_argument('--loss_norm', help='loss norm - NM,SBN,None', default='SBN')
    parser.add_argument('--channels_increase', help='doubling channels after how many layers - 1,2,3,4,5,6', default=5, type=int)
    parser.add_argument('--loss_layers', help='loss to be taken for the first how many layers', default=14, type=int)
    parser.add_argument('--filter_size', help='filter size for the convolutions', default=3, type=int)
    parser.add_argument('--train_from_checkpoint', help='train_from_checkpoint', default=0, type=int)
    parser.add_argument('--epochs', help='number of training epochs', default=2000, type=int)
    parser.add_argument('--type', help='linear/finetune/scratch', default='scratch')
    parser.add_argument('--pretrained_model_path', help='Model Path for the pretrained model', default='../pre-model/pretrained_loss')
    parser.add_argument('--batch_size', help='batch_size', default=16,type=int)
    return parser

args = argument_parser().parse_args()

##Dataset Load
dataset=load_full_data_list()
dataset=split_trainAndtest(dataset)
dataset_train=loadall_audio_train_waveform(dataset)
dataset_test=loadall_audio_test_waveform_batch(dataset)

##Model Params

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

##Model network - lin,fin and scratch
with tf.variable_scope(tf.get_variable_scope()):
    input1_wav=tf.placeholder(tf.float32,shape=[None, None, None,1])
    
    clean1_wav=tf.placeholder(tf.float32,shape=[None, None, None,1])

    keep_prob = tf.placeholder_with_default(1.0, shape=())
    
    enhanced,loss_sum = featureloss_train(input1_wav,clean1_wav,keep_prob,loss_layers=SE_LOSS_LAYERS,n_layers=LOSS_LAYERS, norm_type=LOSS_NORM, base_channels=LOSS_BASE_CHANNELS,blk_channels=LOSS_BLK_CHANNELS,ksz=FILTER_SIZE) 
    
    res=tf.reduce_mean(enhanced,0)
    distance=res
    
    #distance = loss_sum
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


## function for MAP eval
def scores_map(noise):
    
    filename='dataset_test_'+noise+'.txt'
    
    import numpy as np
    if noise!='combined':
            dataset_test=load_full_data_list_test('../',filename)
    elif noise=='combined':
        dataset_test=load_full_data_list_combined_test('../',filename)
    
    output=np.zeros((len(dataset_test["all"]["inname"]),1))
                     
    for id in tqdm(range(0, len(dataset_test["all"]["inname"]))):

        wav_in,wav_out=load_full_data_test_waveform(dataset_test,'all',id)
        a,_= sess.run([distance,dist_sigmoid],feed_dict={input1_wav:wav_in, clean1_wav:wav_out})
        output[id]=a
    
    import numpy as np
    perceptual=[]
    for i in range(len(dataset_test['all']['label'])):
        perceptual.append(float(dataset_test['all']['label'][i]))
    perceptual=(np.array(perceptual))
    perceptual=1-perceptual
    
    label=[]
    for i in range(len(output)):
        label.append(output[i][0])
    label=np.array(label)

    a=np.argsort(label) # numbered lists distance output by the audio metric
    a1=np.sort(label)

    label_sorted=label[a]
    perceptual_sorted = perceptual[a] 

    TPs = np.cumsum(perceptual_sorted)
    FPs = np.cumsum(1-perceptual_sorted)
    FNs = np.sum(perceptual_sorted)-TPs
    TNs = np.sum(1-perceptual_sorted)-FPs

    precs = TPs/(TPs+FPs)
    recs = TPs/(TPs+FNs)
    #print(recs)
    tpr=TPs/(TPs+FNs)
    fpr=FPs/(FPs+TNs)
    #print(output)
    score = voc_ap(recs,precs)
    #print(score) # as high as possible
    from sklearn import metrics
    metrics_points=metrics.auc(fpr, tpr)
    #print(metrics_points) # as high as possible than 0.50 to be meaningful
    return [score,metrics_points]
       
          
##Tensorboard Visualisation
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


def load_full_data_batch(dataset,sets,id_value):
    
    for i in range(len(id_value[0])):
        
        id = id_value[0][i][0]
        
        inputData_wav=dataset[sets]['inaudio'][id]
        outputData_wav=dataset[sets]['outaudio'][id]
        label = np.reshape(np.asarray(dataset[sets]['label'][id]),[-1,1])
        
        if i==0:
            waveform_in=inputData_wav
            waveform_out=outputData_wav
            #spec_in=inputData_spec
            #spec_out=outputData_spec
            labels=label
        elif i!=0:
            waveform_in=np.concatenate((waveform_in,inputData_wav),axis=0)
            waveform_out=np.concatenate((waveform_out,outputData_wav),axis=0)
            labels=np.concatenate((labels,label),axis=0)
    
    return [waveform_in,waveform_out,labels]


##Train and Test Loop
#linear,finetune and scratch
with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    outfolder = args.summary_folder
    
    if args.type=='linear' or args.type=='finetune':
        
        modfolder=args.pretrained_model_path
        loss_saver = tf.train.Saver([var for var in tf.trainable_variables() if var.name.startswith("loss_")])
        loss_saver.restore(sess, "%s/loss_model.ckpt" % modfolder)
        print('Loaded Pretrained Weights')
        
    if args.train_from_checkpoint==0:
        os.mkdir(os.path.join('summaries',outfolder))
    elif args.train_from_checkpoint==1:
        path=os.path.join('summaries',outfolder)
        saver.restore(sess, "%s/my_test_model" % path)
        print('Loaded Checkpoint')
    
    saver = tf.train.Saver(max_to_keep=20, keep_checkpoint_every_n_hours=8)
    summ_writer = tf.summary.FileWriter(os.path.join('summaries',outfolder), sess.graph)  
    
    for epoch in range(epoches):
        loss_epoch=[]
        
        BATCH_SIZE=args.batch_size
        features =  np.arange(0,len(dataset_train['train']['inname']))
        features=np.reshape(features,[-1,1])
        dataset1=tf.data.Dataset.from_tensor_slices((features)).shuffle(1000).batch(BATCH_SIZE)
        iter = dataset1.make_initializable_iterator()
        x = iter.get_next()
        sess.run(iter.initializer)
        
        batches=len(dataset_train['train']['inname'])
        n_batches = batches // BATCH_SIZE
        
        for j in tqdm(range(n_batches)):
            
            a=sess.run([x])
            wav_in,wav_out,labels=load_full_data_batch(dataset_train,'train',a)
                
            y=np.zeros((labels.shape[0],2))
            for i in range(labels.shape[0]):
                if labels[i]=='0':
                    y[i]+=[1,0]
                elif labels[i]=='1':
                    y[i]+=[0,1]
            
            keep_prob_drop=1
            if args.type!='linear' or args.type!='finetune':
                keep_prob_drop=0.70
            
            _,dist,loss_train= sess.run([opt_task,distance,loss_1],feed_dict={input1_wav:wav_in, clean1_wav:wav_out,label_task:y,keep_prob:keep_prob_drop})
            loss_epoch.append(loss_train)
                    
        if epoch%10==0:
            
            loss_epoch_test=[]
            
            BATCH_SIZE=args.batch_size
            features =  np.arange(0,len(dataset_test['test']['inname']))
            features=np.reshape(features,[-1,1])
            dataset1=tf.data.Dataset.from_tensor_slices((features)).shuffle(1000).batch(BATCH_SIZE)
            iter = dataset1.make_initializable_iterator()
            x = iter.get_next()
            sess.run(iter.initializer)

            batches=len(dataset_test['test']['inname'])
            n_batches = batches // BATCH_SIZE
            
            for j in tqdm(range(n_batches)):
                a=sess.run([x])
                wav_in,wav_out,labels=load_full_data_batch(dataset_test,'test',a)
                
                y=np.zeros((labels.shape[0],2))
                for i in range(labels.shape[0]):
                    if labels[i]=='0':
                        y[i]+=[1,0]
                    elif labels[i]=='1':
                        y[i]+=[0,1]
                
                dist,loss_train= sess.run([distance,loss_1],feed_dict={input1_wav:wav_in, clean1_wav:wav_out,label_task:y})
                  
                loss_epoch_test.append(loss_train)
            '''    
            [ap0,auc0]=scores_map('linear')
            [ap1,auc1]=scores_map('reverb')
            [ap2,auc2]=scores_map('mp3')
            [ap3,auc3]=scores_map('combined')

            summ_map_linear = sess.run(performance_summaries_map_linear, feed_dict={tf_loss_ph_map_linear:ap0})
            summ_writer.add_summary(summ_map_linear, epoch)

            summ_map_reverb = sess.run(performance_summaries_map_reverb, feed_dict={tf_loss_ph_map_reverb:ap1})
            summ_writer.add_summary(summ_map_reverb, epoch)

            summ_map_mp3 = sess.run(performance_summaries_map_mp3, feed_dict={tf_loss_ph_map_mp3:ap2})
            summ_writer.add_summary(summ_map_mp3, epoch)
            
            summ_map_combined = sess.run(performance_summaries_map_combined, feed_dict={tf_loss_ph_map_combined:ap3})
            summ_writer.add_summary(summ_map_combined, epoch)
            '''
            summ_test = sess.run(performance_summaries_test, feed_dict={tf_loss_ph_test:sum(loss_epoch_test) / len(loss_epoch_test)})
            summ_writer.add_summary(summ_test, epoch)
            
        summ = sess.run(performance_summaries_train, feed_dict={tf_loss_ph_train: sum(loss_epoch) / len(loss_epoch)})
        summ_writer.add_summary(summ, epoch)
        
        print("Epoch {} Train Loss {}".format(epoch,sum(loss_epoch) / len(loss_epoch)))
        
        if epoch%20==0:
            saver.save(sess, os.path.join('summaries',outfolder,'my_test_model'))