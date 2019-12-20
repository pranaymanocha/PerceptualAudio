import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from helper import *

# FEATURE LOSS NETWORK
def lossnet(input, keep_prob,n_layers=14, training=True, reuse=False, norm_type="SBN",
               ksz=3, base_channels=32, blk_channels=5,lin_layer=0):
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
            
            if lin_layer==1:
                print('lin_layer_taken')
                net=tf.reduce_max(net, reduction_indices=[2], keep_dims=True)
                net=tf.reshape(net,[1,n_channels])
                net = slim.fully_connected(net, 512, activation_fn=lrelu, normalizer_fn=None, scope='loss_fc',reuse=reuse)
                net=tf.reshape(net,[1,1,512,1])
                layers.append(net)
    return layers


def lossnet_spec(input, keep_prob,n_layers=14, training=True, reuse=False, norm_type="SBN",
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
            net = slim.conv2d(input, n_channels, [ksz, ksz], activation_fn=lrelu, normalizer_fn=norm_fn, stride=[2, 2],
                              scope='loss_conv_%d' % id, padding='SAME', reuse=reuse)
            net = slim.dropout(net, keep_prob, scope='Dropout_%d' %id)
            layers.append(net)
        elif id < n_layers - 1:
            net = slim.conv2d(layers[-1], n_channels, [ksz, ksz], activation_fn=lrelu, normalizer_fn=norm_fn,
                              stride=[2, 2], scope='loss_conv_%d' % id, padding='SAME', reuse=reuse)
            net = slim.dropout(net, keep_prob, scope='Dropout_%d' %id)
            layers.append(net)
        else:
            net = slim.conv2d(layers[-1], n_channels, [ksz, ksz], activation_fn=lrelu, normalizer_fn=norm_fn,
                              scope='loss_conv_%d' % id, padding='SAME', reuse=reuse)
            layers.append(net)
    
    return layers


def featureloss(target, current, keep_prob ,loss_layers, n_layers=14, norm_type="SBN", base_channels=32, blk_channels=5,ksz=3):
    
    feat_current = lossnet(current, reuse=False, n_layers=n_layers, norm_type=norm_type,
                         base_channels=base_channels, blk_channels=blk_channels,training=False,ksz=ksz,keep_prob=keep_prob)

    feat_target = lossnet(target, reuse=True, n_layers=n_layers, norm_type=norm_type,
                         base_channels=base_channels, blk_channels=blk_channels,training=False,ksz=ksz,keep_prob=keep_prob)
    
    
    loss_vec = [0]
    channels = np.asarray([base_channels * (2 ** (id // blk_channels)) for id in range(n_layers)])
    
    for id in range(loss_layers):
        a=feat_current[id]-feat_target[id]
        weights = tf.Variable(tf.random_normal([channels[id]]),
                      name="weights_%d" %id, trainable=True)
        a1=tf.transpose(a, [0, 1, 3, 2])
        result=tf.multiply(a1, weights[:,tf.newaxis])
        loss_result=l1_loss_all(result)
        loss_vec.append(loss_result)
        loss_vec[0]+=loss_result
    return loss_vec[1:],loss_vec[0]

def featureloss_pretrained(target, current, keep_prob ,loss_layers, n_layers=14, norm_type="SBN", base_channels=32, blk_channels=5,ksz=3):
    
    feat_current = lossnet(current, reuse=False, n_layers=n_layers, norm_type=norm_type,
                         base_channels=base_channels, blk_channels=blk_channels,training=False,ksz=ksz,keep_prob=keep_prob)

    feat_target = lossnet(target, reuse=True, n_layers=n_layers, norm_type=norm_type,
                         base_channels=base_channels, blk_channels=blk_channels,training=False,ksz=ksz,keep_prob=keep_prob)
    

    loss_vec = [0]
    for id in range(loss_layers):
        loss_vec.append(l1_loss(feat_current[id], feat_target[id]))

    for id in range(1,loss_layers+1):
        loss_vec[0] += loss_vec[id]

    return loss_vec[0],loss_vec[0]

def featureloss_batch(target, current, keep_prob,loss_layers, n_layers=14, norm_type="SBN", base_channels=32, blk_channels=5,ksz=3,lin_layer=0):

    feat_current = lossnet(current, reuse=False, n_layers=n_layers, norm_type=norm_type,
                         base_channels=base_channels, blk_channels=blk_channels,training=False,ksz=ksz,keep_prob=keep_prob,lin_layer=lin_layer)

    feat_target = lossnet(target, reuse=True, n_layers=n_layers, norm_type=norm_type,
                         base_channels=base_channels, blk_channels=blk_channels,training=False,ksz=ksz,keep_prob=keep_prob,lin_layer=lin_layer)
    
    if lin_layer==1:
        
        loss_vec = []
        channels = np.asarray([base_channels * (2 ** (id // blk_channels)) for id in range(n_layers)])
        
        for id in range(loss_layers+1):
            
            a=feat_current[id]-feat_target[id]
            if id==loss_layers:
                weights = tf.Variable(tf.random_normal([1]),name="weights_%d" %id, trainable=True)
            else:
                weights = tf.Variable(tf.random_normal([channels[id]]),name="weights_%d" %id, trainable=True)
            a1=tf.transpose(a, [0, 1, 3, 2])
            result=tf.multiply(a1, weights[:,tf.newaxis])
            loss_result=l1_loss_batch(result)
            loss_vec.append(loss_result)
    else:
        
        loss_vec = []
        channels = np.asarray([base_channels * (2 ** (id // blk_channels)) for id in range(n_layers)])
        
        for id in range(loss_layers):

            a=feat_current[id]-feat_target[id]
            weights = tf.Variable(tf.random_normal([channels[id]]),name="weights_%d" %id, trainable=True)
            a1=tf.transpose(a, [0, 1, 3, 2])
            result=tf.multiply(a1, weights[:,tf.newaxis])
            loss_result=l1_loss_batch(result)
            loss_vec.append(loss_result)
    
    return loss_vec,loss_vec

def featureloss_spec_batch(target, current, keep_prob,loss_layers, n_layers=14, norm_type="SBN", base_channels=32, blk_channels=5,ksz=3):

    feat_current = lossnet_spec(current, reuse=False, n_layers=n_layers, norm_type=norm_type,
                         base_channels=base_channels, blk_channels=blk_channels,training=False,ksz=ksz,keep_prob=keep_prob)

    feat_target = lossnet_spec(target, reuse=True, n_layers=n_layers, norm_type=norm_type,
                         base_channels=base_channels, blk_channels=blk_channels,training=False,ksz=ksz,keep_prob=keep_prob)
    
    loss_vec = []
    
    channels = np.asarray([base_channels * (2 ** (id // blk_channels)) for id in range(n_layers)])
    
    for id in range(loss_layers):
        a=feat_current[id]-feat_target[id]
        weights = tf.Variable(tf.random_normal([channels[id]]),
                      name="weights_%d" %id, trainable=True)
        a1=tf.transpose(a, [0, 1, 3, 2])
        result=tf.multiply(a1, weights[:,tf.newaxis])
        loss_result=l1_loss_batch(result)
        loss_vec.append(loss_result)
    
    return loss_vec,loss_vec

def lossnet_siamese(input, keep_prob,n_layers=14, training=True, reuse=False, norm_type="SBN",
               ksz=3, base_channels=32, blk_channels=5,lin_layer=False):
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
            net = slim.conv2d(net, n_channels, [1, ksz], activation_fn=lrelu, normalizer_fn=norm_fn,
                              stride=[1, 2], scope='loss_conv_%d' % id, padding='SAME', reuse=reuse)
            net = slim.dropout(net, keep_prob, scope='Dropout_%d' %id)
            layers.append(net)
        else:
            net = slim.conv2d(net, n_channels, [1, ksz], activation_fn=lrelu, normalizer_fn=norm_fn,
                              scope='loss_conv_%d' % id, padding='SAME', reuse=reuse)
            if lin_layer==True:
                net=tf.reduce_max(net, reduction_indices=[2], keep_dims=True)
                net=tf.reshape(net,[1,n_channels])
                net = slim.fully_connected(net, 512, activation_fn=lrelu, normalizer_fn=None, scope='loss_fc',reuse=reuse)
            else:
                net = slim.flatten(net)
            layers.append(net)

    return layers


def siamese(target, current, keep_prob ,loss_layers, n_layers=14, norm_type="SBN", base_channels=32, blk_channels=5,ksz=3,lin_layer=False):
    
    feat_current = lossnet_siamese(current, reuse=False, n_layers=n_layers, norm_type=norm_type,
                         base_channels=base_channels, blk_channels=blk_channels,training=False,ksz=ksz,keep_prob=keep_prob,lin_layer=lin_layer)

    feat_target = lossnet_siamese(target, reuse=True, n_layers=n_layers, norm_type=norm_type,
                         base_channels=base_channels, blk_channels=blk_channels,training=False,ksz=ksz,keep_prob=keep_prob,lin_layer=lin_layer)
    
    return feat_current[-1],feat_target[-1]

def lossnet_triplet(input, keep_prob,n_layers=14, training=True, reuse=False, norm_type="SBN",
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
            #net = slim.dropout(net, keep_prob, scope='Dropout_%d' %id)
            layers.append(net)
            
        elif id < n_layers - 1:
            net = slim.conv2d(net, n_channels, [1, ksz], activation_fn=lrelu, normalizer_fn=norm_fn,
                              stride=[1, 2], scope='loss_conv_%d' % id, padding='SAME', reuse=reuse)
            #net = slim.dropout(net, keep_prob, scope='Dropout_%d' %id)
            layers.append(net)
        else:
            net = slim.conv2d(net, n_channels, [1, ksz], activation_fn=lrelu, normalizer_fn=norm_fn,
                              scope='loss_conv_%d' % id, padding='SAME', reuse=reuse)
            net = slim.flatten(net)
            layers.append(net)

    return layers


def triplet(sample1, sample2, anchor, keep_prob ,loss_layers, n_layers=14, norm_type="SBN", base_channels=32, blk_channels=5,ksz=3):
    
    feat_sample1 = lossnet_triplet(sample1, reuse=False, n_layers=n_layers, norm_type=norm_type,
                         base_channels=base_channels, blk_channels=blk_channels,training=False,ksz=ksz,keep_prob=keep_prob)
    feat_sample2 = lossnet_triplet(sample2, reuse=True, n_layers=n_layers, norm_type=norm_type,
                         base_channels=base_channels, blk_channels=blk_channels,training=False,ksz=ksz,keep_prob=keep_prob)
    feat_anchor = lossnet_triplet(anchor, reuse=True, n_layers=n_layers, norm_type=norm_type,
                         base_channels=base_channels, blk_channels=blk_channels,training=False,ksz=ksz,keep_prob=keep_prob)
    
    return feat_sample1[-1],feat_sample2[-1],feat_anchor[-1]


def vggish_model(target, current,maxpool=1,input_mod=0,training=True):
    
    feat_current = lossnet_vggish(current, reuse=False,training=training,maxpool=maxpool,input_mod=input_mod)

    feat_target = lossnet_vggish(target, reuse=True, training=training,maxpool=maxpool,input_mod=input_mod)
    
    loss_vec = []
    
    channels=np.array([64,128,256,512])
        
    for id in range(len(channels)):
        
        a=feat_current[id]-feat_target[id]
        weights = tf.Variable(tf.random_normal([channels[id]]),
                      name="weights_%d" %id, trainable=True)
        a1=tf.transpose(a, [0, 1, 3, 2])
        result=tf.multiply(a1, weights[:,tf.newaxis])
        loss_result=l1_loss_batch(result)
        loss_vec.append(loss_result)
    
    return loss_vec,loss_vec

def lossnet_vggish(features, training=True, reuse=False,maxpool=1,input_mod=0):    
    
    layers=[]

    with slim.arg_scope([slim.conv2d],
                      weights_initializer=tf.truncated_normal_initializer(
                          stddev=0.01),
                      biases_initializer=tf.zeros_initializer(),
                      activation_fn=tf.nn.relu,
                      trainable=training), \
       slim.arg_scope([slim.conv2d],
                      kernel_size=[3, 3], stride=1, padding='SAME'), \
       slim.arg_scope([slim.max_pool2d],
                      kernel_size=[2, 2], stride=2, padding='SAME'), \
       tf.variable_scope('vggish'):
        
        net = slim.conv2d(features, 64, scope='conv1',reuse=reuse)
        if maxpool==1:
            net = slim.max_pool2d(net, scope='pool1')
        layers.append(net)
        net = slim.conv2d(net, 128, scope='conv2',reuse=reuse)
        if maxpool==1:
            net = slim.max_pool2d(net, scope='pool2')
        layers.append(net)
        net = slim.repeat(net, 2, slim.conv2d, 256, scope='conv3',reuse=reuse)
        if maxpool==1:
            net = slim.max_pool2d(net, scope='pool3')
        layers.append(net)
        net = slim.repeat(net, 2, slim.conv2d, 512, scope='conv4',reuse=reuse)
        if maxpool==1:
            net = slim.max_pool2d(net, scope='pool4')
        layers.append(net)
        
    return layers


def lossnet_vggish_def(training=True, reuse=False,maxpool=1,input_mod=0):    
    
    features = tf.placeholder(
        tf.float32, shape=(None, None, None,1),
        name='input_features')
    
    with slim.arg_scope([slim.conv2d],
                      weights_initializer=tf.truncated_normal_initializer(
                          stddev=0.01),
                      biases_initializer=tf.zeros_initializer(),
                      activation_fn=tf.nn.relu,
                      trainable=training), \
       slim.arg_scope([slim.conv2d],
                      kernel_size=[3, 3], stride=1, padding='SAME'), \
       slim.arg_scope([slim.max_pool2d],
                      kernel_size=[2, 2], stride=2, padding='SAME'), \
       tf.variable_scope('vggish'):
        
        net = slim.conv2d(features, 64, scope='conv1',reuse=reuse)
        if maxpool==1:
            net = slim.max_pool2d(net, scope='pool1')

        net = slim.conv2d(net, 128, scope='conv2',reuse=reuse)
        if maxpool==1:
            net = slim.max_pool2d(net, scope='pool2')

        net = slim.repeat(net, 2, slim.conv2d, 256, scope='conv3',reuse=reuse)
        if maxpool==1:
            net = slim.max_pool2d(net, scope='pool3')

        net = slim.repeat(net, 2, slim.conv2d, 512, scope='conv4',reuse=reuse)
        if maxpool==1:
            net = slim.max_pool2d(net, scope='pool4')