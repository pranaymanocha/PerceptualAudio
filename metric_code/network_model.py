import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from helper import *

# FEATURE LOSS NETWORK
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

def featureloss_batch(target, current, keep_prob,loss_layers, n_layers=14, norm_type="SBN", base_channels=32, blk_channels=5,ksz=3):

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
        result=tf.multiply(a1, weights[:,tf.newaxis])
        loss_result=l1_loss_batch(result)
        loss_vec.append(loss_result)
    
    return loss_vec,loss_vec