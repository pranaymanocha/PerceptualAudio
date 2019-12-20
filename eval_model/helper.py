import numpy
import numpy as np
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
import resampy
from sklearn.metrics import f1_score
from voco_MOS import *
from fftnet_MOS import *
from bwe_MOS import *

###########################
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from sklearn.preprocessing import normalize

# LEAKY RELU UNIT
def lrelu(x):
    return tf.maximum(0.2*x,x)


# GENERATE DILATED LAYER FROM 1D SIGNAL
def signal_to_dilated(signal, dilation, n_channels):
    shape = tf.shape(signal)
    pad_elements = dilation - 1 - (shape[2] + dilation - 1) % dilation
    dilated = tf.pad(signal, [[0, 0], [0, 0], [0, pad_elements], [0, 0]])
    dilated = tf.reshape(dilated, [shape[0],-1,dilation,n_channels])
    return tf.transpose(dilated, perm=[0,2,1,3]), pad_elements


# COLLAPSE DILATED LAYER TO 1D SIGNAL
def dilated_to_signal(dilated, pad_elements, n_channels):
    shape = tf.shape(dilated)
    signal = tf.transpose(dilated, perm=[0,2,1,3])
    signal = tf.reshape(signal, [shape[0],1,-1,n_channels])
    return signal[:,:,:shape[1]*shape[2]-pad_elements,:]


# ADAPTIVE BATCH NORMALIZATION LAYER
def nm(x):
    w0=tf.Variable(1.0,name='w0')
    w1=tf.Variable(0.0,name='w1')
    return w0*x+w1*slim.batch_norm(x)


# IDENTITY INITIALIZATION OF CONV LAYERS
def identity_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        array = np.zeros(shape, dtype=float)
        cx, cy = shape[0]//2, shape[1]//2
        for i in range(np.minimum(shape[2],shape[3])):
            array[cx, cy, i, i] = 1
        return tf.constant(array, dtype=dtype)
    return _initializer

def l1_loss_batch(target):
    return tf.reduce_mean(tf.abs(target),axis=[1,2,3])

# L1 LOSS FUNCTION
def l1_loss(target,current):
    return tf.reduce_mean(tf.abs(target-current))

def l1_loss_all(agg):
    return tf.reduce_mean(tf.abs(agg))

def l2_loss_all(agg):
    return tf.reduce_mean(tf.square(agg))

def l1_loss_batch(target):
    return tf.reduce_mean(tf.abs(target),axis=[1,2,3])

def l1_loss_batch(target):
    return tf.reduce_mean(tf.abs(target),axis=[1,2,3])

# L2 LOSS FUNCTION
def l2_loss(target,current):
    return tf.reduce_mean(tf.square(target-current))

def l2_loss_unit(target,current):
    target=tf.linalg.l2_normalize(target,axis=3)
    current = tf.linalg.l2_normalize(current,axis=3)
    return tf.reduce_mean(tf.square(target-current))

###########################

A_triplet_mymetric=[
{ "id": "0", "countPos": 7, "countNeg": 2, "raw": [ 1, 1, 1, 1, 0, 0, 1, 1, 1 ] },
{ "id": "1", "countPos": 7, "countNeg": 2, "raw": [ 1, 1, 1, 1, 0, 1, 1, 0, 1 ] },
{ "id": "10", "countPos": 4, "countNeg": 5, "raw": [ 0, 0, 0, 1, 1, 1, 0, 1, 0 ] },
{ "id": "100", "countPos": 6, "countNeg": 3, "raw": [ 1, 0, 1, 1, 0, 1, 1, 0, 1 ] },
{ "id": "101", "countPos": 8, "countNeg": 1, "raw": [ 1, 1, 1, 1, 1, 0, 1, 1, 1 ] },
{ "id": "102", "countPos": 9, "countNeg": 0, "raw": [ 1, 1, 1, 1, 1, 1, 1, 1, 1 ] },
{ "id": "103", "countPos": 8, "countNeg": 1, "raw": [ 1, 1, 1, 1, 1, 1, 1, 0, 1 ] },
{ "id": "104", "countPos": 2, "countNeg": 7, "raw": [ 0, 0, 0, 0, 1, 0, 0, 1, 0 ] },
{ "id": "105", "countPos": 8, "countNeg": 1, "raw": [ 1, 1, 1, 1, 0, 1, 1, 1, 1 ] },
{ "id": "106", "countPos": 0, "countNeg": 9, "raw": [ 0, 0, 0, 0, 0, 0, 0, 0, 0 ] },
{ "id": "107", "countPos": 2, "countNeg": 7, "raw": [ 0, 0, 0, 0, 0, 1, 1, 0, 0 ] },
{ "id": "108", "countPos": 4, "countNeg": 5, "raw": [ 0, 0, 1, 0, 1, 0, 0, 1, 1 ] },
{ "id": "109", "countPos": 6, "countNeg": 3, "raw": [ 0, 1, 1, 1, 1, 1, 0, 1, 0 ] },
{ "id": "11", "countPos": 3, "countNeg": 6, "raw": [ 1, 0, 0, 0, 1, 0, 0, 1, 0 ] },
{ "id": "110", "countPos": 8, "countNeg": 1, "raw": [ 1, 1, 1, 1, 1, 1, 1, 1, 0 ] },
{ "id": "111", "countPos": 8, "countNeg": 1, "raw": [ 1, 1, 1, 1, 1, 1, 1, 1, 0 ] },
{ "id": "112", "countPos": 8, "countNeg": 1, "raw": [ 1, 1, 1, 1, 1, 1, 1, 1, 0 ] },
{ "id": "113", "countPos": 3, "countNeg": 6, "raw": [ 1, 0, 0, 0, 0, 0, 1, 0, 1 ] },
{ "id": "114", "countPos": 3, "countNeg": 6, "raw": [ 1, 0, 0, 0, 0, 0, 0, 1, 1 ] },
{ "id": "115", "countPos": 4, "countNeg": 5, "raw": [ 0, 1, 1, 0, 1, 0, 0, 0, 1 ] },
{ "id": "116", "countPos": 5, "countNeg": 4, "raw": [ 1, 1, 0, 1, 0, 1, 0, 0, 1 ] },
{ "id": "117", "countPos": 6, "countNeg": 3, "raw": [ 1, 0, 1, 0, 1, 0, 1, 1, 1 ] },
{ "id": "118", "countPos": 6, "countNeg": 3, "raw": [ 1, 1, 1, 0, 1, 1, 1, 0, 0 ] },
{ "id": "119", "countPos": 4, "countNeg": 5, "raw": [ 1, 1, 0, 0, 0, 0, 1, 0, 1 ] },
{ "id": "12", "countPos": 7, "countNeg": 2, "raw": [ 1, 1, 1, 0, 1, 1, 1, 0, 1 ] },
{ "id": "120", "countPos": 3, "countNeg": 6, "raw": [ 0, 1, 0, 0, 0, 1, 1, 0, 0 ] },
{ "id": "121", "countPos": 5, "countNeg": 4, "raw": [ 1, 1, 1, 1, 0, 0, 0, 0, 1 ] },
{ "id": "122", "countPos": 3, "countNeg": 6, "raw": [ 0, 0, 0, 0, 1, 1, 0, 0, 1 ] },
{ "id": "123", "countPos": 6, "countNeg": 3, "raw": [ 0, 1, 1, 1, 1, 0, 1, 0, 1 ] },
{ "id": "124", "countPos": 2, "countNeg": 7, "raw": [ 0, 0, 0, 0, 0, 1, 0, 1, 0 ] },
{ "id": "125", "countPos": 2, "countNeg": 7, "raw": [ 0, 1, 0, 0, 0, 1, 0, 0, 0 ] },
{ "id": "126", "countPos": 7, "countNeg": 2, "raw": [ 1, 1, 1, 1, 1, 1, 0, 0, 1 ] },
{ "id": "127", "countPos": 5, "countNeg": 4, "raw": [ 1, 1, 1, 0, 0, 1, 1, 0, 0 ] },
{ "id": "128", "countPos": 2, "countNeg": 7, "raw": [ 0, 0, 0, 0, 0, 1, 1, 0, 0 ] },
{ "id": "129", "countPos": 3, "countNeg": 6, "raw": [ 0, 0, 1, 0, 0, 1, 1, 0, 0 ] },
{ "id": "13", "countPos": 8, "countNeg": 1, "raw": [ 1, 1, 1, 0, 1, 1, 1, 1, 1 ] },
{ "id": "130", "countPos": 8, "countNeg": 1, "raw": [ 1, 1, 1, 1, 0, 1, 1, 1, 1 ] },
{ "id": "131", "countPos": 8, "countNeg": 1, "raw": [ 1, 1, 1, 1, 1, 1, 0, 1, 1 ] },
{ "id": "132", "countPos": 7, "countNeg": 2, "raw": [ 1, 1, 1, 0, 1, 0, 1, 1, 1 ] },
{ "id": "133", "countPos": 4, "countNeg": 5, "raw": [ 0, 1, 1, 1, 0, 1, 0, 0, 0 ] },
{ "id": "134", "countPos": 5, "countNeg": 4, "raw": [ 0, 0, 1, 0, 0, 1, 1, 1, 1 ] },
{ "id": "135", "countPos": 6, "countNeg": 3, "raw": [ 1, 1, 0, 1, 1, 0, 1, 1, 0 ] },
{ "id": "136", "countPos": 4, "countNeg": 5, "raw": [ 0, 0, 0, 0, 1, 1, 0, 1, 1 ] },
{ "id": "137", "countPos": 1, "countNeg": 8, "raw": [ 0, 1, 0, 0, 0, 0, 0, 0, 0 ] },
{ "id": "138", "countPos": 6, "countNeg": 3, "raw": [ 1, 1, 1, 1, 0, 1, 0, 1, 0 ] },
{ "id": "139", "countPos": 1, "countNeg": 8, "raw": [ 0, 0, 0, 0, 0, 0, 1, 0, 0 ] },
{ "id": "14", "countPos": 7, "countNeg": 2, "raw": [ 1, 1, 1, 0, 0, 1, 1, 1, 1 ] },
{ "id": "140", "countPos": 1, "countNeg": 8, "raw": [ 0, 0, 0, 0, 0, 0, 0, 1, 0 ] },
{ "id": "141", "countPos": 7, "countNeg": 2, "raw": [ 1, 1, 1, 0, 0, 1, 1, 1, 1 ] },
{ "id": "142", "countPos": 3, "countNeg": 6, "raw": [ 0, 0, 0, 1, 0, 0, 1, 1, 0 ] },
{ "id": "143", "countPos": 3, "countNeg": 6, "raw": [ 0, 0, 1, 0, 0, 0, 1, 1, 0 ] },
{ "id": "144", "countPos": 2, "countNeg": 7, "raw": [ 0, 1, 0, 0, 0, 0, 1, 0, 0 ] },
{ "id": "145", "countPos": 2, "countNeg": 7, "raw": [ 0, 0, 0, 0, 0, 1, 0, 1, 0 ] },
{ "id": "146", "countPos": 6, "countNeg": 3, "raw": [ 1, 1, 1, 1, 1, 1, 0, 0, 0 ] },
{ "id": "147", "countPos": 5, "countNeg": 4, "raw": [ 1, 1, 0, 0, 1, 0, 1, 1, 0 ] },
{ "id": "148", "countPos": 6, "countNeg": 3, "raw": [ 0, 0, 1, 1, 0, 1, 1, 1, 1 ] },
{ "id": "149", "countPos": 5, "countNeg": 4, "raw": [ 0, 1, 1, 0, 0, 0, 1, 1, 1 ] },
{ "id": "15", "countPos": 4, "countNeg": 5, "raw": [ 0, 0, 1, 1, 0, 0, 1, 0, 1 ] },
{ "id": "150", "countPos": 5, "countNeg": 4, "raw": [ 0, 1, 1, 0, 0, 1, 1, 0, 1 ] },
{ "id": "151", "countPos": 7, "countNeg": 2, "raw": [ 1, 1, 1, 1, 0, 1, 1, 0, 1 ] },
{ "id": "152", "countPos": 7, "countNeg": 3, "raw": [ 0, 1, 1, 1, 1, 0, 0, 1, 1, 1 ] },
{ "id": "153", "countPos": 6, "countNeg": 4, "raw": [ 1, 1, 1, 0, 1, 0, 1, 0, 1, 0 ] },
{ "id": "154", "countPos": 3, "countNeg": 7, "raw": [ 0, 0, 0, 0, 0, 0, 1, 0, 1, 1 ] },
{ "id": "155", "countPos": 5, "countNeg": 5, "raw": [ 0, 1, 1, 0, 1, 0, 1, 0, 0, 1 ] },
{ "id": "156", "countPos": 4, "countNeg": 6, "raw": [ 1, 0, 0, 0, 1, 0, 0, 1, 1, 0 ] },
{ "id": "157", "countPos": 6, "countNeg": 4, "raw": [ 0, 1, 1, 1, 0, 0, 0, 1, 1, 1 ] },
{ "id": "158", "countPos": 1, "countNeg": 9, "raw": [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 ] },
{ "id": "159", "countPos": 3, "countNeg": 7, "raw": [ 0, 0, 0, 0, 0, 1, 0, 0, 1, 1 ] },
{ "id": "16", "countPos": 5, "countNeg": 5, "raw": [ 1, 1, 0, 0, 1, 0, 0, 0, 1, 1 ] },
{ "id": "160", "countPos": 6, "countNeg": 4, "raw": [ 0, 0, 0, 1, 1, 1, 1, 1, 1, 0 ] },
{ "id": "161", "countPos": 4, "countNeg": 5, "raw": [ 1, 1, 0, 0, 0, 0, 1, 1, 0 ] },
{ "id": "162", "countPos": 3, "countNeg": 6, "raw": [ 0, 0, 1, 0, 0, 0, 1, 0, 1 ] },
{ "id": "163", "countPos": 1, "countNeg": 8, "raw": [ 0, 0, 0, 0, 1, 0, 0, 0, 0 ] },
{ "id": "164", "countPos": 6, "countNeg": 3, "raw": [ 1, 1, 0, 0, 0, 1, 1, 1, 1 ] },
{ "id": "165", "countPos": 3, "countNeg": 6, "raw": [ 0, 0, 0, 1, 0, 1, 0, 0, 1 ] },
{ "id": "166", "countPos": 3, "countNeg": 6, "raw": [ 0, 1, 1, 0, 0, 1, 0, 0, 0 ] },
{ "id": "167", "countPos": 7, "countNeg": 2, "raw": [ 1, 1, 0, 1, 1, 1, 1, 0, 1 ] },
{ "id": "168", "countPos": 9, "countNeg": 0, "raw": [ 1, 1, 1, 1, 1, 1, 1, 1, 1 ] },
{ "id": "169", "countPos": 3, "countNeg": 6, "raw": [ 0, 0, 0, 0, 1, 1, 0, 1, 0 ] },
{ "id": "17", "countPos": 4, "countNeg": 5, "raw": [ 1, 0, 0, 0, 1, 1, 0, 1, 0 ] },
{ "id": "170", "countPos": 2, "countNeg": 7, "raw": [ 0, 0, 0, 0, 0, 1, 0, 0, 1 ] },
{ "id": "171", "countPos": 8, "countNeg": 1, "raw": [ 1, 1, 1, 1, 1, 1, 1, 1, 0 ] },
{ "id": "172", "countPos": 6, "countNeg": 3, "raw": [ 1, 1, 0, 1, 0, 0, 1, 1, 1 ] },
{ "id": "173", "countPos": 4, "countNeg": 5, "raw": [ 0, 0, 0, 0, 0, 1, 1, 1, 1 ] },
{ "id": "174", "countPos": 2, "countNeg": 7, "raw": [ 0, 0, 0, 0, 0, 0, 1, 0, 1 ] },
{ "id": "175", "countPos": 3, "countNeg": 6, "raw": [ 0, 0, 0, 0, 0, 0, 1, 1, 1 ] },
{ "id": "176", "countPos": 4, "countNeg": 5, "raw": [ 0, 1, 0, 1, 1, 0, 1, 0, 0 ] },
{ "id": "177", "countPos": 2, "countNeg": 7, "raw": [ 0, 1, 0, 0, 0, 0, 1, 0, 0 ] },
{ "id": "178", "countPos": 7, "countNeg": 2, "raw": [ 1, 1, 1, 1, 1, 1, 0, 1, 0 ] },
{ "id": "179", "countPos": 9, "countNeg": 0, "raw": [ 1, 1, 1, 1, 1, 1, 1, 1, 1 ] },
{ "id": "18", "countPos": 6, "countNeg": 3, "raw": [ 1, 1, 0, 1, 0, 1, 1, 0, 1 ] },
{ "id": "180", "countPos": 3, "countNeg": 6, "raw": [ 0, 0, 1, 0, 1, 0, 1, 0, 0 ] },
{ "id": "181", "countPos": 7, "countNeg": 2, "raw": [ 0, 1, 0, 1, 1, 1, 1, 1, 1 ] },
{ "id": "182", "countPos": 4, "countNeg": 5, "raw": [ 0, 1, 0, 1, 0, 1, 0, 1, 0 ] },
{ "id": "183", "countPos": 6, "countNeg": 3, "raw": [ 1, 1, 0, 1, 1, 1, 0, 1, 0 ] },
{ "id": "184", "countPos": 3, "countNeg": 6, "raw": [ 0, 0, 1, 0, 1, 0, 1, 0, 0 ] },
{ "id": "185", "countPos": 8, "countNeg": 1, "raw": [ 1, 1, 1, 1, 1, 1, 1, 0, 1 ] },
{ "id": "186", "countPos": 5, "countNeg": 4, "raw": [ 0, 0, 0, 1, 1, 1, 0, 1, 1 ] },
{ "id": "187", "countPos": 4, "countNeg": 5, "raw": [ 1, 0, 0, 0, 1, 0, 1, 1, 0 ] },
{ "id": "188", "countPos": 5, "countNeg": 4, "raw": [ 0, 0, 1, 0, 1, 1, 1, 1, 0 ] },
{ "id": "189", "countPos": 2, "countNeg": 7, "raw": [ 0, 0, 0, 0, 1, 0, 0, 0, 1 ] },
{ "id": "19", "countPos": 7, "countNeg": 2, "raw": [ 1, 1, 1, 1, 1, 0, 0, 1, 1 ] },
{ "id": "190", "countPos": 8, "countNeg": 1, "raw": [ 1, 1, 1, 1, 1, 1, 1, 0, 1 ] },
{ "id": "191", "countPos": 7, "countNeg": 2, "raw": [ 1, 1, 1, 0, 1, 1, 1, 0, 1 ] },
{ "id": "192", "countPos": 4, "countNeg": 5, "raw": [ 0, 0, 0, 1, 0, 1, 1, 1, 0 ] },
{ "id": "193", "countPos": 7, "countNeg": 2, "raw": [ 1, 1, 1, 1, 1, 0, 1, 1, 0 ] },
{ "id": "194", "countPos": 8, "countNeg": 1, "raw": [ 1, 1, 1, 1, 0, 1, 1, 1, 1 ] },
{ "id": "195", "countPos": 1, "countNeg": 8, "raw": [ 0, 0, 0, 0, 0, 0, 0, 1, 0 ] },
{ "id": "196", "countPos": 1, "countNeg": 8, "raw": [ 0, 0, 0, 0, 0, 0, 0, 1, 0 ] },
{ "id": "197", "countPos": 5, "countNeg": 4, "raw": [ 1, 1, 1, 0, 0, 0, 1, 0, 1 ] },
{ "id": "198", "countPos": 3, "countNeg": 7, "raw": [ 0, 1, 0, 0, 0, 0, 0, 1, 0, 1 ] },
{ "id": "199", "countPos": 9, "countNeg": 1, "raw": [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 0 ] },
{ "id": "2", "countPos": 3, "countNeg": 7, "raw": [ 0, 0, 0, 0, 1, 0, 0, 1, 0, 1 ] },
{ "id": "20", "countPos": 4, "countNeg": 6, "raw": [ 1, 1, 0, 0, 0, 0, 0, 1, 1, 0 ] },
{ "id": "200", "countPos": 4, "countNeg": 6, "raw": [ 0, 1, 1, 0, 0, 0, 1, 0, 0, 1 ] },
{ "id": "201", "countPos": 8, "countNeg": 2, "raw": [ 1, 1, 1, 1, 0, 1, 1, 1, 1, 0 ] },
{ "id": "202", "countPos": 6, "countNeg": 4, "raw": [ 1, 1, 1, 0, 0, 0, 0, 1, 1, 1 ] },
{ "id": "203", "countPos": 5, "countNeg": 5, "raw": [ 1, 0, 0, 1, 0, 1, 0, 0, 1, 1 ] },
{ "id": "204", "countPos": 7, "countNeg": 3, "raw": [ 1, 1, 1, 1, 0, 1, 1, 0, 1, 0 ] },
{ "id": "205", "countPos": 0, "countNeg": 10, "raw": [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ] },
{ "id": "206", "countPos": 4, "countNeg": 4, "raw": [ 1, 1, 0, 0, 1, 0, 0, 1 ] },
{ "id": "207", "countPos": 3, "countNeg": 5, "raw": [ 0, 1, 1, 0, 0, 0, 0, 1 ] },
{ "id": "208", "countPos": 5, "countNeg": 3, "raw": [ 1, 1, 1, 0, 1, 0, 0, 1 ] },
{ "id": "209", "countPos": 4, "countNeg": 4, "raw": [ 1, 0, 0, 0, 1, 0, 1, 1 ] },
{ "id": "21", "countPos": 5, "countNeg": 3, "raw": [ 1, 0, 1, 1, 0, 1, 0, 1 ] },
{ "id": "210", "countPos": 1, "countNeg": 7, "raw": [ 1, 0, 0, 0, 0, 0, 0, 0 ] },
{ "id": "211", "countPos": 5, "countNeg": 3, "raw": [ 1, 1, 1, 1, 0, 1, 0, 0 ] },
{ "id": "212", "countPos": 3, "countNeg": 5, "raw": [ 0, 0, 1, 0, 1, 0, 0, 1 ] },
{ "id": "213", "countPos": 6, "countNeg": 2, "raw": [ 1, 1, 1, 1, 1, 0, 1, 0 ] },
{ "id": "214", "countPos": 4, "countNeg": 4, "raw": [ 0, 0, 1, 0, 1, 0, 1, 1 ] },
{ "id": "215", "countPos": 1, "countNeg": 8, "raw": [ 0, 0, 0, 0, 0, 1, 0, 0, 0 ] },
{ "id": "216", "countPos": 7, "countNeg": 2, "raw": [ 0, 1, 1, 1, 1, 1, 0, 1, 1 ] },
{ "id": "217", "countPos": 2, "countNeg": 7, "raw": [ 0, 0, 1, 0, 0, 1, 0, 0, 0 ] },
{ "id": "218", "countPos": 2, "countNeg": 7, "raw": [ 1, 0, 0, 0, 1, 0, 0, 0, 0 ] },
{ "id": "219", "countPos": 8, "countNeg": 1, "raw": [ 1, 1, 1, 1, 1, 1, 0, 1, 1 ] },
{ "id": "22", "countPos": 3, "countNeg": 6, "raw": [ 0, 0, 0, 0, 0, 1, 1, 0, 1 ] },
{ "id": "220", "countPos": 2, "countNeg": 7, "raw": [ 1, 0, 0, 0, 1, 0, 0, 0, 0 ] },
{ "id": "221", "countPos": 8, "countNeg": 1, "raw": [ 1, 1, 1, 1, 1, 0, 1, 1, 1 ] },
{ "id": "222", "countPos": 2, "countNeg": 7, "raw": [ 0, 0, 0, 0, 0, 1, 1, 0, 0 ] },
{ "id": "223", "countPos": 9, "countNeg": 0, "raw": [ 1, 1, 1, 1, 1, 1, 1, 1, 1 ] },
{ "id": "224", "countPos": 3, "countNeg": 6, "raw": [ 0, 0, 0, 1, 1, 1, 0, 0, 0 ] },
{ "id": "225", "countPos": 7, "countNeg": 2, "raw": [ 1, 1, 1, 1, 1, 0, 0, 1, 1 ] },
{ "id": "226", "countPos": 3, "countNeg": 6, "raw": [ 0, 1, 0, 0, 1, 0, 1, 0, 0 ] },
{ "id": "227", "countPos": 6, "countNeg": 3, "raw": [ 1, 1, 1, 1, 1, 0, 0, 0, 1 ] },
{ "id": "228", "countPos": 4, "countNeg": 5, "raw": [ 1, 0, 0, 0, 1, 1, 1, 0, 0 ] },
{ "id": "229", "countPos": 3, "countNeg": 6, "raw": [ 0, 0, 1, 0, 1, 0, 0, 1, 0 ] },
{ "id": "23", "countPos": 5, "countNeg": 4, "raw": [ 1, 1, 1, 1, 0, 0, 0, 0, 1 ] },
{ "id": "230", "countPos": 4, "countNeg": 5, "raw": [ 0, 1, 0, 1, 0, 1, 0, 0, 1 ] },
{ "id": "231", "countPos": 4, "countNeg": 5, "raw": [ 0, 1, 0, 0, 1, 0, 0, 1, 1 ] },
{ "id": "232", "countPos": 1, "countNeg": 8, "raw": [ 0, 0, 0, 1, 0, 0, 0, 0, 0 ] },
{ "id": "233", "countPos": 7, "countNeg": 2, "raw": [ 1, 1, 0, 0, 1, 1, 1, 1, 1 ] },
{ "id": "234", "countPos": 3, "countNeg": 6, "raw": [ 1, 0, 0, 0, 0, 1, 1, 0, 0 ] },
{ "id": "235", "countPos": 8, "countNeg": 1, "raw": [ 1, 1, 1, 1, 1, 0, 1, 1, 1 ] },
{ "id": "236", "countPos": 6, "countNeg": 3, "raw": [ 1, 0, 0, 1, 1, 0, 1, 1, 1 ] },
{ "id": "237", "countPos": 8, "countNeg": 1, "raw": [ 1, 1, 1, 1, 1, 1, 0, 1, 1 ] },
{ "id": "238", "countPos": 2, "countNeg": 7, "raw": [ 0, 0, 0, 0, 0, 1, 1, 0, 0 ] },
{ "id": "239", "countPos": 7, "countNeg": 2, "raw": [ 1, 1, 1, 1, 0, 1, 1, 0, 1 ] },
{ "id": "24", "countPos": 6, "countNeg": 3, "raw": [ 1, 1, 1, 1, 0, 1, 0, 0, 1 ] },
{ "id": "240", "countPos": 4, "countNeg": 5, "raw": [ 1, 0, 1, 1, 0, 0, 0, 1, 0 ] },
{ "id": "241", "countPos": 4, "countNeg": 5, "raw": [ 1, 1, 1, 0, 0, 0, 1, 0, 0 ] },
{ "id": "242", "countPos": 8, "countNeg": 1, "raw": [ 1, 1, 1, 1, 1, 1, 0, 1, 1 ] },
{ "id": "243", "countPos": 6, "countNeg": 3, "raw": [ 1, 1, 0, 1, 1, 1, 0, 0, 1 ] },
{ "id": "244", "countPos": 3, "countNeg": 6, "raw": [ 0, 0, 1, 0, 0, 1, 0, 1, 0 ] },
{ "id": "245", "countPos": 5, "countNeg": 4, "raw": [ 1, 1, 1, 1, 0, 0, 0, 1, 0 ] },
{ "id": "246", "countPos": 7, "countNeg": 2, "raw": [ 1, 0, 1, 1, 1, 1, 0, 1, 1 ] },
{ "id": "247", "countPos": 7, "countNeg": 2, "raw": [ 1, 1, 1, 1, 0, 1, 1, 0, 1 ] },
{ "id": "248", "countPos": 1, "countNeg": 8, "raw": [ 0, 0, 0, 0, 0, 1, 0, 0, 0 ] },
{ "id": "249", "countPos": 2, "countNeg": 7, "raw": [ 0, 0, 0, 0, 1, 0, 0, 1, 0 ] },
{ "id": "25", "countPos": 6, "countNeg": 3, "raw": [ 1, 1, 1, 0, 1, 0, 1, 0, 1 ] },
{ "id": "250", "countPos": 2, "countNeg": 7, "raw": [ 0, 0, 0, 0, 0, 1, 0, 0, 1 ] },
{ "id": "251", "countPos": 4, "countNeg": 5, "raw": [ 0, 1, 0, 1, 0, 1, 0, 1, 0 ] },
{ "id": "252", "countPos": 5, "countNeg": 4, "raw": [ 0, 1, 0, 0, 1, 0, 1, 1, 1 ] },
{ "id": "253", "countPos": 2, "countNeg": 7, "raw": [ 0, 0, 0, 1, 1, 0, 0, 0, 0 ] },
{ "id": "254", "countPos": 7, "countNeg": 2, "raw": [ 1, 1, 1, 0, 1, 0, 1, 1, 1 ] },
{ "id": "255", "countPos": 6, "countNeg": 3, "raw": [ 1, 1, 0, 1, 0, 1, 1, 0, 1 ] },
{ "id": "256", "countPos": 6, "countNeg": 3, "raw": [ 1, 1, 1, 0, 0, 1, 1, 0, 1 ] },
{ "id": "257", "countPos": 7, "countNeg": 2, "raw": [ 1, 1, 1, 0, 1, 1, 1, 0, 1 ] },
{ "id": "258", "countPos": 3, "countNeg": 6, "raw": [ 0, 0, 1, 0, 0, 0, 0, 1, 1 ] },
{ "id": "259", "countPos": 2, "countNeg": 7, "raw": [ 0, 0, 0, 1, 0, 0, 0, 1, 0 ] },
{ "id": "26", "countPos": 4, "countNeg": 5, "raw": [ 1, 0, 0, 1, 1, 0, 0, 0, 1 ] },
{ "id": "260", "countPos": 2, "countNeg": 7, "raw": [ 0, 0, 0, 1, 0, 0, 0, 1, 0 ] },
{ "id": "261", "countPos": 1, "countNeg": 8, "raw": [ 0, 1, 0, 0, 0, 0, 0, 0, 0 ] },
{ "id": "262", "countPos": 3, "countNeg": 6, "raw": [ 1, 0, 0, 1, 0, 0, 0, 0, 1 ] },
{ "id": "263", "countPos": 6, "countNeg": 3, "raw": [ 1, 1, 0, 0, 0, 1, 1, 1, 1 ] },
{ "id": "264", "countPos": 7, "countNeg": 2, "raw": [ 1, 1, 1, 0, 1, 0, 1, 1, 1 ] },
{ "id": "265", "countPos": 6, "countNeg": 3, "raw": [ 1, 0, 1, 0, 1, 1, 1, 1, 0 ] },
{ "id": "266", "countPos": 7, "countNeg": 2, "raw": [ 0, 1, 1, 0, 1, 1, 1, 1, 1 ] },
{ "id": "267", "countPos": 4, "countNeg": 5, "raw": [ 0, 0, 1, 1, 0, 0, 0, 1, 1 ] },
{ "id": "268", "countPos": 4, "countNeg": 5, "raw": [ 0, 0, 1, 1, 0, 0, 1, 0, 1 ] },
{ "id": "269", "countPos": 2, "countNeg": 7, "raw": [ 0, 0, 1, 0, 0, 1, 0, 0, 0 ] },
{ "id": "27", "countPos": 3, "countNeg": 6, "raw": [ 0, 0, 0, 0, 1, 0, 1, 0, 1 ] },
{ "id": "270", "countPos": 4, "countNeg": 5, "raw": [ 1, 0, 0, 0, 1, 1, 0, 0, 1 ] },
{ "id": "271", "countPos": 2, "countNeg": 7, "raw": [ 0, 1, 0, 0, 1, 0, 0, 0, 0 ] },
{ "id": "272", "countPos": 2, "countNeg": 7, "raw": [ 0, 0, 0, 0, 1, 0, 0, 1, 0 ] },
{ "id": "273", "countPos": 2, "countNeg": 7, "raw": [ 0, 0, 0, 1, 0, 1, 0, 0, 0 ] },
{ "id": "274", "countPos": 3, "countNeg": 6, "raw": [ 1, 0, 0, 0, 1, 0, 1, 0, 0 ] },
{ "id": "275", "countPos": 2, "countNeg": 7, "raw": [ 0, 0, 0, 0, 1, 0, 0, 1, 0 ] },
{ "id": "276", "countPos": 6, "countNeg": 3, "raw": [ 0, 1, 1, 1, 1, 1, 0, 0, 1 ] },
{ "id": "277", "countPos": 4, "countNeg": 5, "raw": [ 0, 0, 1, 0, 0, 1, 0, 1, 1 ] },
{ "id": "278", "countPos": 6, "countNeg": 3, "raw": [ 0, 1, 1, 1, 1, 1, 1, 0, 0 ] },
{ "id": "279", "countPos": 2, "countNeg": 7, "raw": [ 0, 0, 0, 0, 0, 0, 0, 1, 1 ] },
{ "id": "28", "countPos": 7, "countNeg": 2, "raw": [ 1, 0, 1, 1, 1, 1, 1, 1, 0 ] },
{ "id": "280", "countPos": 4, "countNeg": 5, "raw": [ 1, 1, 0, 0, 0, 1, 0, 0, 1 ] },
{ "id": "281", "countPos": 3, "countNeg": 6, "raw": [ 0, 1, 0, 0, 0, 1, 1, 0, 0 ] },
{ "id": "282", "countPos": 3, "countNeg": 6, "raw": [ 0, 1, 0, 0, 0, 1, 0, 1, 0 ] },
{ "id": "283", "countPos": 4, "countNeg": 5, "raw": [ 1, 1, 1, 0, 1, 0, 0, 0, 0 ] },
{ "id": "284", "countPos": 6, "countNeg": 3, "raw": [ 1, 1, 1, 1, 0, 1, 1, 0, 0 ] },
{ "id": "285", "countPos": 3, "countNeg": 6, "raw": [ 0, 0, 1, 0, 0, 0, 1, 1, 0 ] },
{ "id": "286", "countPos": 5, "countNeg": 4, "raw": [ 0, 1, 0, 0, 1, 0, 1, 1, 1 ] },
{ "id": "287", "countPos": 5, "countNeg": 4, "raw": [ 1, 0, 1, 0, 1, 1, 0, 1, 0 ] },
{ "id": "288", "countPos": 2, "countNeg": 7, "raw": [ 0, 1, 0, 0, 0, 0, 0, 0, 1 ] },
{ "id": "289", "countPos": 4, "countNeg": 5, "raw": [ 0, 1, 1, 1, 0, 1, 0, 0, 0 ] },
{ "id": "29", "countPos": 2, "countNeg": 7, "raw": [ 1, 0, 0, 0, 0, 0, 0, 1, 0 ] },
{ "id": "290", "countPos": 3, "countNeg": 6, "raw": [ 0, 1, 0, 0, 1, 1, 0, 0, 0 ] },
{ "id": "291", "countPos": 2, "countNeg": 7, "raw": [ 0, 1, 0, 1, 0, 0, 0, 0, 0 ] },
{ "id": "292", "countPos": 9, "countNeg": 0, "raw": [ 1, 1, 1, 1, 1, 1, 1, 1, 1 ] },
{ "id": "293", "countPos": 2, "countNeg": 7, "raw": [ 0, 0, 0, 0, 0, 1, 0, 0, 1 ] },
{ "id": "294", "countPos": 4, "countNeg": 5, "raw": [ 0, 0, 1, 0, 1, 0, 1, 1, 0 ] },
{ "id": "295", "countPos": 3, "countNeg": 6, "raw": [ 1, 0, 0, 1, 0, 0, 1, 0, 0 ] },
{ "id": "296", "countPos": 8, "countNeg": 1, "raw": [ 1, 1, 1, 1, 1, 1, 1, 1, 0 ] },
{ "id": "297", "countPos": 1, "countNeg": 8, "raw": [ 0, 0, 0, 0, 0, 1, 0, 0, 0 ] },
{ "id": "298", "countPos": 1, "countNeg": 8, "raw": [ 0, 0, 0, 0, 1, 0, 0, 0, 0 ] },
{ "id": "299", "countPos": 8, "countNeg": 1, "raw": [ 1, 1, 1, 1, 0, 1, 1, 1, 1 ] },
{ "id": "3", "countPos": 3, "countNeg": 6, "raw": [ 0, 0, 0, 0, 1, 0, 1, 0, 1 ] },
{ "id": "30", "countPos": 6, "countNeg": 3, "raw": [ 1, 1, 0, 1, 0, 1, 1, 1, 0 ] },
{ "id": "31", "countPos": 5, "countNeg": 4, "raw": [ 1, 0, 0, 1, 1, 0, 1, 1, 0 ] },
{ "id": "32", "countPos": 9, "countNeg": 0, "raw": [ 1, 1, 1, 1, 1, 1, 1, 1, 1 ] },
{ "id": "33", "countPos": 5, "countNeg": 4, "raw": [ 1, 0, 0, 0, 0, 1, 1, 1, 1 ] },
{ "id": "34", "countPos": 7, "countNeg": 2, "raw": [ 1, 1, 1, 1, 0, 1, 0, 1, 1 ] },
{ "id": "35", "countPos": 8, "countNeg": 1, "raw": [ 1, 1, 1, 1, 1, 1, 1, 1, 0 ] },
{ "id": "36", "countPos": 5, "countNeg": 4, "raw": [ 1, 1, 1, 0, 1, 0, 1, 0, 0 ] },
{ "id": "37", "countPos": 8, "countNeg": 1, "raw": [ 1, 1, 1, 1, 1, 1, 1, 1, 0 ] },
{ "id": "38", "countPos": 0, "countNeg": 9, "raw": [ 0, 0, 0, 0, 0, 0, 0, 0, 0 ] },
{ "id": "39", "countPos": 6, "countNeg": 3, "raw": [ 0, 1, 1, 0, 1, 1, 1, 1, 0 ] },
{ "id": "4", "countPos": 2, "countNeg": 7, "raw": [ 0, 1, 0, 0, 0, 0, 1, 0, 0 ] },
{ "id": "40", "countPos": 8, "countNeg": 1, "raw": [ 1, 1, 1, 1, 1, 1, 1, 1, 0 ] },
{ "id": "41", "countPos": 5, "countNeg": 4, "raw": [ 0, 0, 0, 1, 0, 1, 1, 1, 1 ] },
{ "id": "42", "countPos": 5, "countNeg": 4, "raw": [ 0, 1, 0, 1, 1, 0, 1, 0, 1 ] },
{ "id": "43", "countPos": 6, "countNeg": 3, "raw": [ 0, 0, 1, 0, 1, 1, 1, 1, 1 ] },
{ "id": "44", "countPos": 6, "countNeg": 3, "raw": [ 1, 0, 1, 1, 1, 1, 1, 0, 0 ] },
{ "id": "45", "countPos": 5, "countNeg": 4, "raw": [ 0, 1, 1, 1, 0, 0, 1, 1, 0 ] },
{ "id": "46", "countPos": 5, "countNeg": 4, "raw": [ 1, 1, 1, 0, 1, 0, 0, 0, 1 ] },
{ "id": "47", "countPos": 5, "countNeg": 4, "raw": [ 1, 1, 1, 0, 1, 1, 0, 0, 0 ] },
{ "id": "48", "countPos": 4, "countNeg": 5, "raw": [ 1, 0, 1, 1, 0, 0, 0, 0, 1 ] },
{ "id": "49", "countPos": 7, "countNeg": 2, "raw": [ 1, 1, 1, 1, 0, 0, 1, 1, 1 ] },
{ "id": "5", "countPos": 6, "countNeg": 3, "raw": [ 1, 1, 1, 1, 0, 1, 1, 0, 0 ] },
{ "id": "50", "countPos": 7, "countNeg": 2, "raw": [ 1, 1, 1, 1, 1, 1, 0, 1, 0 ] },
{ "id": "51", "countPos": 4, "countNeg": 5, "raw": [ 0, 1, 0, 0, 0, 1, 1, 0, 1 ] },
{ "id": "52", "countPos": 4, "countNeg": 5, "raw": [ 0, 1, 0, 1, 1, 0, 0, 0, 1 ] },
{ "id": "53", "countPos": 3, "countNeg": 6, "raw": [ 0, 1, 0, 0, 0, 0, 0, 1, 1 ] }]

A_triplet_fftnet=[
{ "id": "fft1xfft2x2508", "countPos": 4, "countNeg": 9, "raw": [ 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1 ] },
{ "id": "fft1xfft2x2509", "countPos": 3, "countNeg": 10, "raw": [ 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0 ] },
{ "id": "fft1xfft2x2510", "countPos": 5, "countNeg": 8, "raw": [ 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1 ] },
{ "id": "fft1xfft2x2511", "countPos": 6, "countNeg": 7, "raw": [ 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1 ] },
{ "id": "fft1xfft2x2512", "countPos": 5, "countNeg": 8, "raw": [ 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1 ] },
{ "id": "fft1xfft2x2513", "countPos": 2, "countNeg": 11, "raw": [ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1 ] },
{ "id": "fft1xfft2x2514", "countPos": 4, "countNeg": 9, "raw": [ 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1 ] },
{ "id": "fft1xfft2x2515", "countPos": 2, "countNeg": 11, "raw": [ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0 ] },
{ "id": "fft1xfft2x2516", "countPos": 3, "countNeg": 10, "raw": [ 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0 ] },
{ "id": "fft1xfft2x2517", "countPos": 4, "countNeg": 9, "raw": [ 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0 ] },
{ "id": "fft1xfft2x2518", "countPos": 3, "countNeg": 10, "raw": [ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 ] },
{ "id": "fft1xfft2x2519", "countPos": 5, "countNeg": 8, "raw": [ 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0 ] },
{ "id": "fft1xfft2x2520", "countPos": 5, "countNeg": 8, "raw": [ 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0 ] },
{ "id": "fft1xfft2x2521", "countPos": 4, "countNeg": 9, "raw": [ 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1 ] },
{ "id": "fft1xfft2x2522", "countPos": 4, "countNeg": 9, "raw": [ 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0 ] },
{ "id": "fft1xfft2x2523", "countPos": 8, "countNeg": 5, "raw": [ 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1 ] },
{ "id": "fft1xfft2x2524", "countPos": 3, "countNeg": 10, "raw": [ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 ] },
{ "id": "fft1xfft2x2525", "countPos": 6, "countNeg": 7, "raw": [ 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0 ] },
{ "id": "fft1xfft2x2526", "countPos": 4, "countNeg": 9, "raw": [ 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0 ] },
{ "id": "fft1xfft2x2527", "countPos": 3, "countNeg": 10, "raw": [ 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1 ] },
{ "id": "fft1xfft2x2528", "countPos": 5, "countNeg": 8, "raw": [ 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0 ] },
{ "id": "fft1xfft2x2529", "countPos": 5, "countNeg": 8, "raw": [ 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1 ] },
{ "id": "fft1xfft2x2530", "countPos": 5, "countNeg": 8, "raw": [ 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1 ] },
{ "id": "fft1xfft2x2531", "countPos": 3, "countNeg": 10, "raw": [ 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0 ] },
{ "id": "fft1xfft2x2532", "countPos": 1, "countNeg": 12, "raw": [ 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ] },
{ "id": "fft1xfft2x2533", "countPos": 1, "countNeg": 12, "raw": [ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 ] },
{ "id": "fft1xfft2x2534", "countPos": 6, "countNeg": 7, "raw": [ 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1 ] },
{ "id": "fft1xfft2x2535", "countPos": 4, "countNeg": 9, "raw": [ 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0 ] },
{ "id": "fft1xfft2x2536", "countPos": 3, "countNeg": 10, "raw": [ 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1 ] },
{ "id": "fft1xfft2x2537", "countPos": 5, "countNeg": 8, "raw": [ 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1 ] },
{ "id": "fft1xfft2x2538", "countPos": 4, "countNeg": 10, "raw": [ 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0 ] },
{ "id": "fft1xfft2x2539", "countPos": 4, "countNeg": 10, "raw": [ 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0 ] },
{ "id": "fft1xfft2x2540", "countPos": 4, "countNeg": 10, "raw": [ 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0 ] },
{ "id": "fft1xfft2x2541", "countPos": 2, "countNeg": 12, "raw": [ 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0 ] },
{ "id": "fft1xfft2x2542", "countPos": 4, "countNeg": 10, "raw": [ 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0 ] },
{ "id": "fft1xfft2x2543", "countPos": 4, "countNeg": 10, "raw": [ 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0 ] },
{ "id": "fft1xfft2x2544", "countPos": 4, "countNeg": 10, "raw": [ 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1 ] },
{ "id": "fft1xfft2x2545", "countPos": 4, "countNeg": 10, "raw": [ 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0 ] },
{ "id": "fft1xfft2x2546", "countPos": 5, "countNeg": 9, "raw": [ 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1 ] },
{ "id": "fft1xfft2x2547", "countPos": 6, "countNeg": 8, "raw": [ 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0 ] },
{ "id": "fft1xfft2x2548", "countPos": 5, "countNeg": 9, "raw": [ 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0 ] },
{ "id": "fft1xfft2x2549", "countPos": 7, "countNeg": 7, "raw": [ 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1 ] },
{ "id": "fft1xfft2x2550", "countPos": 5, "countNeg": 9, "raw": [ 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0 ] },
{ "id": "fft1xfft2x2551", "countPos": 6, "countNeg": 8, "raw": [ 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1 ] },
{ "id": "fft1xfft2x2552", "countPos": 1, "countNeg": 13, "raw": [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 ] },
{ "id": "fft1xfft2x2553", "countPos": 6, "countNeg": 8, "raw": [ 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1 ] },
{ "id": "fft1xfft2x2554", "countPos": 7, "countNeg": 7, "raw": [ 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0 ] },
{ "id": "fft1xfft2x2555", "countPos": 7, "countNeg": 7, "raw": [ 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0 ] },
{ "id": "fft1xfft2x2556", "countPos": 4, "countNeg": 10, "raw": [ 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0 ] },
{ "id": "fft1xfft2x2557", "countPos": 5, "countNeg": 9, "raw": [ 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1 ] },
{ "id": "fft1xfft2x2558", "countPos": 5, "countNeg": 8, "raw": [ 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1 ] },
{ "id": "fft1xfft2x2559", "countPos": 4, "countNeg": 9, "raw": [ 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1 ] },
{ "id": "fft1xfft2x2560", "countPos": 4, "countNeg": 9, "raw": [ 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1 ] },
{ "id": "fft1xfft2x2561", "countPos": 2, "countNeg": 11, "raw": [ 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0 ] },
{ "id": "fft1xfft2x2562", "countPos": 3, "countNeg": 10, "raw": [ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1 ] },
{ "id": "fft1xfft2x2563", "countPos": 3, "countNeg": 10, "raw": [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0 ] },
{ "id": "fft1xfft2x2564", "countPos": 5, "countNeg": 8, "raw": [ 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1 ] },
{ "id": "fft1xfft2x2565", "countPos": 5, "countNeg": 8, "raw": [ 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1 ] },
{ "id": "fft1xfft2x2566", "countPos": 2, "countNeg": 11, "raw": [ 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0 ] },
{ "id": "fft1xfft2x2567", "countPos": 1, "countNeg": 12, "raw": [ 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 ] },
{ "id": "fft1xfft2x2568", "countPos": 1, "countNeg": 12, "raw": [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 ] },
{ "id": "fft1xfft2x2569", "countPos": 3, "countNeg": 10, "raw": [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1 ] },
{ "id": "fft1xfft2x2570", "countPos": 4, "countNeg": 9, "raw": [ 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0 ] },
{ "id": "fft1xfft2x2571", "countPos": 10, "countNeg": 3, "raw": [ 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1 ] },
{ "id": "fft1xfft2x2572", "countPos": 1, "countNeg": 12, "raw": [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 ] },
{ "id": "fft1xfft2x2573", "countPos": 3, "countNeg": 10, "raw": [ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1 ] },
{ "id": "fft1xmlsax100", "countPos": 4, "countNeg": 9, "raw": [ 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1 ] },
{ "id": "fft1xmlsax101", "countPos": 7, "countNeg": 6, "raw": [ 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1 ] },
{ "id": "fft1xmlsax102", "countPos": 8, "countNeg": 5, "raw": [ 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0 ] },
{ "id": "fft1xmlsax103", "countPos": 8, "countNeg": 5, "raw": [ 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0 ] },
{ "id": "fft1xmlsax104", "countPos": 9, "countNeg": 4, "raw": [ 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1 ] },
{ "id": "fft1xmlsax105", "countPos": 7, "countNeg": 6, "raw": [ 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1 ] },
{ "id": "fft1xmlsax106", "countPos": 4, "countNeg": 9, "raw": [ 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1 ] },
{ "id": "fft1xmlsax107", "countPos": 9, "countNeg": 4, "raw": [ 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1 ] },
{ "id": "fft1xmlsax108", "countPos": 5, "countNeg": 8, "raw": [ 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0 ] },
{ "id": "fft1xmlsax109", "countPos": 6, "countNeg": 7, "raw": [ 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1 ] },
{ "id": "fft1xmlsax110", "countPos": 8, "countNeg": 5, "raw": [ 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1 ] },
{ "id": "fft1xmlsax111", "countPos": 10, "countNeg": 3, "raw": [ 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1 ] },
{ "id": "fft1xmlsax112", "countPos": 10, "countNeg": 3, "raw": [ 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0 ] },
{ "id": "fft1xmlsax113", "countPos": 6, "countNeg": 7, "raw": [ 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1 ] },
{ "id": "fft1xmlsax114", "countPos": 6, "countNeg": 8, "raw": [ 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0 ] },
{ "id": "fft1xmlsax115", "countPos": 5, "countNeg": 9, "raw": [ 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0 ] },
{ "id": "fft1xmlsax116", "countPos": 7, "countNeg": 7, "raw": [ 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0 ] },
{ "id": "fft1xmlsax117", "countPos": 8, "countNeg": 6, "raw": [ 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0 ] },
{ "id": "fft1xmlsax118", "countPos": 7, "countNeg": 7, "raw": [ 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1 ] },
{ "id": "fft1xmlsax119", "countPos": 4, "countNeg": 10, "raw": [ 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0 ] },
{ "id": "fft1xmlsax120", "countPos": 8, "countNeg": 6, "raw": [ 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1 ] },
{ "id": "fft1xmlsax121", "countPos": 4, "countNeg": 10, "raw": [ 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0 ] },
{ "id": "fft1xmlsax122", "countPos": 7, "countNeg": 7, "raw": [ 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1 ] },
{ "id": "fft1xmlsax123", "countPos": 7, "countNeg": 7, "raw": [ 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1 ] },
{ "id": "fft1xmlsax124", "countPos": 6, "countNeg": 7, "raw": [ 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0 ] },
{ "id": "fft1xmlsax125", "countPos": 3, "countNeg": 10, "raw": [ 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 ] },
{ "id": "fft1xmlsax126", "countPos": 6, "countNeg": 7, "raw": [ 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1 ] },
{ "id": "fft1xmlsax127", "countPos": 3, "countNeg": 10, "raw": [ 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1 ] },
{ "id": "fft1xmlsax128", "countPos": 10, "countNeg": 3, "raw": [ 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1 ] },
{ "id": "fft1xmlsax129", "countPos": 7, "countNeg": 6, "raw": [ 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1 ] },
{ "id": "fft1xmlsax130", "countPos": 7, "countNeg": 6, "raw": [ 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0 ] },
{ "id": "fft1xmlsax131", "countPos": 6, "countNeg": 7, "raw": [ 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0 ] },
{ "id": "fft1xmlsax1386", "countPos": 6, "countNeg": 7, "raw": [ 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1 ] },
{ "id": "fft1xmlsax1387", "countPos": 3, "countNeg": 10, "raw": [ 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0 ] },
{ "id": "fft1xmlsax1388", "countPos": 6, "countNeg": 7, "raw": [ 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0 ] },
{ "id": "fft1xmlsax1389", "countPos": 4, "countNeg": 9, "raw": [ 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0 ] },
{ "id": "fft1xmlsax1390", "countPos": 4, "countNeg": 9, "raw": [ 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0 ] },
{ "id": "fft1xmlsax1391", "countPos": 2, "countNeg": 11, "raw": [ 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0 ] },
{ "id": "fft1xmlsax1392", "countPos": 3, "countNeg": 10, "raw": [ 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0 ] },
{ "id": "fft1xmlsax1393", "countPos": 4, "countNeg": 9, "raw": [ 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0 ] },
{ "id": "fft1xmlsax1394", "countPos": 6, "countNeg": 7, "raw": [ 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0 ] },
{ "id": "fft1xmlsax1395", "countPos": 4, "countNeg": 9, "raw": [ 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0 ] },
{ "id": "fft1xmlsax1396", "countPos": 4, "countNeg": 9, "raw": [ 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0 ] },
{ "id": "fft1xmlsax1397", "countPos": 6, "countNeg": 7, "raw": [ 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0 ] },
{ "id": "fft1xmlsax1398", "countPos": 5, "countNeg": 9, "raw": [ 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0 ] },
{ "id": "fft1xmlsax1399", "countPos": 8, "countNeg": 6, "raw": [ 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0 ] },
{ "id": "fft1xmlsax1400", "countPos": 8, "countNeg": 6, "raw": [ 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0 ] },
{ "id": "fft1xmlsax1401", "countPos": 9, "countNeg": 5, "raw": [ 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1 ] },
{ "id": "fft1xmlsax1402", "countPos": 5, "countNeg": 9, "raw": [ 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1 ] },
{ "id": "fft1xmlsax1403", "countPos": 5, "countNeg": 9, "raw": [ 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0 ] },
{ "id": "fft1xmlsax1404", "countPos": 8, "countNeg": 6, "raw": [ 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1 ] },
{ "id": "fft1xmlsax1405", "countPos": 4, "countNeg": 10, "raw": [ 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 ] },
{ "id": "fft1xmlsax1406", "countPos": 6, "countNeg": 8, "raw": [ 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1 ] },
{ "id": "fft1xmlsax1407", "countPos": 9, "countNeg": 5, "raw": [ 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1 ] },
{ "id": "fft1xmlsax1408", "countPos": 6, "countNeg": 8, "raw": [ 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1 ] },
{ "id": "fft1xmlsax1409", "countPos": 10, "countNeg": 4, "raw": [ 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1 ] },
{ "id": "fft1xmlsax1410", "countPos": 8, "countNeg": 6, "raw": [ 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1 ] },
{ "id": "fft1xmlsax1411", "countPos": 10, "countNeg": 4, "raw": [ 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1 ] },
{ "id": "fft1xmlsax1412", "countPos": 10, "countNeg": 4, "raw": [ 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1 ] },
{ "id": "fft1xmlsax1413", "countPos": 10, "countNeg": 4, "raw": [ 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1 ] },
{ "id": "fft1xmlsax1414", "countPos": 5, "countNeg": 9, "raw": [ 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1 ] },
{ "id": "fft1xmlsax1415", "countPos": 10, "countNeg": 4, "raw": [ 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1 ] },
{ "id": "fft1xmlsax1416", "countPos": 12, "countNeg": 2, "raw": [ 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1 ] },
{ "id": "fft1xmlsax1417", "countPos": 7, "countNeg": 7, "raw": [ 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1 ] },
{ "id": "fft1xmlsax1418", "countPos": 6, "countNeg": 7, "raw": [ 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0 ] },
{ "id": "fft1xmlsax1419", "countPos": 5, "countNeg": 8, "raw": [ 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1 ] },
{ "id": "fft1xmlsax1420", "countPos": 6, "countNeg": 7, "raw": [ 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0 ] },
{ "id": "fft1xmlsax1421", "countPos": 10, "countNeg": 3, "raw": [ 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1 ] },
{ "id": "fft1xmlsax1422", "countPos": 5, "countNeg": 8, "raw": [ 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0 ] },
{ "id": "fft1xmlsax1423", "countPos": 3, "countNeg": 10, "raw": [ 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0 ] },
{ "id": "fft1xmlsax1424", "countPos": 7, "countNeg": 6, "raw": [ 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0 ] },
{ "id": "fft1xmlsax1425", "countPos": 5, "countNeg": 8, "raw": [ 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1 ] },
{ "id": "fft1xmlsax1426", "countPos": 6, "countNeg": 7, "raw": [ 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0 ] },
{ "id": "fft1xmlsax1427", "countPos": 5, "countNeg": 8, "raw": [ 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0 ] },
{ "id": "fft1xmlsax1428", "countPos": 3, "countNeg": 10, "raw": [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0 ] },
{ "id": "fft1xmlsax1429", "countPos": 5, "countNeg": 8, "raw": [ 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0 ] },
{ "id": "fft1xmlsax1430", "countPos": 7, "countNeg": 6, "raw": [ 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0 ] },
{ "id": "fft1xmlsax1431", "countPos": 6, "countNeg": 7, "raw": [ 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0 ] },
{ "id": "fft1xmlsax1432", "countPos": 6, "countNeg": 7, "raw": [ 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1 ] },
{ "id": "fft1xmlsax1433", "countPos": 4, "countNeg": 9, "raw": [ 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0 ] },
{ "id": "fft1xmlsax1434", "countPos": 5, "countNeg": 8, "raw": [ 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0 ] },
{ "id": "fft1xmlsax1435", "countPos": 4, "countNeg": 9, "raw": [ 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0 ] },
{ "id": "fft1xmlsax1436", "countPos": 3, "countNeg": 10, "raw": [ 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0 ] },
{ "id": "fft1xmlsax1437", "countPos": 4, "countNeg": 9, "raw": [ 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0 ] },
{ "id": "fft1xmlsax1438", "countPos": 8, "countNeg": 5, "raw": [ 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0 ] },
{ "id": "fft1xmlsax1439", "countPos": 3, "countNeg": 10, "raw": [ 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0 ] },
{ "id": "fft1xmlsax1440", "countPos": 8, "countNeg": 5, "raw": [ 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1 ] },
{ "id": "fft1xmlsax1441", "countPos": 6, "countNeg": 7, "raw": [ 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1 ] },
{ "id": "fft1xmlsax1442", "countPos": 6, "countNeg": 7, "raw": [ 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1 ] },
{ "id": "fft1xmlsax1443", "countPos": 8, "countNeg": 5, "raw": [ 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1 ] },
{ "id": "fft1xmlsax1444", "countPos": 5, "countNeg": 8, "raw": [ 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1 ] },
{ "id": "fft1xmlsax1445", "countPos": 6, "countNeg": 7, "raw": [ 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0 ] },
{ "id": "fft1xmlsax1446", "countPos": 6, "countNeg": 7, "raw": [ 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1 ] },
{ "id": "fft1xmlsax1447", "countPos": 5, "countNeg": 8, "raw": [ 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0 ] },
{ "id": "fft1xmlsax1448", "countPos": 11, "countNeg": 2, "raw": [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1 ] },
{ "id": "fft1xmlsax1449", "countPos": 6, "countNeg": 7, "raw": [ 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0 ] },
{ "id": "fft1xmlsax1450", "countPos": 3, "countNeg": 10, "raw": [ 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0 ] },
{ "id": "fft1xmlsax1451", "countPos": 5, "countNeg": 8, "raw": [ 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0 ] },
{ "id": "fft1xmlsax2046", "countPos": 5, "countNeg": 8, "raw": [ 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0 ] },
{ "id": "fft1xmlsax2047", "countPos": 4, "countNeg": 9, "raw": [ 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1 ] },
{ "id": "fft1xmlsax2048", "countPos": 2, "countNeg": 11, "raw": [ 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0 ] },
{ "id": "fft1xmlsax2049", "countPos": 4, "countNeg": 9, "raw": [ 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1 ] },
{ "id": "fft1xmlsax2050", "countPos": 6, "countNeg": 7, "raw": [ 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1 ] },
{ "id": "fft1xmlsax2051", "countPos": 2, "countNeg": 11, "raw": [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 ] },
{ "id": "fft1xmlsax2052", "countPos": 4, "countNeg": 9, "raw": [ 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0 ] },
{ "id": "fft1xmlsax2053", "countPos": 6, "countNeg": 7, "raw": [ 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1 ] },
{ "id": "fft1xmlsax2054", "countPos": 7, "countNeg": 6, "raw": [ 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0 ] },
{ "id": "fft1xmlsax2055", "countPos": 5, "countNeg": 8, "raw": [ 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0 ] },
{ "id": "fft1xmlsax2056", "countPos": 6, "countNeg": 7, "raw": [ 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1 ] },
{ "id": "fft1xmlsax2057", "countPos": 6, "countNeg": 7, "raw": [ 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0 ] },
{ "id": "fft1xmlsax2058", "countPos": 5, "countNeg": 8, "raw": [ 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1 ] },
{ "id": "fft1xmlsax2059", "countPos": 7, "countNeg": 6, "raw": [ 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1 ] },
{ "id": "fft1xmlsax2060", "countPos": 6, "countNeg": 7, "raw": [ 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1 ] },
{ "id": "fft1xmlsax2061", "countPos": 8, "countNeg": 5, "raw": [ 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0 ] },
{ "id": "fft1xmlsax2062", "countPos": 6, "countNeg": 7, "raw": [ 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1 ] },
{ "id": "fft1xmlsax2063", "countPos": 5, "countNeg": 8, "raw": [ 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0 ] },
{ "id": "fft1xmlsax2064", "countPos": 4, "countNeg": 9, "raw": [ 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1 ] },
{ "id": "fft1xmlsax2065", "countPos": 5, "countNeg": 8, "raw": [ 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0 ] },
{ "id": "fft1xmlsax2066", "countPos": 5, "countNeg": 8, "raw": [ 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1 ] },
{ "id": "fft1xmlsax2067", "countPos": 7, "countNeg": 6, "raw": [ 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1 ] },
{ "id": "fft1xmlsax2068", "countPos": 4, "countNeg": 9, "raw": [ 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1 ] },
{ "id": "fft1xmlsax2069", "countPos": 9, "countNeg": 4, "raw": [ 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1 ] },
{ "id": "fft1xmlsax2070", "countPos": 7, "countNeg": 6, "raw": [ 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0 ] },
{ "id": "fft1xmlsax2071", "countPos": 7, "countNeg": 6, "raw": [ 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1 ] },
{ "id": "fft1xmlsax2072", "countPos": 6, "countNeg": 8, "raw": [ 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1 ] },
{ "id": "fft1xmlsax2073", "countPos": 8, "countNeg": 6, "raw": [ 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1 ] },
{ "id": "fft1xmlsax2074", "countPos": 6, "countNeg": 8, "raw": [ 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1 ] },
{ "id": "fft1xmlsax2075", "countPos": 7, "countNeg": 7, "raw": [ 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1 ] },
{ "id": "fft1xmlsax2076", "countPos": 6, "countNeg": 8, "raw": [ 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1 ] },
{ "id": "fft1xmlsax2077", "countPos": 6, "countNeg": 8, "raw": [ 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1 ] },
{ "id": "fft1xmlsax2078", "countPos": 3, "countNeg": 11, "raw": [ 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1 ] },
{ "id": "fft1xmlsax2079", "countPos": 4, "countNeg": 10, "raw": [ 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0 ] },
{ "id": "fft1xmlsax2080", "countPos": 6, "countNeg": 8, "raw": [ 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1 ] },
{ "id": "fft1xmlsax2081", "countPos": 9, "countNeg": 5, "raw": [ 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1 ] },
{ "id": "fft1xmlsax2082", "countPos": 10, "countNeg": 5, "raw": [ 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1 ] },
{ "id": "fft1xmlsax2083", "countPos": 8, "countNeg": 7, "raw": [ 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0 ] },
{ "id": "fft1xmlsax2084", "countPos": 3, "countNeg": 12, "raw": [ 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0 ] },
{ "id": "fft1xmlsax2085", "countPos": 7, "countNeg": 8, "raw": [ 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0 ] },
{ "id": "fft1xmlsax2086", "countPos": 5, "countNeg": 10, "raw": [ 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0 ] },
{ "id": "fft1xmlsax2087", "countPos": 8, "countNeg": 7, "raw": [ 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1 ] },
{ "id": "fft1xmlsax2088", "countPos": 7, "countNeg": 8, "raw": [ 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1 ] },
{ "id": "fft1xmlsax2089", "countPos": 6, "countNeg": 9, "raw": [ 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0 ] },
{ "id": "fft1xmlsax2090", "countPos": 5, "countNeg": 10, "raw": [ 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1 ] },
{ "id": "fft1xmlsax2091", "countPos": 8, "countNeg": 7, "raw": [ 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1 ] },
{ "id": "fft1xmlsax2092", "countPos": 6, "countNeg": 7, "raw": [ 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0 ] },
{ "id": "fft1xmlsax2093", "countPos": 3, "countNeg": 10, "raw": [ 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1 ] },
{ "id": "fft1xmlsax2094", "countPos": 7, "countNeg": 6, "raw": [ 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1 ] },
{ "id": "fft1xmlsax2095", "countPos": 6, "countNeg": 7, "raw": [ 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1 ] },
{ "id": "fft1xmlsax2096", "countPos": 7, "countNeg": 6, "raw": [ 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1 ] },
{ "id": "fft1xmlsax2097", "countPos": 5, "countNeg": 8, "raw": [ 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1 ] },
{ "id": "fft1xmlsax2098", "countPos": 4, "countNeg": 9, "raw": [ 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1 ] },
{ "id": "fft1xmlsax2099", "countPos": 6, "countNeg": 7, "raw": [ 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0 ] },
{ "id": "fft1xmlsax2100", "countPos": 6, "countNeg": 7, "raw": [ 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0 ] },
{ "id": "fft1xmlsax2101", "countPos": 7, "countNeg": 6, "raw": [ 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1 ] },
{ "id": "fft1xmlsax2102", "countPos": 8, "countNeg": 6, "raw": [ 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1 ] },
{ "id": "fft1xmlsax2103", "countPos": 5, "countNeg": 9, "raw": [ 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0 ] },
{ "id": "fft1xmlsax2104", "countPos": 5, "countNeg": 9, "raw": [ 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1 ] },
{ "id": "fft1xmlsax2105", "countPos": 4, "countNeg": 10, "raw": [ 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0 ] },
{ "id": "fft1xmlsax2106", "countPos": 7, "countNeg": 7, "raw": [ 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0 ] },
{ "id": "fft1xmlsax2107", "countPos": 5, "countNeg": 9, "raw": [ 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1 ] },
{ "id": "fft1xmlsax2108", "countPos": 5, "countNeg": 9, "raw": [ 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1 ] },
{ "id": "fft1xmlsax2109", "countPos": 9, "countNeg": 5, "raw": [ 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1 ] },
{ "id": "fft1xmlsax2110", "countPos": 7, "countNeg": 7, "raw": [ 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1 ] },
{ "id": "fft1xmlsax2111", "countPos": 5, "countNeg": 9, "raw": [ 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1 ] },
{ "id": "fft1xmlsax66", "countPos": 5, "countNeg": 9, "raw": [ 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0 ] },
{ "id": "fft1xmlsax67", "countPos": 7, "countNeg": 7, "raw": [ 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1 ] },
{ "id": "fft1xmlsax68", "countPos": 8, "countNeg": 6, "raw": [ 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0 ] },
{ "id": "fft1xmlsax69", "countPos": 9, "countNeg": 5, "raw": [ 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1 ] },
{ "id": "fft1xmlsax70", "countPos": 7, "countNeg": 7, "raw": [ 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0 ] },
{ "id": "fft1xmlsax71", "countPos": 8, "countNeg": 6, "raw": [ 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0 ] },
{ "id": "fft1xmlsax72", "countPos": 5, "countNeg": 9, "raw": [ 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0 ] },
{ "id": "fft1xmlsax726", "countPos": 9, "countNeg": 5, "raw": [ 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1 ] },
{ "id": "fft1xmlsax727", "countPos": 7, "countNeg": 7, "raw": [ 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1 ] },
{ "id": "fft1xmlsax728", "countPos": 8, "countNeg": 6, "raw": [ 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1 ] },
{ "id": "fft1xmlsax729", "countPos": 7, "countNeg": 6, "raw": [ 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1 ] },
{ "id": "fft1xmlsax73", "countPos": 9, "countNeg": 4, "raw": [ 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0 ] },
{ "id": "fft1xmlsax730", "countPos": 5, "countNeg": 8, "raw": [ 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1 ] },
{ "id": "fft1xmlsax731", "countPos": 6, "countNeg": 7, "raw": [ 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0 ] },
{ "id": "fft1xmlsax732", "countPos": 7, "countNeg": 6, "raw": [ 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1 ] },
{ "id": "fft1xmlsax733", "countPos": 6, "countNeg": 7, "raw": [ 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0 ] },
{ "id": "fft1xmlsax734", "countPos": 10, "countNeg": 3, "raw": [ 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1 ] },
{ "id": "fft1xmlsax735", "countPos": 6, "countNeg": 7, "raw": [ 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1 ] },
{ "id": "fft1xmlsax736", "countPos": 6, "countNeg": 7, "raw": [ 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0 ] },
{ "id": "fft1xmlsax737", "countPos": 8, "countNeg": 5, "raw": [ 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1 ] }]


def load_full_data_list(A_triplet_mymetric,A_triplet_fftnet,args): #check change path names
        
    
    if args.triplet_or_MOS=='triplet':
        list_methods=[]
        dataset={}
        print("Loading Files....")
        dataset['anchor_inname']=[]
        dataset['sample1_inname']=[]
        dataset['sample2_inname']=[]
        
        if args.dataset_used=='mymetric':
            A1=A_triplet_mymetric
            for i in range(len(A1)):
                filename=A1[i]['id']
                path='../../www/mturk_hosts/website_all_combined_final/'
                file_ref=path+'triplets/first_set/triplet_pairs_new/anchor/'+'Audio_anchor_'+str(A1[i]['id'])+'.wav'
                file_sample1=path+'triplets/first_set/triplet_pairs_new/sample1/'+'Audio_sample1_'+str(A1[i]['id'])+'.wav'
                file_sample2=path+'triplets/first_set/triplet_pairs_new/sample2/'+'Audio_sample2_'+str(A1[i]['id'])+'.wav'
                dataset['anchor_inname'].append(file_ref)
                dataset['sample1_inname'].append(file_sample1)
                dataset['sample2_inname'].append(file_sample2)
        elif args.dataset_used=='fftnet':
            A1=A_triplet_fftnet
            for i in range(len(A1)):
                filename=A1[i]['id']
                path='../../fftnet_mos/'
                file_ref=path+'triplet_pairs/anchor/'+'Audio_anchor_'+str(A1[i]['id'])+'.wav'
                file_sample1=path+'triplet_pairs/sample1/'+'Audio_sample1_'+str(A1[i]['id'])+'.wav'
                file_sample2=path+'triplet_pairs/sample2/'+'Audio_sample2_'+str(A1[i]['id'])+'.wav'
                dataset['anchor_inname'].append(file_ref)
                dataset['sample1_inname'].append(file_sample1)
                dataset['sample2_inname'].append(file_sample2)
            
    elif  args.triplet_or_MOS=='MOS':
        
        if args.dataset_used=='voco':
            
            list_methods=numpy.array(["SIRI","CUTE","RG67","RGALT","RGEdit2","REAL"])
            speakers=["CLB","DBL","RMS","SLT"]
            list_pa='/n/fs/percepaudio/zeyu_voco/zeyu_data/'

            dataset={}
            print("Loading Files....")
            for i in list_methods:
                dataset[i]={}
                dataset[i]['inname']=[]

            for speaker in speakers:
                list_path=os.path.join(list_pa,speaker)
                for method in list_methods:
                    path=os.path.join(list_path,method)
                    a1=sorted(os.listdir(path))
                    for i in range(len(a1)):
                        file_names=a1[i]
                        dataset[method]['inname'].append(os.path.join(path,file_names))
        
        elif args.dataset_used=='fftnet':
            list_methods=numpy.array(["mlsa","wn1","fft1","wn2","fft2","real"])
            names=np.array(["01081","01041","01093","01092","01106","01103",
                "01118","01085","01096","01115","01120","01034","01102","01039","01109",
                "01131","01123","01100","01114","01052","01087","01077","01038","01063","01127",
                "01036","01044","01098","01047","01056","01130","01079","01072","01094","01091",
                "01097","01040","01067","01069","01122","01101","01042","01110",
                "01071","01033","01031","01089","01078","01113","01049","01075","01073","01086",
                "01048","01065","01090","01125","01124","01061","01045","01062","01082","01054",
                "01057","01117","01070"])
           
            dataset={}
            print("Loading Files....")
            for i in list_methods:
                dataset[i]={}
                dataset[i]['inname']=[]
            speakers=["slt","bdl","clb","rms"]

            list_pa='/n/fs/percepaudio/fftnet_mos/clips1/data/'
            for speaker in speakers:
                list_path=os.path.join(list_pa,speaker)
                for method in list_methods:
                    path=os.path.join(list_path,method)
                    a1=sorted(os.listdir(path))
                    for i in range(50):
                        file_names=speaker+"_"+method+"_"+str(names[i])+".wav"
                        dataset[method]['inname'].append(os.path.join(path,file_names))
        
        if args.dataset_used=='bwe':
            list_methods=numpy.array(["VALID-F","Kuleshov","SPEC","OUR","VALID-R"])
            speakers=["f10","multi-m","multi-f","m10"]
            list_pa='/n/fs/percepaudio/daps/berthy/mturk-bwe/'

            dataset={}
            print("Loading Files....")
            for i in list_methods:
                dataset[i]={}
                dataset[i]['inname']=[]

            for speaker in speakers:
                list_path=os.path.join(list_pa,speaker)
                for method in list_methods:
                    path=os.path.join(list_path,method)
                    a1=sorted(os.listdir(path))
                    if speaker=='m10' and method=='Kuleshov':
                        for i in range(24):
                            file_names=speaker+"_"+method+"_"+str(i+1).zfill(5)+".wav"
                            dataset[method]['inname'].append(os.path.join(path,file_names))
                    else:
                        for i in range(24):
                            file_names=speaker+"_"+method+"_"+str(i).zfill(5)+".wav"
                            dataset[method]['inname'].append(os.path.join(path,file_names))

    return dataset,list_methods


def load_full_data(dataset,list_methods,args):
    
    
    if args.triplet_or_MOS=='triplet':
        
        list=['anchor_inaudio','sample1_inaudio','sample2_inaudio']
        list_name=['anchor_inname','sample1_inname','sample2_inname']

        for i,method in enumerate(list):

            dataset[method]  = [None]*len(dataset[list_name[i]])

            for id in tqdm(range(len(dataset[list_name[i]]))):

                if dataset[method][id] is None:

                    fs, a1  = wavfile.read(dataset[list_name[i]][id])
                    a1=resampy.resample(a1, fs, 16000)

                    shape1=np.shape(a1)

                    a1  = np.reshape(a1, [-1, 1])

                    shape_wav = np.shape(a1)

                    inputData_wav = np.reshape(a1, [1, 1,shape_wav[0], shape_wav[1]])
                    inputData_wav  = np.float32(inputData_wav)
                    dataset[method][id]  = inputData_wav
                    
    elif args.triplet_or_MOS=='MOS':
        
        for method in list_methods:
            dataset[method]['inaudio']  = [None]*len(dataset[method]['inname'])
            for id in tqdm(range(len(dataset[method]['inname']))):
                if dataset[method]['inaudio'][id] is None:
                    fs, inputData  = wavfile.read(dataset[method]['inname'][id])
                    inputData = resampy.resample(inputData, fs, 16000)
                    fs=16000
                    shape1=np.shape(inputData)
                    inputData_wav  = np.reshape(inputData, [-1, 1])
                    shape_wav = np.shape(inputData_wav)
                    inputData_wav = np.reshape(inputData_wav, [1, 1,shape_wav[0], shape_wav[1]])
                    inputData_wav  = np.float32(inputData_wav)
                    dataset[method]['inaudio'][id]  = inputData_wav

    return dataset


def load_full_data_prepare_triplets(dataset,sets1,id_value):
    #list=['anchor_inaudio','sample1_inaudio','sample2_inaudio']
    
    ref=dataset['anchor_inaudio'][id_value]
    s1=dataset['sample1_inaudio'][id_value]
    s2=dataset['sample2_inaudio'][id_value]
    
    ref=np.reshape(ref,[ref.shape[2]])
    s1=np.reshape(s1,[s1.shape[2]])
    s2=np.reshape(s2,[s2.shape[2]])
    
    shape1=ref.shape[0]
    shape2=s1.shape[0]
    shape3=s2.shape[0]
    
    if shape1>=shape2 and shape1>=shape3:
        #shape1>shape2/shape3
        difference=shape1-shape2
        a=(np.zeros(difference))
        s1=np.append(a,s1,axis=0)
        
        difference=shape1-shape3
        a=(np.zeros(difference))
        s2=np.append(a,s2,axis=0)
        
    
    elif shape1>=shape2 and shape1<=shape3:
        #shape 3 make equal to shape1 shape2
        
        difference=shape3-shape1
        a=(np.zeros(difference))
        ref=np.append(a,ref,axis=0)
        
        difference=shape3-shape2
        a=(np.zeros(difference))
        s1=np.append(a,s1,axis=0)
        
    elif shape1<=shape2 and shape1<=shape3:
        #see which of shape 2 or shape 3 is bigger and update:
        if shape2>=shape3:
            #shape1<shape3<shape2
            difference=shape2-shape1
            a=(np.zeros(difference))
            ref=np.append(a,ref,axis=0)
            
            difference=shape2-shape3
            a=(np.zeros(difference))
            s2=np.append(a,s2,axis=0)
            
        elif shape2<=shape3:
            #abc
            difference=shape3-shape1
            a=(np.zeros(difference))
            ref=np.append(a,ref,axis=0)
            
            difference=shape3-shape2
            a=(np.zeros(difference))
            s1=np.append(a,s1,axis=0)
        
    elif shape1<=shape2 and shape1>=shape3:
        #shape 2 make all equal
        difference=shape2-shape1
        a=(np.zeros(difference))
        ref=np.append(a,ref,axis=0)
        
        difference=shape2-shape3
        a=(np.zeros(difference))
        s2=np.append(a,s2,axis=0)
    
    ref=np.reshape(ref,[1,1,ref.shape[0],1])
    s1=np.reshape(s1,[1,1,s1.shape[0],1])
    s2=np.reshape(s2,[1,1,s2.shape[0],1])
    
    return [ref,s1,s2]


def load_full_data_prepare_MOS(dataset,sets1,id_value,clean_method):
    
    noisy=dataset[sets1]['inaudio'][id_value]
    clean=dataset[clean_method]['inaudio'][id_value]
    
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


def get_score(args,pickle_path,A_triplet_mymetric,A_triplet_fftnet,dataset):
    
    if args.triplet_or_MOS=='triplet':
        
        pickle_in = open(pickle_path,"rb")
        output = pickle.load(pickle_in)
        distance1=output
        
        if args.dataset_used=='mymetric':
            A=A_triplet_mymetric
        elif args.dataset_used=='fftnet':
            A=A_triplet_fftnet
        
        act=[] #actual prediction
        indices=[] #indices of the list
        for i in range(len(A)):
            a=float(A[i]['countPos'])
            b=A[i]['countNeg']
            if a/(a+b)>=args.split_triplet:
                indices.append(i)
                act.append(1)
            if b/(a+b)>=args.split_triplet:
                indices.append(i)
                act.append(0)
               
        try:
            pred=[]
            for i in indices:
                if distance1[i][0]<=distance1[i][1]:
                    pred.append(1)
                else:
                    pred.append(0)
        except:
            pred=[]
            for i in indices:
                if distance1[i][0][0]<=distance1[i][1][0]:
                    pred.append(1)
                else:
                    pred.append(0)
        
        score=f1_score(act,pred)
        
    elif args.triplet_or_MOS=='MOS':
        
        if args.dataset_used=='voco':
            pickle_in = open(pickle_path,"rb")
            output = pickle.load(pickle_in)
            
            final_pesq=[]
            for i in range(len(output)):
                for j in range(len(output[i])):
                    final_pesq.append(output[i][j])
            
            output=np.asarray(final_pesq)
            final_pesq=[]
            for i in range(20):
                final_pesq.append(np.mean(output[i*44:i*44+44]))
            score=voco_mos(final_pesq)
            
        elif args.dataset_used=='fftnet':
            pickle_in = open(pickle_path,"rb")
            output = pickle.load(pickle_in)
            
            final_pesq=[]
            for i in range(len(output)):
                for j in range(len(output[i])):
                    final_pesq.append(output[i][j])
            
            output=np.asarray(final_pesq)
            final_pesq=[]
            for i in range(20):
                final_pesq.append(np.mean(output[i*50:i*50+50]))
            score=fftnet_mos(final_pesq)
            
        elif args.dataset_used=='bwe':
            
            pickle_in = open(pickle_path,"rb")
            output = pickle.load(pickle_in)
            
            final_pesq=[]
            for i in range(len(output)):
                for j in range(len(output[i])):
                    final_pesq.append(output[i][j])
            
            output=np.asarray(final_pesq)
            final_pesq=[]
            for i in range(16):
                final_pesq.append(np.mean(output[i*24:i*24+24]))
            score=bwe_mos(final_pesq,dataset)
            
    return score

##### MAP
def load_full_data_list_test(datafolder='dataset',filename='dataset_test_mp3.txt'): #check change path names

    noises=['applause','blue_noise','brown_noise','crickets','pink_noise','reverb_noise','siren','violet_noise','water_drops','white_noise','mp3']
    #sets=['train','val']
    dataset={}
    dataset['all']={}
    print('Loading files..')
    dataset['all']['inname'] = []
    dataset['all']['outname'] = []
    dataset['all']['label']=[]
    
    for noise in (noises):
        file = open(os.path.join(datafolder,filename), 'r')
        for line in file: 
            split_line=line.split('\t')
            if split_line[3][:-1].strip() == noise:
                #print(split_line[2])
                dataset['all']['inname'].append("%s_list/%s"%(datafolder+noise,split_line[0]))
                dataset['all']['outname'].append("%s_list/%s"%(datafolder+noise,split_line[1]))
                dataset['all']['label'].append(split_line[2])
    return dataset 
    
    
def load_full_data_list_combined_test(datafolder='dataset',filename='dataset_test_mp3.txt'): #check change path names

    dataset={}
    dataset['all']={}
    print('Loading files..')
    dataset['all']['inname'] = []
    dataset['all']['outname'] = []
    dataset['all']['label']=[]
    
    path='../'
    file = open(os.path.join(datafolder,filename), 'r')
    for line in file: 
        split_line=line.split('\t')
        dataset['all']['inname'].append("%s"%(datafolder+split_line[0]))
        dataset['all']['outname'].append("%s"%(datafolder+split_line[1]))
        dataset['all']['label'].append(split_line[2][:-1])
        
    return dataset

def load_full_data_test_waveform(dataset,sets,id_value):
    
    fs, inputData  = wavfile.read(dataset[sets]['inname'][id_value])

    fs, outputData = wavfile.read(dataset[sets]['outname'][id_value])

    shape1=np.shape(inputData)
    shape2=np.shape(outputData)
    
    
    if shape1[0]>shape2[0]:
        a=(np.zeros(shape1[0]-shape2[0]))
        outputData=np.append(a,outputData,axis=0)
                    
    elif shape1[0]<shape2[0]:
        a=(np.zeros(shape2[0]-shape1[0]))
        inputData=np.append(a,inputData,axis=0)
    
    inputData_wav  = np.reshape(inputData, [-1, 1])
    outputData_wav = np.reshape(outputData, [-1, 1])

  
    shape_wav = np.shape(inputData_wav)

    inputData_wav = np.reshape(inputData_wav, [1, 1,shape_wav[0], shape_wav[1]])
    outputData_wav = np.reshape(outputData_wav, [1, 1,shape_wav[0], shape_wav[1]])


    inputData_wav  = np.float32(inputData_wav)
    outputData_wav = np.float32(outputData_wav)
    
    return [inputData_wav,outputData_wav]


def scores_map(args):
        
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
            
        [ap0,auc0]=map_score_calc('linear')
        [ap1,auc1]=map_score_calc('reverb')
        [ap2,auc2]=map_score_calc('mp3')
        [ap3,auc3]=map_score_calc('combined')
    
    return [ap0,auc0,ap1,auc1,ap2,auc2,ap3,auc3]
    
    
def map_score_calc(noise):   
    
    filename='dataset_test_'+noise+'.txt'
    
    import numpy as np
    if noise!='combined':
            dataset_test=load_full_data_list_test('../dataset_collection/',filename)
    elif noise=='combined':
        dataset_test=load_full_data_list_combined_test('../dataset_collection/',filename)
    
    output=np.zeros((len(dataset_test["all"]["inname"]),1))
                     
    for id in tqdm(range(0, len(dataset_test["all"]["inname"]))):
        
        wav_in,wav_out=load_full_data_test_waveform(dataset_test,'all',id)
        a,_= sess.run([dist,dist],feed_dict={input:wav_in, clean1_wav:wav_out})
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