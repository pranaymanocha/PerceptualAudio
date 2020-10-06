from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import resampy
from scipy.io import wavfile
import os, csv

from pip_pasm.pip_pasm import *

def check_wav_mp3(filename):
    count=0
    ext=filename[-3:]
    if ext!='wav':
        count+=1
    return count
    
def convert_mp3_wav(filename,outfile): 
    subprocess.call(['ffmpeg', '-i', filename,'-ar','16000',outfile])
    
def load_audio(path):
    
    check=check_wav_mp3(path)
    
    if check!=0:
        # not wav
        seconds = time.time()
        output_file = path[:-3]+'wav'
        convert_mp3_wav(path,output_file)
        fs, inputData  = wavfile.read(output_file)
    else:
        #wav
        fs, inputData  = wavfile.read(path)
    '''
    if fs!=16000:
        
        inputData = resampy.resample(inputData, fs, 16000)
        fs=16000
    
    maximum = np.amax(inputData)
    
    if maximum<=1:
        inputData = np.rint(inputData*32768)
    '''
    shape = np.shape(inputData)
    inputData = np.reshape(inputData, [1, 1,shape[0], 1])
    
    inputData  = np.float32(inputData)
    
    return inputData


def check_length(audio1,audio2):
    
    noisy=audio1
    clean=audio2
    
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