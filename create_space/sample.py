import soundfile as sf
import os
import numpy
import numpy as np
import math
from scipy import signal as sg
import csv
import soundfile as sf
import os
import numpy
import math
import random
import librosa
import tqdm
import subprocess
import os
import sys
import subprocess
import time

## White noise
def white_noise(audio,level): #from levels 0 - 100 -> higher level, more noise
    
    starting_w=0.0001
    ending_w = starting_w*800.0
    levels = 100.0
    F = (ending_w / starting_w)
    expon = 1 / (levels)
    F1 = F ** (expon)
    jk=level
    noise_level = starting_w * (F1 ** (jk))
    noise = noise_level * np.random.random_sample(audio.shape[0])
    conv = noise + audio
    return conv

## compression
def compression_mp3(audio,level): #from levels 0 - 100 -> higher level, more noise
    
    starting_w=300.0
    ending_w = 1.0
    levels = 100.0
    F = (ending_w / starting_w)
    expon = 1 / (levels)
    F1 = F ** (expon)
    jk=level
    noise_level = starting_w * (F1 ** (jk))
    noise_level=round(noise_level,4)
    
    seconds = time.time()
    usage = str(seconds)
    
    os.mkdir(usage)
    
    sf.write(os.path.join(usage,'3.wav'),audio,44100)
    bashCommand = "ffmpeg -i "+usage+"/3.wav"+" -vn -ar 44100 -b:a " + str(noise_level)+"k "+usage+"/3.mp3"
    print(bashCommand)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    bashCommand_reverse = "ffmpeg -i "+usage+"/3.mp3 "+usage+"/4.wav"
    print(bashCommand_reverse)
    process = subprocess.Popen(bashCommand_reverse.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    
    audio,sr=sf.read(usage+"/4.wav")
    os.remove(usage+"/3.mp3")
    os.remove(usage+"/3.wav")
    os.remove(usage+"/4.wav")
    
    os.rmdir(usage)
    
    return audio

## Pops noise
def pops(level,audio,args=1):
    
    audio_return=np.zeros(audio.shape)
    starting_w=0.0001
    ending_w = 0.10
    levels = 100.0
    F = (ending_w / starting_w)
    expon = 1 / (levels)
    F1 = F ** (expon)
    jk=level
    noise_level = starting_w * (F1 ** (jk))
    noise_level=round(noise_level,4)
    
    p=noise_level
    length=p*len(audio)
    maxi=np.max(audio)
    mini=np.min(audio)
    for i in range(len(audio)):
        audio_return[i]=audio[i]
    
    length1=p*len(audio_return)
    A=np.random.choice(int(len(audio_return)), int(length1), replace=False)
    A1=np.random.choice(A,int(len(A)/2),replace=False)
    A2=np.setdiff1d(A, A1)
    audio_return[A1]=maxi
    audio_return[A2]=mini
    
    return audio_return


## Equalization Noise
def EQ_create(audio,a_levels,fs,args=1):
    
    EQ_bands = '5e2_1e3'
    
    freq_1=float(EQ_bands.split('_')[0])
    freq_2=float(EQ_bands.split('_')[1])
    
    a_levels = a_levels.astype(np.float)
    #EQ CONTROL at 3 bands
    #1. 5e2
    #2. 1e3
    #x1_low
    b, a = sg.butter(4, freq_1 / (fs / 2.), 'low')
    x_fil = sg.filtfilt(b, a, audio)
    #x1_high
    b, a = sg.butter(4, freq_1 / (fs / 2.), 'high')
    x_fil1 = sg.filtfilt(b, a, audio)
    #x2_low
    b, a = sg.butter(4, freq_2 / (fs / 2.), 'low')
    x_fil2 = sg.filtfilt(b, a, x_fil1)
    #x2_high
    b, a = sg.butter(4, freq_2 / (fs / 2.), 'high')
    x_fil3 = sg.filtfilt(b, a, x_fil1)
    #x3_low

    x_level=np.array([x_fil,x_fil2,x_fil3])
    
    starting_w = 1.0
    ending_w = 10.0
    levels = 100.0
    F = (ending_w / starting_w)
    expon = 1 / (levels)
    F1 = F ** (expon)
    jk=a_levels
    noise_level = starting_w * (F1 ** (jk))
    
    sum=0
    for i in range((noise_level.shape[0])):
        sum+=x_level[i]*(1/float(noise_level[i]))
    return sum


## Reverberation
def rt60_calculation(noise_l,audio,audio_ir,sample_rate):
    starting_w=1.0  
    ending_w = 7.0
    levels = 100.0
    F = (ending_w / starting_w)
    expon = 1 / (levels)
    F1 = F ** (expon)
    jk=noise_l
    noise_level = starting_w * (F1 ** (jk))
    y_slow = librosa.effects.time_stretch(audio_ir,1/noise_level)
    conv=np.convolve(audio,y_slow,'full')
    conv1=conv[0:int(sample_rate*2.5)]
    return conv1