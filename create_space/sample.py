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

def white_noise(audio,level):
    
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

def compression_mp3(audio,level):
    
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