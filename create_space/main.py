import numpy
import numpy as np
import random
import os
import argparse
from sample import *
import csv
import tqdm
import soundfile as sf


#load a clean audio file:
audio,fs=sf.read('../sample_audio/2.wav')

#add selected perturbations: for example white noise: # from levels 0-100
audio_new=white_noise(audio,50)
sf.write('audio_50.wav',audio_new,fs)