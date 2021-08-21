from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import resampy
from scipy.io import wavfile
import os, csv

import torch
import sys
import torch.nn.functional as F
import shutil
import scipy.io
import librosa
import os
import numpy as np
import numpy.matlib
import random
import subprocess
import pickle
import os
import argparse
import resampy
import csv
from scipy.io import wavfile
import os, csv

from cdpam.cdpam import *

def load_audio(path):
    
    inputData,fs  = librosa.load(path,sr=22050)
    
    ## convert to 16 bit floating point
    inputData = np.round(inputData.astype(np.float)*32768)
    
    inputData  = np.reshape(inputData, [-1, 1])
    
    shape_wav = np.shape(inputData)
    
    inputData = np.reshape(inputData, [1,shape_wav[0]])
    
    inputData  = np.float32(inputData)
    
    return inputData