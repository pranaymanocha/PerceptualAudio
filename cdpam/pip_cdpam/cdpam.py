from __future__ import absolute_import

import numpy as np
import os
import inspect
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
from cdpam.models import FINnet


class CDPAM():
    def __init__(self, modfolder='CDPAM_trained/scratchJNDdefault_best_model.pth', dev='cuda:0'):
        
        self.device = torch.device(dev)
        encoder_layers = 16
        encoder_filters = 64
        input_size = 512
        proj_ndim = [512,256]
        ndim = [16,6]
        classif_BN = 0
        classif_act = 'no'
        proj_dp=0.1
        proj_BN=1
        classif_dp = 0.05
        
        modfolder = os.path.abspath(os.path.join(inspect.getfile(self.__init__), '..', modfolder))
        #os.path.abspath(os.path.join(inspect.getfile(self.__init__), '..', 'weights/v%s/%s.pth'%(version,net)))
        model = FINnet(dev=device,encoder_layers=encoder_layers,encoder_filters=encoder_filters,ndim=ndim, classif_dp=classif_dp,classif_BN=classif_BN,classif_act=classif_act,input_size=input_size)
        state = torch.load(modfolder,map_location="cpu")['state']
        model.load_state_dict(state)

        model.to(device)
        model.eval()
        self.model = model
    
    def forward(self, wav_in=1, wav_out=1):
         
        # input size accepted is [N x Lsize]
        if torch.is_tensor(wav_in) == False:
            audio1 = torch.from_numpy(wav_in).float().to(self.device)
            audio2 = torch.from_numpy(wav_out).float().to(self.device)
        else:
            audio1 = wav_in.float().to(self.device)
            audio2 = wav_out.float().to(self.device)
        
        _,a1,c1 = self.model.base_encoder.forward(audio1.unsqueeze(1))
        a1 = F.normalize(a1, dim=1)
        _,a2,c2 = self.model.base_encoder.forward(audio2.unsqueeze(1))
        a2 = F.normalize(a2, dim=1)
        dist1 = self.model.model_dist.forward(a1,a2)
        
        return dist1