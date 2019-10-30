import numpy as np

from tqdm import tqdm
from scipy.io import wavfile
import os, csv
import tensorflow as tf
import pickle

from helper import *

import numpy as np

import argparse

def argument_parser():
    """
    Get an argument parser for a training script.
    """
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--layers', help='number of layers in the model', default=14, type=int)
    parser.add_argument('--learning_rate', help='learning rate', default=1e-3, type=float)
    parser.add_argument('--summary_folder', help='summary folder name', default='m21')
    parser.add_argument('--optimiser', help='choose optimiser - gd/adam', default='adam')
    parser.add_argument('--loss_norm', help='loss norm - NM,SBN,None', default='SBN')
    parser.add_argument('--channels_increase', help='doubling channels after how many layers - 1,2,3,4,5,6', default=5, type=int)
    parser.add_argument('--loss_layers', help='loss to be taken for the first how many layers', default=6, type=int)
    parser.add_argument('--filter_size', help='filter size for the convolutions', default=3, type=int)
    parser.add_argument('--train_from_checkpoint', help='train_from_checkpoint', default=0, type=int)
    #loss_layers
    #channels_increase
    #(NM, SBN or None)
    #GradientDescentOptimizer
    return parser

args = argument_parser().parse_args()

print(args.layers)
print(args.learning_rate)
print(args.summary_folder)
print(args.optimiser)
print(args.loss_norm)
print(args.channels_increase)
print(args.loss_layers)

def frame(data, window_length, hop_length):
  """Convert array into a sequence of successive possibly overlapping frames.
  An n-dimensional array of shape (num_samples, ...) is converted into an
  (n+1)-D array of shape (num_frames, window_length, ...), where each frame
  starts hop_length points after the preceding one.
  This is accomplished using stride_tricks, so the original data is not
  copied.  However, there is no zero-padding, so any incomplete frames at the
  end are not included.
  Args:
    data: np.array of dimension N >= 1.
    window_length: Number of samples in each frame.
    hop_length: Advance (in samples) between each window.
  Returns:
    (N+1)-D np.array with as many rows as there are complete frames that can be
    extracted.
  """
  num_samples = data.shape[0]
  num_frames = 1 + int(np.floor((num_samples - window_length) / hop_length))
  shape = (num_frames, window_length) + data.shape[1:]
  strides = (data.strides[0] * hop_length,) + data.strides
  return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def periodic_hann(window_length):
  """Calculate a "periodic" Hann window.
  The classic Hann window is defined as a raised cosine that starts and
  ends on zero, and where every value appears twice, except the middle
  point for an odd-length window.  Matlab calls this a "symmetric" window
  and np.hanning() returns it.  However, for Fourier analysis, this
  actually represents just over one cycle of a period N-1 cosine, and
  thus is not compactly expressed on a length-N Fourier basis.  Instead,
  it's better to use a raised cosine that ends just before the final
  zero value - i.e. a complete cycle of a period-N cosine.  Matlab
  calls this a "periodic" window. This routine calculates it.
  Args:
    window_length: The number of points in the returned window.
  Returns:
    A 1D np.array containing the periodic hann window.
  """
  return 0.5 - (0.5 * np.cos(2 * np.pi / window_length *
                             np.arange(window_length)))


def stft_magnitude(signal, fft_length,
                   hop_length=None,
                   window_length=None):
  """Calculate the short-time Fourier transform magnitude.
  Args:
    signal: 1D np.array of the input time-domain signal.
    fft_length: Size of the FFT to apply.
    hop_length: Advance (in samples) between each frame passed to FFT.
    window_length: Length of each block of samples to pass to FFT.
  Returns:
    2D np.array where each row contains the magnitudes of the fft_length/2+1
    unique values of the FFT for the corresponding frame of input samples.
  """
  frames = frame(signal, window_length, hop_length)
  # Apply frame window to each frame. We use a periodic Hann (cosine of period
  # window_length) instead of the symmetric Hann of np.hanning (period
  # window_length-1).
  window = periodic_hann(window_length)
  windowed_frames = frames * window
  return np.abs(np.fft.rfft(windowed_frames, int(fft_length)))


# Mel spectrum constants and functions.
_MEL_BREAK_FREQUENCY_HERTZ = 700.0
_MEL_HIGH_FREQUENCY_Q = 1127.0


def hertz_to_mel(frequencies_hertz):
  """Convert frequencies to mel scale using HTK formula.
  Args:
    frequencies_hertz: Scalar or np.array of frequencies in hertz.
  Returns:
    Object of same size as frequencies_hertz containing corresponding values
    on the mel scale.
  """
  return _MEL_HIGH_FREQUENCY_Q * np.log(
      1.0 + (frequencies_hertz / _MEL_BREAK_FREQUENCY_HERTZ))


def spectrogram_to_mel_matrix(num_mel_bins=64,
                              num_spectrogram_bins=129,
                              audio_sample_rate=8000,
                              lower_edge_hertz=125.0,
                              upper_edge_hertz=3800.0):
  """Return a matrix that can post-multiply spectrogram rows to make mel.
  Returns a np.array matrix A that can be used to post-multiply a matrix S of
  spectrogram values (STFT magnitudes) arranged as frames x bins to generate a
  "mel spectrogram" M of frames x num_mel_bins.  M = S A.
  The classic HTK algorithm exploits the complementarity of adjacent mel bands
  to multiply each FFT bin by only one mel weight, then add it, with positive
  and negative signs, to the two adjacent mel bands to which that bin
  contributes.  Here, by expressing this operation as a matrix multiply, we go
  from num_fft multiplies per frame (plus around 2*num_fft adds) to around
  num_fft^2 multiplies and adds.  However, because these are all presumably
  accomplished in a single call to np.dot(), it's not clear which approach is
  faster in Python.  The matrix multiplication has the attraction of being more
  general and flexible, and much easier to read.
  Args:
    num_mel_bins: How many bands in the resulting mel spectrum.  This is
      the number of columns in the output matrix.
    num_spectrogram_bins: How many bins there are in the source spectrogram
      data, which is understood to be fft_size/2 + 1, i.e. the spectrogram
      only contains the nonredundant FFT bins.
    audio_sample_rate: Samples per second of the audio at the input to the
      spectrogram. We need this to figure out the actual frequencies for
      each spectrogram bin, which dictates how they are mapped into mel.
    lower_edge_hertz: Lower bound on the frequencies to be included in the mel
      spectrum.  This corresponds to the lower edge of the lowest triangular
      band.
    upper_edge_hertz: The desired top edge of the highest frequency band.
  Returns:
    An np.array with shape (num_spectrogram_bins, num_mel_bins).
  Raises:
    ValueError: if frequency edges are incorrectly ordered or out of range.
  """
  nyquist_hertz = audio_sample_rate / 2.
  if lower_edge_hertz < 0.0:
    raise ValueError("lower_edge_hertz %.1f must be >= 0" % lower_edge_hertz)
  if lower_edge_hertz >= upper_edge_hertz:
    raise ValueError("lower_edge_hertz %.1f >= upper_edge_hertz %.1f" %
                     (lower_edge_hertz, upper_edge_hertz))
  if upper_edge_hertz > nyquist_hertz:
    raise ValueError("upper_edge_hertz %.1f is greater than Nyquist %.1f" %
                     (upper_edge_hertz, nyquist_hertz))
  spectrogram_bins_hertz = np.linspace(0.0, nyquist_hertz, num_spectrogram_bins)
  spectrogram_bins_mel = hertz_to_mel(spectrogram_bins_hertz)
  # The i'th mel band (starting from i=1) has center frequency
  # band_edges_mel[i], lower edge band_edges_mel[i-1], and higher edge
  # band_edges_mel[i+1].  Thus, we need num_mel_bins + 2 values in
  # the band_edges_mel arrays.
  band_edges_mel = np.linspace(hertz_to_mel(lower_edge_hertz),
                               hertz_to_mel(upper_edge_hertz), num_mel_bins + 2)
  # Matrix to post-multiply feature arrays whose rows are num_spectrogram_bins
  # of spectrogram values.
  mel_weights_matrix = np.empty((num_spectrogram_bins, num_mel_bins))
  for i in range(num_mel_bins):
    lower_edge_mel, center_mel, upper_edge_mel = band_edges_mel[i:i + 3]
    # Calculate lower and upper slopes for every spectrogram bin.
    # Line segments are linear in the *mel* domain, not hertz.
    lower_slope = ((spectrogram_bins_mel - lower_edge_mel) /
                   (center_mel - lower_edge_mel))
    upper_slope = ((upper_edge_mel - spectrogram_bins_mel) /
                   (upper_edge_mel - center_mel))
    # .. then intersect them with each other and zero.
    mel_weights_matrix[:, i] = np.maximum(0.0, np.minimum(lower_slope,
                                                          upper_slope))
  # HTK excludes the spectrogram DC bin; make sure it always gets a zero
  # coefficient.
  mel_weights_matrix[0, :] = 0.0
  return mel_weights_matrix


def log_mel_spectrogram(data,
                        audio_sample_rate=44100,
                        log_offset=0.01,
                        window_length_secs=0.025,
                        hop_length_secs=0.010,
                        **kwargs):
  """Convert waveform to a log magnitude mel-frequency spectrogram.
  Args:
    data: 1D np.array of waveform data.
    audio_sample_rate: The sampling rate of data.
    log_offset: Add this to values when taking log to avoid -Infs.
    window_length_secs: Duration of each window to analyze.
    hop_length_secs: Advance between successive analysis windows.
    **kwargs: Additional arguments to pass to spectrogram_to_mel_matrix.
  Returns:
    2D np.array of (num_frames, num_mel_bins) consisting of log mel filterbank
    magnitudes for successive frames.
  """
  window_length_samples = int(round(audio_sample_rate * window_length_secs))
  hop_length_samples = int(round(audio_sample_rate * hop_length_secs))
  fft_length = 2 ** int(np.ceil(np.log(window_length_samples) / np.log(2.0)))
  spectrogram = stft_magnitude(
      data,
      fft_length=fft_length,
      hop_length=hop_length_samples,
      window_length=window_length_samples)
  mel_spectrogram = np.dot(spectrogram, spectrogram_to_mel_matrix(
      num_spectrogram_bins=spectrogram.shape[1],
      audio_sample_rate=audio_sample_rate, **kwargs))
  return np.log(mel_spectrogram + log_offset)

# FEATURE LOSS NETWORK
def waveform2spec_1(input,n_layers,kernel,reuse):
    
    
    layers = []

    for id in range(15,15+n_layers):
        if id == 15:
            net = slim.conv2d(input, 1, [1, kernel], activation_fn=lrelu, stride=[1, 2],
                              scope='loss_conv_%d' % id, padding='VALID', reuse=reuse)
            layers.append(net)
        else:
            net = slim.conv2d(layers[-1], 1, [1, kernel], activation_fn=lrelu,
                              stride=[1, 2], scope='loss_conv_%d' % id, padding='VALID', reuse=reuse)
            layers.append(net)
    return layers


def waveform2spec_2(input,n_layers,kernel,reuse):
    
    layers = []

    for id in range(20,20+n_layers):
        if id == 20:
            net = slim.conv2d(input, 1, [1, kernel], activation_fn=lrelu, stride=[1, 2],
                              scope='loss_conv_%d' % id, padding='VALID', reuse=reuse)
            layers.append(net)
        else:
            net = slim.conv2d(layers[-1], 1, [1, kernel], activation_fn=lrelu,
                              stride=[1, 2], scope='loss_conv_%d' % id, padding='VALID', reuse=reuse)
            layers.append(net)
    return layers


def lossnet(input, n_layers=14, training=True, reuse=False, norm_type="SBN",
               ksz=3, base_channels=32, blk_channels=5):
    
    layers = []

    if norm_type == "NM": # ADAPTIVE BATCH NORM
        norm_fn = nm
    elif norm_type == "SBN": # BATCH NORM
        norm_fn = slim.batch_norm
    else: # NO LAYER NORMALIZATION
        norm_fn = None
    
    for id in range(n_layers):
        
        #n_channels = base_channels * (2 ** (id // blk_channels)) # UPDATE CHANNEL COUNT
        
        if id ==0:
            net = slim.conv2d(input, 128, [1, ksz], activation_fn=lrelu, normalizer_fn=norm_fn,
                              scope='loss_conv_%d' % id, padding='SAME', reuse=reuse)
            layers.append(net)
        
        elif id<14:
            
            net = slim.conv2d(layers[-1], 128, [1, ksz], activation_fn=lrelu, normalizer_fn=norm_fn,
                              scope='loss_conv_%d' % id, padding='SAME', reuse=reuse)
            layers.append(net)
        
        elif id >=14 & id < n_layers-1:
            
            net = slim.conv2d(layers[-1], 128, [1, ksz], activation_fn=lrelu, normalizer_fn=norm_fn,
                              stride=[1, 2], scope='loss_conv_%d' % id, padding='SAME', reuse=reuse)
            layers.append(net)
            
        else:
            net = slim.conv2d(layers[-1], 128, [1, ksz], activation_fn=lrelu, normalizer_fn=norm_fn,
                              scope='loss_conv_%d' % id, padding='SAME', reuse=reuse)
            layers.append(net)

    return layers


def featureloss_train(target, current, loss_weights, loss_layers, n_layers=14, norm_type="SBN", base_channels=32, blk_channels=5,ksz=3):

    feat_current = lossnet(current, reuse=False, n_layers=n_layers, norm_type=norm_type,
                         base_channels=base_channels, blk_channels=blk_channels,training=True,ksz=ksz)

    feat_target = lossnet(target, reuse=True, n_layers=n_layers, norm_type=norm_type,
                         base_channels=base_channels, blk_channels=blk_channels,training=True,ksz=ksz)

    loss_vec = [0]
    #loss_overall=np.zeros((6,1))
    for id in range(loss_layers):
        loss_vec.append(l1_loss_batch(feat_current[id], feat_target[id]))
    #loss_overall[0]=loss_vec[0]
    
    for id in range(1,loss_layers+1):
        loss_vec[0] += loss_vec[id]
        #loss_overall[id]=loss_vec[id]
    return loss_vec[1:]

def load_full_data_list(datafolder='dataset'): #check change path names
    
    #sets=['train','val']
    dataset={}
    dataset['all']={}
    
    print('Loading files..')
    dataset['all']['inname'] = []
    dataset['all']['outname'] = []
    dataset['all']['label']=[]
    
    print("Prefetching the Combined")
    #data_path='prefetch_audio_new_mp3_new_morebandwidth'
    list_path='/n/fs/percepaudio/www/mturk_hosts/website_noise_combination/'
    file = open(os.path.join(datafolder,'dataset_train_combined_all_shuffled.txt'), 'r')
    for line in file:
        split_line=line.split('\t')
        dataset['all']['inname'].append("%s"%(os.path.join(list_path,split_line[0])))
        dataset['all']['outname'].append("%s"%(os.path.join(list_path,split_line[1])))
        dataset['all']['label'].append(split_line[2])
        
    print("Prefetching the Reverb")  
    list_path='/n/fs/percepaudio/www/mturk_hosts/website_MITIR_perturbation/'
    file = open(os.path.join(list_path,'dataset_train_shuffled_reverbBatch.txt'), 'r')
    for line in file:
        split_line=line.split('\t')
        dataset['all']['inname'].append("%s"%(os.path.join(list_path,split_line[0])))
        dataset['all']['outname'].append("%s"%(os.path.join(list_path,split_line[1])))
        dataset['all']['label'].append(split_line[2])
    
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
        
    return dataset
    

def split_data_equally(dataset):
    noises=['applause','blue_noise','brown_noise','crickets','pink_noise','reverb_noise','siren','violet_noise','water_drops','white_noise','mp3']
    count=np.zeros(len(noises))
    for i,noise in enumerate(noises):
        #count the number of files in each, randomise each and then put in the respective dictionary
        count[i]=len(dataset['all'][noise]['inname'])
    count_test=np.floor(0.20*count);
    count_train=np.floor(0.80*count);
    #shuffle the old dataset
    dataset_new={}
    dataset_new['train']={}
    dataset_new['val']={}
    dataset_new['test']={}
    
    #shuffle dataset for each noise type make sure that the labels are correctly there. 
    
    jobs=['train','val','test']
    for job in jobs:
        for noise in noises:
            dataset_new[job][noise]={}
            print('Loading files..')
            dataset_new[job][noise]['inname'] = []
            dataset_new[job][noise]['outname'] = []
            dataset_new[job][noise]['label']=[]
            
    for job in jobs:
        for i,noise in enumerate(noises):
            if job=='train':
                for j in range(int(count_train[i])):
                    dataset_new[job][noise]['inname'].append(dataset['all'][noise]['inname'][j])
                    dataset_new[job][noise]['outname'].append(dataset['all'][noise]['outname'][j])
                    dataset_new[job][noise]['label'].append(dataset['all'][noise]['label'][j])
                    
            elif job=='test':
                for j in range(int(count_train[i]),int(count_train[i]+count_test[i])):
                    dataset_new[job][noise]['inname'].append(dataset['all'][noise]['inname'][j])
                    dataset_new[job][noise]['outname'].append(dataset['all'][noise]['outname'][j])
                    dataset_new[job][noise]['label'].append(dataset['all'][noise]['label'][j])

    return dataset_new


def split_data_trainLinearReverb(dataset):
    noises=['applause','blue_noise','brown_noise','crickets','pink_noise','reverb_noise','siren','violet_noise','water_drops','white_noise','mp3']
    count=np.zeros(len(noises))
    for i,noise in enumerate(noises):
        #count the number of files in each, randomise each and then put in the respective dictionary
        count[i]=len(dataset['all'][noise]['inname'])
    count_valtest=np.round(0.20*count);
    count_train=np.round(0.60*count);
    #shuffle the old dataset
    dataset_new={}
    dataset_new['train']={}
    dataset_new['val']={}
    dataset_new['test']={}
    
    #shuffle dataset for each noise type make sure that the labels are correctly there. 
    
    jobs=['train','val','test']
    for job in jobs:
        for noise in noises:
            dataset_new[job][noise]={}
            print('Loading files..')
            dataset_new[job][noise]['inname'] = []
            dataset_new[job][noise]['outname'] = []
            dataset_new[job][noise]['label']=[]
            
    for job in jobs:
        for i,noise in enumerate(noises):
            if job=='train' and noise!='mp3':
                for j in range(len(dataset['all'][noise]['inname'])):
                    #if noise!='mp3' or noise!='reverb_noise';
                    dataset_new[job][noise]['inname'].append(dataset['all'][noise]['inname'][j])
                    dataset_new[job][noise]['outname'].append(dataset['all'][noise]['outname'][j])
                    dataset_new[job][noise]['label'].append(dataset['all'][noise]['label'][j])
                    
            elif job=='test' and noise=='mp3':
                for j in range(len(dataset['all'][noise]['inname'])):
                    dataset_new[job][noise]['inname'].append(dataset['all'][noise]['inname'][j])
                    dataset_new[job][noise]['outname'].append(dataset['all'][noise]['outname'][j])
                    dataset_new[job][noise]['label'].append(dataset['all'][noise]['label'][j])
            #elif job=='test':
            #    for j in range(int(count_train[i]+count_valtest[i]),int(count_train[i]+2*count_valtest[i])):
            #        dataset_new[job][noise]['inname'].append(dataset['all'][noise]['inname'][j])
            #        dataset_new[job][noise]['outname'].append(dataset['all'][noise]['outname'][j])
            #        dataset_new[job][noise]['label'].append(dataset['all'][noise]['label'][j])
                    
    return dataset_new


def combine_lists_trainLinearReverb(datasets):
    noises=['applause','blue_noise','brown_noise','crickets','pink_noise','reverb_noise','siren','violet_noise','water_drops','white_noise','mp3']
    sets=['train','val','test']
    #count=np.zeros(len(noises))
    #for i,noise in enumerate(noises):
        #count the number of files in each, randomise each and then put in the respective dictionary
    #    count[i]=len(dataset['all'][noise]['inname'])
    #count_valtest=np.round(0.20*count);
    #count_train=np.round(0.60*count);
    #shuffle the old dataset
    dataset_new={}
    dataset_new['train']={}
    dataset_new['val']={}
    dataset_new['test']={}
    
    for set in sets:
        dataset_new[set]['inname'] = []
        dataset_new[set]['outname'] = []
        dataset_new[set]['label']=[]
    
    for set in sets:
        for noise in noises:
            for j in range(len(datasets[set][noise]['inname'])):
                dataset_new[set]['inname'].append(datasets[set][noise]['inname'][j])
                dataset_new[set]['outname'].append(datasets[set][noise]['outname'][j])
                dataset_new[set]['label'].append(datasets[set][noise]['label'][j])
    
    return dataset_new


def split_test(datasets):
    noises=['applause','blue_noise','brown_noise','crickets','pink_noise','reverb_noise','siren','violet_noise','water_drops','white_noise','mp3']
    sets=['linear','reverb','mp3']
    noises_linear=['applause','blue_noise','brown_noise','crickets','pink_noise','siren','violet_noise','water_drops','white_noise']
    noises_reverb=['reverb_noise']
    noises_mp3=['mp3']
    
    dataset_new={}
    dataset_new['linear']={}
    dataset_new['reverb']={}
    dataset_new['mp3']={}
    
    for set in sets:
        dataset_new[set]['inname'] = []
        dataset_new[set]['outname'] = []
        dataset_new[set]['label']=[]
    
    for set in sets:
        if set=='linear':
            for i in noises_linear:
                for j in range(len(datasets['test'][i]['inname'])):
                    dataset_new[set]['inname'].append(datasets['test'][i]['inname'][j])
                    dataset_new[set]['outname'].append(datasets['test'][i]['outname'][j])
                    dataset_new[set]['label'].append(datasets['test'][i]['label'][j])
        elif set=='reverb':
            for i in noises_reverb:
                for j in range(len(datasets['test'][i]['inname'])):
                    dataset_new[set]['inname'].append(datasets['test'][i]['inname'][j])
                    dataset_new[set]['outname'].append(datasets['test'][i]['outname'][j])
                    dataset_new[set]['label'].append(datasets['test'][i]['label'][j])
        elif set=='mp3':
            for i in noises_mp3:
                for j in range(len(datasets['test'][i]['inname'])):
                    dataset_new[set]['inname'].append(datasets['test'][i]['inname'][j])
                    dataset_new[set]['outname'].append(datasets['test'][i]['outname'][j])
                    dataset_new[set]['label'].append(datasets['test'][i]['label'][j])
             
    return dataset_new
            
     
def load_full_data(dataset,sets,id_value):
    
    
    inputData_wav=dataset[sets]['inaudio'][id_value]
    outputData_wav=dataset[sets]['outaudio'][id_value]
    label = np.reshape(np.asarray(dataset[sets]['label'][id_value]),[-1,1])
       
    return [inputData_wav,outputData_wav,label]

    
def loadall_audio_train(dataset):
    
    dataset['train']['inaudio']  = [None]*len(dataset['train']['inname'])
    dataset['train']['outaudio'] = [None]*len(dataset['train']['outname'])
    #dataset['train']['inspec']  = [None]*len(dataset['train']['inname'])
    #dataset['train']['outspec'] = [None]*len(dataset['train']['outname'])

    for id in tqdm(range(len(dataset['train']['inname']))):

        if dataset['train']['inaudio'][id] is None:
            
            try:
                fs, inputData  = wavfile.read(dataset['train']['inname'][id])
                fs, outputData = wavfile.read(dataset['train']['outname'][id])

                shape1=np.shape(inputData)
                shape2=np.shape(outputData)

                if shape1[0]!=110250:
                        a=(np.zeros(110250-shape1[0]))
                        import random
                        a1=random.randint(0,1)
                        if a1==0:
                            inputData=np.append(a,inputData,axis=0)
                        else:
                            inputData=np.append(inputData,a,axis=0)
                            
                if shape2[0]!=110250:
                    a=(np.zeros(110250-shape2[0]))
                    import random
                    a1=random.randint(0,1)
                    if a1==0:
                        outputData=np.append(a,outputData,axis=0)
                    else:
                        outputData=np.append(outputData,a,axis=0)

                #inputData_fr=frame(inputData,int(round(0.02321*fs)),int(round(0.010*fs)))
                #outputData_fr=frame(outputData,int(round(0.02321*fs)),int(round(0.010*fs)))

                #inputData_fr = np.reshape(inputData_fr, [1, 248, 1024,1])
                #outputData_fr = np.reshape(outputData_fr, [1, 248, 1024, 1])

                #inputData_fr  = np.float32(inputData_fr)
                #outputData_fr = np.float32(outputData_fr)

                inputData_wav  = np.reshape(inputData, [-1, 1])
                outputData_wav = np.reshape(outputData, [-1, 1])

                #inputData_spec=log_mel_spectrogram(inputData,audio_sample_rate=44100)
                #outputData_spec=log_mel_spectrogram(outputData,audio_sample_rate=44100)

                #print(outputData.shape)
                #shape_spec = np.shape(inputData_spec)

                #inputData_spec = np.reshape(inputData_spec, [1, shape_spec[0], shape_spec[1],1])
                #outputData_spec = np.reshape(outputData_spec, [1, shape_spec[0], shape_spec[1], 1])

                shape_wav = np.shape(inputData_wav)

                inputData_wav = np.reshape(inputData_wav, [1, 1,shape_wav[0], shape_wav[1]])
                outputData_wav = np.reshape(outputData_wav, [1, 1,shape_wav[0], shape_wav[1]])


                #inputData_spec  = np.float32(inputData_spec)
                #outputData_spec = np.float32(outputData_spec)

                inputData_wav  = np.float32(inputData_wav)
                outputData_wav = np.float32(outputData_wav)

                dataset['train']['inaudio'][id]  = inputData_wav
                #dataset['train']['inspec'][id]  = inputData_spec
                dataset['train']['outaudio'][id] = outputData_wav
                #dataset['train']['outspec'][id] = outputData_spec
                
            except:
                print('Skip->next')
                dataset['train']['inaudio'][id]  = dataset['train']['inaudio'][id-1]
                dataset['train']['outaudio'][id] = dataset['train']['outaudio'][id-1]
                dataset['train']['label'][id] = dataset['train']['label'][id-1]
                
                

    return dataset

def load_full_data_test(dataset,sets,id_value):
    
    
    fs, inputData  = wavfile.read(dataset[sets]['inname'][id_value])

    fs, outputData = wavfile.read(dataset[sets]['outname'][id_value])

    shape1=np.shape(inputData)
    shape2=np.shape(outputData)

    if shape1[0]!=110250:
            a=(np.zeros(110250-shape1[0]))
            inputData=np.append(a,inputData,axis=0)

    if shape2[0]!=110250:
        a=(np.zeros(110250-shape2[0]))
        outputData=np.append(a,outputData,axis=0)
    
    

    inputData_wav  = np.reshape(inputData, [-1, 1])
    outputData_wav = np.reshape(outputData, [-1, 1])

   

    shape_wav = np.shape(inputData_wav)

    inputData_wav = np.reshape(inputData_wav, [1, 1,shape_wav[0], shape_wav[1]])
    outputData_wav = np.reshape(outputData_wav, [1, 1,shape_wav[0], shape_wav[1]])


    inputData_wav  = np.float32(inputData_wav)
    outputData_wav = np.float32(outputData_wav)
    
    
    return [inputData_wav,outputData_wav]



def loadall_audio_test(dataset):

    dataset['test']['inaudio']  = [None]*len(dataset['test']['inname'])
    dataset['test']['outaudio'] = [None]*len(dataset['test']['outname'])
    
    for id in tqdm(range(len(dataset['test']['inname']))):

        if dataset['test']['inaudio'][id] is None:
            
            try:
                fs, inputData  = wavfile.read(dataset['test']['inname'][id])
                fs, outputData = wavfile.read(dataset['test']['outname'][id])

                shape1=np.shape(inputData)
                shape2=np.shape(outputData)

                if shape1[0]!=110250:
                        a=(np.zeros(110250-shape1[0]))
                        inputData=np.append(a,inputData,axis=0)

                if shape2[0]!=110250:
                    a=(np.zeros(110250-shape2[0]))
                    outputData=np.append(a,outputData,axis=0)


                inputData_wav  = np.reshape(inputData, [-1, 1])
                outputData_wav = np.reshape(outputData, [-1, 1])


                shape_wav = np.shape(inputData_wav)

                inputData_wav = np.reshape(inputData_wav, [1, 1,shape_wav[0], shape_wav[1]])
                outputData_wav = np.reshape(outputData_wav, [1, 1,shape_wav[0], shape_wav[1]])


                inputData_wav  = np.float32(inputData_wav)
                outputData_wav = np.float32(outputData_wav)

                dataset['test']['inaudio'][id]  = inputData_wav
                #dataset['train']['inspec'][id]  = inputData_spec
                dataset['test']['outaudio'][id] = outputData_wav
                #dataset['train']['outspec'][id] = outputData_spec
            except:
                dataset['test']['inaudio'][id]  = dataset['test']['inaudio'][id-1]
                dataset['test']['outaudio'][id] = dataset['test']['outaudio'][id-1]
                dataset['test']['label'][id] = dataset['test']['label'][id-1]
    return dataset
    
    
def split_trainAndtest(dataset):
    
    total_examples=len(dataset['all']['inname'])
    count_valtest=np.round(0.20*total_examples);
    count_train=np.round(0.80*total_examples);
    #shuffle the old dataset
    
    import random
    
    a = dataset['all']['inname']
    b = dataset['all']['outname']
    c = dataset['all']['label']

    d = list(zip(a, b, c))
    random.seed(4)
    random.shuffle(d)

    dataset['all']['inname'], dataset['all']['outname'], dataset['all']['label'] = zip(*d)
    
    dataset_new={}
    dataset_new['train']={}
    dataset_new['test']={}
    
    #shuffle dataset for each noise type make sure that the labels are correctly there. 
    
    jobs=['train','test']
    print('Loading files..')
    
    for job in jobs:    
        dataset_new[job]['inname'] = []
        dataset_new[job]['outname'] = []
        dataset_new[job]['label']=[]
            
    for job in jobs:
            if job=='train':
                for j in range(0,int(count_train)):
                    #if noise!='mp3' or noise!='reverb_noise';
                    dataset_new[job]['inname'].append(dataset['all']['inname'][j])
                    dataset_new[job]['outname'].append(dataset['all']['outname'][j])
                    dataset_new[job]['label'].append(dataset['all']['label'][j])
                    
            elif job=='test':
                for j in range(int(count_train),int(count_train)+int(count_valtest)):
                    dataset_new[job]['inname'].append(dataset['all']['inname'][j])
                    dataset_new[job]['outname'].append(dataset['all']['outname'][j])
                    dataset_new[job]['label'].append(dataset['all']['label'][j])
    return dataset_new
    

dataset=load_full_data_list('/n/fs/percepaudio/www/mturk_hosts/website_all_perturbations/')
#split into train and testing: -> two mutuallly exclusive sets
dataset=split_trainAndtest(dataset)

dataset_train=loadall_audio_train(dataset)
dataset_test=loadall_audio_test(dataset)

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

SET_WEIGHT_EPOCH = 10 # NUMBER OF EPOCHS BEFORE FEATURE LOSS BALANCE
SAVE_EPOCHS = 10 # NUMBER OF EPOCHS BETWEEN MODEL SAVES
FILTER_SIZE = args.filter_size

import tensorflow as tf

with tf.variable_scope(tf.get_variable_scope()):
    input1_wav=tf.placeholder(tf.float32,shape=[None, None, None,1])
    
    loss_weights = tf.placeholder(tf.float32, shape=[SE_LOSS_LAYERS])
    
    clean1_wav=tf.placeholder(tf.float32,shape=[None, None, None,1])
    
    enhanced = featureloss_train(input1_wav,clean1_wav,loss_weights,loss_layers=SE_LOSS_LAYERS,n_layers=LOSS_LAYERS, norm_type=LOSS_NORM, base_channels=LOSS_BASE_CHANNELS,blk_channels=LOSS_BLK_CHANNELS,ksz=FILTER_SIZE) 
    
    enhanced1=tf.reshape(enhanced,[-1,len(enhanced)])
    weights = tf.Variable(tf.random_normal([len(enhanced),1]),
                      name="weights",trainable=True)
    weights2=tf.reshape(weights,[-1])
    weights1=tf.nn.softmax(weights2)
    weights3=tf.reshape(weights1,[-1,1])
    distance = tf.matmul(enhanced1, weights3)
    dist_sigmoid=tf.nn.sigmoid(distance)
    dense3=tf.layers.dense(distance,16,activation=tf.nn.relu)
    drop_out = tf.nn.dropout(dense3, 0.50)  # DROP-OUT here
    dense4=tf.layers.dense(drop_out,6,activation=tf.nn.relu)
    drop_out_1 = tf.nn.dropout(dense4, 0.50)  # DROP-OUT here
    dense2=tf.layers.dense(drop_out_1,2,None)
    label_task= tf.placeholder(tf.float32,shape=[None,2])
    net1 = tf.nn.softmax_cross_entropy_with_logits(labels=label_task,logits=dense2)
    loss_1=tf.reduce_mean(net1)
    if args.optimiser=='adam':
        opt_task = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss_1,var_list=[var for var in tf.trainable_variables()])
    elif args.optimiser=='gd':
        opt_task = tf.train.GradientDescentOptimizer(learning_rate=args.learning_rate).minimize(loss_1,var_list=[var for var in tf.trainable_variables()])
    

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


def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        #print('abc')
        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
    
def load_full_data_list_test(datafolder='dataset',filename='dataset_test_mp3.txt'): #check change path names

    noises=['applause','blue_noise','brown_noise','crickets','pink_noise','reverb_noise','siren','violet_noise','water_drops','white_noise','mp3']
    #sets=['train','val']
    dataset={}
    dataset['all']={}
    print('Loading files..')
    dataset['all']['inname'] = []
    dataset['all']['outname'] = []
    dataset['all']['label']=[]
    
    #print(filename)
    for noise in (noises):
        file = open(os.path.join(datafolder,filename), 'r')
        for line in file: 
            split_line=line.split('\t')
            #print(split_line)
            #print(noise)
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
    
    path='/n/fs/percepaudio/www/mturk_hosts/website_noise_combination/'
    file = open(os.path.join(datafolder,filename), 'r')
    for line in file: 
        split_line=line.split('\t')
        dataset['all']['inname'].append("%s"%(path+split_line[0]))
        dataset['all']['outname'].append("%s"%(path+split_line[1]))
        dataset['all']['label'].append(split_line[2][:-1])
    return dataset 


def scores_map(noise='mp3'):
    
    filename='dataset_test_'+noise+'.txt'
    #saver = tf.train.Saver()
    import numpy as np
    if noise!='combined':
        dataset_test=load_full_data_list_test('/n/fs/percepaudio/www/mturk_hosts/website_all_perturbations/',filename)
    elif noise=='combined':
        dataset_test=load_full_data_list_combined_test('/n/fs/percepaudio/www/mturk_hosts/website_all_perturbations/',filename)
    
    output=np.zeros((len(dataset_test["all"]["inname"]),1))
    for id in tqdm(range(0, len(dataset_test["all"]["inname"]))):
        
        loss_w = np.ones(SE_LOSS_LAYERS)
        wav_in,wav_out=load_full_data_test(dataset_test,'all',id)
        a,_= sess.run([distance,enhanced1],feed_dict={input1_wav:wav_in, clean1_wav:wav_out, loss_weights:loss_w})
        output[id]=a[0]

    import numpy as np
    perceptual=[]
    for i in range(len(dataset_test['all']['label'])):
        perceptual.append(float(dataset_test['all']['label'][i]))
    #perceptual=np.transpose(perceptual)
    perceptual=(np.array(perceptual))
    perceptual=1-perceptual
    #print(len(perceptual))

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
    #print(precs)
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


with tf.Session() as sess:
    
    epoches=2000
    sess.run(tf.global_variables_initializer())
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    outfolder = args.summary_folder
    saver = tf.train.Saver()
    
    if args.train_from_checkpoint==0:
        os.mkdir(os.path.join('summaries',outfolder))
    elif args.train_from_checkpoint==1:
        path=os.path.join('summaries',outfolder)
        saver.restore(sess, "%s/my_test_model" % path)
        print('Loaded Checkpoint')
    
    summ_writer = tf.summary.FileWriter(os.path.join('summaries',outfolder), sess.graph)
    
    for epoch in range(epoches):
        loss_epoch=[]
        
        batches=len(dataset_train['train']['inname'])
        n_batches = batches // 1
        
        for j in tqdm(range(batches)):
            wav_in,wav_out,labels=load_full_data(dataset_train,'train',j)
            y=np.zeros((labels.shape[0],2))
            for i in range(labels.shape[0]):
                if float(labels[i])==0:
                    y[i]+=[1,0]
                elif float(labels[i])==1:
                    y[i]+=[0,1]
            loss_ones=np.ones([SE_LOSS_LAYERS])
            _,dist,loss_train= sess.run([opt_task,distance,loss_1],feed_dict={input1_wav:wav_in, clean1_wav:wav_out, loss_weights:loss_ones,label_task:y})
            loss_epoch.append(loss_train)
                    
        if epoch%10==0:
            
            loss_epoch_test=[]
            
            batches=len(dataset_test['test']['inname'])
            n_batches = batches // 1
            for j in tqdm(range(batches)):
    
                wav_in,wav_out,labels=load_full_data(dataset_test,'test',j)
                y=np.zeros((labels.shape[0],2))
                for i in range(labels.shape[0]):
                    if float(labels[i])==0:
                        y[i]+=[1,0]
                    elif float(labels[i])==1:
                        y[i]+=[0,1]
                loss_ones=np.ones([SE_LOSS_LAYERS])
                dist,loss_train= sess.run([distance,loss_1],feed_dict={input1_wav:wav_in, clean1_wav:wav_out, loss_weights:loss_ones,label_task:y})
                loss_epoch_test.append(loss_train)

            [ap0,auc0]=scores_map('linear')
            [ap1,auc1]=scores_map('reverb')
            [ap2,auc2]=scores_map('mp3')
            [ap3,auc3]=scores_map('combined')

            #print(ap0)
            #print(ap1)
            #print(ap2)
            #print(ap3)
            
            summ_map_linear = sess.run(performance_summaries_map_linear, feed_dict={tf_loss_ph_map_linear:ap0})
            summ_writer.add_summary(summ_map_linear, epoch)

            summ_map_reverb = sess.run(performance_summaries_map_reverb, feed_dict={tf_loss_ph_map_reverb:ap1})
            summ_writer.add_summary(summ_map_reverb, epoch)

            summ_map_mp3 = sess.run(performance_summaries_map_mp3, feed_dict={tf_loss_ph_map_mp3:ap2})
            summ_writer.add_summary(summ_map_mp3, epoch)
            
            summ_map_combined = sess.run(performance_summaries_map_combined, feed_dict={tf_loss_ph_map_combined:ap3})
            summ_writer.add_summary(summ_map_combined, epoch)
             
            summ_test = sess.run(performance_summaries_test, feed_dict={tf_loss_ph_test:sum(loss_epoch_test) / len(loss_epoch_test)})
            summ_writer.add_summary(summ_test, epoch)

        summ = sess.run(performance_summaries_train, feed_dict={tf_loss_ph_train: sum(loss_epoch) / len(loss_epoch)})
        summ_writer.add_summary(summ, epoch)

        print("Epoch {} Train Loss {}".format(epoch,sum(loss_epoch) / len(loss_epoch)))
        if epoch%20==0:
            saver.save(sess, os.path.join('summaries',outfolder,'my_test_model'))