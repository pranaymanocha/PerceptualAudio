import os
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
import os, csv
import random
import resampy

def load_full_data_list(dummy_test): #check change path names

    #sets=['train','val']
    dataset={}
    dataset['all']={}
    
    print('Loading files..')
    dataset['all']['inname'] = []
    dataset['all']['outname'] = []
    dataset['all']['label']=[]
    
    if dummy_test==0:
    
        print("Prefetching the Combined_1,2,3")
        #data_path='prefetch_audio_new_mp3_new_morebandwidth'
        list_path='../dataset'
        file = open(os.path.join(list_path,'dataset_combined.txt'), 'r')
        for line in file:
            split_line=line.split('\t')
            dataset['all']['inname'].append("%s"%(os.path.join(list_path,split_line[0])))
            dataset['all']['outname'].append("%s"%(os.path.join(list_path,split_line[1])))
            dataset['all']['label'].append(split_line[2][:-1])

        print("Prefetching the Reverb")  
        list_path='../dataset'
        file = open(os.path.join(list_path,'dataset_reverb.txt'), 'r')
        for line in file:
            split_line=line.split('\t')
            dataset['all']['inname'].append("%s"%(os.path.join(list_path,split_line[0])))
            dataset['all']['outname'].append("%s"%(os.path.join(list_path,split_line[1])))
            dataset['all']['label'].append(split_line[2][:-1])

        print("Prefetching the Linear Noises")
        list_path='../dataset/'
        noises=['applause','blue_noise','brown_noise','crickets','pink_noise','reverb_noise','siren','violet_noise','water_drops','white_noise','mp3']
        for noise in noises:
            file = open(os.path.join(list_path,'dataset_linear.txt'), 'r')
            for line in file: 
                split_line=line.split('\t')
                if split_line[3][:-1].strip()==noise:
                    dataset['all']['inname'].append("%s_list/%s"%(list_path+noise,split_line[0]))
                    dataset['all']['outname'].append("%s_list/%s"%(list_path+noise,split_line[1]))
                    dataset['all']['label'].append(split_line[2])

        print("Prefetching the EQ")
        list_path='../dataset'
        file = open(os.path.join(list_path,'dataset_eq.txt'), 'r')
        for line in file:
            split_line=line.split('\t')
            dataset['all']['inname'].append("%s"%(os.path.join(list_path,split_line[0])))
            dataset['all']['outname'].append("%s"%(os.path.join(list_path,split_line[1])))
            dataset['all']['label'].append(split_line[2][:-1])
            
    elif dummy_test==1:
        
        print("Prefetching the Dummy Test Files")  
        list_path='../dataset'
        file = open(os.path.join(list_path,'dataset_dummy_jnd.txt'), 'r')
        for line in file:
            split_line=line.split('\t')
            dataset['all']['inname'].append("%s"%(os.path.join(list_path,split_line[0])))
            dataset['all']['outname'].append("%s"%(os.path.join(list_path,split_line[1])))
            dataset['all']['label'].append(split_line[2][:-1])
    
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
    
    #shuffle the dataset
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
    


def loadall_audio_train_waveform(dataset,resample=0):
    
    dataset['train']['inaudio']  = [None]*len(dataset['train']['inname'])
    dataset['train']['outaudio'] = [None]*len(dataset['train']['outname'])
    
    for id in tqdm(range(len(dataset['train']['inname']))):

        if dataset['train']['inaudio'][id] is None:
            
            try:
                fs, inputData  = wavfile.read(dataset['train']['inname'][id])
                fs, outputData = wavfile.read(dataset['train']['outname'][id])
                
                if resample==1:
                    
                    inputData = resampy.resample(inputData, fs, 16000)
                    outputData = resampy.resample(outputData, fs, 16000)
                    fs=16000
                
                shape1=np.shape(inputData)
                shape2=np.shape(outputData)
                
                if shape1[0]<shape2[0]:
                        a=(np.zeros(shape2[0]-shape1[0]))
                        inputData=np.append(a,inputData,axis=0)
                        
                elif shape1[0]>shape2[0]:
                        a=(np.zeros(shape1[0]-shape2[0]))
                        outputData=np.append(a,outputData,axis=0)
    
                #print('waveform')
                inputData_wav  = np.reshape(inputData, [-1, 1])
                outputData_wav = np.reshape(outputData, [-1, 1])

                shape_wav = np.shape(inputData_wav)

                inputData_wav = np.reshape(inputData_wav, [1, 1,shape_wav[0], shape_wav[1]])
                outputData_wav = np.reshape(outputData_wav, [1, 1,shape_wav[0], shape_wav[1]])

                inputData_wav  = np.float32(inputData_wav)
                outputData_wav = np.float32(outputData_wav)

                dataset['train']['inaudio'][id]  = inputData_wav
                dataset['train']['outaudio'][id] = outputData_wav

            except:
                print('Skip->next')
                print(dataset['train']['inname'][id])
                dataset['train']['inaudio'][id]  = dataset['train']['inaudio'][id-1]
                dataset['train']['outaudio'][id] = dataset['train']['outaudio'][id-1]
                dataset['train']['label'][id] = dataset['train']['label'][id-1]
    return dataset
    


def loadall_audio_test_waveform(dataset,resample=0):

    dataset['test']['inaudio']  = [None]*len(dataset['test']['inname'])
    dataset['test']['outaudio'] = [None]*len(dataset['test']['outname'])
    
    for id in tqdm(range(len(dataset['test']['inname']))):
        
        if dataset['test']['inaudio'][id] is None:
            
            try:
                fs, inputData  = wavfile.read(dataset['test']['inname'][id])
                fs, outputData = wavfile.read(dataset['test']['outname'][id])
                
                if resample==1:
                    #print('resampled')
                    inputData = resampy.resample(inputData, fs, 16000)
                    outputData = resampy.resample(outputData, fs, 16000)
                    fs=16000
                
                shape1=np.shape(inputData)
                shape2=np.shape(outputData)
                
                if shape1[0]<shape2[0]:
                        a=(np.zeros(shape2[0]-shape1[0]))
                        inputData=np.append(a,inputData,axis=0)
                        
                elif shape1[0]>shape2[0]:
                        a=(np.zeros(shape1[0]-shape2[0]))
                        outputData=np.append(a,outputData,axis=0) 
                
                inputData_wav  = np.reshape(inputData, [-1, 1])
                outputData_wav = np.reshape(outputData, [-1, 1])

                shape_wav = np.shape(inputData_wav)

                inputData_wav = np.reshape(inputData_wav, [1, 1,shape_wav[0], shape_wav[1]])
                outputData_wav = np.reshape(outputData_wav, [1, 1,shape_wav[0], shape_wav[1]])

                inputData_wav  = np.float32(inputData_wav)
                outputData_wav = np.float32(outputData_wav)

                dataset['test']['inaudio'][id]  = inputData_wav
                dataset['test']['outaudio'][id] = outputData_wav
                
            except:
                print('Skip->next')
                print(dataset['test']['inname'][id])
                dataset['test']['inaudio'][id]  = dataset['test']['inaudio'][id-1]
                dataset['test']['outaudio'][id] = dataset['test']['outaudio'][id-1]
                dataset['test']['label'][id] = dataset['test']['label'][id-1]
    return dataset    
    

def load_full_data_batch(dataset,sets,id_value):
    
    highest = []
    #calculate the longest file in the batch
    for i in range(len(id_value[0])):
        id = id_value[0][i][0]
        inputData_wav=dataset[sets]['inaudio'][id]
        inputData_wav  = np.reshape(inputData_wav, [-1, 1])
        shape=np.shape(inputData_wav)[0]
        highest.append(shape)
    
    maximum=max(highest)
    
    for i in range(len(id_value[0])):
        
        id = id_value[0][i][0]
        inputData_wav=dataset[sets]['inaudio'][id]
        inputData_wav  = np.reshape(inputData_wav, [-1])
        shape=np.shape(inputData_wav)[0]
        inputData_wav=append_ends(inputData_wav,maximum-shape)
        inputData_wav  = np.reshape(inputData_wav, [1,1,maximum,1])
        
        outputData_wav=dataset[sets]['outaudio'][id]
        outputData_wav  = np.reshape(outputData_wav, [-1])
        shape=np.shape(outputData_wav)[0]
        outputData_wav=append_ends(outputData_wav,maximum-shape)
        outputData_wav  = np.reshape(outputData_wav, [1,1,maximum,1])
        
        label = np.reshape(np.asarray(dataset[sets]['label'][id]),[-1,1])
        
        if i==0:
            waveform_in=inputData_wav
            waveform_out=outputData_wav
            labels=label
        elif i!=0:
            waveform_in=np.concatenate((waveform_in,inputData_wav),axis=0)
            waveform_out=np.concatenate((waveform_out,outputData_wav),axis=0)
            labels=np.concatenate((labels,label),axis=0)
            
    return [waveform_in,waveform_out,labels]


def append_ends(audio,amount):
    
    a=(np.zeros(amount))
    a1=random.randint(0,1)
    if a1==0:
        inputData=np.append(a,audio,axis=0)
    else:
        inputData=np.append(audio,a,axis=0)
    
    return inputData