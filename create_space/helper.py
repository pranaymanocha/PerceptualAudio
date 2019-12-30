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


def check_valid(new_levels):
    count=0 
    for i in range(len(new_levels)):
        if new_levels[i]<0 or  new_levels[i]>100:
            count=count+1
    return count


def ChooseRandomPoint(dataset='librispeech',pairs=50,min_difference=50):
    
    #linear has one param
    #reverb has 2
    #compression has 1
    #eq has 3
    #misc has 1
    
    list = np.array(['linear','reverb','compression','eq','misc'])
    random.shuffle(list)
    noise_type=np.array(["applause","blue_noise","crickets","pink_noise","violet_noise","water_drops","white_noise"])
    noise_type_compression=np.array(["mp3","mulaw"])
    noise_type_misc=np.array(["pops","griffinlin"])
    
    reverblist=sorted(os.listdir('/n/fs/percepaudio/www/mturk_hosts/RIR_MIT/Audio'))
    if dataset=='librispeech':
        speaker=np.array(sorted(os.listdir('/n/fs/percepaudio/librispeech/LibriSpeech/resized_files_2')))
    elif dataset=='daps':
        speaker=np.array(['m1','m2','m3','m4','m5','m6','m7','m8','m9','m10','f1','f2','f3','f4','f5','f6','f7','f8','f9','f10'])
    overall_list=[]
    b = np.linspace(0,1,64)
    
    for j in range(pairs):
        
        list = [1,0]
        sampling = random.choices(list, k=5)
        rand_speaker=random.choice(speaker)
        rand_reverb=random.choice(reverblist)
        space=sampling.count(1)
        
        while(space<1):
            list = [1,0]
            sampling = random.choices(list, k=5)
            rand_speaker=random.choice(speaker)
            rand_reverb=random.choice(reverblist)
            space=sampling.count(1)
            
        
        if sampling[1]!=0 and sampling[3]!=0:
            
            init=np.random.randint(0,101,size=space+3)
            s = np.random.normal(0, 1, space+3)
            decrease_max_per_hit=np.repeat(min_difference,space+3)
            
        elif sampling[1]!=0 and sampling[3]==0:
            
            init=np.random.randint(0,101,size=space+1)
            s = np.random.normal(0, 1, space+1)
            decrease_max_per_hit=np.repeat(min_difference,space+1)
        
        elif sampling[1]==0 and sampling[3]!=0:
            
            init=np.random.randint(0,101,size=space+2)
            s = np.random.normal(0, 1, space+2)
            decrease_max_per_hit=np.repeat(min_difference,space+2)
    
        else:
            
            init=np.random.randint(0,101,size=space)
            s = np.random.normal(0, 1, space)
            decrease_max_per_hit=np.repeat(min_difference,space)
        
        x_normalized=s/np.linalg.norm(s)
        
        x_normalized_equal1 = x_normalized / np.min(np.abs(x_normalized))
        
        a = np.abs(decrease_max_per_hit / x_normalized_equal1)
        min_steps = np.min(a)
        c = min_steps * x_normalized_equal1
        new_levels=init+c
        
        count = check_valid(new_levels)
        
        while (count!=0):
                
            list = [1,0]
            sampling = random.choices(list, k=5)
            rand_speaker=random.choice(speaker)
            rand_reverb=random.choice(reverblist)
            space=sampling.count(1)

            while(space<1):
                list = [1,0]
                sampling = random.choices(list, k=5)
                rand_speaker=random.choice(speaker)
                rand_reverb=random.choice(reverblist)
                space=sampling.count(1)


            if sampling[1]!=0 and sampling[3]!=0:

                init=np.random.randint(0,101,size=space+3)
                s = np.random.normal(0, 1, space+3)
                decrease_max_per_hit=np.repeat(min_difference,space+3)

            elif sampling[1]!=0 and sampling[3]==0:

                init=np.random.randint(0,101,size=space+1)
                s = np.random.normal(0, 1, space+1)
                decrease_max_per_hit=np.repeat(min_difference,space+1)

            elif sampling[1]==0 and sampling[3]!=0:

                init=np.random.randint(0,101,size=space+2)
                s = np.random.normal(0, 1, space+2)
                decrease_max_per_hit=np.repeat(min_difference,space+2)

            else:

                init=np.random.randint(0,101,size=space)
                s = np.random.normal(0, 1, space)
                decrease_max_per_hit=np.repeat(min_difference,space)

            x_normalized=s/np.linalg.norm(s)

            x_normalized_equal1 = x_normalized / np.min(np.abs(x_normalized))

            a = np.abs(decrease_max_per_hit / x_normalized_equal1)
            min_steps = np.min(a)
            c = min_steps * x_normalized_equal1
            new_levels=init+c

            count = check_valid(new_levels)
        
        pra = np.linalg.norm(c)
        
        
        for i in range(64):
            
            if dataset=='librispeech':
                b0=os.listdir('/n/fs/percepaudio/librispeech/LibriSpeech/resized_files_2/'+rand_speaker)
            elif dataset=='daps':
                b0=os.listdir('/n/fs/percepaudio/www/mturk_hosts/website_perturbation/snr_resize_folder/'+rand_speaker)
            
            b1=np.random.randint(0,len(b0))
            b2=np.round(init+b[i]*pra*x_normalized,2)
            
            save_file=''
            count=0
            
            if sampling[1]!=0 and sampling[3]!=0:
                length=5
                for k in range(length):
                    if k==0:
                        if sampling[k]==1:
                            save_file+=str(b2[count])
                            count+=1
                        else:
                            save_file+=str(0.0)
                    elif k==1:
                        save_file+='_'+str(b2[count])
                        count+=1
                        save_file+='_'+str(b2[count])
                        count+=1
                    elif k==2:
                        if sampling[k]==1:
                            save_file+='_'+str(b2[count])
                            count+=1
                        else:
                            save_file+='_'+str(0.0)
                    elif k==3:
                        save_file+='_'+str(b2[count])
                        count+=1
                        save_file+='_'+str(b2[count])
                        count+=1
                        save_file+='_'+str(b2[count])
                        count+=1
                    elif k==4:
                        if sampling[k]==1:
                            save_file+='_'+str(b2[count])
                            count+=1
                        else:
                            save_file+='_'+str(0.0)
                        
                        
            elif sampling[1]!=0 and sampling[3]==0:
                length=5
                for k in range(length):
                    if k==0:
                        if sampling[k]==1:
                            save_file+=str(b2[count])
                            count+=1
                        else:
                            save_file+=str(0.0)
                    elif k==1:
                        save_file+='_'+str(b2[count])
                        count+=1
                        save_file+='_'+str(b2[count])
                        count+=1
                    elif k==2:
                        if sampling[k]==1:
                            save_file+='_'+str(b2[count])
                            count+=1
                        else:
                            save_file+='_'+str(0.0)
                    elif k==3:
                        save_file+='_'+str(0.0)
                        save_file+='_'+str(0.0)
                        save_file+='_'+str(0.0)
                    elif k==4:
                        if sampling[k]==1:
                            save_file+='_'+str(b2[count])
                            count+=1
                        else:
                            save_file+='_'+str(0.0)
                        
            elif sampling[1]==0 and sampling[3]!=0:
                length=5
                for k in range(length):
                    if k==0:
                        if sampling[k]==1:
                            save_file+=str(b2[count])
                            count+=1
                        else:
                            save_file+=str(0.0)
                            
                    elif k==1:
                        save_file+='_'+str(0.0)
                        save_file+='_'+str(0.0)
                        
                    elif k==2:
                        if sampling[k]==1:
                            save_file+='_'+str(b2[count])
                            count+=1
                        else:
                            save_file+='_'+str(0.0)
                    elif k==3:
                        save_file+='_'+str(b2[count])
                        count+=1
                        save_file+='_'+str(b2[count])
                        count+=1
                        save_file+='_'+str(b2[count])
                        count+=1
                    elif k==4:
                        if sampling[k]==1:
                            save_file+='_'+str(b2[count])
                            count+=1
                        else:
                            save_file+='_'+str(0.0)
                        
            else:
                length=5
                for k in range(length):
                    if k==0:
                        if sampling[k]==1:
                            save_file+=str(b2[count])
                            count+=1
                        else:
                            save_file+=str(0.0)
                            
                    elif k==1:
                        save_file+='_'+str(0.0)
                        save_file+='_'+str(0.0)
                        
                    elif k==2:
                        if sampling[k]==1:
                            save_file+='_'+str(b2[count])
                            count+=1
                        else:
                            save_file+='_'+str(0.0)
                    elif k==3:
                        save_file+='_'+str(0.0)
                        save_file+='_'+str(0.0)
                        save_file+='_'+str(0.0)
                    elif k==4:
                        if sampling[k]==1:
                            save_file+='_'+str(b2[count])
                            count+=1
                        else:
                            save_file+='_'+str(0.0)
            
            overall_list.append(save_file+'_'+b0[b1][:-4]+'_'+str(rand_reverb))
            print(save_file+'_'+b0[b1][:-4]+'_'+str(rand_reverb))
    return overall_list

def SplitList(overall_list):
    normalised_list=[]
    for i in range(0,len(overall_list),64):
        partition_list=[]
        for j in range(0,64):
            if j==0:
                partition_list=overall_list[i]
            else:
                partition_list=partition_list+','+overall_list[i+j]
        normalised_list.append(partition_list)
    return normalised_list

def CreatePHPFile(normalised_list,saved_file):
    #import numpy
    file = open(saved_file, 'w+')
    file.write("<?php \n")
    file.write("$eq = array( \n")
    a1=np.shape(normalised_list)[0]
    for i in range(np.shape(normalised_list)[0]):
        file.write("array(")
        split_string=normalised_list[i].split(',')
        a='"'+split_string[0]+'"'
        for j in range(1,len(split_string)):
            a=a+',"'+split_string[j]+'"'
        if i==(a1-1):
            a=a+')'+' \n'
        else:
            a=a+'),'+' \n'
        file.write(a)
    file.write(");")
    file.close()

    
def CreatePHPFile_finetune(normalised_list,saved_file):
    
    file = open(saved_file,'w+')
    file.write("<?php \n")
    file.write("$eq = array( \n")
    a1=np.shape(normalised_list)[0]
    
    for i in range(np.shape(normalised_list)[0]):
        file.write("array(")
        split_string=normalised_list[i]
        a='"'+split_string[0]+'"'
        for j in range(1,len(split_string)):
            a=a+',"'+split_string[j]+'"'
        if i==(a1-1):
            a=a+')'+' \n'
        else:
            a=a+'),'+' \n'
        file.write(a)
    file.write(");")
    file.close()
    
    
def snr_value(audio,audio_noise):
    audio_mag=np.sum(np.square(audio))
    audio_noise_mag=np.sum(np.square(audio_noise))
    SNR=10*math.log10(audio_mag/audio_noise_mag)
    return SNR

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

def applause(audio,level):

    audio_noise,sr_noise=sf.read('/n/fs/percepaudio/ESC/applause/applause-1.wav')
    audio_noise_shape=audio_noise[44100:44100+np.shape(audio)[0]]
    starting_w=0.0006
    ending_w = starting_w*800.0
    levels = 100.0
    F = (ending_w / starting_w)
    expon = 1 / (levels)
    F1 = F ** (expon)
    jk=level
    noise_level = starting_w * (F1 ** (jk))
    noise = noise_level * audio_noise_shape
    conv = noise + audio
    return conv
    
def blue_noise(audio,level):
    
    audio_noise,sr_noise= sf.read('/n/fs/percepaudio/ESC/blue_noise/audiocheck.net_bluenoise.wav')
    audio_noise_shape=audio_noise[44100:44100+np.shape(audio)[0]]
    starting_w=0.0006
    ending_w = starting_w*800.0
    levels = 100.0
    F = (ending_w / starting_w)
    expon = 1 / (levels)
    F1 = F ** (expon)
    jk=level
    noise_level = starting_w * (F1 ** (jk))
    noise = noise_level * audio_noise_shape
    conv = noise + audio
    return conv
    
def brown_noise(audio,level):
    
    audio_noise,sr_noise= sf.read('/n/fs/percepaudio/ESC/brown_noise/audiocheck.net_brownnoise.wav')
    audio_noise_shape=audio_noise[44100:44100+np.shape(audio)[0]]
    starting_w=0.0006
    ending_w = starting_w*800.0
    levels = 100.0
    F = (ending_w / starting_w)
    expon = 1 / (levels)
    F1 = F ** (expon)
    jk=level
    noise_level = starting_w * (F1 ** (jk))
    noise = noise_level * audio_noise_shape
    conv = noise + audio
    return conv
            
def crickets(audio,level):
    
    audio_noise,sr_noise= sf.read('/n/fs/percepaudio/ESC/crickets/5-216214-A-13.wav')
    audio_noise_shape=audio_noise[44100:44100+np.shape(audio)[0]]
    starting_w=0.0006
    ending_w = starting_w*800.0
    levels = 100.0
    F = (ending_w / starting_w)
    expon = 1 / (levels)
    F1 = F ** (expon)
    jk=level
    noise_level = starting_w * (F1 ** (jk))
    noise = noise_level * audio_noise_shape
    conv = noise + audio
    return conv
        
def pink_noise(audio,level):

    audio_noise,sr_noise= sf.read('/n/fs/percepaudio/ESC/pink_noise/audiocheck.net_pinknoise.wav')
    audio_noise_shape=audio_noise[44100:44100+np.shape(audio)[0]]
    starting_w=0.0006
    ending_w = starting_w*800.0
    levels = 100.0
    F = (ending_w / starting_w)
    expon = 1 / (levels)
    F1 = F ** (expon)
    jk=level
    noise_level = starting_w * (F1 ** (jk))
    noise = noise_level * audio_noise_shape
    conv = noise + audio
    return conv
    
def siren(audio,level):
    
    audio_noise,sr_noise= sf.read('/n/fs/percepaudio/ESC/siren/5-160551-A-42.wav')
    audio_noise_shape=audio_noise[44100:44100+np.shape(audio)[0]]
    starting_w=0.0006
    ending_w = starting_w*800.0
    levels = 100.0
    F = (ending_w / starting_w)
    expon = 1 / (levels)
    F1 = F ** (expon)
    jk=level
    noise_level = starting_w * (F1 ** (jk))
    noise = noise_level * audio_noise_shape
    conv = noise + audio
    return conv
       
def violet_noise(audio,level):
    
    audio_noise,sr_noise= sf.read('/n/fs/percepaudio/ESC/violet_noise/audiocheck.net_violetnoise.wav')
    audio_noise_shape=audio_noise[44100:44100+np.shape(audio)[0]]
    starting_w=0.0006
    ending_w = starting_w*800.0
    levels = 100.0
    F = (ending_w / starting_w)
    expon = 1 / (levels)
    F1 = F ** (expon)
    jk=level
    noise_level = starting_w * (F1 ** (jk))
    noise = noise_level * audio_noise_shape
    conv = noise + audio
    return conv
    
def water_drops(audio,level):
    
    audio_noise,sr_noise= sf.read('/n/fs/percepaudio/ESC/water_drops/4-212604-B-15.wav')
    audio_noise_shape=audio_noise[44100:44100+np.shape(audio)[0]]
    starting_w=0.0006
    ending_w = starting_w*800.0
    levels = 100.0
    F = (ending_w / starting_w)
    expon = 1 / (levels)
    F1 = F ** (expon)
    jk=level
    noise_level = starting_w * (F1 ** (jk))
    noise = noise_level * audio_noise_shape
    conv = noise + audio
    return conv
'''
def pitch(level,audio,fs,args):
    
    starting_w=0.05
    ending_w = 5.0
    levels = 100.0
    noise_level = starting_w + level *((ending_w-starting_w)/100)
    noise_level=round(noise_level,4)
    audio_new = librosa.effects.pitch_shift(audio, fs, n_steps=noise_level)
    return audio_new
'''
def griffin_lim(level,audio,args):
    
    starting_w=500
    ending_w = 1
    levels = 100.0
    noise_level = starting_w + level *((ending_w-starting_w)/100)
    noise_level=round(noise_level,4)
    S = np.abs(librosa.stft(audio))
    y_inv = librosa.griffinlim(S,n_iter=int(noise_level))
    return y_inv

def pops(level,audio,args):
    
    
    starting_w=0.0001
    ending_w = 0.31
    levels = 100.0
    noise_level = starting_w + level *((ending_w-starting_w)/100)
    noise_level=round(noise_level,4)
    p=noise_level
    length=p*len(audio)
    maxi=np.max(audio)
    mini=np.min(audio)

    A=np.random.choice(int(len(audio)), int(length), replace=False)
    A1=np.random.choice(A,int(len(A)/2),replace=False)
    A2=np.setdiff1d(A, A1)
    audio[A1]=maxi
    audio[A2]=mini
    return audio

def compression_mp3(noise_l,audio,args):
    
    #path='/n/fs/percepaudio/www/mturk_hosts/website_noise_combination/prefetch_audio_new_mp3_testing/'
    starting_w=300.0
    ending_w = 1.0
    levels = 100.0
    F = (ending_w / starting_w)
    expon = 1 / (levels)
    F1 = F ** (expon)
    jk=noise_l
    noise_level = starting_w * (F1 ** (jk))
    noise_level=round(noise_level,4)
    sf.write(os.path.join(args.folder,'3.wav'),audio,44100)
    
    bashCommand = "ffmpeg -i "+args.folder+"/3.wav"+" -vn -ar 44100 -b:a " + str(noise_level)+"k "+args.folder+"/3.mp3"
    print(bashCommand)
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    bashCommand_reverse = "ffmpeg -i "+args.folder+"/3.mp3 "+args.folder+"/4.wav"
    print(bashCommand_reverse)
    process = subprocess.Popen(bashCommand_reverse.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    
    audio,sr=sf.read(args.folder+"/4.wav")
    os.remove(args.folder+"/3.mp3")
    os.remove(args.folder+"/3.wav")
    os.remove(args.folder+"/4.wav")
    
    return audio

def audio_make(audio,list,linear_noise_type,noise_compression,noise_misc,noise_list,audio_ir,fs,args):
    
    noise1=noise_list[0]
    noise2=noise_list[1]
    noise3=noise_list[2]
    noise4=noise_list[3]
    
    noise5=noise_list[4]
    noise6=noise_list[5]
    noise7=noise_list[6]
    noise8=noise_list[7]
    
    noise_final=audio
    
    for noise_list in list:
        if noise_list=='linear':
            print('linear')
            if noise1==0.0:
                z=1
            else:
                noise_function=str(linear_noise_type)+'(noise_final,float(noise1))'
                noise_final=eval(noise_function)
                
        elif noise_list=='reverb':
            print('reverb')
            if noise2==0.0:
                z=1
            else:
                audio_drr=drr_calculation(float(noise2),noise_final,audio_ir,fs)
                noise_final=rt60_calculation(float(noise3),noise_final,audio_drr,fs)
                
        elif noise_list=='compression':
            print('compression')
            if noise4==0.0:
                z=1
            else:
                if noise_compression=='mulaw':
                    noise_final=mu_law_selection(float(noise4),noise_final)
                elif noise_compression=='mp3':
                    noise_final=compression_mp3(float(noise4),noise_final,args)
                    
        elif noise_list=='eq':
            print('eq')
            if noise5==0.0:
                z=1
            else:
                eq_law=np.array([float(noise5),float(noise6),float(noise7)])
                noise_final=EQ_create(noise_final,eq_law,fs,args)
            
        elif noise_list=='misc':
            print('misc')
            if noise8==0.0:
                z=1
            else:
                if noise_misc=='pops':
                    noise_final=pops(float(noise8),noise_final,args)
                elif noise_misc=='griffinlin':
                    noise_final=griffin_lim(float(noise8),noise_final,args)
                #elif noise_misc=='pitch':
                    #level,audio,fs,args
                #    noise_final=pitch(float(noise8),noise_final,fs,args)
             
    return noise_final


def mu_law_selection(noise_l,audio):
    starting_w=1.0
    ending_w = 60.0
    levels = 100.0
    F = (ending_w / starting_w)
    expon = 1 / (levels)
    F1 = F ** (expon)
    jk=noise_l
    noise_level = starting_w * (F1 ** (jk))
    audio_final=mu_law(audio,pow(2,noise_level)-1)
    return audio_final

def mu_law(audio,mu):
    audio_new=np.zeros(audio.shape)
    for i in range(audio.shape[0]):
        x=audio[i]
       
        y=(np.sign(x)*(np.log((1+mu*np.absolute(x)))/np.log(1+mu)))
        #audio_new_test[i]=y
        audio_new[i]=np.round(((y+1)/2)*255)/255
    return audio_new


def drr_calculation(noise_l,audio,audio_ir,sample_rate): 
    starting_w = 1.0   
    ending_w = 100.0
    levels = 100.0
    F = (ending_w / starting_w)
    expon = 1 / (levels)
    F1 = F ** (expon)
    jk=noise_l
    noise_level = starting_w * (F1 ** (jk))
    y_max=np.max(audio_ir)
    x_max=np.argmax(audio_ir)
    audio_new=audio_ir[0:x_max+56]
    audio_scaled=numpy.multiply(audio_ir[x_max+56:],1/noise_level)
    audio_final=np.concatenate((audio_new,audio_scaled), axis=0)
    #conv=np.convolve(audio,audio_final,'same')
    return audio_final

#level 0 - 100
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


def EQ_create(audio,a_levels,fs,args):
    
    
    freq_1=float(args.EQ_bands.split('_')[0])
    freq_2=float(args.EQ_bands.split('_')[1])
    
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
    
    starting_w=0.1
    ending_w = 2
    levels = 100.0
    F = (ending_w / starting_w)
    expon = 1 / (levels)
    F1 = F ** (expon)
    jk=a_levels
    noise_level = starting_w * (F1 ** (jk))
    
    sum=0
    for i in range((noise_level.shape[0])):
        sum+=x_level[i]*noise_level[i]
    return sum


def librispeech(args,path,path_new):
    
    count=0
    with open(args.saved_file) as csv_file:
        csv_reader = csv.reader(csv_file,delimiter=',')
        for row in tqdm.tqdm(csv_reader):
            count+=1
            list = np.array(['linear','reverb','compression','eq','misc'])
            noise_type=np.array(["applause","blue_noise","crickets","pink_noise","violet_noise","water_drops","white_noise"])
            random.shuffle(list)
            noise_type_compression=np.array(["mp3","mulaw"])
            noise_type_misc=np.array(["pops","griffinlin"])

            noise=random.choice(noise_type)
            noise_compression=random.choice(noise_type_compression)
            noise_misc=random.choice(noise_type_misc)
            
            if count>=3 and count<=args.numberofpairs+1:
                a=(row[0][7:-1]).split('_')
                noise1_orig=a[0]
                noise2_orig=a[1]
                noise3_orig=a[2]
                noise4_orig=a[3]
                noise5_orig=a[4]
                noise6_orig=a[5]
                noise7_orig=a[6]
                noise8_orig=a[7]

                speaker=a[8].split('-')[0]
                audio_file_ir="_".join(a[11:])
                audio_ir, sample_rate_ir = sf.read(os.path.join('/n/fs/percepaudio/www/mturk_hosts/RIR_MIT/Audio/',audio_file_ir))

                for i in range(len(row)-1):

                    if i==0:
                        a=(row[0][7:-1]).split('_')
                        noise1=a[0]
                        noise2=a[1]
                        noise3=a[2]
                        noise4=a[3]
                        noise5=a[4]
                        noise6=a[5]
                        noise7=a[6]
                        noise8=a[7]

                        speaker=a[8].split('-')[0]
                        filename="_".join(a[8:11])
                        filename_full_ir="_".join(a[8:])
                        
                        audio,sr=sf.read(os.path.join(path,speaker,filename+'.wav'))

                        noise_list=[noise1,noise2,noise3,noise4,noise5,noise6,noise7,noise8]
                        audio_final = audio_make(audio,list,noise,noise_compression,noise_misc,noise_list,audio_ir,sr,args)
                        new_filename=str(noise1)+'_'+str(noise2)+'_'+str(noise3)+'_'+str(noise4)+'_'+str(noise5)+'_'+str(noise6)+'_'+str(noise7)+'_'+str(noise8)+'_'+filename_full_ir
                        sf.write(os.path.join(path_new,new_filename),audio_final,sr)
                        
                        subprocess.call(['ffmpeg', '-i', os.path.join(path_new,new_filename),'-b:a','192000',os.path.join(path_new,new_filename)[:-3]+'mp3'])
                        os.remove(os.path.join(path_new,new_filename))

                    elif i==len(row)-2:

                        a=(row[i][:-1]).split('_')
                        noise1=a[0]
                        noise2=a[1]
                        noise3=a[2]
                        noise4=a[3]
                        noise5=a[4]
                        noise6=a[5]
                        noise7=a[6]
                        noise8=a[7]

                        speaker=a[8].split('-')[0]
                        filename="_".join(a[8:11])
                        filename_full_ir="_".join(a[8:])

                        audio,sr=sf.read(os.path.join(path,speaker,filename+'.wav'))

                        noise_list=[noise1,noise2,noise3,noise4,noise5,noise6,noise7,noise8]
                        audio_final = audio_make(audio,list,noise,noise_compression,noise_misc,noise_list,audio_ir,sr,args)
                        new_filename=str(noise1)+'_'+str(noise2)+'_'+str(noise3)+'_'+str(noise4)+'_'+str(noise5)+'_'+str(noise6)+'_'+str(noise7)+'_'+str(noise8)+'_'+filename_full_ir
                        sf.write(os.path.join(path_new,new_filename),audio_final,sr)
                        subprocess.call(['ffmpeg', '-i', os.path.join(path_new,new_filename),'-b:a','192000',os.path.join(path_new,new_filename)[:-3]+'mp3'])
                        os.remove(os.path.join(path_new,new_filename))
                        
                        if args.job!=1:
                        
                            noise_list=[noise1_orig,noise2_orig,noise3_orig,noise4_orig,noise5_orig,noise6_orig,noise7_orig,noise8_orig]
                            audio_final = audio_make(audio,list,noise,noise_compression,noise_misc,noise_list,audio_ir,sr,args)

                            new_filename=str(noise1_orig)+'_'+str(noise2_orig)+'_'+str(noise3_orig)+'_'+str(noise4_orig)+'_'+str(noise5_orig)+'_'+str(noise6_orig)+'_'+str(noise7_orig)+'_'+str(noise8_orig)+'_'+filename_full_ir

                            sf.write(os.path.join(path_new,new_filename),audio_final,sr)
                            subprocess.call(['ffmpeg', '-i', os.path.join(path_new,new_filename),'-b:a','192000',os.path.join(path_new,new_filename)[:-3]+'mp3'])
                            os.remove(os.path.join(path_new,new_filename))
                        
                        
                    else:
                        a=(row[i]).split('_')
                        noise1=a[0]
                        noise2=a[1]
                        noise3=a[2]
                        noise4=a[3]
                        noise5=a[4]
                        noise6=a[5]
                        noise7=a[6]
                        noise8=a[7]

                        speaker=a[8].split('-')[0]
                        filename="_".join(a[8:11])
                        filename_full_ir="_".join(a[8:])

                        audio,sr=sf.read(os.path.join(path,speaker,filename+'.wav'))

                        noise_list=[noise1,noise2,noise3,noise4,noise5,noise6,noise7,noise8]
                        audio_final = audio_make(audio,list,noise,noise_compression,noise_misc,noise_list,audio_ir,sr,args)

                        new_filename=str(noise1)+'_'+str(noise2)+'_'+str(noise3)+'_'+str(noise4)+'_'+str(noise5)+'_'+str(noise6)+'_'+str(noise7)+'_'+str(noise8)+'_'+filename_full_ir
                        sf.write(os.path.join(path_new,new_filename),audio_final,sr)
                        subprocess.call(['ffmpeg', '-i', os.path.join(path_new,new_filename),'-b:a','192000',os.path.join(path_new,new_filename)[:-3]+'mp3'])
                        os.remove(os.path.join(path_new,new_filename))
                        
                        if args.job!=1:
                            noise_list=[noise1_orig,noise2_orig,noise3_orig,noise4_orig,noise5_orig,noise6_orig,noise7_orig,noise8_orig]
                            audio_final = audio_make(audio,list,noise,noise_compression,noise_misc,noise_list,audio_ir,sr,args)

                            new_filename=str(noise1_orig)+'_'+str(noise2_orig)+'_'+str(noise3_orig)+'_'+str(noise4_orig)+'_'+str(noise5_orig)+'_'+str(noise6_orig)+'_'+str(noise7_orig)+'_'+str(noise8_orig)+'_'+filename_full_ir

                            sf.write(os.path.join(path_new,new_filename),audio_final,sr)
                            subprocess.call(['ffmpeg', '-i', os.path.join(path_new,new_filename),'-b:a','192000',os.path.join(path_new,new_filename)[:-3]+'mp3'])
                            os.remove(os.path.join(path_new,new_filename))
                        
            elif count==args.numberofpairs+2:
                a=(row[0][7:-1]).split('_')
                noise1_orig=a[0]
                noise2_orig=a[1]
                noise3_orig=a[2]
                noise4_orig=a[3]
                noise5_orig=a[4]
                noise6_orig=a[5]
                noise7_orig=a[6]
                noise8_orig=a[7]

                speaker=a[8].split('-')[0]
                audio_file_ir="_".join(a[11:])
                audio_ir, sample_rate_ir = sf.read(os.path.join('/n/fs/percepaudio/www/mturk_hosts/RIR_MIT/Audio/',audio_file_ir))

                for i in (range(0,len(row))):
                    if i==len(row)-1:
                        a=(row[i][:-2]).split('_')
                        noise1=a[0]
                        noise2=a[1]
                        noise3=a[2]
                        noise4=a[3]
                        noise5=a[4]
                        noise6=a[5]
                        noise7=a[6]
                        noise8=a[7]

                        speaker=a[8].split('-')[0]
                        filename="_".join(a[8:11])
                        filename_full_ir="_".join(a[8:])

                        audio,sr=sf.read(os.path.join(path,speaker,filename+'.wav'))

                        noise_list=[noise1,noise2,noise3,noise4,noise5,noise6,noise7,noise8]
                        audio_final = audio_make(audio,list,noise,noise_compression,noise_misc,noise_list,audio_ir,sr,args)
                        new_filename=str(noise1)+'_'+str(noise2)+'_'+str(noise3)+'_'+str(noise4)+'_'+str(noise5)+'_'+str(noise6)+'_'+str(noise7)+'_'+str(noise8)+'_'+filename_full_ir
                        sf.write(os.path.join(path_new,new_filename),audio_final,sr)
                        subprocess.call(['ffmpeg', '-i', os.path.join(path_new,new_filename),'-b:a','192000',os.path.join(path_new,new_filename)[:-3]+'mp3'])
                        os.remove(os.path.join(path_new,new_filename))
                        
                        if args.job!=1:
                        
                            noise_list=[noise1_orig,noise2_orig,noise3_orig,noise4_orig,noise5_orig,noise6_orig,noise7_orig,noise8_orig]
                            audio_final = audio_make(audio,list,noise,noise_compression,noise_misc,noise_list,audio_ir,sr,args)

                            new_filename=str(noise1_orig)+'_'+str(noise2_orig)+'_'+str(noise3_orig)+'_'+str(noise4_orig)+'_'+str(noise5_orig)+'_'+str(noise6_orig)+'_'+str(noise7_orig)+'_'+str(noise8_orig)+'_'+filename_full_ir
                            sf.write(os.path.join(path_new,new_filename),audio_final,sr)
                            subprocess.call(['ffmpeg', '-i', os.path.join(path_new,new_filename),'-b:a','192000',os.path.join(path_new,new_filename)[:-3]+'mp3'])
                            os.remove(os.path.join(path_new,new_filename))
                    elif i==0:
                        a=(row[i][7:-1]).split('_')
                        noise1=a[0]
                        noise2=a[1]
                        noise3=a[2]
                        noise4=a[3]
                        noise5=a[4]
                        noise6=a[5]
                        noise7=a[6]
                        noise8=a[7]

                        speaker=a[8].split('-')[0]
                        filename="_".join(a[8:11])
                        filename_full_ir="_".join(a[8:])
                        audio,sr=sf.read(os.path.join(path,speaker,filename+'.wav'))

                        noise_list=[noise1,noise2,noise3,noise4,noise5,noise6,noise7,noise8]
                        audio_final = audio_make(audio,list,noise,noise_compression,noise_misc,noise_list,audio_ir,sr,args)
                        new_filename=str(noise1)+'_'+str(noise2)+'_'+str(noise3)+'_'+str(noise4)+'_'+str(noise5)+'_'+str(noise6)+'_'+str(noise7)+'_'+str(noise8)+'_'+filename_full_ir
                        sf.write(os.path.join(path_new,new_filename),audio_final,sr)
                        subprocess.call(['ffmpeg', '-i', os.path.join(path_new,new_filename),'-b:a','192000',os.path.join(path_new,new_filename)[:-3]+'mp3'])
                        os.remove(os.path.join(path_new,new_filename))
                    else:
                        a=(row[i]).split('_')
                        noise1=a[0]
                        noise2=a[1]
                        noise3=a[2]
                        noise4=a[3]
                        noise5=a[4]
                        noise6=a[5]
                        noise7=a[6]
                        noise8=a[7]
                        
                        speaker=a[8].split('-')[0]
                        filename="_".join(a[8:11])
                        filename_full_ir="_".join(a[8:])

                        audio,sr=sf.read(os.path.join(path,speaker,filename+'.wav'))

                        noise_list=[noise1,noise2,noise3,noise4,noise5,noise6,noise7,noise8]
                        audio_final = audio_make(audio,list,noise,noise_compression,noise_misc,noise_list,audio_ir,sr,args)

                        new_filename=str(noise1)+'_'+str(noise2)+'_'+str(noise3)+'_'+str(noise4)+'_'+str(noise5)+'_'+str(noise6)+'_'+str(noise7)+'_'+str(noise8)+'_'+filename_full_ir
                        sf.write(os.path.join(path_new,new_filename),audio_final,sr)
                        
                        subprocess.call(['ffmpeg', '-i', os.path.join(path_new,new_filename),'-b:a','192000',os.path.join(path_new,new_filename)[:-3]+'mp3'])
                        os.remove(os.path.join(path_new,new_filename))
                        
                        if args.job!=1:
                        
                            noise_list=[noise1_orig,noise2_orig,noise3_orig,noise4_orig,noise5_orig,noise6_orig,noise7_orig,noise8_orig]
                            audio_final = audio_make(audio,list,noise,noise_compression,noise_misc,noise_list,audio_ir,sr,args)

                            new_filename=str(noise1_orig)+'_'+str(noise2_orig)+'_'+str(noise3_orig)+'_'+str(noise4_orig)+'_'+str(noise5_orig)+'_'+str(noise6_orig)+'_'+str(noise7_orig)+'_'+str(noise8_orig)+'_'+filename_full_ir
                            sf.write(os.path.join(path_new,new_filename),audio_final,sr)
                            subprocess.call(['ffmpeg', '-i', os.path.join(path_new,new_filename),'-b:a','192000',os.path.join(path_new,new_filename)[:-3]+'mp3'])
                            os.remove(os.path.join(path_new,new_filename))
                        
def daps(args,path,path_new):
    
    count=0
    with open(args.saved_file) as csv_file:
        csv_reader = csv.reader(csv_file,delimiter=',')
        for row in tqdm.tqdm(csv_reader):
            count+=1
            list = np.array(['linear','reverb','compression','eq','misc'])
            noise_type=np.array(["applause","blue_noise","crickets","pink_noise","violet_noise","water_drops","white_noise"])
            random.shuffle(list)
            noise_type_compression=np.array(["mp3","mulaw"])
            noise_type_misc=np.array(["pops","griffinlin"])

            noise=random.choice(noise_type)
            noise_compression=random.choice(noise_type_compression)
            noise_misc=random.choice(noise_type_misc)

            if count>=3 and count<=args.numberofpairs+1:
                a=(row[0][7:-1]).split('_')
                noise1_orig=a[0]
                noise2_orig=a[1]
                noise3_orig=a[2]
                noise4_orig=a[3]
                noise5_orig=a[4]
                noise6_orig=a[5]
                noise7_orig=a[6]
                noise8_orig=a[7]

                speaker=a[8]
                audio_file_ir="_".join(a[13:])
                audio_ir, sample_rate_ir = sf.read(os.path.join('/n/fs/percepaudio/www/mturk_hosts/RIR_MIT/Audio/',audio_file_ir))

                for i in range(len(row)-1):
                    if i==0:
                        a=(row[i][7:-1]).split('_')
                        noise1=a[0]
                        noise2=a[1]
                        noise3=a[2]
                        noise4=a[3]
                        noise5=a[4]
                        noise6=a[5]
                        noise7=a[6]
                        noise8=a[7]

                        speaker=a[8]
                        filename="_".join(a[8:13])
                        filename_full_ir="_".join(a[8:])
                        print(filename)
                        audio,sr=sf.read(os.path.join(path,speaker,filename+'.wav'))
                        
                        noise_list=[noise1,noise2,noise3,noise4,noise5,noise6,noise7,noise8]
                        audio_final = audio_make(audio,list,noise,noise_compression,noise_misc,noise_list,audio_ir,sr,args)
                        new_filename=str(noise1)+'_'+str(noise2)+'_'+str(noise3)+'_'+str(noise4)+'_'+str(noise5)+'_'+str(noise6)+'_'+str(noise7)+'_'+str(noise8)+'_'+filename_full_ir
                        sf.write(os.path.join(path_new,new_filename),audio_final,sr)
                        subprocess.call(['ffmpeg', '-i', os.path.join(path_new,new_filename),'-b:a','192000',os.path.join(path_new,new_filename)[:-3]+'mp3'])
                        os.remove(os.path.join(path_new,new_filename))

                    elif i==len(row)-2:

                        a=(row[i][:-1]).split('_')
                        noise1=a[0]
                        noise2=a[1]
                        noise3=a[2]
                        noise4=a[3]
                        noise5=a[4]
                        noise6=a[5]
                        noise7=a[6]
                        noise8=a[7]
                        
                        speaker=a[8]
                        filename="_".join(a[8:13])
                        filename_full_ir="_".join(a[8:])

                        audio,sr=sf.read(os.path.join(path,speaker,filename+'.wav'))
                        #randomly decide between the first two - reverb first or linear noise first

                        noise_list=[noise1,noise2,noise3,noise4,noise5,noise6,noise7,noise8]
                        audio_final = audio_make(audio,list,noise,noise_compression,noise_misc,noise_list,audio_ir,sr,args)

                        #audio_final=mu_law_selection(float(noise3),noise_before)
                        new_filename=str(noise1)+'_'+str(noise2)+'_'+str(noise3)+'_'+str(noise4)+'_'+str(noise5)+'_'+str(noise6)+'_'+str(noise7)+'_'+str(noise8)+'_'+filename_full_ir
                        sf.write(os.path.join(path_new,new_filename),audio_final,sr)
                        subprocess.call(['ffmpeg', '-i', os.path.join(path_new,new_filename),'-b:a','192000',os.path.join(path_new,new_filename)[:-3]+'mp3'])
                        os.remove(os.path.join(path_new,new_filename))
                        
                        if args.job!=1:
                        
                            #print(new_filename)
                            noise_list=[noise1_orig,noise2_orig,noise3_orig,noise4_orig,noise5_orig,noise6_orig,noise7_orig,noise8_orig]
                            audio_final = audio_make(audio,list,noise,noise_compression,noise_misc,noise_list,audio_ir,sr,args)

                            new_filename=str(noise1_orig)+'_'+str(noise2_orig)+'_'+str(noise3_orig)+'_'+str(noise4_orig)+'_'+str(noise5_orig)+'_'+str(noise6_orig)+'_'+str(noise7_orig)+'_'+str(noise8_orig)+'_'+filename_full_ir
                            #print(new_filename)
                            sf.write(os.path.join(path_new,new_filename),audio_final,sr)
                            subprocess.call(['ffmpeg', '-i', os.path.join(path_new,new_filename),'-b:a','192000',os.path.join(path_new,new_filename)[:-3]+'mp3'])
                            os.remove(os.path.join(path_new,new_filename))
                        
                    else:
                        a=(row[i]).split('_')
                        noise1=a[0]
                        noise2=a[1]
                        noise3=a[2]
                        noise4=a[3]
                        noise5=a[4]
                        noise6=a[5]
                        noise7=a[6]
                        noise8=a[7]

                        speaker=a[8]
                        filename="_".join(a[8:13])
                        filename_full_ir="_".join(a[8:])

                        audio,sr=sf.read(os.path.join(path,speaker,filename+'.wav'))
                        #randomly decide between the first two - reverb first or linear noise first

                        noise_list=[noise1,noise2,noise3,noise4,noise5,noise6,noise7,noise8]
                        audio_final = audio_make(audio,list,noise,noise_compression,noise_misc,noise_list,audio_ir,sr,args)

                        #audio_final=mu_law_selection(float(noise3),noise_before)
                        new_filename=str(noise1)+'_'+str(noise2)+'_'+str(noise3)+'_'+str(noise4)+'_'+str(noise5)+'_'+str(noise6)+'_'+str(noise7)+'_'+str(noise8)+'_'+filename_full_ir
                        sf.write(os.path.join(path_new,new_filename),audio_final,sr)
                        subprocess.call(['ffmpeg', '-i', os.path.join(path_new,new_filename),'-b:a','192000',os.path.join(path_new,new_filename)[:-3]+'mp3'])
                        os.remove(os.path.join(path_new,new_filename))
                        
                        if args.job!=1:
                            noise_list=[noise1_orig,noise2_orig,noise3_orig,noise4_orig,noise5_orig,noise6_orig,noise7_orig,noise8_orig]
                            audio_final = audio_make(audio,list,noise,noise_compression,noise_misc,noise_list,audio_ir,sr,args)

                            new_filename=str(noise1_orig)+'_'+str(noise2_orig)+'_'+str(noise3_orig)+'_'+str(noise4_orig)+'_'+str(noise5_orig)+'_'+str(noise6_orig)+'_'+str(noise7_orig)+'_'+str(noise8_orig)+'_'+filename_full_ir
                            #print(new_filename)
                            sf.write(os.path.join(path_new,new_filename),audio_final,sr)
                            subprocess.call(['ffmpeg', '-i', os.path.join(path_new,new_filename),'-b:a','192000',os.path.join(path_new,new_filename)[:-3]+'mp3'])
                            os.remove(os.path.join(path_new,new_filename))
                        
                        
            elif count==args.numberofpairs+2:
                a=(row[0][7:-1]).split('_')
                noise1_orig=a[0]
                noise2_orig=a[1]
                noise3_orig=a[2]
                noise4_orig=a[3]
                noise5_orig=a[4]
                noise6_orig=a[5]
                noise7_orig=a[6]
                noise8_orig=a[7]

                speaker=a[8]
                audio_file_ir="_".join(a[13:])
                audio_ir, sample_rate_ir = sf.read(os.path.join('/n/fs/percepaudio/www/mturk_hosts/RIR_MIT/Audio/',audio_file_ir))

                for i in (range(0,len(row))):
                    if i==len(row)-1:
                        a=(row[i][:-2]).split('_')
                        noise1=a[0]
                        noise2=a[1]
                        noise3=a[2]
                        noise4=a[3]
                        noise5=a[4]
                        noise6=a[5]
                        noise7=a[6]
                        noise8=a[7]

                        speaker=a[8]
                        filename="_".join(a[8:13])
                        filename_full_ir="_".join(a[8:])

                        audio,sr=sf.read(os.path.join(path,speaker,filename+'.wav'))
                        #randomly decide between the first two - reverb first or linear noise first

                        noise_list=[noise1,noise2,noise3,noise4,noise5,noise6,noise7,noise8]
                        audio_final = audio_make(audio,list,noise,noise_compression,noise_misc,noise_list,audio_ir,sr,args)

                        #audio_final=mu_law_selection(float(noise3),noise_before)
                        new_filename=str(noise1)+'_'+str(noise2)+'_'+str(noise3)+'_'+str(noise4)+'_'+str(noise5)+'_'+str(noise6)+'_'+str(noise7)+'_'+str(noise8)+'_'+filename_full_ir
                        sf.write(os.path.join(path_new,new_filename),audio_final,sr)
                        subprocess.call(['ffmpeg', '-i', os.path.join(path_new,new_filename),'-b:a','192000',os.path.join(path_new,new_filename)[:-3]+'mp3'])
                        os.remove(os.path.join(path_new,new_filename))
                        
                        if args.job!=1:
                        #print(new_filename)
                            noise_list=[noise1_orig,noise2_orig,noise3_orig,noise4_orig,noise5_orig,noise6_orig,noise7_orig,noise8_orig]
                            audio_final = audio_make(audio,list,noise,noise_compression,noise_misc,noise_list,audio_ir,sr,args)

                            new_filename=str(noise1_orig)+'_'+str(noise2_orig)+'_'+str(noise3_orig)+'_'+str(noise4_orig)+'_'+str(noise5_orig)+'_'+str(noise6_orig)+'_'+str(noise7_orig)+'_'+str(noise8_orig)+'_'+filename_full_ir
                            #print(new_filename)
                            sf.write(os.path.join(path_new,new_filename),audio_final,sr)
                            subprocess.call(['ffmpeg', '-i', os.path.join(path_new,new_filename),'-b:a','192000',os.path.join(path_new,new_filename)[:-3]+'mp3'])
                            os.remove(os.path.join(path_new,new_filename))

                    elif i==0:
                        a=(row[i][7:-1]).split('_')
                        noise1=a[0]
                        noise2=a[1]
                        noise3=a[2]
                        noise4=a[3]
                        noise5=a[4]
                        noise6=a[5]
                        noise7=a[6]
                        noise8=a[7]

                        speaker=a[8]
                        filename="_".join(a[8:13])
                        filename_full_ir="_".join(a[8:])
                        audio,sr=sf.read(os.path.join(path,speaker,filename+'.wav'))

                        noise_list=[noise1,noise2,noise3,noise4,noise5,noise6,noise7,noise8]
                        audio_final = audio_make(audio,list,noise,noise_compression,noise_misc,noise_list,audio_ir,sr,args)
                        new_filename=str(noise1)+'_'+str(noise2)+'_'+str(noise3)+'_'+str(noise4)+'_'+str(noise5)+'_'+str(noise6)+'_'+str(noise7)+'_'+str(noise8)+'_'+filename_full_ir
                        sf.write(os.path.join(path_new,new_filename),audio_final,sr)
                        subprocess.call(['ffmpeg', '-i', os.path.join(path_new,new_filename),'-b:a','192000',os.path.join(path_new,new_filename)[:-3]+'mp3'])
                        os.remove(os.path.join(path_new,new_filename))

                    else:
                        a=(row[i]).split('_')
                        noise1=a[0]
                        noise2=a[1]
                        noise3=a[2]
                        noise4=a[3]
                        noise5=a[4]
                        noise6=a[5]
                        noise7=a[6]
                        noise8=a[7]
                        
                        speaker=a[8]
                        filename="_".join(a[8:13])
                        filename_full_ir="_".join(a[8:])
                        audio,sr=sf.read(os.path.join(path,speaker,filename+'.wav'))
                        noise_list=[noise1,noise2,noise3,noise4,noise5,noise6,noise7,noise8]
                        audio_final = audio_make(audio,list,noise,noise_compression,noise_misc,noise_list,audio_ir,sr,args)

                        new_filename=str(noise1)+'_'+str(noise2)+'_'+str(noise3)+'_'+str(noise4)+'_'+str(noise5)+'_'+str(noise6)+'_'+str(noise7)+'_'+str(noise8)+'_'+filename_full_ir
                        sf.write(os.path.join(path_new,new_filename),audio_final,sr)
                        subprocess.call(['ffmpeg', '-i', os.path.join(path_new,new_filename),'-b:a','192000',os.path.join(path_new,new_filename)[:-3]+'mp3'])
                        os.remove(os.path.join(path_new,new_filename))
                        
                        if args.job!=1:
                            noise_list=[noise1_orig,noise2_orig,noise3_orig,noise4_orig,noise5_orig,noise6_orig,noise7_orig,noise8_orig]
                            audio_final = audio_make(audio,list,noise,noise_compression,noise_misc,noise_list,audio_ir,sr,args)

                            new_filename=str(noise1_orig)+'_'+str(noise2_orig)+'_'+str(noise3_orig)+'_'+str(noise4_orig)+'_'+str(noise5_orig)+'_'+str(noise6_orig)+'_'+str(noise7_orig)+'_'+str(noise8_orig)+'_'+filename_full_ir
                            sf.write(os.path.join(path_new,new_filename),audio_final,sr)
                            subprocess.call(['ffmpeg', '-i', os.path.join(path_new,new_filename),'-b:a','192000',os.path.join(path_new,new_filename)[:-3]+'mp3'])
                            os.remove(os.path.join(path_new,new_filename))
                                     
def function_map(args):
    print(float(args.EQ_bands.split('_')[0]))
    
    
def check_valid_type(init,new_levels):
    
    count=0 
    for i in range(len(new_levels)):
        if init[i]<=new_levels[i] or new_levels[i]<0 or  new_levels[i]>100:
            count=count+1
    return count


def get_points_finetune(rand_speaker,rand_reverb,b1,args):
    
    #linear has one param
    #reverb has 2
    #compression has 1
    #eq has 3
    
    list = np.array(['linear','reverb','compression','eq','misc'])
    random.shuffle(list)
    noise_type=np.array(["applause","blue_noise","crickets","pink_noise","violet_noise","water_drops","white_noise"])
    noise_type_compression=np.array(["mp3","mulaw"])
    noise_type_misc=np.array(["pops","griffinlin"])
    
    overall_list=[]
    b = np.linspace(0,1,64)
    
    for j in range(1):
        
        list = [1,0]
        sampling = random.choices(list, k=5)
        
        space=sampling.count(1)

        while(space<1):
            list = [1,0]
            sampling = random.choices(list, k=5)
            
            space=sampling.count(1)

        if sampling[1]!=0 and sampling[3]!=0:

            init=np.random.randint(0,101,size=space+3)
            s = np.random.normal(0, 1, space+3)
            decrease_max_per_hit=np.repeat(args.min_difference,space+3)
        
        elif sampling[1]!=0 and sampling[3]==0:

            init=np.random.randint(0,101,size=space+1)
            s = np.random.normal(0, 1, space+1)
            decrease_max_per_hit=np.repeat(args.min_difference,space+1)

        elif sampling[1]==0 and sampling[3]!=0:

            init=np.random.randint(0,101,size=space+2)
            s = np.random.normal(0, 1, space+2)
            decrease_max_per_hit=np.repeat(args.min_difference,space+2)

        else:

            init=np.random.randint(0,101,size=space)
            s = np.random.normal(0, 1, space)
            decrease_max_per_hit=np.repeat(args.min_difference,space)

        x_normalized=s/np.linalg.norm(s)
        
        x_normalized_equal1 = x_normalized / np.min(np.abs(x_normalized))

        a = np.abs(decrease_max_per_hit / x_normalized_equal1)
        min_steps = np.min(a)
        c = min_steps * x_normalized_equal1
        new_levels=init+c

        count = check_valid_type(init,new_levels)

        while (count!=0):

            list = [1,0]
            sampling = random.choices(list, k=5)
            #rand_speaker=random.choice(speaker)
            #rand_reverb=random.choice(reverblist)
            space=sampling.count(1)

            while(space<1):
                list = [1,0]
                sampling = random.choices(list, k=5)
                #rand_speaker=random.choice(speaker)
                #rand_reverb=random.choice(reverblist)
                space=sampling.count(1)


            if sampling[1]!=0 and sampling[3]!=0:

                init=np.random.randint(0,101,size=space+3)
                s = np.random.normal(0, 1, space+3)
                decrease_max_per_hit=np.repeat(args.min_difference,space+3)

            elif sampling[1]!=0 and sampling[3]==0:

                init=np.random.randint(0,101,size=space+1)
                s = np.random.normal(0, 1, space+1)
                decrease_max_per_hit=np.repeat(args.min_difference,space+1)

            elif sampling[1]==0 and sampling[3]!=0:

                init=np.random.randint(0,101,size=space+2)
                s = np.random.normal(0, 1, space+2)
                decrease_max_per_hit=np.repeat(args.min_difference,space+2)

            else:
                
                init=np.random.randint(0,101,size=space)
                s = np.random.normal(0, 1, space)
                decrease_max_per_hit=np.repeat(args.min_difference,space)

            x_normalized=s/np.linalg.norm(s)

            x_normalized_equal1 = x_normalized / np.min(np.abs(x_normalized))

            a = np.abs(decrease_max_per_hit / x_normalized_equal1)
            min_steps = np.min(a)
            c = min_steps * x_normalized_equal1
            new_levels=init+c
            
            count = check_valid_type(init,new_levels)
        
        pra = np.linalg.norm(c)
        random_choice=np.sort(np.random.choice(64, 3, replace=False))[::-1]
        
        for i in range(len(random_choice)):
                
                z1=random_choice[i]
                b2=np.round(init+b[z1]*pra*x_normalized,2)
                save_file=''
                count=0
                
                if sampling[1]!=0 and sampling[3]!=0:
                    length=5
                    for k in range(length):
                        if k==0:
                            if sampling[k]==1:
                                save_file+=str(b2[count])
                                count+=1
                            else:
                                save_file+=str(0.0)
                        elif k==1:
                            save_file+='_'+str(b2[count])
                            count+=1
                            save_file+='_'+str(b2[count])
                            count+=1
                        elif k==2:
                            if sampling[k]==1:
                                save_file+='_'+str(b2[count])
                                count+=1
                            else:
                                save_file+='_'+str(0.0)
                        elif k==3:
                            save_file+='_'+str(b2[count])
                            count+=1
                            save_file+='_'+str(b2[count])
                            count+=1
                            save_file+='_'+str(b2[count])
                            count+=1
                        elif k==4:
                            if sampling[k]==1:
                                save_file+='_'+str(b2[count])
                                count+=1
                            else:
                                save_file+='_'+str(0.0)


                elif sampling[1]!=0 and sampling[3]==0:
                    length=5
                    for k in range(length):
                        if k==0:
                            if sampling[k]==1:
                                save_file+=str(b2[count])
                                count+=1
                            else:
                                save_file+=str(0.0)
                        elif k==1:
                            save_file+='_'+str(b2[count])
                            count+=1
                            save_file+='_'+str(b2[count])
                            count+=1
                        elif k==2:
                            if sampling[k]==1:
                                save_file+='_'+str(b2[count])
                                count+=1
                            else:
                                save_file+='_'+str(0.0)
                        elif k==3:
                            save_file+='_'+str(0.0)
                            save_file+='_'+str(0.0)
                            save_file+='_'+str(0.0)
                        elif k==4:
                            if sampling[k]==1:
                                save_file+='_'+str(b2[count])
                                count+=1
                            else:
                                save_file+='_'+str(0.0)

                elif sampling[1]==0 and sampling[3]!=0:
                    length=5
                    for k in range(length):
                        if k==0:
                            if sampling[k]==1:
                                save_file+=str(b2[count])
                                count+=1
                            else:
                                save_file+=str(0.0)

                        elif k==1:
                            save_file+='_'+str(0.0)
                            save_file+='_'+str(0.0)

                        elif k==2:
                            if sampling[k]==1:
                                save_file+='_'+str(b2[count])
                                count+=1
                            else:
                                save_file+='_'+str(0.0)
                        elif k==3:
                            save_file+='_'+str(b2[count])
                            count+=1
                            save_file+='_'+str(b2[count])
                            count+=1
                            save_file+='_'+str(b2[count])
                            count+=1
                        elif k==4:
                            if sampling[k]==1:
                                save_file+='_'+str(b2[count])
                                count+=1
                            else:
                                save_file+='_'+str(0.0)

                else:
                    length=5
                    for k in range(length):
                        if k==0:
                            if sampling[k]==1:
                                save_file+=str(b2[count])
                                count+=1
                            else:
                                save_file+=str(0.0)

                        elif k==1:
                            save_file+='_'+str(0.0)
                            save_file+='_'+str(0.0)

                        elif k==2:
                            if sampling[k]==1:
                                save_file+='_'+str(b2[count])
                                count+=1
                            else:
                                save_file+='_'+str(0.0)
                        elif k==3:
                            save_file+='_'+str(0.0)
                            save_file+='_'+str(0.0)
                            save_file+='_'+str(0.0)
                        elif k==4:
                            if sampling[k]==1:
                                save_file+='_'+str(b2[count])
                                count+=1
                            else:
                                save_file+='_'+str(0.0)
                
                overall_list.append(save_file+'_'+b1[:-4]+'_'+str(rand_reverb))
        return overall_list
        

def create_triplets_finetune(args):
    
    reverblist=sorted(os.listdir('/n/fs/percepaudio/www/mturk_hosts/RIR_MIT/Audio'))
    
    if args.dataset_used=='librispeech':
        speaker=np.array(sorted(os.listdir('/n/fs/percepaudio/librispeech/LibriSpeech/resized_files_2')))
    elif args.dataset_used=='daps':
        speaker=np.array(sorted(os.listdir('/n/fs/percepaudio/www/mturk_hosts/website_perturbation/snr_resize_folder/')))
    
    overall=[]
    
    for i in range(args.numberofpairs):
        count=0
        rand_speaker=random.choice(speaker)
        
        if args.dataset_used=='librispeech':
            b0=os.listdir('/n/fs/percepaudio/librispeech/LibriSpeech/resized_files_2/'+rand_speaker)
            
        elif args.dataset_used=='daps':
            b0=os.listdir('/n/fs/percepaudio/www/mturk_hosts/website_perturbation/snr_resize_folder/'+rand_speaker)
                
        b1=np.random.randint(0,len(b0))
        rand_reverb=random.choice(reverblist)
        sample1=get_points_finetune(rand_speaker,rand_reverb,b0[b1],args)
        overall.append(sample1)
    return overall
