def load_full_data_list(datafolder='../'): #check change path names

    #sets=['train','val']
    dataset={}
    dataset['all']={}
    
    print('Loading files..')
    dataset['all']['inname'] = []
    dataset['all']['outname'] = []
    dataset['all']['label']=[]
    
    print("Prefetching the Combined")
    #data_path='prefetch_audio_new_mp3_new_morebandwidth'
    list_path='../'
    file = open(os.path.join(datafolder,'dataset_train_combined_all_shuffled.txt'), 'r')
    for line in file:
        split_line=line.split('\t')
        dataset['all']['inname'].append("%s"%(os.path.join(list_path,split_line[0])))
        dataset['all']['outname'].append("%s"%(os.path.join(list_path,split_line[1])))
        dataset['all']['label'].append(split_line[2][:-1])
        
    print("Prefetching the Reverb")  
    list_path='../'
    file = open(os.path.join(list_path,'dataset_train_shuffled_reverbBatch.txt'), 'r')
    for line in file:
        split_line=line.split('\t')
        dataset['all']['inname'].append("%s"%(os.path.join(list_path,split_line[0])))
        dataset['all']['outname'].append("%s"%(os.path.join(list_path,split_line[1])))
        dataset['all']['label'].append(split_line[2][:-1])
    
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
    

def loadall_audio_train_waveform(dataset):
    
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
                dataset['train']['inaudio'][id]  = dataset['train']['inaudio'][id-1]
                dataset['train']['outaudio'][id] = dataset['train']['outaudio'][id-1]
                dataset['train']['label'][id] = dataset['train']['label'][id-1]
                
                

    return dataset  


def loadall_audio_train_spectrogram(dataset):
    
    dataset['train']['inspec']  = [None]*len(dataset['train']['inname'])
    dataset['train']['outspec'] = [None]*len(dataset['train']['outname'])

    for id in tqdm(range(len(dataset['train']['inname']))):

        if dataset['train']['inspec'][id] is None:
            
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


                inputData_spec=log_mel_spectrogram(inputData,audio_sample_rate=44100)
                outputData_spec=log_mel_spectrogram(outputData,audio_sample_rate=44100)

                shape_spec = np.shape(inputData_spec)

                inputData_spec = np.reshape(inputData_spec, [1, shape_spec[0], shape_spec[1],1])
                outputData_spec = np.reshape(outputData_spec, [1, shape_spec[0], shape_spec[1], 1])


                inputData_spec  = np.float32(inputData_spec)
                outputData_spec = np.float32(outputData_spec)

                #dataset['train']['inaudio'][id]  = inputData_wav
                dataset['train']['inspec'][id]  = inputData_spec
                #dataset['train']['outaudio'][id] = outputData_wav
                dataset['train']['outspec'][id] = outputData_spec
            except:
                dataset['train']['inspec'][id]  = dataset['train']['inspec'][id-1]
                #dataset['train']['outaudio'][id] = outputData_wav
                dataset['train']['outspec'][id] = dataset['train']['outspec'][id-1]
                dataset['train']['label'][id] = dataset['train']['label'][id-1]
                

    return dataset

def loadall_audio_train_both(dataset):
    
   
    dataset['train']['inspec']  = [None]*len(dataset['train']['inname'])
    dataset['train']['outspec'] = [None]*len(dataset['train']['outname'])
    dataset['train']['inaudio']  = [None]*len(dataset['train']['inname'])
    dataset['train']['outaudio'] = [None]*len(dataset['train']['outname'])

    for id in tqdm(range(len(dataset['train']['inname']))):

        if dataset['train']['inspec'][id] is None:
            
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

                inputData_fr=frame(inputData,int(round(0.02321*fs)),int(round(0.010*fs)))
                outputData_fr=frame(outputData,int(round(0.02321*fs)),int(round(0.010*fs)))

                inputData_fr = np.reshape(inputData_fr, [1, -1, 1024,1])
                outputData_fr = np.reshape(outputData_fr, [1, -1, 1024, 1])

                inputData_fr  = np.float32(inputData_fr)
                outputData_fr = np.float32(outputData_fr)

                #inputData_wav  = np.reshape(inputData, [-1, 1])
                #outputData_wav = np.reshape(outputData, [-1, 1])

                inputData_spec=log_mel_spectrogram(inputData,audio_sample_rate=fs)
                outputData_spec=log_mel_spectrogram(outputData,audio_sample_rate=fs)

                #print(outputData.shape)
                shape_spec = np.shape(inputData_spec)

                inputData_spec = np.reshape(inputData_spec, [1, shape_spec[0], shape_spec[1],1])
                outputData_spec = np.reshape(outputData_spec, [1, shape_spec[0], shape_spec[1], 1])

                #shape_wav = np.shape(inputData_wav)

                #inputData_wav = np.reshape(inputData_wav, [1, 1,shape_wav[0], shape_wav[1]])
                #outputData_wav = np.reshape(outputData_wav, [1, 1,shape_wav[0], shape_wav[1]])


                inputData_spec  = np.float32(inputData_spec)
                outputData_spec = np.float32(outputData_spec)

                #inputData_wav  = np.float32(inputData_wav)
                #outputData_wav = np.float32(outputData_wav)

                #dataset['train']['inaudio'][id]  = inputData_wav
                dataset['train']['inspec'][id]  = inputData_spec
                #dataset['train']['outaudio'][id] = outputData_wav
                dataset['train']['outspec'][id] = outputData_spec
                
                dataset['train']['inaudio'][id]  = inputData_fr
                #dataset['train']['outaudio'][id] = outputData_wav
                dataset['train']['outaudio'][id] = outputData_fr
                
            except:
                dataset['train']['inspec'][id]  = dataset['train']['inspec'][id-1]
                #dataset['train']['outaudio'][id] = outputData_wav
                dataset['train']['outspec'][id] = dataset['train']['outspec'][id-1]
                dataset['train']['inaudio'][id]  = dataset['train']['inaudio'][id-1]
                #dataset['train']['outaudio'][id] = outputData_wav
                dataset['train']['outaudio'][id] = dataset['train']['outaudio'][id-1]
                dataset['train']['label'][id] = dataset['train']['label'][id-1]
                
    return dataset


def loadall_audio_test_waveform(dataset):

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
                dataset['test']['outaudio'][id] = outputData_wav
                
            except:
                
                dataset['test']['inaudio'][id]  = dataset['test']['inaudio'][id-1]
                dataset['test']['outaudio'][id] = dataset['test']['outaudio'][id-1]
                dataset['test']['label'][id] = dataset['test']['label'][id-1]
    return dataset

def loadall_audio_test_spectrogram(dataset):

    dataset['test']['inspec']  = [None]*len(dataset['test']['inname'])
    dataset['test']['outspec'] = [None]*len(dataset['test']['outname'])

    for id in tqdm(range(len(dataset['test']['inname']))):

        if dataset['test']['inspec'][id] is None:
            
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


                inputData_spec=log_mel_spectrogram(inputData,audio_sample_rate=fs)
                outputData_spec=log_mel_spectrogram(outputData,audio_sample_rate=fs)

                shape_spec = np.shape(inputData_spec)

                inputData_spec = np.reshape(inputData_spec, [1, shape_spec[0], shape_spec[1],1])
                outputData_spec = np.reshape(outputData_spec, [1, shape_spec[0], shape_spec[1], 1])


                inputData_spec  = np.float32(inputData_spec)
                outputData_spec = np.float32(outputData_spec)

                dataset['test']['inspec'][id]  = inputData_spec
                dataset['test']['outspec'][id] = outputData_spec
            except:
                dataset['test']['inspec'][id]  = dataset['test']['inspec'][id-1]
                dataset['test']['outspec'][id] = dataset['test']['outspec'][id-1]
                dataset['test']['label'][id]=dataset['test']['label'][id-1]
                
    return dataset

def loadall_audio_test_both(dataset):

    dataset['test']['inspec']  = [None]*len(dataset['test']['inname'])
    dataset['test']['outspec'] = [None]*len(dataset['test']['outname'])
    dataset['test']['inaudio']  = [None]*len(dataset['test']['inname'])
    dataset['test']['outaudio'] = [None]*len(dataset['test']['outname'])

    for id in tqdm(range(len(dataset['test']['inname']))):

        if dataset['test']['inspec'][id] is None:
            
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

                inputData_fr=frame(inputData,int(round(0.02321*fs)),int(round(0.010*fs)))
                outputData_fr=frame(outputData,int(round(0.02321*fs)),int(round(0.010*fs)))

                inputData_fr = np.reshape(inputData_fr, [1, -1, 1024,1])
                outputData_fr = np.reshape(outputData_fr, [1, -1, 1024, 1])

                inputData_fr  = np.float32(inputData_fr)
                outputData_fr = np.float32(outputData_fr)

                inputData_spec=log_mel_spectrogram(inputData,audio_sample_rate=fs)
                outputData_spec=log_mel_spectrogram(outputData,audio_sample_rate=fs)

                shape_spec = np.shape(inputData_spec)

                inputData_spec = np.reshape(inputData_spec, [1, shape_spec[0], shape_spec[1],1])
                outputData_spec = np.reshape(outputData_spec, [1, shape_spec[0], shape_spec[1], 1])



                inputData_spec  = np.float32(inputData_spec)
                outputData_spec = np.float32(outputData_spec)


                dataset['test']['inspec'][id]  = inputData_spec
                dataset['test']['outspec'][id] = outputData_spec
                
                dataset['test']['inaudio'][id]  = inputData_fr
                dataset['test']['outaudio'][id] = outputData_fr
            except:
                dataset['test']['inspec'][id]  = dataset['test']['inspec'][id-1]
                dataset['test']['outspec'][id] = dataset['test']['outspec'][id-1]
                
                dataset['test']['inaudio'][id]  = dataset['test']['inwav'][id-1]
                dataset['test']['outaudio'][id] = dataset['test']['outwav'][id-1]
                dataset['test']['label'][id]=dataset['test']['label'][id-1]
                
    return dataset



def loadall_audio_test_spectrogram(dataset):

    dataset['test']['inspec']  = [None]*len(dataset['test']['inname'])
    dataset['test']['outspec'] = [None]*len(dataset['test']['outname'])

    for id in tqdm(range(len(dataset['test']['inname']))):

        if dataset['test']['inspec'][id] is None:
            
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


                inputData_spec=log_mel_spectrogram(inputData,audio_sample_rate=44100)
                outputData_spec=log_mel_spectrogram(outputData,audio_sample_rate=44100)

                shape_spec = np.shape(inputData_spec)

                inputData_spec = np.reshape(inputData_spec, [1, shape_spec[0], shape_spec[1],1])
                outputData_spec = np.reshape(outputData_spec, [1, shape_spec[0], shape_spec[1], 1])


                inputData_spec  = np.float32(inputData_spec)
                outputData_spec = np.float32(outputData_spec)

                dataset['test']['inspec'][id]  = inputData_spec
                dataset['test']['outspec'][id] = outputData_spec
            except:
                dataset['test']['inspec'][id]  = dataset['test']['inspec'][id-1]
                dataset['test']['outspec'][id] = dataset['test']['outspec'][id-1]
                dataset['test']['label'][id]=dataset['test']['label'][id-1]
                
    return dataset


def load_full_data_test_waveform(dataset,sets,id_value):
    
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
 
    
def load_full_data_test_spectrogram(dataset,sets,id_value):
    
    
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
    

    inputData_spec=log_mel_spectrogram(inputData,audio_sample_rate=44100)
    outputData_spec=log_mel_spectrogram(outputData,audio_sample_rate=44100)

    shape_spec = np.shape(inputData_spec)

    inputData_spec = np.reshape(inputData_spec, [1, shape_spec[0], shape_spec[1],1])
    outputData_spec = np.reshape(outputData_spec, [1, shape_spec[0], shape_spec[1], 1])



    inputData_spec  = np.float32(inputData_spec)
    outputData_spec = np.float32(outputData_spec)
    
    
    return [inputData_spec,outputData_spec]
    
    
def load_full_data_test_both(dataset,sets,id_value):
    

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
    
    inputData_fr=frame(inputData,int(round(0.02321*fs)),int(round(0.010*fs)))
    outputData_fr=frame(outputData,int(round(0.02321*fs)),int(round(0.010*fs)))

    inputData_fr = np.reshape(inputData_fr, [1, -1, 1024,1])
    outputData_fr = np.reshape(outputData_fr, [1, -1, 1024, 1])
    
    inputData_fr  = np.float32(inputData_fr)
    outputData_fr = np.float32(outputData_fr)

    inputData_spec=log_mel_spectrogram(inputData,audio_sample_rate=fs)
    outputData_spec=log_mel_spectrogram(outputData,audio_sample_rate=fs)

   
    shape_spec = np.shape(inputData_spec)

    inputData_spec = np.reshape(inputData_spec, [1, shape_spec[0], shape_spec[1],1])
    outputData_spec = np.reshape(outputData_spec, [1, shape_spec[0], shape_spec[1], 1])

    inputData_spec  = np.float32(inputData_spec)
    outputData_spec = np.float32(outputData_spec)

    return [inputData_spec,outputData_spec,inputData_fr,outputData_fr]
    
    
def load_full_data_waveform(dataset,sets,id_value):
        
    inputData_wav=dataset[sets]['inaudio'][id_value]
    outputData_wav=dataset[sets]['outaudio'][id_value]
    label = np.reshape(np.asarray(dataset[sets]['label'][id_value]),[-1,1])

    return [inputData_wav,outputData_wav,label]


def load_full_data_spectrogram(dataset,sets,id_value):
    
    inputData_wav=dataset[sets]['inspec'][id_value]
    outputData_wav=dataset[sets]['outspec'][id_value]
    label = np.reshape(np.asarray(dataset[sets]['label'][id_value]),[-1,1])
       
    return [inputData_wav,outputData_wav,label]

def load_full_data_both(dataset,sets,id_value):
      
    inputData_spec=dataset[sets]['inspec'][id_value]
    outputData_spec=dataset[sets]['outspec'][id_value]
    inputData_wav=dataset[sets]['inaudio'][id_value]
    outputData_wav=dataset[sets]['outaudio'][id_value]
    label = np.reshape(np.asarray(dataset[sets]['label'][id_value]),[-1,1])
    
    return [inputData_spec,outputData_spec,inputData_wav,outputData_wav,label]

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