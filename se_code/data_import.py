import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
import os, csv

# DATA LOADING - LOAD FILE LISTS
def load_full_data_list(datafolder='dataset'):#check change path names

    sets = ['train', 'val']
    dataset = {}
    datafolders = {}
    for setname in sets:
        dataset[setname] = {}
        datafolders[setname] = datafolder + '/' + setname + 'set'

    print "Loading files..."
    for setname in sets:
        foldername = datafolders[setname]

        dataset[setname]['innames'] = []
        dataset[setname]['outnames'] = []
        dataset[setname]['shortnames'] = []

        filelist = os.listdir("%s_noisy"%(foldername))
        filelist = [f for f in filelist if f.endswith(".wav")]
        for i in tqdm(filelist):
            dataset[setname]['innames'].append("%s_noisy/%s"%(foldername,i))
            dataset[setname]['outnames'].append("%s_clean/%s"%(foldername,i))
            dataset[setname]['shortnames'].append("%s"%(i))
    return dataset['train'], dataset['val']


# DATA LOADING - LOAD FILE DATA
def load_full_data(trainset, valset):

    for dataset in [trainset, valset]:

        dataset['inaudio']  = [None]*len(dataset['innames'])
        dataset['outaudio'] = [None]*len(dataset['outnames'])

        for id in tqdm(range(len(dataset['innames']))):

            if dataset['inaudio'][id] is None:
                fs, inputData  = wavfile.read(dataset['innames'][id])
                fs, outputData = wavfile.read(dataset['outnames'][id])

                inputData  = np.reshape(inputData, [-1, 1])
                outputData = np.reshape(outputData, [-1, 1])

                shape = np.shape(inputData)

                inputData = np.reshape(inputData, [1, 1, shape[0], shape[1]])
                outputData = np.reshape(outputData, [1, 1, shape[0], shape[1]])

                dataset['inaudio'][id]  = np.float32(inputData)
                dataset['outaudio'][id] = np.float32(outputData)

    return trainset, valset

# DATA LOADING - LOAD FILE LISTS
def load_noisy_data_list(valfolder = ''):#check change path names

    sets = ['val']
    dataset = {'val': {}}
    datafolders = {'val': valfolder}

    print "Loading files..."
    for setname in sets:
        foldername = datafolders[setname]

        dataset[setname]['innames'] = []
        dataset[setname]['shortnames'] = []

        filelist = os.listdir("%s"%(foldername))
        filelist = [f for f in filelist if f.endswith(".wav")]
        for i in tqdm(filelist):
            dataset[setname]['innames'].append("%s/%s"%(foldername,i))
            dataset[setname]['shortnames'].append("%s"%(i))

    return dataset['val']


# DATA LOADING - LOAD FILE DATA
def load_noisy_data(valset):

    for dataset in [valset]:

        dataset['inaudio']  = [None]*len(dataset['innames'])

        for id in tqdm(range(len(dataset['innames']))):

            if dataset['inaudio'][id] is None:
                fs, inputData  = wavfile.read(dataset['innames'][id])

                inputData  = np.reshape(inputData, [-1, 1])
                shape = np.shape(inputData)

                inputData = np.reshape(inputData, [1, 1, shape[0], shape[1]])

                dataset['inaudio'][id]  = np.float32(inputData)

    return valset

# ACOUSTIC SCENE CLASSIFICATION - LOAD DATA
def load_asc_data(ase_folder):

    sets = ['train', 'val']
    folders = {}
    for setname in sets:
        folders[setname] = ase_folder + "/" + setname + "set"
    labels = {}
    names = {}
    datasets = {}

    for setname in sets:
        foldername = folders[setname]

        labels[setname] = []
        names[setname] = []
        datasets[setname] = []

        n = []
        l = []

        with open('%s/meta.txt' % foldername, 'rb') as csvfile:
            metareader = csv.reader(csvfile, delimiter='\t', quotechar='|')
            for row in metareader:
                n.append(row[0][6:])
                l.append(row[1])

        for i in tqdm(range(len(n))):
            filename = n[i]
            #print(foldername + '/' + filename)
            fs, inputAudio = wavfile.read(foldername + '/' + filename)
            #print(inputAudio.shape)
            if not (fs == 16000):
                raise ValueError('Sample frequency is not 16kHz')
            shape = np.shape(inputAudio)
            if len(shape) > 1 and shape[1] > 1:
                for j in range(shape[1]):
                    inputData = np.reshape(inputAudio[:, j], [1, 1, shape[0], 1])
                    datasets[setname].append(inputData)
                    labels[setname].append(l[i])
                    names[setname].append(n[i])
            else:
                inputData = np.reshape(inputAudio, [1, 1, shape[0], 1])
                datasets[setname].append(inputData)
                labels[setname].append(l[i])
                names[setname].append(n[i])

    label_list = list(set(labels[sets[0]]))

    return datasets, labels, names, label_list


# DOMESTIC AUDIO TAGGING - LOAD DATA
def load_dat_data(dat_folder='dataset/dat'):

    sets = ['train', 'val']
    csv_files = {}
    csv_files[sets[0]] = dat_folder + "/development_chunks_refined.csv"
    csv_files[sets[1]] = dat_folder + "/evaluation_chunks_refined.csv"
    labels = {}
    names = {}
    datasets = {}

    for setname in sets:

        labels[setname] = []
        names[setname] = []
        datasets[setname] = []

        n = []
        l = []

        with open(csv_files[setname], 'rb') as csvfile:
            metareader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in metareader:
                n.append(row[1] + ".wav")
                #print(row[1])
                with open('%s/%s.csv' % (dat_folder, row[1]), 'rb') as csvfile2:
                    metareader2 = csv.reader(csvfile2, delimiter=',', quotechar='|')
                    for row in metareader2:
                        if row[0] == 'majorityvote':
                            l.append(row[1])

        for i in tqdm(range(len(n))):
            filename = n[i]
            fs, inputAudio = wavfile.read(dat_folder + '/' + filename)
            if not (fs == 16000):
                raise ValueError('Sample frequency is not 16kHz')
            shape = np.shape(inputAudio)
            if len(shape) > 1 and shape[1] > 1:
                for j in range(shape[1]):
                    inputData = np.reshape(inputAudio[:, j], [1, 1, shape[0], 1])
                    datasets[setname].append(inputData)
                    labels[setname].append(l[i])
                    names[setname].append(n[i])
            else:
                inputData = np.reshape(inputAudio, [1, 1, shape[0], 1])
                datasets[setname].append(inputData)
                labels[setname].append(l[i])
                names[setname].append(n[i])

    label_list = []
    for label in labels[sets[0]]:
        for ch in list(label):
            if not (label == 'S'):
                label_list.append(ch)
    label_list = list(set(label_list))

    return datasets, labels, names, label_list