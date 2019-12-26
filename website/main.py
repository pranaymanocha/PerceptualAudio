import os
import csv
import argparse

'''
1) Change the numbers on line 46,47,48: splitted. splitted1 etc
'''

def argument_parser():
    """
    Get an argument parser for a training script.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--text_file_name', help='name of the txt file which has the location', default='dataset_train_shuffled.txt', type=str)
    parser.add_argument('--audio_folder', help='folder which contains the pre-made audio', default='prefetched/', type=str)    
    parser.add_argument('--results_folder', help='location of the results folder which has all txt files', default='results/results_noise_librispeech/', type=str)
    
    return parser

args = argument_parser().parse_args()

count_files=0

file = open('dataset_train_all_librispeech.txt','a+') 
folders = [args.results_folder]
locations=[args.audio_folder]

for p,folder in enumerate(folders):
    path=folder
    for filename in os.listdir(folder):
        a=filename.split('_')
        if a[0] !='log' and a[0] !='responses' and filename !='.ipynb_checkpoints' and filename!='.csv' and filename!='.nfs0000000283187e0c00000cfe':
            with open(os.path.join(path,filename)) as csv_file:
                csv_reader = csv.reader(csv_file,delimiter=',')
                s=','
                for row in csv_reader:
                    answer_string=s.join(row).split('::')
                    count=0
                    for i in range(3):
                        reference=answer_string[9+i]
                        if int(answer_string[12+i].split(',')[0])==0 and int(answer_string[12+i].split(',')[1])==1:
                            for iterate in range(10):
                                position=round(float(answer_string[9+i].split(',')[iterate])*63)
                                question1=answer_string[6+i].split(',')[position]
                                
                                splitted="_".join(question1.split('_')[7:12])
                                splitted1="_".join(question1.split('_')[12:])
                                splitted0="_".join(question1.split('_')[:7])
                                
                                question_path=os.path.join(locations[p],splitted0+'_'+splitted+'_'+splitted1)
                                print(question_path)
                                
                                s1='_'
                                
                                file_name=s1.join(question1.split('_')[7:])
                                
                                reference=answer_string[6+i].split(',')[0]
                                file_values=s1.join(reference.split('_')[:7])
                                
                                reference_filename=os.path.join(locations[p],file_values+'_'+file_name)
                                print(reference_filename)
                                answer=answer_string[12+i].split(',')[iterate]
                                string_save=reference_filename+'\t'+question_path+'\t'+answer
                                file.write(string_save)
                                file.write('\n')
                        else:
                            count+=10
                    
file.close()



import random
with open('dataset_train_all_librispeech.txt','r') as source:
    data = [ (random.random(), line) for line in source ]

data.sort()
with open(args.text_file_name,'w') as target:
    for _, line in data:
        target.write(line)
        
os.remove('dataset_train_all_librispeech.txt')