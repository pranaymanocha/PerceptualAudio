from network_model import *
from data_import import *
from network_model import *

import sys, getopt
import argparse

def argument_parser():
    """
    Get an argument parser for a training script.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_folder', help='full path of the se model parameter', default='../pre-model/se_model/se_model.ckpt')
    parser.add_argument('--model_name', help='name of the SE model used for infer', default='m_sample')
    
    return parser

args = argument_parser().parse_args()

valfolder = "dataset/valset_noisy"
modfolder = args.model_folder

print 'Input folder is "' + valfolder + '/"'
print 'Model folder is "' + modfolder + '/"'

if valfolder[-1] == '/':
    valfolder = valfolder[:-1]

if not os.path.exists(valfolder+'_'+args.model_name+'_denoised'):
    os.makedirs(valfolder+'_'+args.model_name+'_denoised')

# SPEECH ENHANCEMENT NETWORK
SE_LAYERS = 13 # NUMBER OF INTERNAL LAYERS
SE_CHANNELS = 64 # NUMBER OF FEATURE CHANNELS PER LAYER
SE_NORM = "NM" # TYPE OF LAYER NORMALIZATION (NM, SBN or None)

fs = 16000

# SET LOSS FUNCTIONS AND PLACEHOLDERS
with tf.variable_scope(tf.get_variable_scope()):
    input=tf.placeholder(tf.float32,shape=[None,1,None,1])
    clean=tf.placeholder(tf.float32,shape=[None,1,None,1]) 
    enhanced=senet(input, n_layers=SE_LAYERS, norm_type=SE_NORM, n_channels=SE_CHANNELS)

# LOAD DATA
valset = load_noisy_data_list(valfolder = valfolder)
valset = load_noisy_data(valset)

# BEGIN SCRIPT #########################################################################################################

# INITIALIZE GPU CONFIG
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
sess=tf.Session(config=config)

print "Config ready"

sess.run(tf.global_variables_initializer())

print "Session initialized"

saver = tf.train.Saver([var for var in tf.trainable_variables() if var.name.startswith("se_")])
saver.restore(sess, str(modfolder))

#####################################################################################

for id in tqdm(range(0, len(valset["innames"]))):

    i = id # NON-RANDOMIZED ITERATION INDEX
    inputData = valset["inaudio"][i] # LOAD DEGRADED INPUT

    # VALIDATION ITERATION
    output = sess.run([enhanced],
                        feed_dict={input: inputData})
    output = np.reshape(output, -1)
    wavfile.write("%s_%s_denoised/%s" % (valfolder,args.model_name,valset["shortnames"][i]), fs, output)