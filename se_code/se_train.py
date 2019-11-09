from helper import *
from data_import import *
from network_model import *

import sys, getopt
import argparse

def argument_parser():
    """
    Get an argument parser for a training script.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--loss_layers', help='number of layers in the model', default=14, type=int)
    parser.add_argument('--out_folder', help='summary folder name', default='m_example')
    parser.add_argument('--loss_norm', help='loss norm - NM,SBN,None', default='SBN')
    parser.add_argument('--channels_increase', help='doubling channels after how many layers - 1,2,3,4,5,6', default=5, type=int)
    parser.add_argument('--model_folder', help='path of the loss model parameters saved location (without the / at the end )', default='')
    parser.add_argument('--learning_rate', help='learning rate', default=1e-4,type=float)
    parser.add_argument('--feature_loss_layers', help='number of feature loss layers used', default=14,type=int)
    parser.add_argument('--kernel_size', help='kernel convolution size',default=3,type=int)
    
    return parser

args = argument_parser().parse_args()

# SPEECH ENHANCEMENT NETWORK
SE_LAYERS = 13 # NUMBER OF INTERNAL LAYERS
SE_CHANNELS = 64 # NUMBER OF FEATURE CHANNELS PER LAYER
SE_LOSS_LAYERS = args.feature_loss_layers # NUMBER OF FEATURE LOSS LAYERS
SE_NORM = "NM" # TYPE OF LAYER NORMALIZATION (NM, SBN or None)
SE_LOSS_TYPE = "FL" # TYPE OF TRAINING LOSS (L1, L2 or FL)

# FEATURE LOSS NETWORK
LOSS_LAYERS = args.loss_layers # NUMBER OF INTERNAL LAYERS
LOSS_BASE_CHANNELS = 32 # NUMBER OF FEATURE CHANNELS PER LAYER IN FIRT LAYER
LOSS_BLK_CHANNELS = args.channels_increase # NUMBER OF LAYERS BETWEEN CHANNEL NUMBER UPDATES
LOSS_NORM =  args.loss_norm # TYPE OF LAYER NORMALIZATION (NM, SBN or None)

SET_WEIGHT_EPOCH = 10 # NUMBER OF EPOCHS BEFORE FEATURE LOSS BALANCE
SAVE_EPOCHS = 25 # NUMBER OF EPOCHS BETWEEN MODEL SAVES

# COMMAND LINE OPTIONS
datafolder = "dataset"
modfolder = args.model_folder
outfolder = args.out_folder

print('Data folder is "' + datafolder + '/"')
print('Loss model folder is "' + modfolder + '/"')
print('Output model folder is "' + outfolder + '/"')

# SET LOSS FUNCTIONS AND PLACEHOLDERS
with tf.variable_scope(tf.get_variable_scope()):
    input=tf.placeholder(tf.float32,shape=[None,1,None,1])
    clean=tf.placeholder(tf.float32,shape=[None,1,None,1])
    enhanced=senet(input, n_layers=SE_LAYERS, norm_type=SE_NORM, n_channels=SE_CHANNELS)
    
    if SE_LOSS_TYPE == "L1": # L1 LOSS
        loss_fn = l1_loss(clean, enhanced)
    elif SE_LOSS_TYPE == "L2": # L2 LOSS
        loss_fn = l2_loss(clean, enhanced)
    else: # FEATURE LOSS
        keep_prob = tf.placeholder_with_default(1.0, shape=())
        
        if args.type=='pretrained':
            enhanced,sum_total = featureloss_pretrained(clean,enhanced,keep_prob,loss_layers=SE_LOSS_LAYERS,n_layers=LOSS_LAYERS, norm_type=LOSS_NORM, base_channels=LOSS_BASE_CHANNELS,blk_channels=LOSS_BLK_CHANNELS,ksz=args.kernel_size)
            distance = sum_total
            loss_fn = distance
        else:
            enhanced,sum_total = featureloss(clean,enhanced,keep_prob,loss_layers=SE_LOSS_LAYERS,n_layers=LOSS_LAYERS, norm_type=LOSS_NORM, base_channels=LOSS_BASE_CHANNELS,blk_channels=LOSS_BLK_CHANNELS,ksz=args.kernel_size)
            distance = sum_total
            loss_fn = distance
        
# LOAD DATA
trainset, valset = load_full_data_list(datafolder = datafolder)
trainset, valset = load_full_data(trainset, valset)

# TRAINING OPTIMIZER
opt=tf.train.AdamOptimizer(learning_rate=args.learning_rate).\
    minimize(loss_fn,var_list=[var for var in tf.trainable_variables() if var.name.startswith("se_")])

# Log at tensorboard
with tf.name_scope('performance'):
    
    tf_loss_ph_train = tf.placeholder(tf.float32,shape=None,name='loss_summary_train')
    tf_loss_summary_train = tf.summary.scalar('loss_train', tf_loss_ph_train)
 
    tf_loss_ph_valid = tf.placeholder(tf.float32,shape=None,name='loss_summary_valid')
    tf_loss_summary_valid = tf.summary.scalar('loss_valid', tf_loss_ph_valid)
    
    tf_loss_ph_valid_distance = tf.placeholder(tf.float32,shape=None,name='loss_summary_valid_distance')
    tf_loss_summary_valid_distance = tf.summary.scalar('loss_valid_distance', tf_loss_ph_valid_distance)

performance_summaries_train = tf.summary.merge([tf_loss_summary_train])
performance_summaries_valid = tf.summary.merge([tf_loss_summary_valid])
performance_summaries_valid_distance = tf.summary.merge([tf_loss_summary_valid_distance])

# BEGIN SCRIPT #########################################################################################################

# INITIALIZE GPU CONFIG
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
sess=tf.Session(config=config)

print("Config ready")

sess.run(tf.global_variables_initializer())

print("Session initialized")

# LOAD FEATURE LOSS
if SE_LOSS_TYPE == "FL":
    if args.pretrained=='pretrained':
        loss_saver = tf.train.Saver([var for var in tf.trainable_variables() if (var.name.startswith("loss_") or var.name=="weights")])
        loss_saver.restore(sess, "%s/loss_model.ckpt" % modfolder)
    else:
        loss_saver = tf.train.Saver([var for var in tf.trainable_variables() if (var.name.startswith("loss_") or var.name=="weights")])
        loss_saver.restore(sess, "%s/my_test_model" % modfolder)

Nepochs = 400
saver = tf.train.Saver(var_list=[var for var in tf.trainable_variables() if var.name.startswith("se_")])

########################################################################################################################

if SE_LOSS_TYPE == "FL":
    loss_train = np.zeros((len(trainset["innames"]),SE_LOSS_LAYERS+1))
    loss_val = np.zeros((len(valset["innames"]),SE_LOSS_LAYERS+1))
else:
    loss_train = np.zeros((len(trainset["innames"]),1))
    loss_val = np.zeros((len(valset["innames"]),1))
    
os.mkdir(os.path.join('summaries',outfolder))
    
#####################################################################################
summ_writer = tf.summary.FileWriter(os.path.join('summaries',outfolder), sess.graph)
for epoch in range(1,Nepochs+1):

    print("Epoch no.%d"%epoch)
    # TRAINING EPOCH ################################################################

    ids = np.random.permutation(len(trainset["innames"])) # RANDOM FILE ORDER

    for id in tqdm(range(0, len(ids)), file=sys.stdout):

        i = ids[id] # RANDOMIZED ITERATION INDEX
        inputData = trainset["inaudio"][i] # LOAD DEGRADED INPUT
        outputData = trainset["outaudio"][i] # LOAD GROUND TRUTH

        # TRAINING ITERATION
        _, loss_vec = sess.run([opt, loss_fn],
                                feed_dict={input: inputData, clean: outputData})

        # SAVE ITERATION LOSS
        loss_train[id,0] = loss_vec
        
    # PRINT EPOCH TRAINING LOSS AVERAGE
    str1 = "T: %d\t " % (epoch)
    if SE_LOSS_TYPE == "FL":
        str1 += ", %10.6e"%(np.mean(loss_train, axis=0)[0])
    else:
        str1 += ", %10.6e"%(np.mean(loss_train, axis=0)[0])

    # SAVE MODEL EVERY N EPOCHS
    if epoch % SAVE_EPOCHS != 0:
        continue
    
    import time
    seconds = time.time()
    
    saver.save(sess, os.path.join('summaries',outfolder,'se_model_'+str(seconds)+'.ckpt'))
    
    # VALIDATION EPOCH ##############################################################

    print("Validation epoch")

    for id in tqdm(range(0, len(valset["innames"])), file=sys.stdout):

        i = id # NON-RANDOMIZED ITERATION INDEX
        inputData = valset["inaudio"][i] # LOAD DEGRADED INPUT
        outputData = valset["outaudio"][i] # LOAD GROUND TRUTH

        # VALIDATION ITERATION
        output, loss_vec = sess.run([enhanced, loss_fn],
                            feed_dict={input: inputData, clean: outputData})

        # SAVE ITERATION LOSS
        loss_val[id,0] = loss_vec

    
    # PRINT VALIDATION EPOCH LOSS AVERAGE
    str1 = "V: %d " % (epoch)
    if SE_LOSS_TYPE == "FL":
        str1 += ", %10.6e"%(np.mean(loss_val, axis=0)[0])
    else:
        str1 += ", %10.6e"%(np.mean(loss_val, axis=0)[0])

    summ_train = sess.run(performance_summaries_train, feed_dict={tf_loss_ph_train:np.mean(loss_train, axis=0)[0]})
    summ_writer.add_summary(summ_train, epoch)

    summ_valid = sess.run(performance_summaries_valid, feed_dict={tf_loss_ph_valid:np.mean(loss_val, axis=0)[0]*1e9})
    summ_writer.add_summary(summ_valid, epoch)