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


def lossnet_waveform(input, n_layers=14, training=True, reuse=False, norm_type="SBN",
               ksz=3, base_channels=32, blk_channels=5):
    layers = []

    if norm_type == "NM": # ADAPTIVE BATCH NORM
        norm_fn = nm
    elif norm_type == "SBN": # BATCH NORM
        norm_fn = slim.batch_norm
    else: # NO LAYER NORMALIZATION
        norm_fn = None

    for id in range(n_layers):

        n_channels = base_channels * (2 ** (id // blk_channels)) # UPDATE CHANNEL COUNT

        if id == 0:
            net = slim.conv2d(input, n_channels, [1, ksz], activation_fn=lrelu, normalizer_fn=norm_fn, stride=[1, 2],
                              scope='loss_conv_%d' % id, padding='SAME', reuse=reuse)
            layers.append(net)
        elif id < n_layers - 1:
            net = slim.conv2d(layers[-1], n_channels, [1, ksz], activation_fn=lrelu, normalizer_fn=norm_fn,
                              stride=[1, 2], scope='loss_conv_%d' % id, padding='SAME', reuse=reuse)
            layers.append(net)
        else:
            net = slim.conv2d(layers[-1], n_channels, [1, ksz], activation_fn=lrelu, normalizer_fn=norm_fn,
                              scope='loss_conv_%d' % id, padding='SAME', reuse=reuse)
            layers.append(net)

    return layers

def lossnet_spectrogram(input, n_layers=14, training=True, reuse=False, norm_type="SBN",
               ksz=3, base_channels=32, blk_channels=5):
    layers = []

    if norm_type == "NM": # ADAPTIVE BATCH NORM
        norm_fn = nm
    elif norm_type == "SBN": # BATCH NORM
        norm_fn = slim.batch_norm
    else: # NO LAYER NORMALIZATION
        norm_fn = None

    for id in range(n_layers):

        n_channels = base_channels * (2 ** (id // blk_channels)) # UPDATE CHANNEL COUNT

        if id == 0:
            net = slim.conv2d(input, n_channels, [ksz, ksz], activation_fn=lrelu, normalizer_fn=norm_fn, stride=[1, 2],
                              scope='loss_conv_%d' % id, padding='SAME', reuse=reuse)
            layers.append(net)
        elif id < n_layers - 1:
            net = slim.conv2d(layers[-1], n_channels, [ksz, ksz], activation_fn=lrelu, normalizer_fn=norm_fn,
                              stride=[1, 2], scope='loss_conv_%d' % id, padding='SAME', reuse=reuse)
            layers.append(net)
        else:
            net = slim.conv2d(layers[-1], n_channels, [ksz, ksz], activation_fn=lrelu, normalizer_fn=norm_fn,
                              scope='loss_conv_%d' % id, padding='SAME', reuse=reuse)
            layers.append(net)

    return layers


def featureloss_waveform(target, current, loss_weights, loss_layers, n_layers=14, norm_type="SBN", base_channels=32, blk_channels=5,ksz=3):

    feat_current = lossnet_waveform(current, reuse=False, n_layers=n_layers, norm_type=norm_type,
                         base_channels=base_channels, blk_channels=blk_channels,training=True,ksz=ksz)

    feat_target = lossnet_waveform(target, reuse=True, n_layers=n_layers, norm_type=norm_type,
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

def featureloss_spectrogram(target, current, loss_weights, loss_layers, n_layers=14, norm_type="SBN", base_channels=32, blk_channels=5,ksz=3):

    feat_current = lossnet_spectrogram(current, reuse=False, n_layers=n_layers, norm_type=norm_type,
                         base_channels=base_channels, blk_channels=blk_channels,training=True,ksz=ksz)

    feat_target = lossnet_spectrogram(target, reuse=True, n_layers=n_layers, norm_type=norm_type,
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