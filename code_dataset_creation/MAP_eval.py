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
   
def scores_map(noise='mp3',model_input):
    
    filename='dataset_test_'+noise+'.txt'
    #saver = tf.train.Saver()
    import numpy as np
    if noise!='combined':
        dataset_test=load_full_data_list_test('/n/fs/percepaudio/www/mturk_hosts/website_all_perturbations/',filename)
    elif noise=='combined':
        dataset_test=load_full_data_list_combined_test('/n/fs/percepaudio/www/mturk_hosts/website_all_perturbations/',filename)
    
    output=np.zeros((len(dataset_test["all"]["inname"]),1))
    for id in tqdm(range(0, len(dataset_test["all"]["inname"]))):
        
        loss_ones=np.ones([SE_LOSS_LAYERS])
        if args.model_input=='waveform':
            wav_in,wav_out=load_full_data_test_waveform(dataset_test,'all',id)
            a,_= sess.run([distance,enhanced1],feed_dict={input1_wav:wav_in, clean1_wav:wav_out, loss_weights:loss_ones})
        elif args.model_input=='logmelspec':
            spec_in,spec_out=load_full_data_test_spectrogram(dataset_test,'all',id)
            a,_= sess.run([distance,enhanced1],feed_dict={input1_spec:spec_in, clean1_spec:spec_out, loss_weights:loss_ones})
        elif args.model_input=='downsampled':
            spec_in,spec_out,wav_in,wav_out=load_full_data_test_both(dataset_test,'all',j)
            a,_= sess.run([distance,enhanced1],feed_dict={input1_spec:spec_in, clean1_spec:spec_out, input1_wav:wav_in, clean1_wav:wav_out,loss_weights:loss_ones})
        output[id]=a[0]
        

    import numpy as np
    perceptual=[]
    for i in range(len(dataset_test['all']['label'])):
        perceptual.append(float(dataset_test['all']['label'][i]))
    perceptual=(np.array(perceptual))
    perceptual=1-perceptual

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