import os
from random import shuffle
import pandas as pd
import tensorflow as tf
import numpy as np

def get_global_mean_std(params, train_batch = None, imgloader = None):
    """
    Input: 
        params: tf.flags with image location
        train_batch: a tensor for training data
        !!! warning, turn off augmentation in generating train_batch for 1st pass !!!
    Return: 
        mean: a list of length c
        std: a list of length c
        where c is number of colors
    """
    mode = params.norm_mode

    if mode == 'per_color':
        if params.data_mode == 'folder':
            if params.folders == '':
                # change ../../data/bytes to 1-1-data-bytes
                folder = params.rootfolder.split('/')
                folder = [i if '.' not in i else '1' for i in folder]
                name = '-'.join(folder)

        filename = '%s/%s_%s_mean_std.csv'%(params.cache, name, mode)
        if os.path.exists(filename):
            s = pd.read_csv(filename)
            mean = s['mean'].tolist()
            std = s['std'].tolist()
        else:
            print("\nMean-Std not existd. Calculating ...")
            mean, std = run_per_color_mean_std(train_batch, imgloader)
            s = pd.DataFrame({"mean":mean, "std":std})
            s.to_csv(filename,index=False)

        return mean, std
    else:
        raise(NotImplementedError)

def run_per_color_mean_std(train_batch, imgloader):
    """
    train_batch: 4-D tensor with NHWC format
    """
    image_batch, _ = train_batch
    image_batch = tf.cast(image_batch, tf.float32)
    ysum = tf.reduce_mean(image_batch, axis=[0,1,2])
    ysum2 = tf.reduce_mean(image_batch*image_batch, axis=[0,1,2])
    rsum, rsum2 = [],[]
    NUM = imgloader.get_num_samples(is_train=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        imgloader.init()
        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        count = 0
        while count<NUM:
            sum_, sum2_ = sess.run([ysum, ysum2])
            rsum.append(sum_)
            rsum2.append(sum2_)
            count += imgloader.batch
            if count%1024 == 0:
                print("Processed %d images"%count, sum_, sum2_)
        print()
        coord.request_stop()
        coord.join(threads)
    rsum, rsum2 = np.vstack(rsum), np.vstack(rsum2)
    #print(rsum.shape, rsum2.shape)
    mean = np.mean(rsum, axis=0) #*1.0 / imgloader.batch
    mean2 = np.mean(rsum2, axis=0) #*1.0 / imgloader.batch
    var = mean2 - mean*mean
    std = var**0.5
    return mean, std


def get_all_files(folders=[], imgtype='jpg', rootfolder=None, testfolder=''):
    if rootfolder is not None:
        folders = ['%s/%s'%(rootfolder,i) for i in os.listdir(rootfolder) 
                if os.path.isdir('%s/%s'%(rootfolder,i)) and i!=testfolder.split('/')[-1]]
    data = []
    for f in folders:
        data.extend(['%s/%s'%(f,i) for i in os.listdir(f) if i.endswith('.%s'%imgtype)])
    return data

def split_data(files, ratio):
    labels = [i.split('/')[-2] for i in files]
    dic = {}
    for i,j in zip(labels,files):
        if i not in dic:
            dic[i] = []
        dic[i].append(j)

    for i in dic:
        shuffle(dic[i])

    train = []
    test = []
    for i in dic:
        bar = int(len(dic[i])*ratio)
        #print("length %d bar %d"%(len(dic)))
        test.extend(dic[i][:bar])
        train.extend(dic[i][bar:])
    return train, test

def write_submission(ids, pred, real=None, out='out.csv'):
    """
    ids: ("id", id_array)
    """
    dic = {ids[0]:ids[1], pred[0]:pred[1]}
    if real:
        dic[real[0]] = real[1]
    s = pd.DataFrame(dic)
    s.to_csv(out, index=False)


def write_prob_submission(ids, prob, classes, real = None, out='prob.csv'):
    s = pd.DataFrame(prob, columns = classes)
    s[ids[0]] = ids[1]
    if real:
        s[real[0]] = real[1]
    s.to_csv(out, index=False)
