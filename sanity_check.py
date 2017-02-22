import tensorflow as tf
import numpy as np
from utils.image_loader import ImageLoader
from utils.utils import write_submission
from tqdm import tqdm
from utils.data_augmentation import ImageAugmentation
from models.cnn_v1.cnn_v1 import CNN_V1

def sanity_check():

    aug = ImageAugmentation()
    aug.add_random_flip_leftright()
    aug.add_random_brightness(0.3)
    aug.add_random_contrast(0.7, 1.3)
    aug.add_random_saturation(0.7, 1.3)

    batch_size = 128
    num_epochs = 1000

    #######################################################################
    imgloader = ImageLoader(imgtype = 'png', mode='folder', width = 128,
                             height = 32, color = 3, batch=128, imgAug = aug)
    imgloader.setup(folders=['../data/sample'], resize=True,
        task = 'local_split')
    #######################################################################

    #######################################################################
    """
    imgloader = ImageLoader(imgtype = 'jpg', mode='folder', width = 64,
        height = 128, color = 1, batch=batch_size, imgAug = None)
    #imgloader.setup(folders=['../data/bytes/5_Simda'], resize=True,
    #    task = 'local_split')

    #imgloader.setup(rootfolder='../data/bytes', resize=True,
    #    task = 'local_split')
    imgloader.setup(rootfolder='../data/bytes', resize=True,
        task = 'train_test', testfolder = '../data/bytes/9_Gatak')
    """
    #######################################################################


    #######################################################################
    #imgloader = ImageLoader(imgtype = 'jpg', mode='folder', width = 64,
    #    height = 64, color = 3, batch=batch_size, imgAug = None)

    #imgloader.setup(rootfolder='../../kaggle/fish/input/train', resize=True,
    #    task = 'local_split')

    #imgloader.setup(rootfolder='../../kaggle/fish/input/train', resize=True,
    #    task = 'train_test', testfolder = '../../kaggle/fish/input/test_stg1')
    #######################################################################

    imgs_valid, labels_valid = imgloader.get_batch_tensor(is_train=False)
    imgs_train, labels_train = imgloader.get_batch_tensor(is_train=True)
    model = CNN_V1()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        imgloader.init()
        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        count = 0
        num_epochs = (imgloader.get_num_samples(is_train=False)+batch_size-1)//batch_size

        for epoch_no in tqdm(range(10), desc='Epoch', maxinterval=86400, ncols=100):
            x,y = sess.run([imgst, labelst])
            print("\nTrain x",x.shape, y.shape)

        pred = []
        for epoch_no in tqdm(range(num_epochs), desc='Epoch', maxinterval=86400, ncols=100):
            x,y = sess.run([imgs, labels])
            count += x.shape[0]
            pred.append(np.argmax(y,axis=1))
            print("\n",x.shape, y.shape, type(x), type(y), pred[-1].shape)
            if count == imgloader.get_num_samples(is_train=False):
                break
            
        coord.request_stop()
        coord.join(threads)
        print (type(threads), threads)

    ids = ("img", imgloader.test_files)
    pred = ("predict", np.concatenate(pred)+1)
    #truth = [int(i.split('/')[-2].split('_')[0]) for i in imgloader.test_files]
    real = ("real", pred[1])
    print(len(ids[1]), pred[1].shape, len(real[1]))
    write_submission(ids, pred, real, "out.csv")
