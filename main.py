import tensorflow as tf
import os
from utils.image_loader import ImageLoader
from utils.data_augmentation import ImageAugmentation
from utils.utils import write_prob_submission, get_global_mean_std
from models.cnn_v1.malware_cnn_v1 import MALWARE_CNN_V1

flags = tf.app.flags
flags.DEFINE_string('model_name', 'MW_cnn_v1', 'Model type - cnn_v1 [Default: cnn_v1]')
flags.DEFINE_integer('batch_size', 128, 'Batch size during training and testing [128]')
flags.DEFINE_string('summary_verbose', '', 'Summary verbose')
flags.DEFINE_string('image_type', 'jpg', 'Image type - jpg, png, jpeg [Default: jpg]')
flags.DEFINE_integer('width', 128, 'Image width resize to [Default: 128]')
flags.DEFINE_integer('height', 128, 'Image height resize to [Default: 128]')
flags.DEFINE_integer('color', 1, 'Image color [Default: 1]')
flags.DEFINE_string('task', 'local_validation', 'Task - [Default: local_validation]')
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate [0.01]')
flags.DEFINE_float('weight_decay', 0.001, 'Weight decay - 0 to turn off L2 regularization [0.001]')
flags.DEFINE_integer('num_epochs', 1, 'Number of epochs [Default: 1]')
flags.DEFINE_integer('test_samples', 1, 'Number of test samples [Default: 1]')
flags.DEFINE_integer('train_samples', 1, 'Number of train samples [Default: 1]')
flags.DEFINE_string('rootfolder', '../input/train', 'root folder of images - [Default: ../data]')
flags.DEFINE_string('testfolder', '../input/test_stg1', 'folder of test images - [Default: ../data/test]')
flags.DEFINE_string('data_mode', 'folder', 'how data is organized - [Default: folder]')
flags.DEFINE_string('folders', '', 'folders of images - [Default: ]')
flags.DEFINE_string('cache', 'cache', 'cache folder for binaries and metadata - [Default: cache]')
flags.DEFINE_string('logs', 'logs', 'log folder for tensorboard - [Default: logs]')
flags.DEFINE_string('norm_mode', 'per_color', 'Modes of normalization - [Default: per_color]')
flags.DEFINE_string('load_model', '', 'Path of models to load - [Default: ]')
flags.DEFINE_string('save_model', '', 'Path of models to save - [Default: ]')
flags.DEFINE_integer('test_only', 0, 'Just test [Default: 0]')
flags.DEFINE_integer('validation', 1, 'do validation [Default: 1]')

FLAGS = flags.FLAGS

def get_imgAug():
    aug = ImageAugmentation()
    aug.add_random_flip_leftright()
    aug.add_random_brightness(0.3)
    aug.add_random_contrast(0.7, 1.3)
    aug.add_random_saturation(0.7, 1.3)
    return aug

def main(_):
    imgloader = ImageLoader(imgtype = FLAGS.image_type, mode=FLAGS.data_mode, width = FLAGS.width,
        height = FLAGS.height, color = FLAGS.color, batch=FLAGS.batch_size, imgAug = None)
    """
    !!! imgAug = None for the 1st pass !!!
    for getting mean and std
    """
    imgloader.setup(rootfolder=FLAGS.rootfolder, resize=True,
        task = FLAGS.task, testfolder=FLAGS.testfolder)
   
    FLAGS.test_samples = imgloader.get_num_samples(is_train=False)
    FLAGS.train_samples = imgloader.get_num_samples(is_train=True)
    train_batch = imgloader.get_batch_tensor("train")
    valid_batch = imgloader.get_batch_tensor("valid")
    test_batch = imgloader.get_batch_tensor("test")
    mean, std = get_global_mean_std(FLAGS, train_batch, imgloader)
    print("{} Mean {} Std {}".format(FLAGS.norm_mode, mean, std))
    FLAGS.mean = mean
    FLAGS.std = std
    #saver = tf.train.Saver(tf.trainable_variables()) 
    with tf.Session() as sess:
        model = MALWARE_CNN_V1(FLAGS)
        model.build(train_batch, valid_batch, test_batch)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        imgloader.init()
        # Start populating the filename queue.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        epochs = 0
        if FLAGS.load_model!='':
            checkpoint = tf.train.get_checkpoint_state(FLAGS.load_model)
            cpath = checkpoint.model_checkpoint_path
            saver.restore(sess, cpath)
            epochs = int(cpath.split('.')[-1].split('-')[-1])
        if not FLAGS.test_only:
            model.train(sess)
            if FLAGS.save_model!='':
                if FLAGS.task == 'local_validation':
                    path = '%s/loss_%.3f'%(FLAGS.save_model, model.va_loss)
                else:
                    path = FLAGS.save_model
                saver.save(sess, path, FLAGS.num_epochs + epochs)
        pred = model.test(sess)

        coord.request_stop()
        coord.join(threads)
    imgs = [i.split('/')[-1] for i in imgloader.test_files]
    if FLAGS.task == 'local_validation':
        truth = [imgloader.class2num[i.split('/')[-2]] for i in imgloader.test_files]
        write_prob_submission(['image', imgs],
                pred, imgloader.classes, ['real', truth], 'result/cv.csv')
    elif FLAGS.task == 'train_test':
        write_prob_submission(ids = ['image', imgs],
                prob = pred, classes = imgloader.classes, out = 'result/sub.csv')
if __name__ == '__main__':

    tf.app.run()
