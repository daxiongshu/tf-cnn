from tqdm import tqdm
from termcolor import colored
import tensorflow as tf
import numpy as np
from tflearn import summarize_activations, summarize_gradients, summarize_variables
class BaseModel(object):

    def __init__(self, params):
        self.params = params

    def build(self, train_batch, valid_batch = None, test_batch = None):
        """
        train_batch/test_batch : tensor!!
        """
        model_name = self.params.model_name
        tf.GraphKeys.MYACTIVATIONS = 'myactivations'
        tf.GraphKeys.MYGRAPHS = 'mygraphs'
        tf.GraphKeys.MYSCALARS = 'myscalars'
        with tf.variable_scope(model_name) as scope:

            self.global_step = tf.Variable(0, name='global_step',
                    trainable=False)
            self._build_network(train_batch, type_ = 'train')
            if valid_batch is not None:
                scope.reuse_variables()
                self._build_network(valid_batch, type_ = 'valid')
            if test_batch is not None:
                scope.reuse_variables()
                self._build_network(test_batch, type_ = 'test') 
            print("Build %s done!"%model_name)        
            self.create_summaries()

    def _build_network(self, batch, type_ = 'train'):
        """
        batch: (image_batch, label_batch)
        """
        raise(NotImplementedError)

    def train_batch(self, sess):
        if self.summ_op is not None:
            return sess.run([self.loss, self.acc, self.global_step,
                self.opt_op, self.summ_op])
        else:
            return sess.run([self.loss, self.acc, self.global_step,
                self.opt_op])

    def test_batch(self, sess):
        return sess.run(self.prediction)

    def validate_batch(self, sess):
        return sess.run([self.valid_loss, self.valid_acc])

    def train(self, sess):
        params = self.params
        num_epochs = params.num_epochs
        print("\nTraining %d epochs ...\n"%num_epochs)
        count = 0
        summary_writer = tf.summary.FileWriter(params.logs, sess.graph)
        for epoch_no in tqdm(range(num_epochs), desc='Epoch', maxinterval=86400, ncols=100):
            tr_loss, tr_acc = [], []
            while count<params.train_samples:
                if self.summ_op is not None:
                    loss, acc, global_step, _, summary = self.train_batch(sess)
                    summary_writer.add_summary(summary, global_step)
                else:
                    loss, acc, global_step, _ = self.train_batch(sess)

                tr_loss.append(loss)
                tr_acc.append(acc)
                count += params.batch_size
            print(colored("\nEpoch %d Step %d Training loss %f Acc %f"%(
                epoch_no, global_step, np.mean(tr_loss), np.mean(tr_acc)), "green"))

            if True:
                va_loss, va_acc = [], []
                count = 0
                while count<params.test_samples:
                    loss, acc = self.validate_batch(sess)
                    va_loss.append(loss)
                    va_acc.append(acc)
                    count += params.batch_size
                print(colored("Epoch %d Step %d Validation loss %f Acc %f"%(
                    epoch_no, global_step, np.mean(va_loss), np.mean(va_acc)), "yellow"))
                self.va_loss = np.mean(va_loss)

        self.gstep = global_step

    def test(self, sess):
        params = self.params
        print("\nTesting ...")
        count = 0
        pred = []
        while count<params.test_samples:
            yp = self.test_batch(sess)
            pred.append(yp)
            count += params.batch_size
        return np.vstack(pred)

    def create_summaries(self):

        """ Create summaries with `verbose` level """

        verbose = self.params.summary_verbose
        opts = verbose.split(',')
        summ_collection = self.params.model_name + "_training_summaries"
        #if self.image_summary is not None:
        if "images" in opts:
            for i in tf.get_collection(tf.GraphKeys.MYGRAPHS):
                tf.add_to_collection(summ_collection, i)
        
        for i in tf.get_collection(tf.GraphKeys.MYSCALARS):
            tf.add_to_collection(summ_collection, i)

        if "activation" in opts:
            for i in tf.get_collection(tf.GraphKeys.MYACTIVATIONS):
                tf.add_to_collection(summ_collection, i)

        if "weights" in opts:
            summarize_variables(None, summ_collection)

        if len(verbose)<=1:
            self.summ_op = None
        else:
            self.summ_op = tf.summary.merge(tf.get_collection(summ_collection))


