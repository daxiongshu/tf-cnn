import tensorflow as tf
from random import uniform
class DataAugmentation(object):
    def __init__(self):
        self.methods = []
        self.args = []

    def apply(self, batch):
        # batch is a image batch tensor
        for i,m in enumerate(self.methods):
            if self.args[i]:
                batch = m(batch, *self.args[i])
            else:
                batch = m(batch)

        return batch

class ImageAugmentation(DataAugmentation):

    def __init__(self, batched=False):
        super(ImageAugmentation, self).__init__()
        self.batched = batched

    def add_random_flip_leftright(self):
        self.methods.append(self._random_flip_leftright)
        self.args.append(None)

    def add_random_brightness(self, delta):
        self.methods.append(self._random_brightness)
        self.args.append([delta])

    def add_random_contrast(self, minc,maxc):
        self.methods.append(self._random_contrast)
        self.args.append([minc,maxc])

    def add_random_saturation(self, low, up):
        self.methods.append(self._random_saturation)
        self.args.append([low, up]) 
           

    def _random_flip_leftright(self, batch):
        if not self.batched:
            return tf.image.random_flip_left_right(batch)

        batch = tf.map_fn(lambda x: tf.image.random_flip_left_right(x),
                batch)
        return batch

    def _random_brightness(self, batch, delta):
        if not self.batched:
            return tf.image.random_brightness(batch,delta)

        batch = tf.map_fn(lambda x: tf.image.random_brightness(x,delta),
                batch)
        return batch

    def _random_contrast(self,batch,minc,maxc):
        if not self.batched:
            return tf.image.random_contrast(batch, minc, maxc)

        batch = tf.map_fn(lambda x: tf.image.random_contrast(x, minc, maxc),
                batch)
        return batch

    def _random_saturation(self, batch, low, up):
        if not self.batched:
            return tf.image.random_contrast(batch, low, up)

        batch = tf.map_fn(lambda x: tf.image.random_contrast(x, low, up),
                batch)
        return batch

