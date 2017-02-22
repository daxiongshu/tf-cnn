import tensorflow as tf
import os
from utils.utils import get_all_files, split_data 
from tensorflow.python.framework import dtypes
class ImageLoader:

    def __init__(self, mode="folder", imgtype = "jpg", 
            width = 32, height = 32, color = 1,
            batch = 128, imgAug = None):        
        self.mode = mode
        self.imgtype = imgtype
        self.width, self.height, self.color = width,height,color
        self.batch = batch
        self.imgAug = imgAug
        self.help()
        return

    def init(self):
        self.table.init.run()

    def help(self, entry = "all"):
        print("\nImage Loader:")
        if entry == "mode" or entry == "all":
            if self.mode == "folder":
                print("Mode: folder")
                print("Please place images in separate folders named by their class names")
        print("Resize Width: %d Height: %d Color: %d"%(self.width,self.height,self.color))
        print()        

    def setup(self, folders = [], rootfolder = None, testfolder = 'test', 
            resize=True, task = 'train_test', split_ratio = 0.3):

        """
        task : 'train_test' or 'local_split'
        folders/rootfolder/testfolder: relative path!!

        """

        imgtype = self.imgtype
        mode = self.mode

        if mode == 'folder':
            files = get_all_files(folders=folders, imgtype=imgtype, 
                    rootfolder=rootfolder, testfolder=testfolder)
            assert len(files)
            if len(folders) == 0:
                folders = os.listdir(rootfolder)
                self.classes = sorted([i for i in folders if os.path.isdir('%s/%s'%(rootfolder,i)) 
                    and i!=testfolder.split('/')[-1]])
            else:
                self.classes = [i.split('/')[-1] for i in folders]
                self.classes = [i for i in self.classes if i!=testfolder.split('/')[-1]]

            self.num_classes = len(self.classes)
            print("Classes: %s"%self.classes)
            print("Encoding: %s"%(' '.join(['%s:%d'%(i,c) for c,i in enumerate(self.classes)])))
            self.class2num = {}
            for c,i in enumerate(self.classes):
                self.class2num[i] = c
            # setup lookup table to class name encoding     
            keys = tf.constant(self.classes)
            values = tf.constant(list(range(self.num_classes)), tf.int64)
            self.table = tf.contrib.lookup.HashTable(
                tf.contrib.lookup.KeyValueTensorInitializer(keys, values,
                dtypes.string, dtypes.int64), 0)  
            # !!! test data will be labeled as class 0 !!!
            # !!! this label is not used !!!

        else:
            print("Unsupported loading mode: %s"%mode)
            raise(NotImplementedError)

        if task == 'local_validation':
            self.train_files, self.test_files = split_data(files, split_ratio)

        elif task == 'train_test':
            self.train_files = files
            self.test_files = get_all_files(folders=[testfolder], imgtype=imgtype,
                    rootfolder=None, testfolder='')
        else:
            raise(NotImplementedError)

        print("Number of Files: train %d test %d\n"%(len(self.train_files), len(self.test_files))) 

        self.train_image_tensor, self.train_label_tensor = self.build_single_image_tensor(
            self.train_files, imgAug = self.imgAug, resize=resize)

        self.valid_image_tensor, self.valid_label_tensor = self.build_single_image_tensor(
            self.test_files, imgAug = None, resize=resize)

        self.test_image_tensor, self.test_label_tensor = self.build_single_image_tensor(
            self.test_files, imgAug = None, resize=resize, epochs = 1, 
            shuffle = False)

        self.train_images_batch, self.train_labels_batch = self.build_image_batch_tensor(
            self.train_image_tensor, self.train_label_tensor, is_test=False)

        self.valid_images_batch, self.valid_labels_batch = self.build_image_batch_tensor(
            self.valid_image_tensor, self.valid_label_tensor, is_test=False)

        self.test_images_batch, self.test_labels_batch = self.build_image_batch_tensor(
            self.test_image_tensor, self.test_label_tensor, is_test=True)
    
    def build_single_image_tensor(self, files, imgAug = None, resize = True, 
            epochs=None, shuffle=True):

        filequeue = tf.train.string_input_producer(files, num_epochs=epochs, shuffle = shuffle)
        reader = tf.WholeFileReader()
    
        key, value = reader.read(filequeue)
        imgtype = self.imgtype
        mode = self.mode

        if imgtype == 'jpg' or imgtype == 'jpeg':
            my_img = tf.image.decode_jpeg(value, channels = self.color)
        elif imgtype == 'png':
            my_img = tf.image.decode_png(value, channels = self.color)
        else:
            print("Unsupported image type: %s"%imgtype)
            raise(NotImplementedError)

        # encode image classname to a number
        if mode == 'folder':
            key = tf.string_split(tf.reshape(key,[1]), delimiter='/').values[-2]
            key = self.table.lookup(key)
            label_tensor = key
        else:
            raise(NotImplementedError)

        if resize:
            img_tensor = tf.image.resize_images(my_img, [self.height, self.width], method = 1)
        else:
            img_tensor = my_img

        if imgAug:
            img_tensor = self.imgAug.apply(img_tensor)

        return img_tensor, label_tensor


    def get_image_tensor(self, type_ = 'train'):
        if type_ == "train":
            return self.train_image_tensor, self.train_label_tensor
        elif type_ == 'test':
            return self.test_image_tensor, self.test_label_tensor
        elif type_ == 'valid':
            return self.valid_image_tensor, self.valid_label_tensor

        
    def get_batch_tensor(self, type_ = 'train'):

        if type_ == 'train':
            return self.train_images_batch, self.train_labels_batch
        elif type_ == 'test':
            return self.test_images_batch, self.test_labels_batch
        elif type_ == 'valid':
            return self.valid_images_batch, self.valid_labels_batch

    def build_image_batch_tensor(self, image_tensor, label_tensor, is_test = False):

        label = tf.one_hot(label_tensor, depth = self.num_classes, dtype=tf.int32)
        image = tf.reshape(image_tensor, [self.height, self.width, self.color])
        label = tf.reshape(label, [self.num_classes])

        if is_test == False:
            images_batch, labels_batch = tf.train.shuffle_batch(
                [image, label], batch_size=self.batch,
                capacity=self.batch*6, num_threads=4,
                min_after_dequeue=self.batch*3, allow_smaller_final_batch=False)
        else:
            images_batch, labels_batch = tf.train.batch([image, label], batch_size=self.batch, 
                num_threads=1, capacity=self.batch*3, allow_smaller_final_batch=True)

        return images_batch, labels_batch

    def get_num_samples(self, is_train=False):
        if is_train:
            return len(self.train_files)
        else:
            return len(self.test_files)


