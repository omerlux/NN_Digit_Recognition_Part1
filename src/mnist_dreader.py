import gzip
import numpy as np
import matplotlib.pyplot as plot
# from mnist import MNIST


class dataReader(object):
    def __init__(self):
        """ Intializing the data reader object.
        55k train with labels,
        5k valid with labels
        and 10k test with labels"""
        print ('Initializing data reader object...')
        image_size = 28  # each image is 28x28

        num_images = 60000  # there are 60k images
        f = gzip.open('../samples/train-images-idx3-ubyte.gz', 'r')  # 60k train & valid
        f.read(16)  # reading by 16-byte double
        buffer = f.read(image_size * image_size * num_images)
        data = np.frombuffer(buffer, dtype=np.uint8).astype(np.int32)  # translating into 0 to 255
        data = data.reshape(num_images, image_size, image_size, 1).squeeze()  # data = 60k x 28 x 28 with 1 value in it
        training_55k_x = [a/255.0 for a in data[0:55000]]            # vector normalized
        validation_5k_x = [a/255.0 for a in data[55000: 60000]]
        f.close()
        f = gzip.open('../samples/train-labels-idx1-ubyte.gz', 'r')  # 60k train & valid - labels
        f.read(8)  # reading by 16-byte double
        buffer = f.read(num_images)
        data = np.frombuffer(buffer, dtype=np.uint8).astype(np.int32)  # translating into 0 to 255
        training_55k_y = data[0:55000]
        validation_5k_y = data[55000: 60000]
        f.close()

        self.train = zip(training_55k_x, training_55k_y)
        self.valid = zip(validation_5k_x, validation_5k_y)

        num_images = 10000  # there are 10k images
        f = gzip.open('../samples/t10k-images-idx3-ubyte.gz', 'r')  # 10k tests
        f.read(16)  # reading by 16-byte double
        buffer = f.read(image_size * image_size * num_images)
        data = np.frombuffer(buffer, dtype=np.uint8).astype(np.uint8)  # translating into 0 to 255
        data = data.reshape(num_images, image_size, image_size, 1).squeeze()  # data = 60k x 28 x 28 with
        test_10k_x = [a/255.0 for a in data[0: 10000]]
        f.close()
        f = gzip.open('../samples/t10k-labels-idx1-ubyte.gz', 'r')  # 10k tests - lbles
        f.read(8)  # reading by 16-byte double
        buffer = f.read(num_images)
        data = np.frombuffer(buffer, dtype=np.uint8).astype(np.int32)  # translating into 0 to 255
        test_10k_y = data[0: 10000]
        f.close()

        self.test = zip(test_10k_x, test_10k_y)

        self.batchNum = -1
        print ('Initialized!')

    def get_batch(self, m, typ):  # m is the number of samples in one batch, type is the type of the samples
        """ m - the amount of the examples in a batch,
        typ - what kind of examples it'll return,
        :return a list of m examples of the specified type"""
        self.batchNum += 1
        switcher = {
            'train': self.train,
            'valid': self.valid,
            'test': self.test
        }
        max_batch_num = np.floor(len(switcher.get(typ)) / m)  # maximum different batches
        k = int((self.batchNum % max_batch_num) * m)  # index is modulo from 0 to the length of the samples array
        l = min(k + m, len(switcher.get(typ)))       # maximum index is the length of the array
        # print ('Getting batch examples of {} sized {}, from [{},{}].'.format(typ, m, k, l))
        return switcher.get(typ)[k:l]   # return the batch

    def shuffle_train(self):    # shuffle the train exmaples only
        """ Shuffles the 55k train vector. Will change the next batch that will be given."""
        from random import shuffle
        shuffle(self.train)

    def reset_batch_num(self):
        self.batchNum = -1
