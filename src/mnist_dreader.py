import gzip
import numpy as np
import matplotlib.pyplot as plot
# from mnist import MNIST


class dataReader(object):
    def __init__(self):
        print ('Initializing data reader object...')
        # mndata = MNIST('samples')
        # images, labels = mndata.load_training()
        # self.train(images[0:55000], labels[0:55000])
        # self.valid(images[55000:60000], labels[55000:60000])
        # images, labels = mndata.load_testing()
        # self.test(images[0:10000], labels[0:10000])


        image_size = 28  # each image is 28x28

        num_images = 60000  # there are 60k images
        f = gzip.open('../samples/train-images-idx3-ubyte.gz', 'r')  # 60k train & valid
        f.read(16)  # reading by 16-byte double
        buffer = f.read(image_size * image_size * num_images)
        data = np.frombuffer(buffer, dtype=np.uint8).astype(np.int32)  # translating into 0 to 255
        data = data.reshape(num_images, image_size, image_size, 1).squeeze()  # data = 60k x 28 x 28 with 1 value in it
        training_55k_x = data[0:55000]
        validation_5k_x = data[55000: 60000]
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
        test_10k_x = data[0: 10000]
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

    def get_batch(self, m, type):  # m is the number of samples in one batch, type is the type of the samples
        self.batchNum += 1
        switcher = {
            'train': self.train,
            'valid': self.valid,
            'test': self.test
        }
        max_batch_num = np.floor(len(switcher.get(type)) / m)  # maximum different batches
        k = int((self.batchNum % max_batch_num) * m)  # index is modulo from 0 to the length of the samples array
        l = min(k + m, len(switcher.get(type)))       # maximum index is the length of the array
        print ('Getting batch examples of {} sized {}, from [{},{}].'.format(type, m, k, l))
        return switcher.get(type)[k:l]   # return the batch

    def shuffle_train(self):    # shuffle the train exmaples only
        from random import shuffle
        shuffle(self.train)
