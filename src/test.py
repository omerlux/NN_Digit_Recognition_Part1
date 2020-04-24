import numpy as np
import gzip
import matplotlib.pyplot as plot
import mnist_dreader
import network_model

dreader = mnist_dreader.dataReader()  # constructing mnist data reader object

# # Test 1 - circular batches
# for i in range(1, 12):
#     a = dreader.get_batch(1000, 'train')  # test for modulo batch

# # Test 2 - shuffle train
# fig = plot.figure()
# for i in range(0, 3):
#     a = dreader.train[i][0]
#     plot.imshow(a)  # test random
#     plot.show()
# dreader.shuffle_train()
# for i in range(0, 3):
#     a = dreader.train[i][0]
#     plot.imshow(a)  # test random
#     plot.show()
# plot.close(fig)

# # Test 3 - print batch num and label
# batch = dreader.get_batch(5, 'train')
# for i in range(0, 5):
#     a = batch[i][0]
#     plot.imshow(a)
#     plot.show()
#     print(batch[i][1])

network = network_model.network([784, 5, 10])
batch = dreader.get_batch(10, 'train')
[mean_square_error, negative_log_loss] = network.feed_forward(batch)


