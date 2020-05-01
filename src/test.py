from __future__ import division

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

network = network_model.network([784, 30, 10])

# # Test 4 - feew_forward
# batch = dreader.get_batch(10, 'train')
# [output, mean_square_error, negative_log_loss] = network.feed_forward(batch)

# # Test 5 - back_prop checking - debuging gradient
# batch = dreader.get_batch(30, 'train')
# # nabla_wj ~ (C(w + epsilon*e_j) - C(w))/epsilon    *when e_j is the unit vector j
# gradient_biases, gradient_weights, loss_mse, loss_nll = network.back_prop(batch)   # should be the real derivative of the w
# epsilon = 10**-4
# gradient_w_avg_relative_error = 0
# gradient_w_sum_relative_error = 0
# max_error = 0
# i = 0
# for w_layer in range (0, len(network.weights)):
#     for w_neuron in range (0, len(network.weights[w_layer])):
#         for w in range (0, len(network.weights[w_layer][w_neuron])):
#             # for each weight, will add epsilon for the w only, and calculate (C(w + epsilon*e_j) - C(w))/epsilon
#             cost_func_normal = network.feed_forward(batch)[1]       # normal cost
#             network.weights[w_layer][w_neuron][w] += epsilon
#             cost_func_epsilon = network.feed_forward(batch)[1]      # epsilon cost
#             network.weights[w_layer][w_neuron][w] -= epsilon
#
#             estimated_gradient = (cost_func_epsilon - cost_func_normal)/epsilon     # estimated gradient
#             backprop_gradient = gradient_weights[w_layer][w_neuron][w]              # backpropagation gradient
#
#             if backprop_gradient != estimated_gradient:
#                 i += 1
#                 if estimated_gradient == 0:
#                     gradient_w_sum_relative_error += backprop_gradient             # estimated is 0, relative error isn't right
#                     max_error = max(max_error, backprop_gradient)
#                 else:
#                     relative_error = np.abs(estimated_gradient - backprop_gradient) / np.abs(estimated_gradient + backprop_gradient)  # |X-Y|/|X+Y|
#                     gradient_w_sum_relative_error += relative_error
#         gradient_w_avg_relative_error = gradient_w_sum_relative_error / i         # very low number -> estimation and real are close
# print(max_error)


