from __future__ import division

import random as rnd
import numpy as np
import matplotlib.pyplot as plot
import mnist_dreader
import network_model
import sgd_optimizer

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

# network = network_model.network([784, 400, 200, 50, 10])  # making the network

# # Test 4 - feew_forward
# batch = dreader.get_batch(10, 'train')
# [output, mean_square_error, negative_log_loss] = network.feed_forward(batch, 0)

# # Test 5 - back_prop checking - debuging gradient
# batch = dreader.get_batch(30, 'train')
# # nabla_wj ~ (C(w + epsilon*e_j) - C(w))/epsilon    *when e_j is the unit vector j
# gradient_biases, gradient_weights = network.back_prop(batch)  # should be the real derivative of the w
# epsilon = 10 ** -4
# gradient_w_avg_relative_error = 0
# gradient_w_sum_relative_error = 0
# max_relative_error = 0
#
# for j in range(0, (784 * 30 + 30 * 10) // 10):  # 10% of the weights
#     w_layer = rnd.randint(0, len(network.weights) - 1)  # random layer
#     w_neuron = rnd.randint(0, len(network.weights[w_layer]) - 1)  # random neuron
#     w = rnd.randint(0, len(network.weights[w_layer][w_neuron]) - 1)  # random weight
#     # for each weight, will add epsilon for the w only, and calculate (C(w + epsilon*e_j) - C(w))/epsilon
#     out, _, _ = network.feed_forward(batch, 0)  # normal cost
#     network.weights[w_layer][w_neuron][w] += epsilon
#     out_epsilon, _, _ = network.feed_forward(batch, 0)  # epsilon cost
#     network.weights[w_layer][w_neuron][w] -= epsilon
#
#     # calculating the quadratic cost function for a batch (with or without epsilon)
#     cost_func_normal = np.sum([network_model.cost_func_single(a, y) for a, (_, y) in zip(out, batch)])
#     cost_func_epsilon = np.sum([network_model.cost_func_single(a, y) for a, (_, y) in zip(out_epsilon, batch)])
#
#     estimated_gradient = (cost_func_epsilon - cost_func_normal) / epsilon  # estimated gradient
#     backprop_gradient = gradient_weights[w_layer][w_neuron][w]  # backpropagation gradient
#
#     if backprop_gradient != estimated_gradient:
#         relative_error = (np.abs(estimated_gradient - backprop_gradient)
#                           / np.abs(estimated_gradient + backprop_gradient))  # |X-Y|/|X+Y|
#         gradient_w_sum_relative_error += relative_error
#         max_relative_error = max(max_relative_error, relative_error)
#         if j != 0:
#             gradient_w_avg_relative_error = gradient_w_sum_relative_error / j  # very low number -> estimation and real are close
#
# print(gradient_w_avg_relative_error)
# print(max_relative_error)

# # Working learning MNIST #1
# network = network_model.network([784, 400, 200, 50, 10])  # making the network
# learning_rate = 3.0
# batch_size = 10
# epochs = 10
# # making Stochastic gradient descent optimizer
# SGD_optimizer = sgd_optimizer.SGD(network, learning_rate, batch_size, epochs)
# SGD_optimizer.training_program()

# # Comparing SGD to Momentum
# batch_size = 100
# epochs = 10
#
# # Success test - do not delete - testing the no-improv ETA change by 3
# epochs = 20
# learning_rate = 0.5        # recommended
#
# network_SGD_Confidence_Penalty = \
#     network_model.network([784, 30, 10], 'cross-entropy', regularization='Confidence Penalty', reg_lambda=0.3)  # making the network SGD
# SGD_optimizer = sgd_optimizer.SGD(network_SGD_Confidence_Penalty, learning_rate, batch_size, epochs)
# SGD_optimizer.training_program('SGD')   # SGD optimizer chosengit
#
# network_SGD_Weights_Constraints = \
#     network_model.network([784, 30, 10], 'cross-entropy', regularization='Weights Constraints', reg_lambda=1.5)  # making the network SGD
# SGD_optimizer = sgd_optimizer.SGD(network_SGD_Weights_Constraints, learning_rate, batch_size, epochs)
# SGD_optimizer.training_program('SGD')   # SGD optimizer chosen
#
# network_SGD_compareto = network_model.network([784,30, 10], 'cross-entropy', regularization='none')  # making the network SGD
# SGD_optimizer = sgd_optimizer.SGD(network_SGD_compareto, learning_rate, batch_size, epochs)
# SGD_optimizer.training_program('SGD', noimpin_n=5, eta_modify=3)   # SGD optimizer chosen


# # Success test - do not delete - testing the no-improv ETA change by 3
# epochs = 30
# eta_cng = 3
# learning_rate = 0.5        # recommended
# network_SGD_no_imp_in_n = network_model.network([784, 30, 10], 'cross-entropy', regularization='L1', reg_lambda=0.1)  # making the network SGD
# SGD_optimizer = sgd_optimizer.SGD(network_SGD_no_imp_in_n, learning_rate, batch_size, epochs)
# SGD_optimizer.training_program('SGD', eta_modify=eta_cng)   # SGD optimizer chosen

# # Success test - do not delete - testing the no-improvment-in-5
# epochs = 30
# n = 5
# batch_size = 100
# learning_rate = 0.5        # recommended
# network_SGD_no_imp_in_n = network_model.network([784, 30, 10], 'cross-entropy', 'L1', 0.1)  # making the network SGD
# SGD_optimizer = sgd_optimizer.SGD(network_SGD_no_imp_in_n, learning_rate, batch_size, epochs)
# SGD_optimizer.training_program('SGD', n)   # SGD optimizer chosen


# # Success test - do not delete -  finding eta, learning rate
# learning_rate = 5.0
# network_SGD_CE_eta5 = network_model.network([784, 30, 10], 'cross-entropy')  # making the network SGD
# SGD_optimizer = sgd_optimizer.SGD(network_SGD_CE_eta5, learning_rate, batch_size, epochs)
# SGD_optimizer.training_program('SGD')   # SGD optimizer chosen
#
# network_SGD_CE_eta2 = network_model.network([784, 30, 10], 'cross-entropy')  # making the network SGD
# SGD_optimizer = sgd_optimizer.SGD(network_SGD_CE_eta2, learning_rate, batch_size, epochs)
# SGD_optimizer.training_program('SGD')   # SGD optimizer chosen
#
# network_SGD_CE_eta1 = network_model.network([784, 30, 10], 'cross-entropy')  # making the network SGD
# SGD_optimizer = sgd_optimizer.SGD(network_SGD_CE_eta1, learning_rate, batch_size, epochs)
# SGD_optimizer.training_program('SGD')   # SGD optimizer chosen
#
# network_SGD_CE_eta05 = network_model.network([784, 30, 10], 'cross-entropy')  # making the network SGD
# SGD_optimizer = sgd_optimizer.SGD(network_SGD_CE_eta05, learning_rate, batch_size, epochs)
# SGD_optimizer.training_program('SGD')   # SGD optimizer chosen
#
# network_SGD_CE_eta01 = network_model.network([784, 30, 10], 'cross-entropy')  # making the network SGD
# SGD_optimizer = sgd_optimizer.SGD(network_SGD_CE_eta01, learning_rate, batch_size, epochs)
# SGD_optimizer.training_program('SGD')   # SGD optimizer chosen




# # Success test - do not delete - finding lambda for regularization
# learning_rate = 3.0         # recommended
# network_SGD_CE_L1reg_lambda5p0 = network_model.network([784, 30, 10], 'cross-entropy', 'L1', 5.0)  # making the network SGD
# SGD_optimizer = sgd_optimizer.SGD(network_SGD_CE_L1reg_lambda5p0, learning_rate, batch_size, epochs)
# SGD_optimizer.training_program('SGD')   # SGD optimizer chosen
#
# network_SGD_CE_L1reg_lambda1p0 = network_model.network([784, 30, 10], 'cross-entropy', 'L1', 1.0)  # making the network SGD
# SGD_optimizer = sgd_optimizer.SGD(network_SGD_CE_L1reg_lambda1p0, learning_rate, batch_size, epochs)
# SGD_optimizer.training_program('SGD')   # SGD optimizer chosen
#
# network_SGD_CE_L1reg_lambda0p5 = network_model.network([784, 30, 10], 'cross-entropy', 'L1', 0.5)  # making the network SGD
# SGD_optimizer = sgd_optimizer.SGD(network_SGD_CE_L1reg_lambda0p5, learning_rate, batch_size, epochs)
# SGD_optimizer.training_program('SGD')   # SGD optimizer chosen
#
# network_SGD_CE_L1reg_lambda0p1 = network_model.network([784, 30, 10], 'cross-entropy', 'L1', 0.1)  # making the network SGD
# SGD_optimizer = sgd_optimizer.SGD(network_SGD_CE_L1reg_lambda0p1, learning_rate, batch_size, epochs)
# SGD_optimizer.training_program('SGD')   # SGD optimizer chosen


# # Success test - do not delete
# network_SGD_quad = network_model.network([784, 30, 10], 'quadratic', 'none')  # making the network SGD
# network_SGD_CE = network_model.network([784, 30, 10], 'cross-entropy', 'none')  # making the network SGD
# network_Momentum = network_model.network([784, 30, 10], 'cross-entropy', 'none')  # making the network SGD
# network_RMSProp = network_model.network([784, 30, 10], 'cross-entropy', 'none')  # making the network SGD
#
# learning_rate = 0.01         # doesnt matter
# SGD_optimizer = sgd_optimizer.SGD(network_RMSProp, learning_rate, batch_size, epochs)
# SGD_optimizer.training_program('RMSProp')   # SGD optimizer chosen
#
# learning_rate = 1.0         # recommended
# SGD_optimizer = sgd_optimizer.SGD(network_Momentum, learning_rate, batch_size, epochs)
# SGD_optimizer.training_program('Momentum')   # SGD optimizer chosen
#
# learning_rate = 3.0         # recommended
# SGD_optimizer = sgd_optimizer.SGD(network_SGD_quad, learning_rate, batch_size, epochs)
# SGD_optimizer.training_program('SGD')   # SGD optimizer chosen
#
# SGD_optimizer = sgd_optimizer.SGD(network_SGD_CE, learning_rate, batch_size, epochs)
# SGD_optimizer.training_program('SGD')   # SGD optimizer chosen



# Overall test

# test - do not delete - testing the no-improv ETA change by 3
epochs = 30
batch_size = 100

# Weights constraints regularization
for i in range(1, 10):
    learning_rate = 0.5  # recommended
    network_SGD_Weights_Constraints = \
        network_model.network([784, 30, 10], 'cross-entropy', regularization='Weights Constraints', reg_lambda=0.25*i)
    SGD_optimizer = sgd_optimizer.SGD(network_SGD_Weights_Constraints, learning_rate, batch_size, epochs)
    SGD_optimizer.training_program('SGD')   # SGD optimizer chosen

# L1 regularization
for i in range(0, 10):
    learning_rate = 0.5        # recommended
    network_SGD_L1 = network_model.network([784, 30, 10], 'cross-entropy', regularization='L1', reg_lambda=0.1*i)
    SGD_optimizer = sgd_optimizer.SGD(network_SGD_L1, learning_rate, batch_size, epochs)
    SGD_optimizer.training_program('SGD')   # SGD optimizer chosen

# L2 regularization
for i in range(0, 10):
    learning_rate = 0.5        # recommended
    network_SGD_L2 = network_model.network([784, 30, 10], 'cross-entropy', regularization='L2', reg_lambda=0.1*i)
    SGD_optimizer = sgd_optimizer.SGD(network_SGD_L2, learning_rate, batch_size, epochs)
    SGD_optimizer.training_program('SGD')   # SGD optimizer chosen

# No improvment in n - eta modify:
for i in range(0, 7):
    network_SGD_etaMod = network_model.network([784,30, 10], 'cross-entropy', regularization='none')  # making the network SGD
    SGD_optimizer = sgd_optimizer.SGD(network_SGD_etaMod, learning_rate, batch_size, epochs)
    SGD_optimizer.training_program('SGD', noimpin_n=5, eta_modify=i)   # SGD optimizer chosen