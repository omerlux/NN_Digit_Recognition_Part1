import numpy as np
import matplotlib.pyplot as plot
import scipy
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss


def sigmoid(z):
    """ Sigmoid function"""
    return 1.0 / (1.0 + np.exp(-z))
    # return scipy.special.expit(z)


def sigmoid_derivative(z):
    """ Sigmoid derivative by z"""
    return sigmoid(z) * (1 - sigmoid(z))


def quadratic_cost_derivative(output, y, z, network):
    """for the cost function (1/2)*(a-y)^2
    the derivative is a-y"""
    # Regularization Confidence Penalty
    if network.regularization == 'Confidence Penalty':
        network.confidence_penalty = (output - y - network.reg_lambda * network.confidence_penalty) * sigmoid_derivative(z)
        return network.confidence_penalty
    else:
        return (output - y) * sigmoid_derivative(z)


def CE_cost_derivative(output, y, z, network):
    """for the cost function [ y*ln(a) + (1-y)*ln(1-a) ]
    the derivative is a-y"""
    # Regularization Confidence Penalty
    if network.regularization == 'Confidence Penalty':
        network.confidence_penalty = output - y - network.reg_lambda * network.confidence_penalty
        return network.confidence_penalty
    else:
        return output - y


def vectorized_res(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere. This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


def cost_func_single(out, y):
    """ :returns the quadratic cost function for single sample"""
    y_vec = vectorized_res(y)
    cost_j = [np.power(aj - yj, 2) for aj, yj in zip(out, y_vec)]
    return (1 / 2.0) * np.sum(cost_j)


class network(object):
    def __init__(self, sizes, cost='cross-entropy', regularization='none',
                 reg_lambda='0'):  # sizes=[#inputs, #2nd layer, .... , #outputs]
        """ Initializing the network object
        sizes is array that will contain the number of neurons in each layer.
        the weights and biases for each neuron will be randomly chosen from a Norm~(meu,sigma) cdf"""
        print ('Initializing network model object...')
        self.sizes = sizes
        self.num_of_layers = len(sizes)
        self.cost = cost
        self.confidence_penalty = 0
        self.regularization = regularization
        self.reg_lambda = reg_lambda
        # Biases initiate - randomize by normal distribution biases as the numbers of neurons in the network (without
        # input neurons)
        sigma = 1
        meu = 0
        self.biases = [sigma * np.random.randn(dimensions, 1) + meu for dimensions in sizes[1:]]
        self.delta_biases = [np.zeros((dimensions, 1)) for dimensions in sizes[1:]]
        self.delta_sqr_biases = [np.zeros((dimensions, 1)) for dimensions in sizes[1:]]
        # dimensions: [#2nd layer, ... , #outputs], so we have X biases as the numbers of neurons except the inputs
        # Weights initiate - randomize by normal distribution weights for each neuron
        sigma2 = 1
        meu2 = 0
        # /np.sqrt(dim1) - smart weights initialization
        self.weights = [sigma2 * np.random.randn(dim2, dim1) / np.sqrt(dim1) + meu2 for (dim1, dim2) in
                        zip(sizes[:-1], sizes[1:])]
        self.delta_weights = [np.zeros((dim2, dim1)) for (dim1, dim2) in zip(sizes[:-1], sizes[1:])]
        self.delta_sqr_weights = [np.zeros((dim2, dim1)) for (dim1, dim2) in zip(sizes[:-1], sizes[1:])]
        # size[:-1], size[1:] creates pairs of elements in the sequence sizes with the next element,
        # because you pair up all elements except the last (sizes[:-1]) with all elements except the first (sizes[1:])
        self.biases_num = sum(sizes) - sizes[0]
        self.weights_num = sum(x * y for (x, y) in zip(sizes[:-1], sizes[1:]))

    def feed_forward(self, batch, only_out):
        """ Feeding forward the network with an intput of batch.
        That means we feed the network m times for an image which is a 784 input vector.
        :return the model's output - ONLY output if bool_only_out=1 else, adding:
        the loss functions. 1. mean square error, 2. negative log loss"""
        # creating the function for each loss function
        out = []
        for (x, y) in batch:
            x = np.reshape(x, (784, 1))
            for (b, w) in zip(self.biases, self.weights):
                x = sigmoid(np.dot(w, x) + b)  # a is now the output of the curr layer, and input of the next
            out.append(x)  # np.reshape(x, len(batch))
        if only_out:
            return out
        else:
            return [out,  # this is the output of the network
                    self.loss_mse(out, [y for (_, y) in batch]),
                    self.loss_nll(out, [y for (_, y) in batch])]

    def loss_mse(self, out, y):
        """ :return the mean square error of @ out compared to @ y"""
        # y - is the labels, out is the network output
        mse = 0  # mean square error
        for res, y_res in zip(out, y):
            mse += mean_squared_error(vectorized_res(y_res), res)  # calculating error from each label and output
        return mse / len(y)  # for the mean square error fix

    def loss_nll(self, out, y):
        """ :return the negative log loss of @ out compared to @ y"""
        # y - is the labels, out is the network output
        nll = 0  # negative log los = -log P(yt|yp) = -(yt log(yp) + (1 - yt) log(1 - yp))
        for res, y_res in zip(out, y):
            nll += log_loss(vectorized_res(y_res), res)  # calculating error from each label and output
        return nll / len(y)  # for the mean square error fix

    def back_prop(self, batch):
        """ batch - is a batch of input examples and labels.
        :returns the gradient vector of the batch"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]  # initializing the changes in b and w
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in batch:  # for every example with label do:
            x = np.reshape(x, (784, 1))
            # FOR EACH EXMAPLE: calculating the delta in b and w for single exmaple
            nabla_b_single, nabla_w_single = self.back_prop_single(x, y)

            # NEXT: adding the nabla_x_single to the nabla_x of all exmaples
            nabla_b = [nb + delta_nb for nb, delta_nb in zip(nabla_b, nabla_b_single)]  # summing the nablas
            nabla_w = [nw + delta_nw for nw, delta_nw in zip(nabla_w, nabla_w_single)]

        return [nabla_b, nabla_w]

    def back_prop_RMSprop(self, batch):
        """ batch - is a batch of input examples and labels - FOR ADAM OPTIMIZER
        :returns the gradient vector of the batch
        and the gradient^2 of it"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]  # initializing the changes in b and w
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        nabla_sqr_b = [np.zeros(b.shape) for b in self.biases]  # initializing the changes in b and w squared
        nabla_sqr_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in batch:  # for every example with label do:
            x = np.reshape(x, (784, 1))
            # FOR EACH EXMAPLE: calculating the delta in b and w for single exmaple
            nabla_b_single, nabla_w_single = self.back_prop_single(x, y)

            # NEXT: adding the nabla_x_single to the nabla_x of all exmaples
            nabla_b = [nb + delta_nb for nb, delta_nb in zip(nabla_b, nabla_b_single)]  # summing the nablas
            nabla_w = [nw + delta_nw for nw, delta_nw in zip(nabla_w, nabla_w_single)]

            # for ADAM optimizer - square calculate
            nabla_sqr_b = [nb + delta_nb ** 2 for nb, delta_nb in zip(nabla_sqr_b, nabla_b_single)]  # nablas square
            nabla_sqr_w = [nw + delta_nw ** 2 for nw, delta_nw in zip(nabla_sqr_w, nabla_w_single)]

        return [nabla_b, nabla_w, nabla_sqr_b, nabla_sqr_w]

    def back_prop_single(self, x, y):
        """ :returns the back propagation gradient of a SINGLE example"""
        # FEEDING FORWARD the network:
        nabla_b_single = [np.zeros(b.shape) for b in
                          self.biases]  # initializing the changes in b and w for single input
        nabla_w_single = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]  # saving activations for all the layers
        zs = []  # saving z vectors for all layers
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b  # setting new z= w*a+b
            zs.append(z)  # saving z
            activation = sigmoid(z)  # setting new output of the current layer
            activations.append(activation)  # saving activations
        # --> we saved all the activations and zs for each layer

        # TODO: Soft-max - NOT WORKING
        # activations[-1] = np.exp(activations[-1]) / np.sum(np.exp(activations[-1]))

        # GOING BACKWARD to calculate the cost change by:
        # BP1 -> delta_L = Nabla_C_a . sigmoid_derivative(z_L)
        # BP2 -> delta_l = ((w_l+1)T * delta_l+1) . sigmoid_derivative(z_l)
        # BP3 -> Nabla_C_bj_l = delta_j_l
        # BP4 -> Nabla_C_wjk_l = delta_j_l * activation_k_l-1
        if self.cost == 'quadratic':
            delta = quadratic_cost_derivative(activations[-1], vectorized_res(y), zs[-1], self)
            # by BP1 - check func* sigmoid_derivative(zs[-1])  # by BP1 equation
        else:  # self.cost == 'cross-entropy':
            delta = CE_cost_derivative(activations[-1], vectorized_res(y), zs[-1], self)
            # by BP1 - check func
        nabla_b_single[-1] = delta  # by BP3
        nabla_w_single[-1] = np.dot(delta, np.transpose(activations[-2]))  # by BP4
        # calculating all the nabla_w when walking backwards in the network
        for l in xrange(2, self.num_of_layers):
            z = zs[-l]
            delta = np.dot(np.transpose(self.weights[-l + 1]), delta) * sigmoid_derivative(z)  # by BP2
            nabla_b_single[-l] = delta  # by BP3
            nabla_w_single[-l] = np.dot(delta, np.transpose(activations[-l - 1]))  # by BP4
        # --> we saved all the changes in 'w' and 'b' for the specific example.
        return nabla_b_single, nabla_w_single

    def update_wb(self, nabla_biases, nabla_weights, eta, len_batch, len_training_set):
        n = len_training_set
        # normalizing the results as the number of exmaples
        if self.regularization == 'none' or self.regularization == 'Confidence Penalty':
            self.biases = [b - (eta / len_batch) * nb
                           for b, nb in zip(self.biases, nabla_biases)]
            self.weights = [w - (eta / len_batch) * nw
                            for w, nw in zip(self.weights, nabla_weights)]
        # implementing regularization (using reg_lambda )
        # L1 regularization -
        if self.regularization == 'L1':
            # print ("Regularization L1 - lambda {}\n").format(self.reg_lambda)
            self.biases = [b - (eta / len_batch) * nb
                           for b, nb in zip(self.biases, nabla_biases)]
            self.weights = [w - ((eta * self.reg_lambda) / n) * np.sign(w) - (eta / len_batch) * nw
                            for w, nw in zip(self.weights, nabla_weights)]
        # L2 regularization -
        elif self.regularization == 'L2':
            # print ("Regularization L2 - lambda {}\n").format(self.reg_lambda)
            self.biases = [b - (eta / len_batch) * nb
                           for b, nb in zip(self.biases, nabla_biases)]
            self.weights = [(1 - (eta * self.reg_lambda) / n) * w - (eta / len_batch) * nw
                            for w, nw in zip(self.weights, nabla_weights)]

        # Weights constraints regularization -
        elif self.regularization == 'Weights Constraints':
            self.biases = [b - (eta / len_batch) * nb
                           for b, nb in zip(self.biases, nabla_biases)]
            self.weights = [w - (eta / len_batch) * nw
                            for w, nw in zip(self.weights, nabla_weights)]
            self.weights = [np.minimum(w, self.reg_lambda) for w in self.weights]
            self.weights = [np.maximum(w, (-1)*(self.reg_lambda)) for w in self.weights]

    def update_wb_momentum(self, nabla_biases, nabla_weights, eta, len_batch):
        """ check the website https://ruder.io/optimizing-gradient-descent/index.html#rmsprop to see momentum
        and https://towardsdatascience.com/10-gradient-descent-optimisation-algorithms-86989510b5e9"""
        # normalizing the results as the number of exmaples
        beta = 0.9
        # mt = beta * mt-1 + (1-beta) * g
        self.delta_biases = [v * beta + (1 - beta) * (nb / len_batch)
                             for v, nb in zip(self.delta_biases, nabla_biases)]
        self.delta_weights = [v * beta + (1 - beta) * (nw / len_batch)
                              for v, nw in zip(self.delta_weights, nabla_weights)]
        # wt = wt-1 - alpha * mt
        self.biases = [b - eta * nb
                       for b, nb in zip(self.biases, self.delta_biases)]
        self.weights = [w - eta * nw
                        for w, nw in zip(self.weights, self.delta_weights)]

    def update_wb_RMSprop(self, nabla_biases, nabla_weights, nabla_sqr_b, nabla_sqr_w, eta, len_batch):
        """ check the website https://ruder.io/optimizing-gradient-descent/index.html#rmsprop to see RMSprop
        and https://towardsdatascience.com/10-gradient-descent-optimisation-algorithms-86989510b5e9"""
        # normalizing the results as the number of exmaples
        beta = 0.9  # recommended
        alpha = 0.001  # recommended
        epsilon = 10 ** -6  # recommended
        # vt = beta*bt-1 + (1-beta) * g^2
        self.delta_sqr_biases = [v_prev * beta + (1 - beta) * (nb2 / len_batch) ** 2
                                 for v_prev, nb2 in zip(self.delta_sqr_biases, nabla_sqr_b)]
        self.delta_sqr_weights = [v_prev * beta + (1 - beta) * (nw2 / len_batch) ** 2
                                  for v_prev, nw2 in zip(self.delta_sqr_weights, nabla_sqr_w)]
        # wt = wt-1 - [alpha / sqrt(vt +epsilon)] * g
        self.biases = [b - (alpha / (np.sqrt(vt + epsilon))) * (g / len_batch)
                       for b, vt, g in zip(self.biases, self.delta_sqr_biases, nabla_biases)]
        self.weights = [w - (alpha / (np.sqrt(vt + epsilon))) * (g / len_batch)
                        for w, vt, g in zip(self.weights, self.delta_sqr_weights, nabla_weights)]

        # ADAM - not working...
        # beta1 = 0.9
        # beta2 = 0.999
        # epsilon = 10 ** -8
        # # mt = b1 * mt-1 + (1-b1) * gradient  ---> MEAN
        # self.delta_biases = [beta1 * m_prev + (1 - beta1) * (nb / len_batch)
        #                      for m_prev, nb in zip(self.delta_biases, nabla_biases)]
        # self.delta_weights = [beta1 * m_prev + (1 - beta1) * (nw / len_batch)
        #                       for m_prev, nw in zip(self.delta_weights, nabla_weights)]
        # # vt = b2 * vt-1 + (1-b2) * gradient^2 ---> Square MEAN
        # self.delta_sqr_biases = [beta2 * v_prev + (1 - beta2) * (nb2 / len_batch) ** 2
        #                          for v_prev, nb2 in zip(self.delta_sqr_biases, nabla_sqr_b)]
        # self.delta_sqr_weights = [beta2 * v_prev + (1 - beta2) * (nw2 / len_batch) ** 2
        #                           for v_prev, nw2 in zip(self.delta_sqr_weights, nabla_sqr_w)]
        # # updating by:  new = old - eta/ sqrt(sqr_new/1-beta1,2) * new
        # self.biases = [b - (eta / (np.sqrt(v / (1 - beta2)) + epsilon)) * (m / 1 - beta1)
        #                for b, v, m in zip(self.biases, self.delta_sqr_biases, self.delta_biases)]
        # self.weights = [w - (eta / (np.sqrt(v / (1 - beta2)) + epsilon)) * (m / 1 - beta1)
        #                 for w, v, m in zip(self.weights, self.delta_sqr_weights, self.delta_weights)]
