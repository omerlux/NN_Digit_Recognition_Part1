# Neural-Network---Part-1---Digit-Recognition
Machine Learning Digit Recognition

I made a network to recognize handwritten digits.
Network - The network is the neural network, which can be calculated by 2 cost function, quadratic and cross-entropy.
The network can build up with as many layers as you wish, when the input is 784 neurons and the output is 10.

SGD optimizer - The optimizer can be (for now) 3 different optimizers, that are established on the back propagation algorithm for gradient decent.
The optimizers are:
1. Stochastic Gradient descent.
2. Momentum.
3. RMSProp.

Added regularization methods:
1. L1.
2. L2.
3. Early stopping - no-improment-in-n epochs will stop the learning.
4. Early stopping - no-improment-in-n will decrese learning rate until eta is eta/128. (After that the learning will stop)
5. Weights constraints - limited to range (-1,1)

Results are in "Few Tests.pdf".

To know more about the project:    http://neuralnetworksanddeeplearning.com/index.html by Michael Nielson / Dec 2019
To check out different optimizers: https://towardsdatascience.com/10-gradient-descent-optimisation-algorithms-86989510b5e9
