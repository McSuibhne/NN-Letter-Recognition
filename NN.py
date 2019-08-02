'''
Jonathan Sweeney
14343826
COMP30230 Programming Assignment
'''
from numpy import *


# Multi-Layer Perceptron using one hidden layer
class MLP:

    # Initialises network attributes using parameter values and sets weights to 0
    def __init__(self, inputs, targets, hidden_layer_size, momentum_rate):

        self.I = inputs
        self.T = targets
        self.H_layer_size = hidden_layer_size
        self.I_layer_size = shape(inputs)[1]
        self.O_layer_size = shape(targets)[1]
        self.examples = shape(inputs)[0]
        self.M = momentum_rate

        self.I = concatenate((self.I, -ones((self.examples, 1))), axis=1)
        self.H, self.O, self.W1, self.W2, self.dW1, self.dW2  = None, None, None, None, None, None

    # Sets weights to random values between -1 and 1, initialises all update weights to 0
    def randomize(self):    
        self.W1 = (random.rand(self.I_layer_size + 1, self.H_layer_size) - 0.5) * 2 / sqrt(self.I_layer_size)
        self.W2 = (random.rand(self.H_layer_size + 1, self.O_layer_size) - 0.5) * 2 / sqrt(self.H_layer_size)
        self.dW1 = zeros((shape(self.W1)))
        self.dW2 = zeros((shape(self.W2)))

    # Feeds the input set forward through the network, calculating results from weights at each node
    def forward(self, inputs):    
        self.H = dot(inputs, self.W1)
        self.H = 1.0 / (1.0 + exp(-self.H))
        self.H = concatenate((self.H, -ones((shape(inputs)[0], 1))), axis=1)
        self.O = dot(self.H, self.W2)
        return self.O
        
    # Calculates values needed for backprop, finding network error and weight changes
    def backwards(self):
        Z2 = (self.O - self.T) / self.examples
        Z1 = self.H * (1.0 - self.H) * (dot(Z2, transpose(self.W2)))
        self.dW1 = dot(transpose(self.I), Z1[:, :-1]) + self.M * self.dW1
        self.dW2 = dot(transpose(self.H), Z2) + self.M * self.dW2

        error = 0.5 * sum((self.O - self.T) ** 2)

        return error

    # Updates weights based on backprop calculation in backwards.
    def update_weights(self, lrate):
        self.W1 -= lrate * self.dW1
        self.W2 -= lrate * self.dW2
