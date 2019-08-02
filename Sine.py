'''
Jonathan Sweeney
14343826
COMP30230 Programming Assignment
'''

from numpy import *
import numpy as np
from NN import MLP

# Generates combinations of sine values and then uses NN class to build prediction model

I_layer_size = 4
H_layer_size = 10
O_layer_size = 1
lrate = 0.2
epochs = 1000
momentum = 0.9
np.random.seed(3)

examples = 50

inputs = 2 * np.random.random((examples, 4)) - 1
targets = []
for array in inputs:
    total_sum = array[0] - array[1] + array[2] - array[3]
    targets.append([math.sin(total_sum)])

targets = np.array(targets)
train_range = int((examples / 5) * 4)
test_range = examples / 5

train_set = inputs[0:train_range, :]
train_outputs = targets[0:train_range, :]

test_set = inputs[train_range:examples, :]
test_outputs = targets[train_range:examples, :]

NN = MLP(train_set, train_outputs, H_layer_size, momentum)
NN.randomize()

for each in range(epochs):
    NN.forward(NN.I)
    error = NN.backwards()
    NN.update_weights(lrate)

    if mod(each, 50) == 0:
        print("Error at epoch: ", each, ": ", error)

print("lRate:", lrate, "M:", momentum)
print("epochs:", epochs, "hidden units:", H_layer_size)

examples = shape(test_set)[0]
test_input = concatenate((test_set, -ones((examples, 1))), axis=1)
expected_output = NN.forward(test_input)
error = 0.5 * sum((expected_output - test_outputs) ** 2)

print()
print("Test set error: ", error)


