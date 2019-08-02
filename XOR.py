'''
Jonathan Sweeney
14343826
COMP30230 Programming Assignment
'''

from numpy import *
import numpy as np
from NN import MLP

# Uses NN class to build XOR prediction model

I_layer_size = 2
H_layer_size = 2
O_layer_size = 1
lrate = 0.2
epochs = 1000
momentum = 0.9
np.random.seed(3)

examples = 4

inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])


NN = MLP(inputs, targets, H_layer_size, momentum)
NN.randomize()

for each in range(epochs):
    NN.forward(NN.I)
    error = NN.backwards()
    NN.update_weights(lrate)
    if mod(each, 100) == 0:
        print("Error at epoch: ", each, ": ", error)

print("lRate:", lrate, "M:", momentum)
print("epochs:", epochs, "hidden units:", H_layer_size)

test_input = concatenate((inputs, -ones((examples, 1))), axis=1)
test_output = NN.forward(test_input)

print()
print("Output: ")
for line in test_output:
    print(line)


    
