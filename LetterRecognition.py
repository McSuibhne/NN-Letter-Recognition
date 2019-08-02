'''
Jonathan Sweeney
14343826
COMP30230 Programming Assignment
'''

from numpy import *
import numpy as np
from NN import MLP

# Reads in 20000 letter examples from a text file and parses them into vector models for use in the neural network
# Trains the network on 80% of the dataset, uses the rest to test the prediction model

I_layer_size = 16
H_layer_size = 35
O_layer_size = 26
lrate = 0.15
epochs = 10000
momentum = 0.5
np.random.seed(3)

examples = 20000
filename = "letter-recognition.data.txt"
inputs = []
targets = []

letters = []
with open(filename) as f:
    for line in f:
        letters.append([str(n) for n in line.strip().split(',')])
for each in letters:
    try:
        alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        char_to_int = dict((c, i) for i, c in enumerate(alphabet))
        integer_encoded = [char_to_int[each[0]]]
        onehot_encoded = []
        for value in integer_encoded:
            letter = [0 for _ in range(len(alphabet))]
            letter[value] = 1
            onehot_encoded = letter
        targets.append(onehot_encoded)
        inputs.append([int(each[1]), int(each[2]), int(each[3]), int(each[4]), int(each[5]), int(each[6]),
                       int(each[7]), int(each[8]), int(each[9]), int(each[10]), int(each[11]), int(each[12]),
                       int(each[13]), int(each[14]), int(each[15]), int(each[16])])
    except IndexError:
        print("A line in the file doesn't have enough entries.")

targets = np.array(targets)
train_range = int((examples / 5) * 4)
test_range = examples / 5
train_set = inputs[:train_range]
train_outputs = targets[:train_range]
test_set = inputs[train_range:examples]
test_outputs = targets[train_range:examples]

NN = MLP(train_set, train_outputs, H_layer_size, momentum)
NN.randomize()

for each in range(epochs):
    NN.forward(NN.I)
    error = NN.backwards()
    NN.update_weights(lrate)

    if mod(each, 100) == 0:
        print("Error at epoch: ", each, ": ", error)

print("lRate:", lrate, "M:", momentum)
print("epochs:", epochs, "hidden units:", H_layer_size)

examples = shape(test_set)[0]
test_input = concatenate((test_set, -ones((examples, 1))), axis=1)
expected_output = NN.forward(test_input)
error = 0.5 * sum((expected_output - test_outputs) ** 2)

print()
print("Test set error: ", error)
