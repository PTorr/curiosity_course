import numpy as np
from part1_bayse import data_generator as dg

# alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
alphas = [0.01]

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output * (1 - output)

def layer_2_activation_func(d):
    mu = [0,8,22.5,37]
    sig = [1,3,4,7]
    return (np.exp(-(d - mu[0]) ** 2.0 / (2.0 * sig[0] ** 2.0))) / (sig[0] * np.sqrt(2.0 * np.pi))\
            +(np.exp(-(d - mu[1]) ** 2.0 / (2.0 * sig[1] ** 2.0))) / (sig[1] * np.sqrt(2.0 * np.pi))\
            +(np.exp(-(d - mu[2]) ** 2.0 / (2.0 * sig[2] ** 2.0))) / (sig[2] * np.sqrt(2.0 * np.pi))\
            +(np.exp(-(d - mu[3]) ** 2.0 / (2.0 * sig[3] ** 2.0))) / (sig[3] * np.sqrt(2.0 * np.pi))\


# size of the data
N = 1000
X = np.zeros([N, 2])
y = np.zeros([N, 1])
for i in range(0, N):
    theta = dg.choose_theta()
    energy = dg.choose_energy()
    v0 = dg.initial_velocity(energy)
    time = dg.time_calculator(v0, theta)
    distance = dg.distance_calculator(v0, theta, time)
    X[i,:] = [energy,theta]
    y[i] = distance

# X = np.array([[0, 0, 1],
#               [0, 1, 1],
#               [1, 0, 1],
#               [1, 1, 1]])
#
# y = np.array([[0],
#               [1],
#               [1],
#               [0]])

for alpha in alphas:
    print "\nTraining With Alpha:" + str(alpha)
    np.random.seed(1)

    # randomly initialize our weights with mean 0
    synapse_0 = 2 * np.random.random((2, 10)) - 1
    synapse_1 = 2 * np.random.random((10, 1)) - 1

    for j in xrange(100):

        # Feed forward through layers 0, 1, and 2
        layer_0 = X
        layer_1 = sigmoid(np.dot(layer_0, synapse_0))
        layer_2 = layer_2_activation_func(np.dot(layer_1, synapse_1)) # Think what the activation function we want for the last layer

        # how much did we miss the target value?
        layer_2_error = layer_2 - y

        if (j % 10000) == 0:
            print "Error after " + str(j) + " iterations:" + str(np.mean(np.abs(layer_2_error)))

        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        # layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)
        layer_2_delta = layer_2_error

        # how much did each l1 value contribute to the l2 error (according to the weights)?
        layer_1_error = layer_2_delta.dot(synapse_1.T)

        # in what direction is the target l1?
        # were we really sure? if so, don't change too much.
        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)

        synapse_1 -= alpha * (layer_1.T.dot(layer_2_delta))
        synapse_0 -= alpha * (layer_0.T.dot(layer_1_delta))

# print layer_2