from __future__ import division
import numpy as np
from part1_bayse import data_generator as dg
import matplotlib.pyplot as plt

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output
# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output * (1 - output)

def squre_root(x):
    return np.sqrt(x)
def squre_root_to_derivative(x):
    return 0.5/np.sqrt(x)

def ReLU(x):
    x[x<0]=0
    x[x>=0] = x[x>=0]
    return x
def ReLU_to_derivative(x):
    x[x < 0] = 0
    x[x >= 0] = 1
    return x

def initialize_synapses(hls,input_size):
    np.random.seed(1)
    synapse_0 = 2 * np.random.random((input_size, hls[0])) - 1
    synapse_1 = 2 * np.random.random((hls[0] + 1, hls[1])) - 1
    synapse_2 = 2 * np.random.random((hls[1] + 1, hls[2])) - 1
    synapse_3 = 2 * np.random.random((hls[2] + 1, 1)) - 1

    return synapse_0,synapse_1,synapse_2,synapse_3

def ann(x,y,synapse_0,synapse_1,synapse_2,synapse_3):
    # Feed forward through layers 0, 1, and 2
    layer_0 = x.T
    layer_1 = sigmoid(np.dot(layer_0, synapse_0))
    layer_1 = np.hstack((np.ones([len(layer_1), 1]), layer_1))
    layer_2 = sigmoid(np.dot(layer_1, synapse_1))
    layer_2 = np.hstack((np.ones([len(layer_2), 1]), layer_2))
    layer_3 = ReLU(np.dot(layer_2, synapse_2))
    layer_3 = np.hstack((np.ones([len(layer_3), 1]), layer_3))
    layer_4 = ReLU(np.dot(layer_3, synapse_3))

    layer_4_error = layer_4 - y

    return layer_0, layer_1, layer_2, layer_3, layer_4, layer_4_error


def train_ann(hl,num_of_iterations,alphas,x,y,synapse_0,synapse_1,synapse_2,synapse_3):
    # for k in hl:
    #     hls = hl[k]
    for alpha in alphas:
        # randomly initialize our weights with mean 0
        input_size = len(x)

        training_error = np.zeros([num_of_iterations, 2])

        for j in xrange(num_of_iterations):

            [layer_0, layer_1, layer_2, layer_3, layer_4, layer_4_error] = ann(x, y,synapse_0,synapse_1,synapse_2,synapse_3)
            print 'prediction = ',layer_4, 'real = ' ,y
            training_error[j,:] = [j,np.sum(layer_4_error**2)/(2.0*len(x))]

            # layer_4_delta = training_error[j,1] * ReLU_to_derivative(layer_4)
            layer_4_delta = layer_4_error * ReLU_to_derivative(layer_4)


            layer_3_error = layer_4_delta.dot(synapse_3.T)
            layer_3_delta = layer_3_error * ReLU_to_derivative(layer_3)

            layer_2_error = layer_3_delta[:,1:].dot(synapse_2.T)
            layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)

            layer_1_error = layer_2_delta[:,1:].dot(synapse_1.T)
            layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)

            synapse_3 -= alpha * (layer_3.T.dot(layer_4_delta))
            synapse_2 -= alpha * (layer_2.T.dot(layer_3_delta[:,1:]))
            synapse_1 -= alpha * (layer_1.T.dot(layer_2_delta[:,1:]))
            synapse_0 -= alpha * (layer_0.T.dot(layer_1_delta[:,1:]))

    return training_error[j,1],synapse_0,synapse_1,synapse_2,synapse_3

def learner(synapse_0,synapse_1,synapse_2,synapse_3):
    alphas = [1e-6]
    num_of_iterations = 1
    hl = {1: [50, 100, 140]}


    theta = dg.choose_theta()
    energy = dg.choose_energy()
    v0 = dg.initial_velocity(energy)
    time = dg.time_calculator(v0,theta)
    distance = dg.distance_calculator(v0,theta,time)

    X = np.array([energy/25.,theta/(np.pi/2.)])
    y = distance/45.
    input_size = len(X)
    hls = hl[1]


    l4_error,synapse_0,synapse_1,synapse_2,synapse_3 = train_ann(hl,num_of_iterations,alphas,X,y,synapse_0,synapse_1,synapse_2,synapse_3)

    return l4_error,synapse_0,synapse_1,synapse_2,synapse_3

if __name__ == '__main__':
    # for first run
    hl = {1: [50, 100, 140]}
    hls = hl[1]
    input_size = 2
    synapse_0, synapse_1, synapse_2, synapse_3 = initialize_synapses(hls, input_size)

    for i in range(1000):
        l4_error, synapse_0, synapse_1, synapse_2, synapse_3 = learner(synapse_0, synapse_1, synapse_2, synapse_3)
        # print l4_error