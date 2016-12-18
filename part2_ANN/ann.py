from __future__ import division
import numpy as np
from part1_bayse import data_generator as dg
import matplotlib.pyplot as plt


# alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
alphas = [0.0000001]

# compute sigmoid nonlinearity
def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output

# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output * (1 - output)


def ann(x,y,synapse_0,synapse_1):
    # Feed forward through layers 0, 1, and 2
    layer_0 = x
    layer_1 = sigmoid(np.dot(layer_0, synapse_0))
    layer_1 = np.hstack((np.ones([len(layer_1), 1]), layer_1))
    layer_2 = sigmoid(np.dot(layer_1, synapse_1))  # Think what the activation function we want for the last layer

    layer_2_error = layer_2 - y

    return layer_0, layer_1, layer_2, layer_2_error

# size of the data

M = 10 # for the binning vector

data = np.genfromtxt('/home/torr/PycharmProjects/curiosity_course/part1_bayse/generated_data.csv',delimiter=',',skip_header=0)
data = data[1:len(data)+1,:]
input_size = 200
X = data[:,0:input_size]
X = np.hstack((np.ones([len(X),1]),X))
y = data[:,input_size:len(data.T)+1]
N = len(data)
x_train = X[0:N-0.2*N, :]
y_train = y[0:N-0.2*N, :]
x_valid = X[N-0.2*N:N-0.1*N, :]
y_valid = y[N-0.2*N:N-0.1*N, :]
x_test = X[N-0.1*N:N, :]
y_test = y[N-0.1*N:N, :]


for alpha in alphas:
    print "\nTraining With Alpha:" + str(alpha)
    np.random.seed(1)

    # randomly initialize our weights with mean 0
    hls = 100 # hidden_layer_size
    synapse_0 = 2 * np.random.random((input_size+1, hls)) - 1
    synapse_1 = 2 * np.random.random((hls+1, M)) - 1

    num_of_iterations = 1000
    training_error = np.zeros([num_of_iterations, 2])
    validation_error = np.zeros([num_of_iterations, 2])
    test_error = np.zeros([num_of_iterations, 2])

    for j in xrange(num_of_iterations):

        [layer_0, layer_1, layer_2, layer_2_error] = ann(x_train, y_train,synapse_0,synapse_1)
        training_error[j,:] = [j,np.sum(layer_2_error**2)/(2.0*len(x_train))]

        [layer_0v, layer_1v, layer_2v, layer_2v_error] = ann(x_valid, y_valid,synapse_0,synapse_1)
        validation_error[j,:] = [j,np.sum(layer_2v_error**2)/(2.0*len(x_valid))]

        [layer_0t, layer_1t, layer_2t, layer_2t_error] = ann(x_test, y_test,synapse_0,synapse_1)
        test_error[j, :] = [j, np.sum(layer_2t_error ** 2) / (2.0 * len(x_test))]


        # if (j % 1000) == 0:
        #     print "Error after " + str(j) + " iterations:" + str(np.mean(np.abs(layer_2_error)))

        layer_2_delta = layer_2_error

        layer_1_error = layer_2_delta.dot(synapse_1.T)

        layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)

        synapse_1 -= alpha * (layer_1.T.dot(layer_2_delta))
        synapse_0 -= alpha * (layer_0.T.dot(layer_1_delta[:,1:]))

        if validation_error[j,1] > validation_error[j-1,1]:
            pass

    plt.figure('errors')
    plt.plot(training_error[:,0],training_error[:,1],'b', label = 'train')
    plt.plot(validation_error[:,0],validation_error[:,1],'r', label = 'validation')
    plt.plot(test_error[:, 0], test_error[:, 1], 'g',label = 'test')
    plt.legend()
    plt.ylim(0,np.max(training_error[:,1]))
    # plt.show()

    plt.figure('layer2t')
    plt.pcolor(layer_2t[1:10,:])
    plt.figure('y')
    plt.pcolor(y_test[1:10,:])
    # print layer_2t.argmax(axis=1)
    plt.show()

    # [layer_0t, layer_1t, layer_2t, layer_2t_error] = ann([25,90], [1,0,0,0,0,0,0,0,0,0])
    # print layer_2t

# print layer_2
