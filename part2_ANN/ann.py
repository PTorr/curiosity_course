from __future__ import division
import numpy as np
from part1_bayse import data_generator as dg
import matplotlib.pyplot as plt


# alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
alphas = [1e-8]

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


def ann(x,y,synapse_0,synapse_1):
    # Feed forward through layers 0, 1, and 2
    layer_0 = x
    layer_1 = sigmoid(np.dot(layer_0, synapse_0))
    layer_1 = np.hstack((np.ones([len(layer_1), 1]), layer_1))
    # layer_2 = sigmoid(np.dot(layer_1, synapse_1))  # sigmoid
    # layer_2 = np.dot(layer_1, synapse_1) # linear
    layer_2 = squre_root(np.dot(layer_1, synapse_1)) # sqrt

    layer_2_error = layer_2 - y

    return layer_0, layer_1, layer_2, layer_2_error

# size of the data

M = 10 # for the binning vector

data = np.genfromtxt('/home/torr/PycharmProjects/curiosity_course/part1_bayse/generated_data.csv',delimiter=',',skip_header=0)
data = data[1:len(data)+1,:]
X = data[:,0:2]
X = np.hstack((np.ones([len(X),1]),X))
y = np.array([data[:,2]]).T
y /= 1.1*np.max(y)
N = len(data)
x_train = X[0:N-0.2*N, :]
y_train = np.array([y[0:N-0.2*N,0]]).T
x_valid = X[N-0.2*N:N-0.1*N, :]
y_valid = np.array([y[N-0.2*N:N-0.1*N,0]]).T
x_test = X[N-0.1*N:N, :]
y_test = np.array([y[N-0.1*N:N,0]]).T


for alpha in alphas:
    print "\nTraining With Alpha:" + str(alpha)
    np.random.seed(1)

    # randomly initialize our weights with mean 0
    hls = 140 # hidden_layer_size
    input_size = len(x_train.T)
    synapse_0 = 2 * np.random.random((input_size, hls)) - 1
    synapse_1 = 2 * np.random.random((hls+1, 1)) - 1

    num_of_iterations = 5000
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


        # layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2) # sigmoid
        # layer_2_delta = layer_2_error * layer_2 #linear
        layer_2_delta = layer_2_error * squre_root_to_derivative(layer_2) #sqrt

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
    a = str(alphas)
    a = a.replace('.', '_')
    fig_name = 'learning_rate' + str(a) + ' iterations' + str(num_of_iterations) + ' hls' + str(hls)
    txt_name = fig_name + '.csv'
    np.savetxt(txt_name, [training_error[-1, 1], validation_error[-1, 1], test_error[-1, 1]], delimiter=",",
               header='train,valid,test')
    plt.savefig(fig_name)

    # plt.figure('layer2t')
    # plt.bar(np.linspace(0,len(layer_2t),len(layer_2t)), layer_2t)
    # plt.figure('y')
    # plt.bar(np.linspace(0,len(layer_2t),len(layer_2t)),y_test)
    # # print np.mean(y_test)
    # plt.figure('compare')
    # plt.plot(np.linspace(0,10,100),layer_2t[0:100],'b')
    # plt.plot(np.linspace(0,10,100),y_train[0:100],'r')
    plt.show()


print layer_2t[0:10].T
print y_train[0:10].T


# print layer_2
