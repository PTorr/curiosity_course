from __future__ import division
import numpy as np
from part1_bayse import data_generator as dg
import matplotlib.pyplot as plt


alphas = [1e-1,1e-3,1e-5,1e-7,1e-9] # learning rate
# alphas = [1e-6]
num_of_iterations = 1000
hl = {1: [140, 100, 50],2: [100, 140, 50],3:[50,100,140], 4: [150,100,50]}
# hl = {1:[50, 100, 140]}  # hidden_layer_size
# hls = [50,20,10]  # hidden_layer_size

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

class layer:
    def __init__(self):
        self.neurons = None
        self.sigmoid = None
        self.sigmoid_deriv = None
        self.weights = None
        self.error = None
        self.delta = None

    def update_weights(self,m,n):
        self.weights = 2 * np.random.random((m, n)) - 1

    def update_layer(self,layer_n,weights_n,func):
        # self.sigmoid_func(np.dot(layer_n, weights_n))
        func(np.dot(layer_n, weights_n))
        self.neurons = np.hstack((np.ones([len(self.neurons), 1]), self.neurons))

    def update_error(self,latern_delta):
        self.error = latern_delta.dot(self.weights.T)
        self.delta = self.error * sigmoid_output_to_derivative(self.neurons)


def ann(x,y,synapse_0,synapse_1):
    # Feed forward through layers 0, 1, and 2
    layer_0 = x
    layer_1 = sigmoid(np.dot(layer_0, synapse_0))
    # layer_11.update_layer(layer_11,layer_0,synapse_0,sigmoid())
    layer_1 = np.hstack((np.ones([len(layer_1), 1]), layer_1))
    layer_2 = sigmoid(np.dot(layer_1, synapse_1))
    layer_2 = np.hstack((np.ones([len(layer_2), 1]), layer_2))
    layer_3 = ReLU(np.dot(layer_2, synapse_2))
    layer_3 = np.hstack((np.ones([len(layer_3), 1]), layer_3))
    layer_4 = ReLU(np.dot(layer_3, synapse_3))

    layer_4_error = layer_4 - y

    return layer_0, layer_1, layer_2, layer_3, layer_4, layer_4_error

data = np.genfromtxt('../part1_bayse/generated_data.csv',delimiter=',',skip_header=0)
data = data[1:len(data)+1,:]
X = data[:,0:2]
# X -= np.amin(X,axis=0)
X /= (np.amax(X,axis=0))
X = np.hstack((np.ones([len(X),1]),X))
y = np.array([data[:,2]]).T
y /= (np.max(y))
N = len(data)
x_train = X[0:N-0.2*N, :]
y_train = np.array([y[0:N-0.2*N,0]]).T
x_valid = X[N-0.2*N:N-0.1*N, :]
y_valid = np.array([y[N-0.2*N:N-0.1*N,0]]).T
x_test = X[N-0.1*N:N, :]
y_test = np.array([y[N-0.1*N:N,0]]).T

layer_11 = layer()

for k in hl:
    hls = hl[k]
    min_validation = 1e20
    print "\nTraining With hls: "+ str(hls)
    for alpha in alphas:
        print "\nTraining With Alpha:" + str(alpha)
        np.random.seed(1)

        # randomly initialize our weights with mean 0
        input_size = len(x_train.T)
        synapse_0 = 2 * np.random.random((input_size, hls[0])) - 1
        synapse_1 = 2 * np.random.random((hls[0]+1, hls[1])) - 1
        # layer_11.update_weights([0]+1, hls[1])
        synapse_2 = 2 * np.random.random((hls[1]+1, hls[2])) - 1
        synapse_3 = 2 * np.random.random((hls[2]+1, 1)) - 1

        training_error = np.zeros([num_of_iterations, 2])
        validation_error = np.zeros([num_of_iterations, 2])
        test_error = np.zeros([num_of_iterations, 2])

        for j in xrange(num_of_iterations):

            [layer_0, layer_1, layer_2, layer_3, layer_4, layer_4_error] = ann(x_train, y_train,synapse_0,synapse_1)
            training_error[j,:] = [j,np.sum(layer_4_error**2)/(2.0*len(x_train))]

            [layer_0v, layer_1v, layer_2v, layer_3v, layer_4v, layer_4v_error] = ann(x_valid, y_valid,synapse_0,synapse_1)
            validation_error[j,:] = [j,np.sum(layer_4v_error**2)/(2.0*len(x_valid))]

            [layer_0t, layer_1t, layer_2t, layer_3t, layer_4t, layer_4t_error] = ann(x_test, y_test,synapse_0,synapse_1)
            test_error[j, :] = [j, np.sum(layer_4t_error ** 2) / (2.0 * len(x_test))]

            # layer_4_delta = layer_4_error * sigmoid_output_to_derivative(layer_4) # sigmoid
            layer_4_delta = layer_4_error * ReLU_to_derivative(layer_4)
            # layer_4_delta = layer_4_error * 0.5 #linear

            layer_3_error = layer_4_delta.dot(synapse_3.T)
            # layer_3_delta = layer_3_error * sigmoid_output_to_derivative(layer_3)
            layer_3_delta = layer_3_error * ReLU_to_derivative(layer_3)

            layer_2_error = layer_3_delta[:,1:].dot(synapse_2.T)
            layer_2_delta = layer_2_error * sigmoid_output_to_derivative(layer_2)

            layer_1_error = layer_2_delta[:,1:].dot(synapse_1.T)
            layer_1_delta = layer_1_error * sigmoid_output_to_derivative(layer_1)

            synapse_3 -= alpha * (layer_3.T.dot(layer_4_delta))
            synapse_2 -= alpha * (layer_2.T.dot(layer_3_delta[:,1:]))
            synapse_1 -= alpha * (layer_1.T.dot(layer_2_delta[:,1:]))
            synapse_0 -= alpha * (layer_0.T.dot(layer_1_delta[:,1:]))

            if validation_error[j,1] < min_validation:
                min_validation = validation_error[j,1]
            elif validation_error[j,1] - min_validation > 1:
                break
            if j!=0:
                if training_error[j-1,1]-training_error[j,1] < 1e-5:
                    break
        plt.figure('errors')
        plt.plot(training_error[:,0],training_error[:,1],'b', label = 'train')
        plt.plot(validation_error[:,0],validation_error[:,1],'r', label = 'validation')
        plt.plot(test_error[:, 0], test_error[:, 1], 'g',label = 'test')
        plt.legend()
        plt.ylim(0,np.max(training_error[:,1]))
        plt.xlabel('Epochs')
        plt.ylabel('Loss function')
        a = str(alpha)
        a = a.replace('.','_')
        fig_name = 'learning_rate' + str(a) + ' iterations' + str(num_of_iterations) + ' hls' + str(hls)
        txt_name = fig_name+'.csv'
        np.savetxt(txt_name, [training_error[-1,1],validation_error[-1,1],test_error[-1, 1]], delimiter=",", header='train,valid,test')
        plt.savefig(fig_name)
        plt.close("all")
        # plt.show()

        # plt.figure('layer_out')
        # plt.bar(np.linspace(0,len(layer_4t),len(layer_4t)), layer_4t,color = 'blue',alpha = 0.1)
        # plt.bar(np.linspace(0,len(layer_4t),len(layer_4t)), y_test,color = 'red',alpha = 0.2)



print layer_2t[0:10].T
print y_train[0:10].T