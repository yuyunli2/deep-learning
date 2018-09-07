import numpy as np
import h5py
import time
import copy
from random import randint
#load MNIST data
MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:] )
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
print('x_train', x_train, x_train.shape)
print('y_train', y_train, y_train.shape)
print('len(x_train)', len(x_train))
print(np.unique(y_train))
x_test = np.float32( MNIST_data['x_test'][:] )
y_test = np.int32( np.array( MNIST_data['y_test'][:,0] ) )
MNIST_data.close()
#Implementation of stochastic gradient descent algorithm

h = 40
#number of inputs
num_inputs = 28*28
#number of outputs
num_outputs = 10
model = {}
model['W1'] = np.random.randn(num_inputs,h) / np.sqrt(num_inputs)
model_grads = copy.deepcopy(model)
model['b1'] = np.random.randn(1, h) / np.sqrt(num_inputs)
model['b2'] = np.random.randn(1, num_outputs) / np.sqrt(num_inputs)
model['C'] = np.random.randn(h, num_outputs) / np.sqrt(num_inputs)
model['pu'] = np.random.randn(1, num_outputs) / np.sqrt(num_inputs)
model['pb2'] = np.random.randn(1, num_outputs) / np.sqrt(num_inputs)
model['pc'] = np.random.randn(h, num_outputs) / np.sqrt(num_inputs)
model['H'] = np.random.randn(1,h) / np.sqrt(num_inputs)

def softmax_function(z):
    ZZ = np.exp(z) / np.sum(np.exp(z))
    return ZZ

def forward(x,y, model):
    model['z'] = np.dot(x, model['W1']) + model['b1']
    model['H'] = 1/(1+np.exp(-model['z']))
    model['U'] = np.dot(model['H'], model['C']) + model['b2']
    f = softmax_function(model['U'])
    return f

def backward(x,y,f, model, model_grads):
    pu = f
    pu[0,y] = pu[0,y] - 1
    pb2 = pu
    pc = np.dot(model['H'].T, pu)
    delta = np.dot(model['C'], pu.T)
    dsigma = model['H'] - model['H']**2
    pb1 = delta * dsigma.T
    pw = np.dot(pb1, x)

    return pc, pb2, pb1, pw
import time
time1 = time.time()
LR = .001
num_epochs = 100
for epochs in range(num_epochs):
    #Learning rate schedule
    total_correct = 0
    for n in range( len(x_train)):
        n_random = randint(0,len(x_train)-1 )
        y = y_train[n_random]
        x = x_train[n_random][:].reshape(1,784)
        p = forward(x, y, model)
        prediction = np.argmax(p)
        if (prediction == y):
            total_correct += 1

        pc, pb2, pb1, pw = backward(x,y,p, model, model_grads)
        model['C'] = model['C'] - LR*pc
        model['b2'] = model['b2'] - LR*pb2
        model['b1'] = model['b1'] - LR*pb1.T
        model['W1'] = model['W1'] - LR*pw.T
    if epochs % 10 == 0:
        print(epochs)
    print(total_correct/np.float(len(x_train) ) )
time2 = time.time()
print(time2-time1)
######################################################
#test data
total_correct = 0
for n in range( len(x_test)):
    y = y_test[n]
    x = x_test[n][:]
    p = forward(x, y, model)
    prediction = np.argmax(p)
    if (prediction == y):
        total_correct += 1
print(total_correct/np.float(len(x_test) ) )
