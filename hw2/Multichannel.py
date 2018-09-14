import numpy as np
import h5py

import time
import copy
from random import randint
#load MNIST data
MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:] )
# X = np.array(x_train)
# print(X.shape)
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

model['K'] = np.random.randn(3,3,5) / 10
model_grads = copy.deepcopy(model['K'])
model['Z'] = np.random.randn(26,26,5) / np.sqrt(num_inputs)
model['H1'] = np.random.randn(26,26,5) / np.sqrt(num_inputs)
model['W1'] = np.random.randn(num_outputs, 26,26,5) / np.sqrt(num_inputs)
model['b1'] = np.random.randn(1, num_outputs) / np.sqrt(num_inputs)
model['U1'] = np.random.randn(1, num_outputs) / np.sqrt(num_inputs)


def softmax_function(z):
    ZZ = np.exp(z) / np.sum(np.exp(z))
    return ZZ



def ReLU(Z):
    ans = np.copy(Z)
    ans[ans < 0] = 0
    return ans

def ReLUp(Z):
    ans = np.copy(Z)
    ans[ans >= 0] = 1
    ans[ans < 0] = 0
    return ans

def forward(x,y, model):
    for p in range(5):
        for i in range(26):
            for j in range(26):
                model['Z'][i, j, p] = np.tensordot(model['K'][:,:,p], x[i:i + 3, j:j + 3])

    model['H1'] = ReLU(model['Z'])
    for i in range(10):
        x_sum2 = np.tensordot(model['W1'][i,:,:,:], model['H1'],((0,1,2),(0,1,2,)))
        model['U1'][0,i] = x_sum2 + model['b1'][0,i]

    f = softmax_function(model['U1'])
    return f

def backward(x,y,f, model, model_grads):
    pu = f
    pu[0,y] = pu[0,y] - 1
    pb1 = pu
    pw1 = np.tensordot(pu.T, model['H1'].reshape(1,26,26,5),(1,0))
    dsigma = ReLUp(model['Z'])
    ph1 = np.tensordot(pu, model['W1'],(1,0)).reshape(26,26,5)
    pb2 = ph1 * dsigma
    pk = np.random.randn(3,3,5) / np.sqrt(num_inputs)
    for p in range(5):
        for i in range(3):
            for j in range(3):
                pk[i,j,p] = np.tensordot(pb2[:,:,p], x[i:i+26,j:j+26])


    return pw1, pb1, pk
import time
time1 = time.time()
LR = 0.0009
num_epochs = 7
for epochs in range(num_epochs):
    #Learning rate schedule
    total_correct = 0
    for n in range( len(x_train)):
        n_random = randint(0,len(x_train)-1 )
        y = y_train[n_random]
        x = x_train[n_random][:].reshape(28,28)
        p = forward(x, y, model)
        prediction = np.argmax(p)
        if (prediction == y):
            total_correct += 1

        pw1, pb1, pk = backward(x,y,p, model, model_grads)

        model['b1'] = model['b1'] - LR*pb1.T
        model['W1'] = model['W1'] - LR*pw1
        model['K'] = model['K'] - LR*pk

    # print(epochs)
    #     if n%10000 == 0:
    print(total_correct/np.float(len(x_train) ) )
time2 = time.time()
print(time2-time1)
######################################################
#test data
total_correct = 0
for n in range( len(x_test)):
    y = y_test[n]
    x = x_test[n][:].reshape(28,28)
    p = forward(x, y, model)
    prediction = np.argmax(p)
    if (prediction == y):
        total_correct += 1
print(total_correct/np.float(len(x_test) ) )
