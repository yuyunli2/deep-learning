import numpy as np
import h5py
import time
import copy
from random import randint
#load MNIST data
MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:] )
X = np.array(x_train)
print(X.shape)
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

model['K'] = np.random.randn(5,5) / np.sqrt(num_inputs)
model_grads = copy.deepcopy(model['K'])
model['Z'] = np.random.randn(780,780) / np.sqrt(num_inputs)
model['H1'] = np.random.randn(780,780) / np.sqrt(num_inputs)
model['W1'] = np.random.randn(num_outputs, 780, 780) / np.sqrt(num_inputs)
model['b1'] = np.random.randn(1, num_outputs) / np.sqrt(num_inputs)
model['U1'] = np.random.randn(1, num_outputs) / np.sqrt(num_inputs)
model['U2'] = np.random.randn(1, num_outputs) / np.sqrt(num_inputs)
model['W2'] = np.random.randn(num_outputs, num_outputs) / np.sqrt(num_inputs)
model['b2'] = np.random.randn(1, num_outputs) / np.sqrt(num_inputs)

model['pu2'] = np.random.randn(1, num_outputs) / np.sqrt(num_inputs)
model['pb2'] = np.random.randn(1, num_outputs) / np.sqrt(num_inputs)
model['pw2'] = np.random.randn(1, num_outputs) / np.sqrt(num_inputs)
model['pb1'] = np.random.randn(1, num_outputs) / np.sqrt(num_inputs)
model['pb2'] = np.random.randn(1, num_outputs) / np.sqrt(num_inputs)


def softmax_function(z):
    ZZ = np.exp(z) / np.sum(np.exp(z))
    return ZZ


for i in range(780):
    for j in range(780):
        # y_sum = np.sum(np.tensordot(model['K'], X[i:i+5,j:j+5]), axis=0)
        # # print('y_sum', y_sum)
        # # print("y_sum.shape", y_sum.shape)
        # x_sum = np.sum(y_sum, axis=1)
        model['Z'][i,j] = np.tensordot(model['K'], X[i:i+5,j:j+5])
model['H1'] = 1/(1+np.exp(-model['Z']))

def forward(x,y, model):
    for i in range(780):
        for j in range(780):
            # y_sum = np.sum(np.tensordot(model['K'], X[i:i+5,j:j+5]), axis=0)
            # # print('y_sum', y_sum)
            # # print("y_sum.shape", y_sum.shape)
            # x_sum = np.sum(y_sum, axis=1)
            model['Z'][i, j] = np.tensordot(model['K'], X[i:i + 5, j:j + 5])
    model['H1'] = 1 / (1 + np.exp(-model['Z']))

    for i in range(10):
        # y_sum2 = np.sum(np.tensordot(model['W1'][i,:,:], model['H1']))
        # x_sum2 = np.sum(y_sum2, axis=1)
        x_sum2 = np.tensordot(model['W1'][i,:,:], model['H1'])
        model['U1'][0,i] = x_sum2 + model['b1'][0,i]
    model['H2'] = 1/(1+np.exp(-model['U1']))
    # print('U1', model['U1'].shape)
    # print('H2', model['H2'].shape)
    for i in range(10):
        model['U2'][0,i] = np.dot(model['W2'][i,:], model['H2'].T) + model['b2'][0,i]
    f = softmax_function(model['U2'])
    return f

def backward(x,y,f, model, model_grads):
    pu = f
    pu[0,y] = pu[0,y] - 1
    pb2 = pu
    # pc = np.dot(model['H'].T, pu)
    pw2 = np.dot(model['H2'].T, pu)
    # print('pw2',pw2.shape)
    dsigma = model['H2'] - model['H2']**2
    pb1 = np.dot(pw2,  dsigma.T)
    # pw1 = np.dot(pb1, x)
    pw1 = np.random.randn(num_outputs, 780, 780) / np.sqrt(num_inputs)
    for i in range(10):
        pw1[i,:,:] = (pb1[i,0] * model['H1']).reshape(1,780,780)
    pk1 = pw1

    return pw2, pb2, pb1, pw1
import time
time1 = time.time()
LR = .001
num_epochs = 100
for epochs in range(num_epochs):
    #Learning rate schedule
    total_correct = 0
    n_random = randint(0,len(x_train)-1 )
    y = y_train[n_random]
    x = x_train[n_random][:].reshape(1,784)
    p = forward(x, y, model)
    prediction = np.argmax(p)
    if (prediction == y):
        total_correct += 1

    pc, pb2, pb1, pw1 = backward(x,y,p, model, model_grads)
    model['W2'] = model['W2'] - LR*pc
    model['b2'] = model['b2'] - LR*pb2
    model['b1'] = model['b1'] - LR*pb1.T
    model['W1'] = model['W1'] - LR*pw1
    # if epochs % 10 == 0:
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
