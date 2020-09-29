import cupy as cp
import pandas as pd
import pathlib as Path
import numpy as np
import matplotlib.pyplot as plt

layer = [100,50]
parameters = {}
cache = {}
cache_d = {}
l = len(layer) + 1
iteration = 10000
plot_train = []
plot_test = []
plot_iteration = []
plot_accuracy = []
cache_test = {}


def softmax(x):
    x_max = cp.amax(x, axis=0, keepdims=True)
    sum = cp.sum(cp.exp(x), axis=0)
    ans = cp.exp(x) / sum
    return ans


def load(pathx, pathy):
    x = cp.load(pathx, 'r')
    y = cp.load(pathy, 'r')
    return x, y


def initial(x, y):
    l1 = x.shape[0]
    ln = y.shape[0]
    a0 = l1
    for i in range(l):
        if i == l - 1:
            a1 = ln
        else:
            a1 = layer[i]
        w = cp.random.rand(a1, a0)*0.01
        b = cp.zeros((a1, 1))
        if i != l - 1:
            a0 = layer[i]
        parameters['w' + str(i + 1)] = w
        parameters['b' + str(i + 1)] = b
    return parameters

def forward():
    for i in range(l):
        a = cache['a' + str(i)]
        w = parameters['w' + str(i + 1)]
        b = parameters['b' + str(i + 1)]

        z = cp.dot(w, a) + b
        if i != l - 1:
            a = cp.tanh(z)
        else:
            a = softmax(z)
        cache['a' + str(i + 1)] = a
        cache['z' + str(i + 1)] = z
    return a


def softmax_dev(a_true, a_hat):
    return a_hat - a_true


def tanh_dev(z):
    return 1 - cp.tanh(z) * cp.tanh(z)


def loss(a_true, a_hat):
    loss = -cp.sum(a_true * cp.log(a_hat)) / a_true.shape[1]
    return loss


def backward(y):
    for i in range(l, 0, -1):
        w = parameters['w' + str(i)]
        a_ = cache['a' + str(i - 1)]
        if i == l:
            a = cache['a' + str(i)]
            dz = softmax_dev(y, a)
        else:
            da = cache_d['da' + str(i)]
            z = cache['z' + str(i)]
            dz = da * tanh_dev(z)
        dw = cp.dot(dz, cp.transpose(a_)) / y.shape[1]
        db = cp.sum(dz, axis=1, keepdims=True) / y.shape[1]
        da_ = cp.dot(cp.transpose(w), dz)
        cache_d['dw' + str(i)] = dw
        cache_d['db' + str(i)] = db
        cache_d['da' + str(i - 1)] = da_


def update(learningRate):
    for i in range(l):
        w = parameters['w' + str(i + 1)]
        b = parameters['b' + str(i + 1)]
        dw = cache_d['dw' + str(i + 1)]
        db = cache_d['db' + str(i + 1)]
        w -= dw * learningRate
        b -= db * learningRate
        parameters['w' + str(i + 1)] = w
        parameters['b' + str(i + 1)] = b


def check(a_hat):
    return cp.argmax(a_hat, axis=0, keepdims=True)


def accuracy(a_hat, a_true):
    result = cp.equal(a_hat, a_true)
    right = cp.count_nonzero(result)
    return right / a_hat.shape[1]


def forward_y(test_y):
    for i in range(l):
        a = cache_test['a' + str(i)]
        w = parameters['w' + str(i + 1)]
        b = parameters['b' + str(i + 1)]

        z = cp.dot(w, a) + b
        if i != l - 1:
            a = cp.tanh(z)
        else:
            a = softmax(z)
        cache_test['a' + str(i + 1)] = a
        cache_test['z' + str(i + 1)] = z
    return a


train_x = cp.load('train_x.npy')
train_y = cp.load('train_y.npy')
train_y = cp.transpose(train_y)
train_x = cp.transpose(train_x)

test_x, test_y = load('test_x.npy', 'test_y.npy')

test_y = cp.transpose(test_y)
test_x = cp.transpose(test_x)
test_y_ans = cp.argmax(test_y, axis=0, keepdims=True)
m = test_x.shape[1]
test_x = test_x / 128
cache['a0'] = test_x

train_y_ans = cp.argmax(train_y, axis=0, keepdims=True)

m = train_x.shape[1]

train_x = train_x / 128
# in order to have a better perfomrance ==> z is at (0,1)

cache['a0'] = train_x
cache_test['a0'] = test_x
initial(train_x, train_y)
for i in range(iteration):
    a = forward()
    da = loss(train_y, a)
    cache_d['loss'] = da
    backward(train_y)
    update(0.1)
    if i % 100 == 0:
        a_ = forward_y(test_y)
        da_ = loss(test_y, a_)
        a_hat = check(a_)
        acc = accuracy(a_hat, test_y_ans)
        print(str(i) + ': train_loss = ' + str(da))
        print(str(i) + ': test_loss = ' + str(da_))
        print((str(i) + ': test_accuracy = ' + str(acc)))
        plot_iteration.append(i)
        plot_test.append(da_)
        plot_train.append(da)
        plot_accuracy.append(acc)

np.save('parameters.npy', parameters)
np.save('cache.npy', cache)
np.save('cache_d.npy', cache_d)

print(plot_test)
print(plot_iteration)
print(plot_train)
print(plot_accuracy)

ln1, = plt.plot(plot_iteration, plot_test, color='red', linewidth=3.0, linestyle='--')
ln2, = plt.plot(plot_iteration, plot_train, color='blue', linewidth=2.0, linestyle='-.')
ln3, = plt.plot(plot_iteration, plot_accuracy, color='green', linewidth=3.0, linestyle='--')

plt.legend(handles=[ln1, ln2, ln3], labels=['test_loss', 'train_loss', 'test_accuracy'])

plt.show()

np.save('plot_test.npy', plot_test)
np.save('plot_iteration.npy', plot_iteration)
np.save('plot_train.npy', plot_train)
np.save('plot_accuracy.npy', plot_accuracy)
