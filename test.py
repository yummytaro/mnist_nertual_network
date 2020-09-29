import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

dictionary = {'hello':'world'}
np.save('my_file.npy', dictionary)
read_dictionary = np.load('my_file.npy',allow_pickle=True).item()
print(read_dictionary['hello']) # displays "world"

parameters = np.load('parameters.npy',allow_pickle=True).item()
cache = np.load('cache.npy',allow_pickle=True).item()
cache_d = np.load('cache_d.npy',allow_pickle=True).item()


layer = [100, 50]
l = len(layer) + 1
iteration = 10000


def check(a_hat):
    return cp.argmax(a_hat, axis=0, keepdims=True)


def accuracy(a_hat, a_true):
    result = cp.equal(a_hat, a_true)
    right = cp.count_nonzero(result)
    return right / m


def softmax(x):
    x_max = cp.amax(x, axis=0, keepdims=True)
    sum = cp.sum(cp.exp(x), axis=0)
    ans = cp.exp(x) / sum
    return ans


def load(pathx, pathy):
    x = cp.load(pathx, 'r')
    y = cp.load(pathy, 'r')
    return x, y


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


test_x, test_y = load('test_x.npy', 'test_y.npy')

test_y = cp.transpose(test_y)
test_x = cp.transpose(test_x)
print(test_x.shape)
print(test_y.shape)
test_y_ans = cp.argmax(test_y, axis=0, keepdims=True)
m = test_x.shape[1]
test_x = test_x / 128

cache['a0'] = test_x

a_hat = forward()
a_hat = check(a_hat)
print(a_hat)
print(test_y_ans)
accuracy = accuracy(a_hat, test_y_ans)
print(accuracy)
