import numpy as np
import pandas as pd

f_test = np.loadtxt('mnist_test.csv', dtype=np.int, delimiter=',', unpack=False)
y_test = f_test[:, 0]
x_test = f_test[:, 1:]
y_test = y_test.flatten()


y_test = pd.get_dummies(y_test)

np.save('test_x.npy', x_test)
np.save('test_y.npy', y_test)

print(y_test.shape)
print(x_test.shape)


f_train = np.loadtxt('mnist_train.csv', dtype=np.int, delimiter=',', unpack=False)

y_train = f_train[:, 0]
x_train = f_train[:, 1:]
y_train = pd.get_dummies(y_train)

np.save('train_x.npy', x_train)
np.save('train_y.npy', y_train)

print(y_train.shape)
print(y_test.shape)

print("succeed initial data")
