import numpy as np
import matplotlib.pyplot as plt

test = np.load('plot_test.npy', allow_pickle=True)
iteration = np.load('plot_iteration.npy', allow_pickle=True)
train = np.load('plot_train.npy', allow_pickle=True)
accuracy = np.load('plot_accuracy.npy', allow_pickle=True)

j = 0
for i in test:
    print("(" + str(j) + " " + str(np.around(i, decimals=2)) + ")")
    j += 100

j = 0
for i in train:
    print("(" + str(j) + " " + str(np.around(i, decimals=2)) + ")")
    j += 100

j = 0
for i in accuracy:
    print("(" + str(j) + " " + str(np.around(i, decimals=2)) + ")")
    j += 100

ln1, = plt.plot(iteration, test, color='red', linewidth=3.0, linestyle='--')
ln2, = plt.plot(iteration, train, color='blue', linewidth=2.0, linestyle='-.')
ln3, = plt.plot(iteration, accuracy, color='green', linewidth=3.0, linestyle='--')
plt.legend(handles=[ln1, ln2, ln3], labels=['test_loss', 'train_loss', 'test_accuracy'])
plt.show()
