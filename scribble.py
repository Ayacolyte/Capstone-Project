import os
import pickle
import matplotlib as plt
import numpy as np

cwd = os.getcwd()
# with open(cwd + '/data/NN_output_50epochs.pkl', 'rb') as file:
#     data = pickle.load(file)

#     print(data)

# with open(cwd + '/data/NN_output_200epochs.pkl', 'rb') as file:
#     data = pickle.load(file)

#     print(data)

with open(cwd + '/data/NN_output.pkl', 'rb') as file:
    data = pickle.load(file)

    print(data)

N_epoch = 50
x = np.arange(1, N_epoch + 1)
labels = ['lr=0.1', 'lr = 0.01', 'lr = 0.001', 'lr = 0.0001']
for i in range(data.shape[0]):
    plt.plot(x, data[i], label=labels[i])