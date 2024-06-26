import os
import pickle

cwd = os.getcwd()
with open(cwd + '/data/NN_output.pkl', 'rb') as file:
    data = pickle.load(file)

    print(data)