import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
#from LNL import W_d
import math

import torch
import torch.nn as nn
#from LNL import LNL_model_path  # Import LNL_model from LNL script
#from Full_model import AutoEncoder
cwd = os.getcwd()
# with open(cwd + '/data/NN_output_50epochs.pkl', 'rb') as file:
#     data = pickle.load(file)

#     print(data)

# with open(cwd + '/data/NN_output_200epochs.pkl', 'rb') as file:
#     data = pickle.load(file)

#     print(data)
def show_error(data_path,model_descrip):
    with open(data_path, 'rb') as file:
        data = pickle.load(file)

        #print(data[0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    N_epoch = data[0].shape[0]
    x = np.arange(1, N_epoch + 1)
    labels = ['lr=0.01', 'lr = 0.001', 'lr = 0.0001', 'lr = 0.00001']


    log_data = data
    for i in range(data[0].shape[1]) :
        for j in range(data[0].shape[0]):

            log_data[0][j,i] = math.log(data[0][j,i])
        #print(i)
        ax1.plot(x, log_data[0][:,i], label=labels[i])

    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Log Scale Error')
    ax1.set_title(f'Training Loss over 50 Epochs: {model_descrip}')
    ax1.set_ylim(-5, 0)
    ax1.legend()

    for i in range(data[1].shape[1]) :
        for j in range(data[1].shape[0]):

            log_data[1][j,i] = math.log(data[1][j,i])
        #print(i)
        ax2.plot(x, log_data[1][:,i], label=labels[i])

    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Log Scale Error')
    ax2.set_title(f'Validation Loss over 50 Epochs: {model_descrip}')
    ax2.set_ylim(-5, 0)
    ax2.legend()

    # Adjust layout
    plt.tight_layout()
    plt.savefig(f'{model_descrip}_Error.png', format='png')
    # Show plot
    plt.show(block=True)

def comp_error(data_path,model_descrip):
    with open(data_path, 'rb') as file:
        data = pickle.load(file)

        N_epoch = data[0].shape[0]
        x = np.arange(1, N_epoch + 1)
        print(data[0])
        log_data = data
        for i in range(data[0].shape[1]) :
            for j in range(data[0].shape[0]):

                log_data[0][j,i] = math.log(data[0][j,i])
            #print(i)

        for i in range(data[1].shape[1]) :
            for j in range(data[1].shape[0]):

                log_data[1][j,i] = math.log(data[1][j,i])
            #print(i)
        plt.figure(figsize=(10, 5))
        plt.plot(x,log_data[0][:], label='Training Error', color='blue')  # Skip the first entry if it contains the mean
        plt.plot(x,log_data[1][:], label='Validation Error', color='red')  # Skip the first entry if it contains the mean
        
        plt.title('Training and Validation Error over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Log Scale Error')
        plt.legend()
        plt.savefig(f'{model_descrip}_Error_compare.png', format='png')
        plt.show()
# # Load the model state dictionary
# model_path = cwd+f'/data/model_1.pth'
# state_dict = torch.load(model_path)

# # Instantiate a new model of the same architecture
# model_loaded = AutoEncoder()

# # Load the state dictionary into the model
# model_loaded.load_state_dict(state_dict)

# # Access the weights of LNL layer
# weights_LNL = model_loaded.LNL_model.weight

# print("Original Weights:")
# print(W_d[3:13,0])
# print("Weights of LNL layer:")
# print(weights_LNL[3:13,0].detach().numpy())

# assert np.array_equal(W_d, weights_LNL.detach().numpy()) 
