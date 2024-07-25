from Full_model import AutoEncoder
from Natural_Images import cifar100_test,loader2tensor
import os
import torch
import torch.nn as nn
import matplotlib.pyplot  as plt
import math
import numpy as np

# Load the model state dictionary
cwd = os.getcwd()
model_path = cwd+f'/data/model_lr = 0.0001.pth'
state_dict = torch.load(model_path)

# Instantiate a new model of the same architecture
model_loaded = AutoEncoder()

# Load the state dictionary into the model
model_loaded.load_state_dict(state_dict)

cifar100_test_tsr_flat = loader2tensor(cifar100_test,flatten=True)
cifar100_test_tsr= loader2tensor(cifar100_test,flatten=False)
cifar100_test_np = cifar100_test_tsr.detach().numpy()

output_test_flat, layer1_flat, layer2_flat = model_loaded(cifar100_test_tsr_flat)

side_dim = int(math.sqrt(output_test_flat.shape[1]))
output = output_test_flat.view(output_test_flat.shape[0],side_dim,side_dim)
output_np = output.detach().numpy()

side_dim = int(math.sqrt(layer1_flat.shape[1]))
layer1 = layer1_flat.view(layer1_flat.shape[0],side_dim,side_dim)
layer1_np = layer1.detach().numpy()

side_dim = int(math.sqrt(layer2_flat.shape[1]))
layer2 = layer2_flat.view(layer2_flat.shape[0],side_dim,side_dim)
layer2_np = layer2.detach().numpy()
#########################################################################################################################
# Code block below is for graphical representation of results
#########################################################################################################################
# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def double_sigmoid(x, alpha1, alpha2, beta1, beta2):
    return (1 / (1 + np.exp(alpha1*(x-beta1))))   *   (1 / (1 + np.exp(alpha2*(x-beta2)))) 

def linear(x):
    return x

def ReLU(x):
    return np.maximum(0, x)


# Define the range for x-axis
x = np.linspace(-10, 10, 400)

# Calculate the activation function values
alpha1, alpha2, beta1, beta2 = 5,-5,0.1,-0.1
activation_2sig = double_sigmoid(x,alpha1, alpha2, beta1, beta2)
activation_linear= linear(x)
activation_ReLU = ReLU(x)


def visualize_weights(model):
    layers = [model.layer1, model.LNL_model, model.layer3]
    activs = [activation_linear, activation_2sig, activation_ReLU]
    fig, axs = plt.subplots(1, len(layers), figsize=(20, 5))

    for i, layer in enumerate(layers):
        weights = layer.weight.data.numpy().flatten()
        #axs[i].imshow(weights, aspect='auto', cmap='gray')
        axs[i].hist(weights, density = True, bins=50, color='blue', edgecolor='black', label='Weights Distribution')
        axs[i].set_title(f'Layer {i + 1} Weights')
        axs[i].set_xlabel('Weights')
        axs[i].set_ylabel('Normalised Frequency')
        axs[i].set_ylim(-0.5, 65)
        axs[i].set_xlim(np.min(weights), np.max(weights))

        ax2 = axs[i].twinx()
        ax2.plot(x, activs[i], 'r', label='Activation Function')
        ax2.set_ylim(-0.5, 1)

        axs[i].legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize='small')
        ax2.legend(loc='upper right', bbox_to_anchor=(1, 0.92), fontsize='small')

    plt.suptitle('Weights Distribution and Activation Functions: Linear+2sig ')
    plt.show()

# Visualize the weights of the model
visualize_weights(model_loaded)
