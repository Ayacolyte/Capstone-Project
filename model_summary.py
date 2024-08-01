#from Full_model import AutoEncoder
#from Natural_Images import cifar100_test,loader2tensor
import os
import torch
import torch.nn as nn
import matplotlib.pyplot  as plt
import math
import numpy as np


def show_lnr_combination(model_path,AutoEncoder, activ_funcs, magnitude, shift, model_descrip):
    # import input images
    from Natural_Images import cifar100_test,loader2tensor

    state_dict = torch.load(model_path)
    model_loaded = AutoEncoder()

    # Load the state dictionary into the model
    model_loaded.load_state_dict(state_dict)

    cifar100_test_tsr_flat = loader2tensor(cifar100_test,flatten=True)
    cifar100_test_np_flat = cifar100_test_tsr_flat.detach().numpy()

    output_test_flat, layer1_flat, layer2_flat = model_loaded(cifar100_test_tsr_flat)

    side_dim = int(math.sqrt(output_test_flat.shape[1]))
    output = output_test_flat.view(output_test_flat.shape[0],side_dim,side_dim)
    output_np = output.detach().numpy()

    side_dim = int(math.sqrt(layer1_flat.shape[1]))
    layer1_flat_np = layer1_flat.detach().numpy()

    side_dim = int(math.sqrt(layer2_flat.shape[1]))
    layer2_flat_np = layer2_flat.detach().numpy()
    #########################################################################################################################
    # Code block below is for graphical representation of results
    #########################################################################################################################
    # Define the sigmoid activation function

    def linear(x):
        return x

    def ReLU(x):
        return np.maximum(0, x)

    # Define the range for x-axis
    x = np.linspace(-10, 10, 10000)


    def activ_func(activ_func1, magnitude, shift):
        if activ_func1 == "ReLU":
            return ReLU(x)
        elif activ_func1 == "linear":
            return linear(x)
        elif activ_func1 == "2sig":
            sigmoid1 = 1/(1+np.exp(magnitude * (x + shift)))
            sigmoid2 = 1/(1+np.exp(-1*magnitude * (x - shift)))
            return sigmoid1 + sigmoid2
        
    activs = []
    for curr_activ_func in activ_funcs:
        activs.append(activ_func(curr_activ_func, magnitude, shift))

    def visualize_weights(model, activs):
        layers = [model.layer1, model.LNL_model, model.layer3]
        inputs = [cifar100_test_np_flat[0], layer1_flat_np[0], layer2_flat_np[0]]
        #activs = [activation_linear, activation_2sig, activation_ReLU]
        fig, axs = plt.subplots(1, len(layers), figsize=(20, 5))

        for i, layer in enumerate(layers):
            weights = layer.weight.data.numpy()
            input = inputs[i]*weights[0,:]
            #axs[i].imshow(weights, aspect='auto', cmap='gray')
            axs[i].hist(input, density = True, bins=50, color='blue', edgecolor='black', label='Weighed Input Distribution')
            axs[i].set_title(f'Layer {i + 1} Weights')
            axs[i].set_xlabel('Weights')
            axs[i].set_ylabel('Normalised Frequency')
            #axs[i].set_ylim(-0.5, 65)
            axs[i].set_xlim(np.min(weights[0,:]), np.max(weights[0,:]))

            ax2 = axs[i].twinx()
            ax2.plot(x, activs[i], 'r', label='Activation Function')
            ax2.set_ylim(-0.2, 0.5)

            axs[i].legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize='small')
            ax2.legend(loc='upper right', bbox_to_anchor=(1, 0.92), fontsize='small')

        plt.suptitle(f"Weighed Input Distribution and Activation Functions:{model_descrip}")
        plt.show()

    # Visualize the weights of the model
    visualize_weights(model_loaded, activs)





