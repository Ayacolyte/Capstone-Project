#from Full_model import AutoEncoder
#from Natural_Images import cifar100_test,loader2tensor
import os
import torch
import torch.nn as nn
import matplotlib.pyplot  as plt
import math
import numpy as np

def visualize_weights(model,model_path):
    state_dict = torch.load(model_path)
    model_loaded = model()

    # Load the state dictionary into the model
    model_loaded.load_state_dict(state_dict)
    layers = [model_loaded.layer1, model_loaded.LNL_model, model_loaded.layer3]
    fig, axs = plt.subplots(1, len(layers), figsize=(20, 5))

    for i, layer in enumerate(layers):
        weights = layer.weight.data.numpy().transpose()
        axs[i].imshow(weights, cmap='gray')
        axs[i].set_title(f'Layer {i + 1} Weights')
        axs[i].set_xlabel('Neurons')
        axs[i].set_aspect('equal')
        axs[i].set_ylabel('Input Features')
        fig.colorbar(axs[i].images[0], ax=axs[i])

    plt.show()

def show_generator(model_path,AutoEncoder, activ_funcs, magnitude, shift, model_descrip,execution_profile):
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

    # Define the range for x-axis
    x = np.linspace(-10, 10, 10000)




    def visualize_weights(model):

        layers = [model.layer1, model.LNL_model, model.layer3]
        inputs = [cifar100_test_np_flat, layer1_flat_np, layer2_flat_np]
        #activs = [activation_linear, activation_2sig, activation_ReLU]
        fig, axs = plt.subplots(1, len(layers), figsize=(20, 5))

        for i, layer in enumerate(layers):
            if i == 0 and (execution_profile == "CNN" or execution_profile == "CNN_pool"):
                gnrtr_mat = 0
            else:

                weights = layer.weight.data.numpy()
                if i == 1:
                    biases = [0]
                else:
                    biases =  layer.bias.data.numpy()
                input = inputs[i]
                gnrtr_mat = []
        
                for j in range(input.shape[0]):

                    curr_generator_1stelec = np.dot(input[j,:],weights[0,:]) + biases[0]
                    gnrtr_mat.append(curr_generator_1stelec)
            x = np.linspace(-1*max(np.abs(np.max(gnrtr_mat)), np.abs(np.min(gnrtr_mat))),max(np.abs(np.max(gnrtr_mat)), np.abs(np.min(gnrtr_mat))),40000)
            def activ_func(af, magnitude, shift):
                if af == "ReLU":
                    return np.maximum(0, x)
                elif af == "linear":
                    return x
                elif af == 'sig':
                    return 1/(1+np.exp(-1*magnitude * (x - shift)))
                elif af == "2sig":
                    sigmoid1 = 1/(1+np.exp(magnitude * (x + shift)))
                    sigmoid2 = 1/(1+np.exp(-1*magnitude * (x - shift)))
                    return sigmoid1 + sigmoid2
            
            activs = []
            for curr_activ_func in activ_funcs:
                activs.append(activ_func(curr_activ_func, magnitude, shift))
            #print(weights.shape)
            #axs[i].imshow(weights, aspect='auto', cmap='gray')
            axs[i].hist(gnrtr_mat, density = True, bins=50, color='blue', edgecolor='black', label='Generator Function')
            axs[i].set_title(f'Layer {i + 1} Generator Function')
            axs[i].set_xlabel('Generator Function')
            axs[i].set_ylabel('Normalised Frequency')
            #axs[i].set_ylim(-0.5, 65)
            axs[i].set_xlim(-1 * max(np.abs(np.max(gnrtr_mat)), np.abs(np.min(gnrtr_mat))),max(np.abs(np.max(gnrtr_mat)), np.abs(np.min(gnrtr_mat))))

            ax2 = axs[i].twinx()
            ax2.plot(x, activs[i], 'r', label='Activation Function')
            #ax2.set_ylim(-0.2, 2)

            axs[i].legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize='small')
            ax2.legend(loc='upper right', bbox_to_anchor=(1, 0.92), fontsize='small')

        plt.suptitle(f"Generator Function Distribution 1st electrode and Activation Functions:{model_descrip}")
        plt.savefig(f"{model_descrip}_GFandAF.png", format='png')
        plt.show()

    # Visualize the weights of the model
    visualize_weights(model_loaded)



# def show_LNL_weights(model_path,AutoEncoder, activ_funcs, magnitude, shift, model_descrip):
#     # import input images
#     from Natural_Images import cifar100_test,loader2tensor

#     state_dict = torch.load(model_path)
#     model_loaded = AutoEncoder()

#     # Load the state dictionary into the model
#     model_loaded.load_state_dict(state_dict)

#     cifar100_test_tsr_flat = loader2tensor(cifar100_test,flatten=True)
#     cifar100_test_np_flat = cifar100_test_tsr_flat.detach().numpy()

#     output_test_flat, layer1_flat, layer2_flat = model_loaded(cifar100_test_tsr_flat)

#     side_dim = int(math.sqrt(output_test_flat.shape[1]))
#     output = output_test_flat.view(output_test_flat.shape[0],side_dim,side_dim)
#     output_np = output.detach().numpy()

#     side_dim = int(math.sqrt(layer1_flat.shape[1]))
#     layer1_flat_np = layer1_flat.detach().numpy()

#     side_dim = int(math.sqrt(layer2_flat.shape[1]))
#     layer2_flat_np = layer2_flat.detach().numpy()
#     #########################################################################################################################
#     # Code block below is for graphical representation of results
#     #########################################################################################################################
#     # Define the sigmoid activation function

#     # Define the range for x-axis
#     x = np.linspace(-10, 10, 10000)


#     def activ_func(activ_func1, magnitude, shift):
#         if activ_func1 == "ReLU":
#             return np.maximum(0, x)
#         elif activ_func1 == "linear":
#             return x
#         elif activ_func1 == "2sig":
#             sigmoid1 = 1/(1+np.exp(magnitude * (x + shift)))
#             sigmoid2 = 1/(1+np.exp(-1*magnitude * (x - shift)))
#             return sigmoid1 + sigmoid2
        
#     activs = []
#     for curr_activ_func in activ_funcs:
#         activs.append(activ_func(curr_activ_func, magnitude, shift))

#     def visualize_weights(model, activs):
#         layers = [model.LNL_model]
#         #inputs = [cifar100_test_np_flat[0], layer1_flat_np[0], layer2_flat_np[0]]
#         #activs = [activation_linear, activation_2sig, activation_ReLU]
#         fig, axs = plt.subplots(1, 3, figsize=(20, 5))

#         for i, layer in enumerate(layers):
#             weights = layer.weight.data.numpy()
#             input = inputs[i]*weights[0,:]
#             #axs[i].imshow(weights, aspect='auto', cmap='gray')
#             axs[i].hist(input, density = True, bins=50, color='blue', edgecolor='black', label='Weighed Input Distribution')
#             axs[i].set_title(f'Layer {i + 1} Weights')
#             axs[i].set_xlabel('Weights')
#             axs[i].set_ylabel('Normalised Frequency')
#             #axs[i].set_ylim(-0.5, 65)
#             axs[i].set_xlim(np.min(input[:]), np.max(input[:]))

#             ax2 = axs[i].twinx()
#             ax2.plot(x, activs[i], 'r', label='Activation Function')
#             ax2.set_ylim(-0.2, 0.5)

#             axs[i].legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize='small')
#             ax2.legend(loc='upper right', bbox_to_anchor=(1, 0.92), fontsize='small')

#         plt.suptitle(f"Weighed Input Distribution and Activation Functions:{model_descrip}")
#         plt.show()

#     # Visualize the weights of the model
#     visualize_weights(model_loaded, activs)



