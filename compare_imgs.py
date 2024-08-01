#from Full_model import AutoEncoder

import os
import torch
import torch.nn as nn
import matplotlib.pyplot  as plt
import math
# #from LNL import elec_side_dim
# elec_side_dim = 6
# # Load the model state dictionary
# cwd = os.getcwd()
# model_path = cwd+f'/data/240729_data/model_lr = 0.0001_lnr_6elec.pth'

#########################################################################################################################
# Code block below is for graphical representation of results
#########################################################################################################################

# # Plotting the images
# fig, axes = plt.subplots(1, 10, figsize=(15, 1.5))
# for i in range(10):
#     img = axes[i].imshow(cifar100_test_np[i],cmap='gray')
#     #axes[i].set_title('%s' % cifar100_train.classes[cifar100_train[i][1]])
#     axes[i].axis('off')
#     fig.colorbar(img, ax=axes[i], fraction=0.046,aspect=20)
# fig.subplots_adjust(wspace=1)  # Increase the width space
# plt.show()
def show_img_compare(model_path, AutoEncoder,model_descrip):
    from Natural_Images import cifar100_test,loader2tensor
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

    # plot reconstructions
    fig, axs = plt.subplots(4, 5,figsize=(15, 6))
    for i in range(4):
        for j in range(5):
            if  i == 3:
                #img = convert
                img = axs[i,j].imshow(output_np[4*j],cmap='gray')
                axs[i,j].axis('off')
                axs[i,j].set_title('Reconstruction')
            elif i == 2:
                img = axs[i,j].imshow(layer2_np[4*j],cmap='gray')
                axs[i,j].axis('off')
                axs[i,j].set_title(f'Neural Activation')
            elif i == 1:
                img = axs[i,j].imshow(layer1_np[4*j],cmap='gray')
                axs[i,j].axis('off')
                axs[i,j].set_title(f'Electrodes Activation')
            elif i == 0:
                img = axs[i,j].imshow(cifar100_test_np[4*j],cmap='gray')
                axs[i,j].axis('off')
                axs[i,j].set_title(f'original image')
            fig.colorbar(img, ax=axs[i, j], fraction=0.046, pad=0.04)    
    fig.suptitle(f"Image Reconstruction Comparison:{model_descrip}", fontsize=16) 
    plt.show()


def visualize_weights(model):
    layers = [model.layer1, model.LNL_model, model.layer3]
    fig, axs = plt.subplots(1, len(layers), figsize=(20, 5))

    for i, layer in enumerate(layers):
        weights = layer.weight.data.numpy()
        axs[i].imshow(weights.transpose(), aspect='equal', cmap='gray')
        axs[i].set_title(f'Layer {i + 1} Weights')
        axs[i].set_xlabel('Neurons')
        axs[i].set_ylabel('Input Features')
        fig.colorbar(axs[i].images[0], ax=axs[i])

    plt.show()


 #visualize_weights(model_loaded)

def visualize_img_recep(model_path, AutoEncoder, img_side_dim, elec_side_dim):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    state_dict = torch.load(model_path)

    # Instantiate a new model of the same architecture
    model_loaded = AutoEncoder()

    # Load the state dictionary into the model
    model_loaded.load_state_dict(state_dict)

    fig, axs = plt.subplots(elec_side_dim, elec_side_dim,figsize=(25, 25))
    axs = axs.flatten()
    for i in range(elec_side_dim**2):
        weights = model_loaded.layer1.weight.data.numpy()
        curr_recep = np.squeeze(weights[i,:])
        curr_recep = curr_recep.reshape(img_side_dim, img_side_dim)
        img = axs[i].imshow(curr_recep, aspect='equal', cmap='gray')
        axs[i].set_title(f'Electrode {i + 1}')
        #axs[i].set_xlabel('Neurons')
        #axs[i].set_ylabel('Input Features')
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        divider = make_axes_locatable(axs[i])
        
        # Append an axes to the right of the current axes with the same height
        cax = divider.append_axes("right", size="5%", pad=0.05)
        
        # Create the colorbar in the new axes
        fig.colorbar(img, cax=cax)
    fig.suptitle('Image Receptive Fields', fontsize=16)
    plt.tight_layout()  # Add space for the title
    plt.show()

# Visualize the weights of the model
    #visualize_img_recep(model_loaded)
