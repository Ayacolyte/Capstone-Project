from Full_model import AutoEncoder
from Natural_Images import cifar100_test,loader2tensor
import os
import torch
import torch.nn as nn
import matplotlib.pyplot  as plt
import math
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

# # Plotting the images
# fig, axes = plt.subplots(1, 10, figsize=(15, 1.5))
# for i in range(10):
#     img = axes[i].imshow(cifar100_test_np[i],cmap='gray')
#     #axes[i].set_title('%s' % cifar100_train.classes[cifar100_train[i][1]])
#     axes[i].axis('off')
#     fig.colorbar(img, ax=axes[i], fraction=0.046,aspect=20)
# fig.subplots_adjust(wspace=1)  # Increase the width space
# plt.show()

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
fig.suptitle('Image Reconstruction Comparison: Linear+modifiedERF', fontsize=16) 
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

# Visualize the weights of the model
visualize_weights(model_loaded)
