"""
Created on Mon May 12 16:57 2024

Get Natural Images

@author: David
"""
import torch
from torchvision import datasets, transforms
from torchvision.transforms import Lambda
from torch.utils.data import TensorDataset, DataLoader
import os
import matplotlib.pyplot as plt
import numpy as np

# Load CIFAR-10 training data
cwd = os.getcwd()

# Custom transformation: Convert an RGB tensor to grayscale and drop the channel dimension
class ToGrayScale(torch.nn.Module):
    def __init__(self):
        super(ToGrayScale, self).__init__()

    def forward(self, img):
        # Use the formula to convert RGB image to grayscale
        r, g, b = img[0], img[1], img[2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray  # Return 2D tensor without channel dimension

transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    ToGrayScale(),
    Lambda(lambda x: (x - torch.min(x)) / (torch.max(x) - torch.min(x)))  # Normalize the images between 0 and 1
])

cifar100_train = datasets.CIFAR100(root=cwd+'/data', train=True, download=True,transform=transform)

# Load CIFAR-10 test data
cifar100_test = datasets.CIFAR100(root=cwd+'/data', train=False, download=True,transform=transform)

#train_dataset = TensorDataset(cifar100_train, cifar100_train)
train_loader = DataLoader(cifar100_train, batch_size=32, shuffle=True)
#test_dataset = TensorDataset(cifar100_test, cifar100_test)
test_loader = DataLoader(cifar100_test, batch_size=32, shuffle=True)

# ==============================================================================================================================
# SECTION: Visualise 1st 10 images to check data
# ==============================================================================================================================

def convert_img_to_tensor(data):
    # load data
    loader = torch.utils.data.DataLoader(data, batch_size=len(data),shuffle=False)

    # Extract all images and labels
    images, _ = next(iter(loader))
    # Remove the channel dimension and convert to numpy array
    images_tsr = images.view(images.shape[0],-1)

    return images_tsr

cifar100_train_tsr = convert_img_to_tensor(cifar100_train)
cifar100_test_tsr = convert_img_to_tensor(cifar100_test)
# Print the shape of the numpy array
#print(cifar100_train_np.shape)  # Should be (50000, 32, 32)

# Plotting the images
# fig, axes = plt.subplots(1, 10, figsize=(15, 1.5))
# for i in range(10):
#     img = axes[i].imshow(cifar100_train_np[i],cmap='gray')
#     axes[i].set_title('Label: %s' % cifar100_train.classes[cifar100_train[i][1]])
#     axes[i].axis('off')
#     fig.colorbar(img, ax=axes[i], fraction=0.046,aspect=20)
# fig.subplots_adjust(wspace=1)  # Increase the width space
# plt.show()


# cwd = os.getcwd()
# # Check if the directory exists where the file will be saved; create it if it does not exist
# save_path_train = cwd + '/train2017'
# save_path_train_label = cwd + '/train2017_label/'
# # List of all paths you need to check/create
# paths = [save_path_train, save_path_train_label]

# # Loop through each path and create the directory if it doesn't exist
# for path in paths:
#     if not os.path.exists(path):
#         os.makedirs(path, exist_ok=True)

# Ensure you replace 'path_to_images' and 'path_to_annotations' with your local dataset paths
# coco_train = CocoDetection(root=cwd + '/train2017',
#                            annFile=cwd + '/train2017_label/instances_train2017.json',)

# coco_val = CocoDetection(root='path_to_images/val2017',
#                          annFile='path_to_annotations/annotations/instances_val2017.json',
#                          transform=transform)