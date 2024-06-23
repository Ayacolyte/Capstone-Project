"""
Created on Mon May 12 16:57 2024

Get Natural Images

@author: David
"""
import torch
from torchvision import datasets, transforms
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
    ToGrayScale()
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the images
])

cifar10_train = datasets.CIFAR10(root=cwd+'/data', train=True, download=True,transform=transform)

# Load CIFAR-10 test data
cifar10_test = datasets.CIFAR10(root=cwd+'/data', train=False, download=True,transform=transform)
# ==============================================================================================================================
# SECTION: Convert to Gray Scale
# ==============================================================================================================================


def convert_img_to_numpy(data):
    # load data
    loader = torch.utils.data.DataLoader(data, batch_size=len(data),
                                          shuffle=False)

    # Extract all images and labels
    images, _ = next(iter(loader))
    # Remove the channel dimension and convert to numpy array
    images_np = images.squeeze(1).numpy()

    return images_np

cifar10_train_np = convert_img_to_numpy(cifar10_train)
cifar10_test_np = convert_img_to_numpy(cifar10_test)
# Print the shape of the numpy array
#print(cifar10_train_np.shape)  # Should be (50000, 32, 32)

# ==============================================================================================================================
# SECTION: Visualise 1st 10 images to check data
# ==============================================================================================================================
def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    return

# Plotting the images
fig, axes = plt.subplots(1, 10, figsize=(15, 1.5))
for i in range(10):
    axes[i].imshow(cifar10_train[i][0].numpy())
    axes[i].set_title('Label: %s' % cifar10_train.classes[cifar10_train[i][1]])
    axes[i].axis('off')
plt.show()


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