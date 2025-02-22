"""
Created on Mon May 12 16:57 2024

Get Natural Images

@author: David
"""
import torch
from torchvision import datasets
from torchvision.transforms import v2
from torchvision.transforms.v2 import Lambda
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
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
normalise = Lambda(lambda x: (x - torch.min(x)) / (torch.max(x) - torch.min(x)))
transform = v2.Compose([
    v2.ToTensor(),  # Convert images to PyTorch tensors
    ToGrayScale(),
    normalise # Normalize the images between 0 and 1
])

# Mirrored and rotated (90 degrees) transformation
mirrored_transform = v2.Compose([v2.ToTensor(), v2.RandomVerticalFlip(p=1.0), ToGrayScale(),normalise])
rotation_transform = v2.Compose([v2.ToTensor(),  v2.RandomRotation((90,90)),  ToGrayScale(), normalise])

augment = False
if augment:
    cifar100_train_og = datasets.CIFAR100(root=cwd+'/data', train=True, download=True,transform=transform)
    cifar100_train_mir = datasets.CIFAR100(root=cwd+'/data', train=True, download=True,transform=mirrored_transform)
    #cifar100_train_rot = datasets.CIFAR100(root=cwd+'/data', train=True, download=True,transform=rotation_transform)
    cifar100_train = ConcatDataset([cifar100_train_og, cifar100_train_mir])
    # Load CIFAR-10 test data
    cifar100_test_og = datasets.CIFAR100(root=cwd+'/data', train=False, download=True,transform=transform)
    cifar100_test_mir = datasets.CIFAR100(root=cwd+'/data', train=False, download=True,transform=mirrored_transform)
    #cifar100_test_rot = datasets.CIFAR100(root=cwd+'/data', train=False, download=True,transform=rotation_transform)
    cifar100_test = ConcatDataset([cifar100_test_og, cifar100_test_mir])
else:
    cifar100_train = datasets.CIFAR100(root=cwd+'/data', train=True, download=True,transform=transform)
    # Load CIFAR-10 test data
    cifar100_test = datasets.CIFAR100(root=cwd+'/data', train=False, download=True,transform=transform)

#train_dataset = TensorDataset(cifar100_train, cifar100_train)
train_loader = DataLoader(cifar100_train, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
#test_dataset = TensorDataset(cifar100_test, cifar100_test)
test_loader = DataLoader(cifar100_test, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
print('DataLoaders ready')
# ==============================================================================================================================
# SECTION: Visualise 1st 10 images to check data
# ==============================================================================================================================

def loader2tensor(data, flatten=False):
    loader = torch.utils.data.DataLoader(data, batch_size=1024, shuffle=False, pin_memory=True, num_workers=4)
    images = []

    for batch, _ in loader:
        images.append(batch)

    images = torch.cat(images, dim=0)  # Concatenate all batches
    if flatten:
        images = images.view(images.shape[0], -1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = 'cpu'
    return images.to(device, non_blocking=True)




cifar100_train_tsr = loader2tensor(cifar100_train,flatten=False)
cifar100_test_tsr = loader2tensor(cifar100_test,flatten=False)
cifar100_train_tsr_flat = cifar100_train_tsr.view(cifar100_train_tsr.size(0), -1)
cifar100_test_tsr_flat = cifar100_test_tsr.view(cifar100_test_tsr.size(0), -1)

print(f'Convesion to tensor on {cifar100_train_tsr_flat.device} completed')
# Print the shape of the numpy array
#print(cifar100_train_np.shape)  # Should be (50000, 32, 32)

# #Plotting the images
# fig, axes = plt.subplots(1, 10, figsize=(15, 1.5))
#cifar100_train_np = cifar100_train_tsr.numpy()
#cifar100_test_np = cifar100_test_tsr.numpy()
# for i in range(10):
#     img = axes[i].imshow(cifar100_train_np[i],cmap='gray')
#     axes[i].set_title('%s' % cifar100_train.classes[cifar100_train[i][1]])
#     axes[i].axis('off')
#     fig.colorbar(img, ax=axes[i], fraction=0.046,aspect=20)
# fig.subplots_adjust(wspace=1)  # Increase the width space
# plt.show()




