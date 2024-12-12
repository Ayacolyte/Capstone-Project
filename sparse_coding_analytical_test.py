import scipy.io
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy import signal
from sklearn.decomposition import FastICA
# Load the .mat file
mat_file = scipy.io.loadmat('IMAGES_SparseCoding.mat')

# Inspect keys to find the image data
print(mat_file.keys())

# Assuming the image data is stored under the key 'image_data'
image_data = mat_file['IMAGES_WHITENED']
num_images = image_data.shape[2]
fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

# Display each image in a subplot
for i in range(num_images):
    axes[i].imshow(image_data[:, :, i], cmap='gray')  # Use cmap='gray' for grayscale images
    axes[i].axis('off')  # Turn off axis labels
    axes[i].set_title(f"Image {i + 1}")

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()


image_size = image_data.shape[0]
sz = 16
batch_size = 2000
n_epoch = 1
BUFF = 4
X_data = np.zeros((sz**2,batch_size))
for i_epoch in range(n_epoch):
    i = math.ceil(num_images*np.random.uniform(0, 1)) - 1
    this_image = image_data[:,:,i]
    
    # Randomly extract some image patches from this image to generate input vector x
    for i in range(batch_size):
        # Choose the left-up point of the image
        r = BUFF + math.ceil((image_size-sz-2*BUFF)*np.random.uniform(0, 1))
        c = BUFF + math.ceil((image_size-sz-2*BUFF)*np.random.uniform(0, 1))
        
        # Shape the image patch into vector form (sz*sz, 1) where L = sz * sz
        X_data[:,i] = np.reshape(this_image[r:r+sz,c:c+sz], sz**2)

W_ica = FastICA(n_components=sz**2, random_state=42,tol=1e-8)
S1 = W_ica.fit_transform(X_data.T) # Sources (sparse codes)
A = W_ica.mixing_  # Mixing matrix (basis vectors)

import matplotlib.pyplot as plt

def visualize_filters(filters, patch_size):
    n_filters = filters.shape[0]
    grid_size = int(np.ceil(np.sqrt(n_filters)))

    fig, axs = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    axs = axs.ravel()

    for i in range(n_filters):
        if i < len(axs):
            axs[i].imshow(filters[i].reshape(patch_size, patch_size), cmap='gray')
            axs[i].axis('off')
    plt.show()

visualize_filters(A.T, sz)