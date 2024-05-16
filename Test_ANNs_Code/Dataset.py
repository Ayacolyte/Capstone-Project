# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 12:05:16 2022

Prepare Dataset

@author: Domingos
"""

from tensorflow.keras import datasets
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib


#%% From RGB to grayscale
def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


#%%============================================================================
# Download an prepare images
# =============================================================================
import pickle

def Create_Dataset(train_val_split, ret_side, augmentation = False):
    
    # Load Dataset:
    (train_images_all, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    print(train_images_all.shape)
    print(test_images.shape)
    # Normalize pixel values to be between 0 and 1:
    train_images_all, test_images = train_images_all / 255.0, test_images / 255.0 

    # Pre Process - cropped BW images:
    nTrainImg = train_images_all.shape[0]
    nTestImg  = test_images.shape[0]
    
    train_images_all_bw = np.zeros((nTrainImg, ret_side, ret_side, 1))
    test_images_bw = np.zeros((nTestImg, ret_side, ret_side, 1))

    for i in range(nTrainImg):
        train_images_all_bw[i, :, :, 0] = rgb2gray(train_images_all[i, int(16-ret_side/2):int(16+ret_side/2), int(16-ret_side/2):int(16+ret_side/2),:])
        
    for i in range(nTestImg):
        test_images_bw[i, :, :, 0]  = rgb2gray(test_images[i, int(16-ret_side/2):int(16+ret_side/2), int(16-ret_side/2):int(16+ret_side/2),:])


    #%% Data Augmentation:
    if augmentation == 'large':     
        # Rotations:
        rot_90 = np.rot90(train_images_all_bw, 1, axes = (1,2))
        rot_180 = np.rot90(train_images_all_bw, 2, axes = (1,2))
        rot_270 = np.rot90(train_images_all_bw, 3, axes = (1,2))
        rot_rev = np.fliplr(train_images_all_bw)
        rot_rev_90 = np.rot90(rot_rev, 1, axes = (1,2))
        rot_rev_180 = np.rot90(rot_rev, 2, axes = (1,2))
        # rot_rev_270 = np.rot90(rot_rev, 3, axes = (1,2))
        
        aug_dataset = np.concatenate((train_images_all_bw, rot_90, rot_180, rot_270, rot_rev, rot_rev_90, rot_rev_180), axis = 0)#, rot_rev_270 
        
        # Negative:
        aug_dataset_neg = 1-aug_dataset
        
        # Mixed:
        flipped = np.flip(aug_dataset, axis = 0)
        mixed_pos_neg = (aug_dataset_neg + flipped)/2
        mixed_pos_pos = (aug_dataset + flipped)/2
        
        train_images_all_bw = np.concatenate((aug_dataset, aug_dataset_neg, mixed_pos_neg, mixed_pos_pos), axis = 0)


    elif augmentation == 'small':   
        train_images_all_bw = np.concatenate((train_images_all_bw, 1-train_images_all_bw), axis = 0)        
        train_images_all_bw = np.concatenate((train_images_all_bw, np.fliplr(train_images_all_bw)), axis = 0)    
        
    nTrainImg = train_images_all_bw.shape[0]

                                    
    # Train Validation Split
    nVal_images = int(nTrainImg * train_val_split)
    val_images_bw = train_images_all_bw[0:nVal_images]    
    train_images_bw = train_images_all_bw[nVal_images::]
    

    nTrainImg = train_images_bw.shape[0]
    nValImg = val_images_bw.shape[0]
    nTestImg = test_images_bw.shape[0]
    
    # Flatten images for training and testing:
    train_images_vec = np.reshape(train_images_bw, (nTrainImg, ret_side**2))
    val_images_vec = np.reshape(val_images_bw, (nValImg, ret_side**2))
    test_images_vec = np.reshape(test_images_bw, (nTestImg, ret_side**2))    
    
    return train_images_bw, val_images_bw, test_images_bw, train_images_vec, val_images_vec, test_images_vec


#%% 
# from skimage.transform import rotate

# def Create_Artificial_Dataset(ret_side):   
    
#     inner_rs = np.linspace(0, 20, 11)
#     outter_r_extras = np.linspace(4, 10, 7)
#     center_xs = np.linspace(4, 20, 9)
#     center_ys = np.linspace(4, 20, 9)
    
#     line_widths =   np.linspace(2, 30, 29)
#     line_start_xs = np.linspace(1, 16, 16)
       
    
#     # Create Images:
#     all_images = []
    
#     angles = np.linspace(0, 350, 10)
    
#     # Circles:    
#     for angle in angles:
#         print(angle)
#         for inner_r in inner_rs:
#             for outter_r_extra in outter_r_extras:
#                 for center_x in center_xs:
#                     for center_y in center_ys: 
                        
#                         image = np.ones((ret_side,ret_side))
#                         for x in range(ret_side):
#                             for y  in range(ret_side):
#                                 if (x-center_x)**2 + (y-center_y)**2 < (inner_r + outter_r_extra)**2 and (x-center_x)**2 + (y-center_y)**2 > inner_r**2:
#                                     image[x,y] = 0
                                    
#                         all_images.append(image) 
    
#     angles = np.linspace(0, 350, 176)
#     for angle in angles:
#         print(angle)
#         for line_width in line_widths:   
#             for line_start_x in line_start_xs: 
#                     # Vertical Lines: 
#                     image = np.ones((ret_side,ret_side))
#                     image[:, int(line_start_x):int(line_start_x+line_width)] = 0   
#                     image = rotate(image, angle, cval=1)
                    
#                     all_images.append(image) 
                    
#                     # Horizontal Lines:  
#                     image = np.ones((ret_side,ret_side))
#                     image[int(line_start_x):int(line_start_x+line_width), :] = 0     
#                     image = rotate(image, angle, cval=1)
                    
#                     all_images.append(image)         
                                
#                     # Vertical and horizontal Lines: 
#                     image = np.ones((ret_side,ret_side))
#                     image[:, int(line_start_x):int(line_start_x+line_width)] = 0    
#                     image[int(line_start_x):int(line_start_x+line_width), :] = 0  
#                     image = rotate(image, angle, cval=1)
#                     all_images.append(image)     
                            
#     with open('artificial_dataset.pkl', 'wb') as file:  
#       pickle.dump(all_images, file)
      
#     return all_images

    
#%% 
import random  


def Load_artificial_dataset(train_val_split):
        
    with open('artificial_dataset.pkl', 'rb') as file:  
        data = pickle.load(file)
    
    nimgs = len(data)
    nVal = int(nimgs * train_val_split)
    
    random.shuffle(data)
    
    val_data = data[:nVal]
    train_data = data[nVal:]

    return train_data, val_data



train_val_split = 0.25
ret_side = 32
 
train_img_bw, val_img_bw, test_img_bw, train_img_vec, val_img_vec, test_img_vec = Create_Dataset(train_val_split, ret_side, augmentation=False)

nTrainImg = train_img_bw.shape[0]
nValImg = val_img_bw.shape[0]
nTestImg = test_img_bw.shape[0]