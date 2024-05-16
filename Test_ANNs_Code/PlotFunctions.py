# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 15:27:36 2022
Plot functions
@author: Domingos
"""


import tensorflow as tf
# import tensorflow.keras.backend as K
# from tensorflow.keras import datasets, layers, models, optimizers
# from tensorflow.keras.layers import Lambda
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
# from PIL import Image
import numpy as np
# import math
import random
import numpy.matlib
# from PIL import Image
# import seaborn as sns

from Dataset import *
from RetinaFun import *

#%%============================================================================
# Plot Training Loss
# =============================================================================

def show_training_loss(history, save_path = None):

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    
    plt.subplot(2,1,2)
    min_val = np.min(history.history['val_loss'])
    ylims = (min_val*0.9, min_val*1.3)
    plt.ylim(ylims)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    
    plt.title('Zoom')    
    if save_path != None:
        plt.savefig(save_path + '/Training_Loss.pdf')

#%%============================================================================
# Plot Predictions:
# =============================================================================

def show_prediction_examples(test_img_vec, stimulus_patterns, retina_output_vec, ret_side, rand_imgs = True, save_path = None, title = None, close_fig = False):
    
    
    nTestImg = test_img_vec.shape[0]    
    retina_output = np.reshape(retina_output_vec, (nTestImg, ret_side, ret_side))  
    test_img_bw = np.reshape(test_img_vec, (nTestImg, ret_side, ret_side))          
    
    fig = plt.figure(figsize=(10, 12), dpi=80)    
    
    # imgs = (4665,8170,345,497,1904,240,8865)
    imgs = (7373, 4933, 6389, 5592, 2672, 240, 8126, 9548, 5682)
    nImgs = len(imgs)
    
    
    for i in range(nImgs):
        if rand_imgs:
            img = random.randint(0, nTestImg)
        else:
            img = imgs[i]
        
        plt.subplot(nImgs, 3,(i*3)+1)
        plt.imshow(test_img_bw[img], cmap = 'gray')
        vmin, vmax = plt.gci().get_clim()
        plt.colorbar()
        if i == 0:
            plt.title('Target Img')
        
        plt.subplot(nImgs,3,(i*3)+2)
        plt.imshow(stimulus_patterns[img], cmap = 'bwr', clim = (-1,1))
        plt.colorbar()
        if i == 0:
            plt.title('Stimulus')
        
        plt.subplot(nImgs,3,(i*3)+3)
        plt.imshow(retina_output[img], cmap = 'gray', clim=(vmin, vmax))        
        plt.colorbar()
        _, RMSE_i = activationError(retina_output_vec[img], test_img_vec[img])
        plt.ylabel('RMSE: ' + str(int(RMSE_i*1000)/1000))
        
        if i == 0:
            all_RMSE, mean_RMSE = activationError(retina_output_vec, test_img_vec)
            plt.title('RMSE ('  + str(nTestImg) +  ' imgs): ' + str(int(mean_RMSE*1000)/1000))
            
    if title != None:
        fig.suptitle(title)
                        
    if save_path != None:
        plt.savefig(save_path + '/Predictions_Examples__ ' + title + '.pdf')       
                 
        
    if close_fig:
        plt.close(fig)
        
    return all_RMSE





def show_prediction_examples_rand_ret(x_ret, y_ret, test_img_vec, stimulus_patterns, retina_output_vec, ret_side, rand_imgs = True, save_path = None, title = None, close_fig = False):
    
    nTestImg = test_img_vec.shape[0]    
    test_img_bw = np.reshape(test_img_vec, (nTestImg, ret_side, ret_side))  
    
    
    fig = plt.figure(figsize=(10, 12), dpi=80)
    
    nImgs = 7
    # imgs = (4665,8170,345,497,1904,240,8865)
    imgs = (7373, 4933, 6389, 5592, 2672, 240, 8126, 9548, 5682)
    # imgs = [1300, 1300]
    nImgs = len(imgs)
    
    ret_order = np.arange(len(x_ret))
    np.random.shuffle(ret_order)    
        
    for i in range(nImgs):
        if rand_imgs:
            img = random.randint(0, nTestImg)
        else:
            img = imgs[i]
        
        plt.subplot(nImgs, 3,(i*3)+1)
        plt.imshow(test_img_bw[img], cmap = 'gray', clim=(0, 1)), 
        plt.colorbar()
        if i == 0:
            plt.title('Target Img')
        
        plt.subplot(nImgs,3,(i*3)+2)
        plt.imshow(stimulus_patterns[img], cmap = 'bwr', clim = (-1,1))
        plt.colorbar()
        if i == 0:
            plt.title('Stimulus')
        
        # plt.subplot(nImgs,3,(i*3)+3)
        # plt.imshow(retina_output[img], cmap = 'gray') # clim=(0, 1),        
        # plt.colorbar()
        # _, RMSE_i = activationError(retina_output_vec[img], test_img_vec[img])
        # plt.ylabel('RMSE: ' + str(int(RMSE_i*1000)/1000))        
        # if i == 0:
        #             all_RMSE, mean_RMSE = activationError(retina_output_vec, test_img_vec)
        #             plt.title('RMSE ('  + str(nTestImg) +  ' imgs): ' + str(int(mean_RMSE*1000)/1000))
        ax = fig.add_subplot(nImgs,3,(i*3)+3)
        plt.scatter( x_ret[ret_order], y_ret[ret_order], c = retina_output_vec[img,ret_order], s = 1,  alpha=0.5, edgecolors=None, cmap = 'gray',  vmin=0, vmax=1 ) 
        ax.set_aspect('equal')      
        plt.gca().invert_yaxis()
        plt.colorbar()    
        plt.axis('off')
        

    if title != None:
        fig.suptitle(title)
        
    if save_path != None:
        plt.savefig(save_path + '/Predictions_Examples__ ' + title + '.pdf')                        

    if close_fig:
        plt.close(fig)    
        
    # return all_RMSE



    
#%%============================================================================
# Plot Methods Comparisons:
# =============================================================================
def show_methods_comparisons(target, s_conv, s_lininv, s_nn, result_conv, result_lininv, result_nn, ret_side, save_path = None):
        
    target_vec = target.reshape(ret_side*ret_side,1)
    # result_conv_vec = tf.reshape(result_conv, (ret_side*ret_side,1))
    result_lininv_vec = tf.reshape(result_lininv, (ret_side*ret_side,1))
    result_nn_vec = tf.reshape(result_nn, (ret_side*ret_side,1))


    plt.figure(figsize=(10, 6), dpi=80)
    
    # # Conventional
    # _, RMSE_conv = activationError(result_conv, target_vec)
    # plt.subplot(3,3,1)
    # plt.imshow(target, cmap = 'gray') # clim=(0, 1), 
    # plt.colorbar()
    # plt.title('Conventional: ' + str(int(RMSE_conv*1000)/1000))
    # plt.ylabel('Target')
    
    # plt.subplot(3,3,4)
    # plt.imshow(s_conv, cmap = 'bwr'), plt.colorbar()
    # plt.ylabel('Stimulus')
     
    # plt.subplot(3,3,7)
    # plt.imshow(result_conv, cmap = 'gray') # clim=(0, 1), 
    # plt.colorbar()
    # plt.ylabel('Activation')
        
    # linear Inversion
    _, RMSE_inv = activationError(result_lininv_vec, target_vec)
    plt.subplot(3,3,2)
    plt.imshow(target, cmap = 'gray', interpolation = 'none') # clim=(0, 1), 
    plt.colorbar()
    plt.title('Linear Inver: ' + str(int(RMSE_inv*1000)/1000))
    plt.ylabel('Target')
    
    plt.subplot(3,3,5)
    plt.imshow(s_lininv, cmap = 'bwr', interpolation = 'none'), plt.colorbar()
    plt.ylabel('Stimulus')
     
    plt.subplot(3,3,8)
    plt.imshow(result_lininv, cmap = 'gray', interpolation = 'none') # clim=(0, 1), 
    plt.colorbar()
    plt.ylabel('Activation')
    
    
    # Neural Nets
    _, RMSE_nn = activationError(result_nn_vec, target_vec)
    plt.subplot(3,3,3)
    plt.imshow(target, cmap = 'gray', interpolation = 'none') # clim=(0, 1), 
    plt.colorbar()
    plt.title('Neural Net: ' + str(int(RMSE_nn*1000)/1000))
    plt.ylabel('Target')
    
    plt.subplot(3,3,6)
    plt.imshow(s_nn, cmap = 'bwr', interpolation = 'none'), plt.colorbar()
    plt.ylabel('Stimulus')
     
    plt.subplot(3,3,9)
    plt.imshow(result_nn, cmap = 'gray', interpolation = 'none') # clim=(0, 1), 
    plt.colorbar()
    plt.ylabel('Activation')

    if save_path != None:
        plt.savefig(save_path + '/Different_Methods.pdf') 
    
    
#%%============================================================================
# Plot Compiled Results
# =============================================================================

def plot_metric_vs_param(metric, param, metric_name, param_name, nTestImg, save_path = None):  
    
    plt.figure()
    plt.violinplot(metric, param, showmeans=True, widths=1/len(param))
    plt.xlabel(param_name)
    plt.ylabel(metric_name + ' (' + str(nTestImg) + ' imgs)')
    
    if save_path != None:
        plt.savefig(save_path + '/COMPILED__' + metric_name + '_vs_' + param_name + '.pdf') 

    

def plot_histories(histories, params, param_name, save_path = None):
    
    # History vs param
    plt.figure()
    color = cm.brg(np.linspace(0, 1, len(params)))
    
    for param_i, c in zip(range(len(params)), color):
        plt.plot(histories[param_i].history['val_loss'], label= param_name + ' = '+str(params[param_i]), c=c)

    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.legend(loc='upper right')
    
    if save_path != None:
        plt.savefig(save_path + '/COMPILED__History_vs_' + param_name + '.pdf') 



#%%============================================================================
# Show MEA
# =============================================================================

def show_MEA(ret_side, elec_side, elec_size):

    ret_space = 1
    elec_space = ret_side*ret_space/elec_side;
    x_elec_pos = np.arange(elec_space/2, elec_space*elec_side, elec_space)
    x_elec = np.matlib.repmat(x_elec_pos , elec_side,1)
    x_elec = x_elec.flatten()
    
    y_elec_pos = np.arange(elec_space/2, elec_space*elec_side, elec_space)
    y_elec = np.matlib.repmat(np.reshape(y_elec_pos, (elec_side, 1)), 1, elec_side)
    y_elec = y_elec.flatten()

    Ne = elec_side**2

    for elec_i in range(Ne):
        plt.plot(x_elec[elec_i], y_elec[elec_i], 'ko', markersize=elec_size)  
        plt.plot(x_elec[elec_i], y_elec[elec_i], 'wo', markersize=elec_size*0.7) 



#%%============================================================================
# Show Retina and Electrodes
# =============================================================================

def show_Retina_MEA(ret_side, elec_side, Wt, spread, Bs, Cs, save_path = None, both = True, nonLin = 'sigmoid'):
    
    ret_space = 1
    # Coordinates:
    x_ret = np.matlib.repmat(np.arange( ret_space/2, ret_space*ret_side, ret_space ), ret_side,1)    
    x_ret = x_ret.flatten()
    y_ret = np.matlib.repmat(np.reshape(np.arange(ret_space/2, ret_space*ret_side, ret_space ), (ret_side, 1)), 1, ret_side)
    y_ret = y_ret.flatten()
    
    elec_space = ret_side*ret_space/elec_side;
    x_elec_pos = np.arange(elec_space/2, elec_space*elec_side, elec_space)
    x_elec = np.matlib.repmat(x_elec_pos , elec_side,1)
    x_elec = x_elec.flatten()
    
    y_elec_pos = np.arange(elec_space/2, elec_space*elec_side, elec_space)
    y_elec = np.matlib.repmat(np.reshape(y_elec_pos, (elec_side, 1)), 1, elec_side)
    y_elec = y_elec.flatten()
        
    stim = np.zeros((elec_side, elec_side))
    stim[int(elec_side/2), int(elec_side/2)] = 1;


    Nt = ret_side**2
    Ne = elec_side**2
    
    
    # Define Positive Stimulus:
    stim = np.zeros((elec_side, elec_side));
    pos_i = ( int(np.floor(elec_side*1/3)), int(np.floor(elec_side*1/3)) )
    stim[pos_i] = 1;
    
    plt.figure(figsize=(12, 4))  
    
    Wt_tensor = np.array([Wt])
    stim_tensor = np.array([stim.flatten()])
    if nonLin  == 'sigmoid':
        activation = LNL(stim_tensor, Wt_tensor, Bs, Cs)  
    elif nonLin == 'gaussian':
        activation = LNL_gauss(stim_tensor, Wt_tensor, Bs, Cs)  
    
    activation = tf.reshape(activation, (ret_side, ret_side))
    
    if both:
        plt.subplot(1,2,1)    
        
    plt.imshow(activation, extent=[x_ret[0]-ret_space/2, x_ret[-1]+ret_space/2, y_ret[0]-ret_space/2, y_ret[-1]+ret_space/2], clim = (0,1), origin='lower', cmap = 'gray')
    for elec_i in range(Ne):
        plt.plot(x_elec[elec_i], y_elec[elec_i], 'ko', markersize=10)  
        plt.plot(x_elec[elec_i], y_elec[elec_i], 'wo', markersize=7)  
        
    plt.plot(x_elec_pos[pos_i[0]], y_elec_pos[pos_i[1]], 'ro', markersize=7)
    plt.title('Retina Activation - spread: ' + str(spread))
   
    
    if both:
        # Define Positive and Negative Stimulus:
        stim = np.zeros((elec_side, elec_side));
        neg_i = ( int(np.ceil(elec_side*2/3)), int(np.ceil(elec_side*2/3)) )
        pos_i = ( int(np.floor(elec_side*1/3)), int(np.floor(elec_side*1/3)) )
        stim[neg_i] = -1;
        stim[pos_i] = 1;
        
        Wt_tensor = np.array([Wt])
        stim_tensor = np.array([stim.flatten()])    
        if nonLin  == 'sigmoid':
                activation = LNL(stim_tensor, Wt_tensor, Bs, Cs)  
        elif nonLin == 'gaussian':
            activation = LNL_gauss(stim_tensor, Wt_tensor, Bs, Cs)             
        activation = tf.reshape(activation, (ret_side, ret_side))
        
        plt.subplot(1,2,2)
        plt.imshow(activation, extent=[x_ret[0]-ret_space/2, x_ret[-1]+ret_space/2, y_ret[0]-ret_space/2, y_ret[-1]+ret_space/2], clim = (0,1), origin='lower', cmap = 'gray')
        for elec_i in range(Ne):
            plt.plot(x_elec[elec_i], y_elec[elec_i], 'ko', markersize=10)  
            plt.plot(x_elec[elec_i], y_elec[elec_i], 'wo', markersize=7)  
            
        plt.plot(x_elec_pos[neg_i[0]], y_elec_pos[neg_i[1]], 'bo', markersize=7)
        plt.plot(x_elec_pos[pos_i[0]], y_elec_pos[pos_i[1]], 'ro', markersize=7)
        plt.title('Retina Activation')
        
    if save_path != None:
        plt.savefig(save_path + '/Simple_Stimuli.pdf')  
  
    
#%%============================================================================
# Show Retina and Electrodes - random retina positions
# =============================================================================
      
def show_Retina_MEA_rand(x_ret, y_ret, x_elec, y_elec, ret_side, elec_side, Wt, spread, Bs, Cs, save_path=None, color_map = 'gray', size = 15, alpha = 0.5, nonLin = 'sigmoid', both = True, close_fig = False):
        
    Nt = x_ret.shape[0]
    Ne = elec_side**2
    
    # Define Positive Stimulus:
    stim = np.zeros((elec_side, elec_side));
    pos_i = ( int(np.ceil(elec_side*1/3)), int(np.ceil(elec_side*1/3)) )
    stim[pos_i] = 1;
    
    fig = plt.figure(figsize=(12, 4))  
    
    Wt_tensor = np.array([Wt])
    stim_tensor = np.array([stim.flatten()])
    if nonLin  == 'sigmoid':
        activation_1 = LNL(stim_tensor, Wt_tensor, Bs, Cs)  
    elif nonLin == 'gaussian':
        activation_1 = LNL_gauss(stim_tensor, Wt_tensor, Bs, Cs)        
    activation_1 = activation_1.numpy()
    
    if both:
        ax = fig.add_subplot(121)  
    else:
        ax = plt.gca()

    ret_order = np.arange(len(x_ret))
    np.random.shuffle(ret_order)    
    
    plt.scatter( x_ret[ret_order], y_ret[ret_order], c = activation_1[0,ret_order] , s = size, alpha=alpha, edgecolors=None, cmap = color_map  ) 
    for elec_i in range(Ne):
        plt.plot(x_elec[elec_i], y_elec[elec_i], 'ko', markersize=10)
        plt.plot(x_elec[elec_i], y_elec[elec_i], 'wo', markersize=7)  
        
    plt.plot(numpy.unique(x_elec)[pos_i[0]], numpy.unique(y_elec)[pos_i[1]], 'ro', markersize=7)
    ax.set_aspect('equal')
    plt.title('Retina Activation - spread: ' + str(spread))
          
    
    if both:
        # Define Positive and Negative Stimulus:
        stim = np.zeros((elec_side, elec_side));
        neg_i = ( int(np.floor(elec_side*2/3)), int(np.floor(elec_side*2/3)) )
        pos_i = ( int(np.floor(elec_side*1/3)), int(np.floor(elec_side*1/3)) )   
        stim[neg_i] = -1;
        stim[pos_i] = 1;
        
        
        # #######################################################################    
        # neg_i_ = ( int(np.floor(elec_side*1/3)), int(np.floor(elec_side*2/3)) )
        # pos_i_ = ( int(np.floor(elec_side*2/3)), int(np.floor(elec_side*1/3)) ) 
        # stim[neg_i_] = -1;
        # stim[pos_i_] = 1;
        # #######################################################################  
        
        
        Wt_tensor = np.array([Wt])
        stim_tensor = np.array([stim.flatten()])
        if nonLin  == 'sigmoid':
            activation_2 = LNL(stim_tensor, Wt_tensor, Bs, Cs)  
        elif nonLin == 'gaussian':
            activation_2 = LNL_gauss(stim_tensor, Wt_tensor, Bs, Cs)    
        activation_2 = activation_2.numpy()
        
        ax = fig.add_subplot(122)
        plt.scatter( x_ret[ret_order], y_ret[ret_order], c = activation_2[0,ret_order] ,s = size, alpha=alpha, edgecolors=None, cmap = color_map) 
        
        for elec_i in range(Ne):
            plt.plot(x_elec[elec_i], y_elec[elec_i], 'ko', markersize=10)  
            plt.plot(x_elec[elec_i], y_elec[elec_i], 'wo', markersize=7)
    
        plt.plot(numpy.unique(x_elec)[neg_i[0]], numpy.unique(y_elec)[neg_i[1]], 'bo', markersize=7)
        plt.plot(numpy.unique(x_elec)[pos_i[0]], numpy.unique(y_elec)[pos_i[1]], 'ro', markersize=7)
        
        
        #############################################################################################
        # plt.plot(numpy.unique(x_elec)[neg_i_[0]], numpy.unique(y_elec)[neg_i_[1]], 'bo', markersize=7)
        # plt.plot(numpy.unique(x_elec)[pos_i_[0]], numpy.unique(y_elec)[pos_i_[1]], 'ro', markersize=7)
        # ##############################################################################################
        
        
        ax.set_aspect('equal')
        plt.title('Retina Activation')
    else:
        activation_2 = []
    
    if save_path != None:
        plt.savefig(save_path + '/Simple_Stimuli.pdf')  
  
    if close_fig:
        plt.close(fig)  
  
    return activation_1, activation_2


#%% Plot Electrodes:
def plot_elecs(ax, values, ret_side, elec_side, radius, cmap, clim = (0,1), alpha = 1):   
    _,_, x_elec, y_elec, _, _, _, _ = ret_and_elec_pos(elec_side*elec_side, ret_side, elec_side)
    plt.scatter(x_elec, y_elec, s = radius, c = values, cmap = cmap, edgecolors = 'k', vmin = clim[0], vmax = clim[1], alpha = alpha)    
    ax.set_aspect('equal')
    plt.axis('off')