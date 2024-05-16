# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 11:03:45 2022

Functions related with Retina: Transfer Matrix, Forward Model, Inverse Models, Situmulus test,...

@author: Domingos
"""

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import random
import numpy.matlib
# from PIL import Image
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA


#%% Loss Function:
# def SSIMLoss(y_true, y_pred):
    
#     print(y_true)
#     print(y_pred)
    
#     return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0, filter_size=3))


 #%%===========================================================================
 # Linear Non-Linear Model - Sigmoid Function
 # ============================================================================
def LNL(stim_tensor, Wt_tensor, b = 10, c = 0.5, random_noise = 0):   # b = 5.5, c = 0.5): # 
      
      if random_noise != 0:
          g = tf.random.get_global_generator()
          noise_mat = g.normal(shape=( tf.shape(Wt_tensor)[1], tf.shape(Wt_tensor)[2]))
          Wt_tensor = Wt_tensor +  random_noise*noise_mat
          
      a = 1      
      
      G = layers.Dot(axes=(2,1))([Wt_tensor, stim_tensor])
      
      result_vec =  a/(1 + tf.math.exp( tf.math.multiply(-b, (abs(G) -c) ) ) )
      
      return result_vec



#%%============================================================================
 # Linear Non-Linear Model - Double Gaussian Function
 # double gaussian activaation instead of double sigmoid: Too much current (positive or negative) also inhibits the cells 
 # ============================================================================
 # def LNL_gauss(stim_tensor, Wt_tensor, b = 10, c = 0.5, random_noise = 0):   # b = 5.5, c = 0.5): # 

 #      G = layers.Dot(axes=(2,1))([Wt_tensor, stim_tensor])      
      
 #      exponent = -tf.math.square(abs(G) - b) / (2 * tf.math.square(c))
 #      result_vec = tf.math.exp(exponent)
      
 #      return result_vec



#%%============================================================================
 # Linear Non-Linear Model - Double Gaussian Function
 # ============================================================================
def LNL(stim_tensor, Wt_tensor, b = 10, c = 0.5, random_noise = 0):   # b = 5.5, c = 0.5): # 
      
      if random_noise != 0:
          g = tf.random.get_global_generator()
          noise_mat = g.normal(shape=( tf.shape(Wt_tensor)[1], tf.shape(Wt_tensor)[2]))
          Wt_tensor = Wt_tensor +  random_noise*noise_mat
          
      a = 1      
      
      G = layers.Dot(axes=(2,1))([Wt_tensor, stim_tensor])
      
      result_vec =  a/(1 + tf.math.exp( tf.math.multiply(-b, (abs(G) -c) ) ) )
      
      return result_vec

#%%============================================================================
# =============================================================================
def LNL_abs(stim_tensor, Wt_tensor, random_noise = 0):
      
      if random_noise != 0:
          noise_mat = random_noise*(random.random(Wt_tensor.shape())*2-1) # random_noise*random.random(Wt_tensor.shape())
          Wt_tensor = Wt_tensor + noise_mat
          
      result_vec = K.abs( layers.Dot(axes=(2,1))([Wt_tensor, stim_tensor]))      
      return result_vec


#%%============================================================================
# Electrode Measurement Functions
# =============================================================================
def Elec_measure(ret_signal_vec, x_ret, y_ret, x_elec, y_elec, z_elec, elec_side, nSignals, kernel_power = 1, elec_radius = 1):
    Ne = elec_side**2
    elec_signal_vec = np.zeros((nSignals, Ne))
    
    
    # # ------------------------------------------------------------------------------
    # plt.figure()
    # ret_order = np.arange(len(x_ret))
    # np.random.shuffle(ret_order)  
    # plt.scatter(x_ret[ret_order], y_ret[ret_order], c = 'k', alpha = 0.4)
    # #-------------------------------------------------------------------------------
    
    
    for elec_i in range(Ne):    
         coords_i = np.unravel_index(elec_i, x_elec.shape)
         #print('-------------------')
         #print('coord i:',coords_i)
         if elec_radius == 0:
             # Electrode is a point - follow 1/r rule:
             kernel = elec_measure_kernel(x_ret, y_ret, x_elec[coords_i], y_elec[coords_i], z_elec, kernel_power)    
             #print('----------------------------------------------------------------') 
             #print('redius 0')
             #print(kernel)
             elec_signal_vec[:, elec_i] = np.sum(np.multiply(ret_signal_vec, kernel.flatten()) , axis=1)
             
         else:
             # Electrode is a surface - average recorded retina cells:
             #print('----------------------------------------------------------------') 
             #print('redius not 0')
             #print('ret_signal_vec:',ret_signal_vec)
             ret_idx = rets_above_elec(x_ret, y_ret,  x_elec[coords_i], y_elec[coords_i], elec_radius)             
             elec_signal_vec[:, elec_i] = np.mean(ret_signal_vec[:,ret_idx], axis = 1)
             
             
# #------------------------------------------------------------------------------
#              # Plot
#              plt.plot(x_elec, y_elec, 'c*')
#              plt.scatter(x_ret[ret_idx], y_ret[ret_idx], c = 'r', alpha = 1)
#              ax = plt.gca()
#              ax.set_aspect('equal')
#              plt.title('Elec Radius: ' + str(elec_radius))
             
 
    
#     plt.figure()
#     plt.subplot(1,2,1)
#     plt.scatter(x_ret[ret_order], y_ret[ret_order], c = ret_signal_vec[0,ret_order], alpha = 0.5, cmap = 'gray', vmin = 0, vmax = 1) 
#     plt.plot(x_elec, y_elec, 'c*')         
#     ax = plt.gca()
#     ax.set_aspect('equal')            
#     plt.colorbar()
    
#     # plt.subplot(1,3,3)
#     plt.subplot(1,2,2)
#     plt.imshow(elec_signal_vec.reshape((elec_side, elec_side)), cmap = 'gray', clim = (0,1), origin = 'lower')   
#     plt.title('Max pixel: ' + str(np.around(np.max(elec_signal_vec)*100)/100) )
#     plt.colorbar()     
#    #------------------------------------------------------------------------------   

    if nSignals > 1:
        elec_signal_mat = np.reshape(elec_signal_vec, (nSignals, elec_side, elec_side) )
    else:   
        elec_signal_mat = np.reshape(elec_signal_vec.transpose(), (elec_side, elec_side) )
        
    return elec_signal_vec, elec_signal_mat


def rets_above_elec(x_ret, y_ret, x_elec, y_elec, R):
    distances = np.sqrt((x_ret -  x_elec)**2 + (y_ret -  y_elec)**2)
    rets_idx = np.where(distances <= R)
    return rets_idx[0]


def elec_measure_kernel(x_ret, y_ret, x_elec, y_elec, z_elec, kernel_power = 1):
    # a distance based measurement, where kernel = 1/ distance between electrode and retina location ^ kernel power (default 1)
    dists = np.sqrt( np.square(x_ret-x_elec) + np.square(y_ret - y_elec) + np.square(0 - z_elec) )
    kernel = np.divide(1, dists**kernel_power)
    kernel = kernel/np.sum(kernel)              
    return kernel


#%% ===========================================================================
# Double Sigmoid Non Linearity
# =============================================================================
def DoubleSig(xx, b, c, nonLin = 'sigmoid'):
    
    if nonLin == 'sigmoid':
      return np.divide(1 , (1 + np.exp( np.multiply(-b, (abs(xx).transpose()-c)))))
  
    elif nonLin == 'gaussian':
      exponent = -np.square(abs(xx).transpose() - b) / (2 * np.square(c))
      return np.exp(exponent)

    
    #%% ===========================================================================
    # Real Retina model:
    # =============================================================================
def BuildRetinaModel(Nt, ret_side, elec_side, base_spread, mode = 'jitter', nonLin = 'sigmoid') :
    
    Ne = elec_side*elec_side
    
    
    # ERFs Parameters:
    spread = elec_side / 8 * base_spread
    
    min_spread = spread - spread/4
    max_spread = spread + spread/4
    noise = 0.0
    

    #####################################################################################
    spreads_x = np.random.uniform(low=min_spread, high=max_spread, size=(Nt, 1))
    spreads_y = np.random.uniform(low=min_spread, high=max_spread, size=(Nt, 1))        
    angs = np.random.uniform(-np.pi, np.pi, size=(Nt, 1))      
    # #####################################################################################
    # print('------------------------------')
    # print('!!!  Personalized Retina  !!! ')
    # print('------------------------------')
    
    # spreads_x = np.random.uniform(low=min_spread, high=max_spread, size=(Nt, 1))
    # spreads_y = np.random.uniform(low=min_spread*0.8, high=max_spread*0.8, size=(Nt, 1))
    # angs = np.reshape(np.linspace(np.pi/4 - 0.5, np.pi/4 + 0.5, Nt), (Nt, 1))
    #####################################################################################


    x_ret, y_ret, x_elec, y_elec,_,_,_,_ = ret_and_elec_pos(Nt, ret_side, elec_side)
    z_elec = 0.0
    
    if mode == 'random':
        x_ret  = np.random.uniform(0.0, ret_side, size=(Nt,1))
        y_ret = np.random.uniform(0.0, ret_side, size=(Nt,1))
    elif mode == 'jitter':
        x_ret = x_ret + np.random.normal(loc=0.0, scale=0.05, size=(Nt,1))
        y_ret = y_ret + np.random.normal(loc=0.0, scale=0.05, size=(Nt,1))
        
    
    # Transfer Matrix:
    elec_space = x_elec[1] - x_elec[0]
    Wt, Wt_tensor = Wt_mat_ell(x_ret, y_ret, x_elec, y_elec, ret_side, elec_side, spreads_x, spreads_y, elec_space, angs, noise,  normalization = False)
    
    if nonLin == 'sigmoid':
        # Non-Linearity Parameters:
        Bs = tf.convert_to_tensor( np.random.uniform(low=5, high=10, size=(1,Nt)), dtype = 'float32' ) 
        Cs = tf.convert_to_tensor( np.random.uniform(low=0.4, high=0.6, size=(1,Nt)), dtype = 'float32' ) 
        
        ## Sweeping NonLinearity:
        # Bs = tf.convert_to_tensor( np.linspace(5, 20, Nt), dtype = 'float32' ) 
        # Cs = tf.convert_to_tensor( np.linspace(0.4, 0.8, Nt), dtype = 'float32' )   
        
        ## Kinda linear:
        # Bs = tf.convert_to_tensor( np.random.uniform(low=5.5, high=5.5, size=(1,Nt)), dtype = 'float32' ) 
        # Cs = tf.convert_to_tensor( np.random.uniform(low=0.5, high=0.5, size=(1,Nt)), dtype = 'float32' ) 
     
    elif nonLin == 'gaussian':
        Bs = tf.convert_to_tensor( np.random.uniform(low=0.5, high=0.7, size=(1,Nt)), dtype = 'float32' ) 
        Cs = tf.convert_to_tensor( np.random.uniform(low=0.1, high=0.15, size=(1,Nt)), dtype = 'float32' ) 
    
    return x_ret, y_ret, x_elec, y_elec, z_elec, Wt, Wt_tensor, Bs, Cs
    

#%%============================================================================
# Matias Method:
# =============================================================================
def LNL_Elec_Estimation(ret_side, elec_side, Wt_RET, b, c, x_ret, y_ret, x_elec, y_elec, z_elec, to_plot = False, stims = [], kernel_power = 1, elec_radius = 1):       
    
    Ne = elec_side**2
    Nt = len(x_ret)
    
    if len(stims) == 0:
        # Gaussian Stims:
        nStims = 10000
        stims = np.random.normal(0, 0.3, (elec_side, elec_side, nStims) )
    else:
        nStims = stims.shape[0]
    
    stims_v = np.reshape(stims, (nStims, Ne) ).transpose()
    
    
    #%% Retina Responses
    
    # Real Retina Activations:
    G_ret = np.dot(Wt_RET, stims_v)
    activation_RET = DoubleSig(G_ret, b, c)
       
    
    # Measured Activation:
    activation_ELEC,  activation_ELEC_mat = Elec_measure(activation_RET, x_ret, y_ret, x_elec, y_elec, z_elec, elec_side, nStims, kernel_power, elec_radius=elec_radius)

    
    #%% Estimated LNL for Electrodes
    Wt_mdl_ELEC, Gs_mdl_ELEC, Bs_mdl_ELEC, Cs_mdl_ELEC = PCA_method_LNL( activation_ELEC, stims_v )
    
    # plt.figure()
    # plt.scatter(Gs_mdl_ELEC[1, :], activation_ELEC[1, :], s = 1) 
    
    # Model Prediction:
    mdl_activation_ELEC = DoubleSig(Gs_mdl_ELEC.transpose(), Bs_mdl_ELEC, Cs_mdl_ELEC)        
    mdl_activation_ELEC_mat = np.reshape( mdl_activation_ELEC.transpose(), (nStims, elec_side, elec_side) )
    total_mean_error = np.mean(np.abs(mdl_activation_ELEC_mat - activation_ELEC_mat))
    
    
    #%%  Plot Measured Activity VS Estimation of Measured Activity   
    
    if to_plot:
               
        fig = plt.figure(figsize=(16, 10), dpi=80)
        nimgs = 5
        
        for i in range(nimgs):
            plt.subplot(nimgs,5, i*5 + 1)
            plt.imshow(stims[i,:,:], cmap = 'bwr', interpolation = 'none', clim = (-1,1))
            plt.title('Stimulus')
            plt.colorbar()
            plt.axis('off')
                                   
            if len(numpy.unique(x_ret)) == ret_side:
                plt.subplot(nimgs,5, i*5 + 2)
                activation_RET_mat = np.reshape(activation_RET, (nStims, ret_side, ret_side))
                plt.imshow(activation_RET_mat[i],clim=(0,1), cmap = 'gray', origin='lower')
            else:
                ax = fig.add_subplot(nimgs,5, i*5 + 2)  
                plt.scatter( x_ret, y_ret, c = activation_RET[i] , alpha=0.5, edgecolors=None, cmap = 'gray' )  
                ax.set_aspect('equal')
                
            plt.title('Retina Activation')    
            plt.gca().invert_yaxis()
            plt.colorbar()
            plt.axis('off')
            
            plt.subplot(nimgs,5, i*5 + 3)
            plt.imshow(activation_ELEC_mat[i],clim=(0,1), cmap = 'gray')
            plt.title('Measured Act.')
            plt.colorbar()
            plt.axis('off')
            
            plt.subplot(nimgs,5, i*5 + 4)
            plt.imshow(mdl_activation_ELEC_mat[i],clim=(0,1), cmap = 'gray')        
            plt.title('Estimated Act.')
            plt.colorbar()
            plt.axis('off')
            
            plt.subplot(nimgs,5, i*5 + 5)
            error_mat = np.abs(mdl_activation_ELEC_mat[i] - activation_ELEC_mat[i])
            plt.imshow(error_mat, cmap = 'gray')        
            plt.title('Error: ' + str(np.round(np.mean(error_mat)*1000)/1000 ))
            plt.colorbar()
            plt.axis('off')
                        
        fig.suptitle('Total error (' + str(nStims) + ' stims): '  + str(np.round(total_mean_error*100000)/100000))


    Bs_mdl_ELEC_tf = tf.convert_to_tensor(np.reshape(Bs_mdl_ELEC, (Ne,)), dtype = 'float32' )
    Cs_mdl_ELEC_tf = tf.convert_to_tensor(np.reshape(Cs_mdl_ELEC, (Ne,)), dtype = 'float32' )
    Wt_mdl_ELEC_tf = np.array([Wt_mdl_ELEC], dtype = 'float32' )
    
    return Wt_mdl_ELEC, Bs_mdl_ELEC, Cs_mdl_ELEC, Wt_mdl_ELEC_tf, Bs_mdl_ELEC_tf, Cs_mdl_ELEC_tf, total_mean_error


 #%%============================================================================
 # PCA :
# =============================================================================   
def PCA_method_LNL( activations, stims_v ):
    
    Nt = activations.shape[0]
    Ne = stims_v.shape[0]
    nStims = activations.shape[1]
    
    
    Wt_mdl = np.zeros( (Nt, Ne) )
    Gs_mdl = np.zeros( (Nt, nStims) )
    Bs_mdl = np.zeros( (Nt, 1) )
    Cs_mdl = np.zeros( (Nt, 1) )
    
    rand_numbers = np.random.uniform(0,1, activations.shape)
    
    for i in range(Nt):
        
        print(i)
        spk_stims_i = np.where(activations[i, :] > rand_numbers[i, :])
        
        Sd = np.squeeze(stims_v[:,spk_stims_i])
        
        pca = PCA()
        pca.fit(np.cov(Sd))
        
        # 1D Non Linearity:
        Wt_mdl[i, :] = pca.components_[0,:]
        Gs_mdl[i, :] = np.dot(Wt_mdl[i, :], stims_v)
        
        popt, pcov = curve_fit(DoubleSig, Gs_mdl[i, :], activations[i, :], p0 = [8, 0.5], maxfev=5000 )
        
        Bs_mdl[i] = popt[0]
        Cs_mdl[i] = popt[1]
        

    return Wt_mdl, Gs_mdl, Bs_mdl, Cs_mdl


#%%============================================================================
# Retina and Electrodes positions:
# =============================================================================
def ret_and_elec_pos(Nt, ret_side, elec_side):
    

    rets_per_row =  int(np.sqrt(Nt))
    ret_space = ret_side / rets_per_row

    Ne = elec_side*elec_side
    elec_space = ret_side/elec_side
        
    x_ret = np.matlib.repmat(np.arange( ret_space/2, ret_side, ret_space ), rets_per_row,1)
    y_ret = np.matlib.repmat(np.reshape(np.arange(ret_space/2, ret_side, ret_space ), (rets_per_row, 1)), 1, rets_per_row)
    
    elec_space = ret_side/elec_side;
    x_elec = np.matlib.repmat(np.arange(elec_space/2, elec_space*elec_side, elec_space), elec_side,1)
    y_elec = np.matlib.repmat(np.reshape(np.arange(elec_space/2, elec_space*elec_side, elec_space), (elec_side, 1)), 1, elec_side)

    if any(np.in1d(x_ret, x_elec)):
        x_elec = x_elec + 0.001 # if positions are the same the 1/r kernel explodes
        y_elec = y_elec + 0.001 # if positions are the same the 1/r kernel explodes

    x_ret_flat = x_ret.reshape((Nt,1))
    y_ret_flat = y_ret.reshape((Nt,1))
    x_elec_flat = x_elec.reshape((Ne,1))
    y_elec_flat = y_elec.reshape((Ne,1))


    return x_ret_flat , y_ret_flat, x_elec_flat, y_elec_flat, x_ret, y_ret, x_elec, y_elec


#%% ===========================================================================
# Transfer Matrix
# =============================================================================
def Wt_mat(ret_side, elec_side, spread, noise, ret_space = 1):
        
    Nt = ret_side**2
    Ne = elec_side**2
        
    Wt = np.zeros((Nt,Ne))    
        
    x_ret, y_ret, x_elec, y_elec,_,_,_,_ = ret_and_elec_pos(ret_side, elec_side)    
    
    elec_space = ret_side*ret_space/elec_side
    sigma = spread * elec_space
    
    
    for ret_i in range(Nt):
        for elec_j in range(Ne):
            Wt[ret_i, elec_j] =  (1+noise*(random.random()*2-1))*math.exp( (-(x_ret[ret_i] - x_elec[elec_j])**2 - (y_ret[ret_i] - y_elec[elec_j])**2) / (2*sigma**2) )       

    Wt_tensor = np.array([Wt])
    
    return Wt, Wt_tensor, x_elec, y_elec

    
#%%============================================================================
# Transfer Matrix
#==============================================================================
def Wt_mat_ell(ret_side, elec_side, spreads_x, spreads_y = None, angs = 0, noise = 0, ret_space = 1, normalization = False):
    
    Nt = ret_side**2
    Ne = elec_side**2
        
    Wt = np.zeros((Nt,Ne))  
    
    elec_space = ret_side*ret_space/elec_side
    sigmas_x = spreads_x * elec_space
    sigmas_y = spreads_y * elec_space
        
    x_ret, y_ret, x_elec, y_elec,_,_,_,_ = ret_and_elec_pos(ret_side, elec_side)
           
    a = np.divide( np.square(np.cos(angs)), 2*np.square(sigmas_x) ) + np.divide( np.square(np.sin(angs)), 2* np.square(sigmas_y) )
    b = np.divide( np.sin(2*angs), 4*np.square(sigmas_x) ) - np.divide( np.sin(2*angs), 4*np.square(sigmas_y) );
    c = np.divide(np.square(np.sin(angs)), 2*np.square(sigmas_x)) + np.divide(np.square(np.cos(angs)), 2*np.square(sigmas_y));
    
    if a.size == 1:
        a = np.matlib.repmat(a, Nt, 1)
    
    if b.size == 1:
        b = np.matlib.repmat(b, Nt, 1)
        
    if c.size == 1:
        c = np.matlib.repmat(c, Nt, 1)
    
    for ret_i in range(Nt):
        for elec_j in range(Ne):
            Wt[ret_i,elec_j] = ( (1+noise*(random.random()*2-1)) * math.exp( -(a[ret_i]*(x_ret[ret_i]-x_elec[elec_j])**2) - 
                        2*b[ret_i]*(x_ret[ret_i]-x_elec[elec_j]) *(y_ret[ret_i]-y_elec[elec_j]) - 
                        c[ret_i]*(y_ret[ret_i]-y_elec[elec_j])**2 ) )
        
        # Normalization...
        if normalization:
            Wt[ret_i,:] = Wt[ret_i,:] / np.linalg.norm(Wt[ret_i,:]) 
                        
    Wt_tensor = np.array([Wt])    
    return Wt, Wt_tensor#, x_elec, y_elec

#%%============================================================================
# Transfer Matrix - genralized for random retina positions
#==============================================================================
def Wt_mat_ell(x_ret, y_ret, x_elec, y_elec, ret_side, elec_side, spreads_x, spreads_y, elec_space, angs = 0, noise = 0, normalization = False):
    print('-------------------------------------------------------------')
    print('latter one called?')
    Nt = x_ret.shape[0]
    Ne = elec_side**2
        
    Wt = np.zeros((Nt,Ne))  
    
    sigmas_x = spreads_x * elec_space
    sigmas_y = spreads_y * elec_space
               
    a = np.divide( np.square(np.cos(angs)), 2*np.square(sigmas_x) ) + np.divide( np.square(np.sin(angs)), 2* np.square(sigmas_y) )
    b = np.divide( np.sin(2*angs), 4*np.square(sigmas_x) ) - np.divide( np.sin(2*angs), 4*np.square(sigmas_y) );
    c = np.divide(np.square(np.sin(angs)), 2*np.square(sigmas_x)) + np.divide(np.square(np.cos(angs)), 2*np.square(sigmas_y));
    
    if a.size == 1:
        a = np.matlib.repmat(a, Nt, 1)
    
    if b.size == 1:
        b = np.matlib.repmat(b, Nt, 1)
        
    if c.size == 1:
        c = np.matlib.repmat(c, Nt, 1)
    
    for ret_i in range(Nt):
        for elec_j in range(Ne):
            Wt[ret_i,elec_j] = ( (1+noise*(random.random()*2-1)) * math.exp( -(a[ret_i]*(x_ret[ret_i]-x_elec[elec_j])**2) - 
                        2*b[ret_i]*(x_ret[ret_i]-x_elec[elec_j]) *(y_ret[ret_i]-y_elec[elec_j]) - 
                        c[ret_i]*(y_ret[ret_i]-y_elec[elec_j])**2 ) )
        
        # Normalization...
        if normalization:
            Wt[ret_i,:] = Wt[ret_i,:] / np.linalg.norm(Wt[ret_i,:]) 
                        
    Wt_tensor = np.array([Wt])    
    return Wt, Wt_tensor


#%% ===========================================================================
# Linear Inversion
# =============================================================================
def LinearInv(target_activation_vec, Wt, eig_lim = 0.01):
    Wt_inv = np.linalg.pinv(Wt, rcond=eig_lim)
    print( 'eig lim: ' + str(eig_lim) )
    s = np.dot(Wt_inv, target_activation_vec)
    return s
     
    
def LinearInv_transp(target_activation_vec, Wt, eig_lim = 0.01):    
    Wt_inv = np.linalg.pinv(Wt, rcond=eig_lim)
    print( 'eig lim: ' + str(eig_lim) )
    s = np.dot(Wt_inv.transpose(), target_activation_vec.transpose())
    return s
    
         
#%% ===========================================================================
# Test Simple Stimulus
# =============================================================================     
def make_S_tensor(s):
    s = s.flatten()
    s = s.reshape(s.shape[0], 1)
    s_tensor =  np.array([s])
    return s_tensor
    

def tensorVec2mat(tensorVec, Nt):
     return tf.reshape(tensorVec, (Nt,Nt))
    

def show_SimpleStims(ret_side, Wt_tensor):
    plt.figure()
    
    # Positive Pulse
    s = np.zeros((ret_side,ret_side))
    s[5, 5] = 1
    s_tensor = make_S_tensor(s)
    activation_vec = LNL(s_tensor, Wt_tensor)
    activation = tensorVec2mat(activation_vec, ret_side)
    plt.subplot(5,2,1)
    plt.imshow(s, clim=(-1, 1))
    plt.subplot(5,2,2)
    plt.imshow(activation, clim=(0, 1) )
    
    
    # Positive and Negative Pulses faw away
    s = np.zeros((ret_side,ret_side))
    s[2, 2] = 1
    s[7, 7] = 1
    s_tensor = make_S_tensor(s)
    activation_vec = LNL(s_tensor, Wt_tensor)
    activation = tensorVec2mat(activation_vec, ret_side)
    plt.subplot(5,2,3)
    plt.imshow(s, clim=(-1, 1))
    plt.subplot(5,2,4)
    plt.imshow(activation, clim=(0, 1))
    
    # Positive and Negative Pulses close
    s = np.zeros((ret_side,ret_side))
    s[5, 4] = 1
    s[5, 6] = 1
    s_tensor = make_S_tensor(s)
    activation_vec = LNL(s_tensor, Wt_tensor)
    activation = tensorVec2mat(activation_vec, ret_side)
    
    plt.subplot(5,2,5)
    plt.imshow(s, clim=(-1, 1))
    plt.subplot(5,2,6)
    plt.imshow(activation, clim=(0, 1))
    
    
    # Positive and Negative Pulses faw away
    s = np.zeros((ret_side,ret_side))
    s[2, 2] = 1
    s[7, 7] = -1
    s_tensor = make_S_tensor(s)
    activation_vec = LNL(s_tensor, Wt_tensor)
    activation = tensorVec2mat(activation_vec, ret_side)
    plt.subplot(5,2,7)
    plt.imshow(s, clim=(-1, 1)) 
    plt.subplot(5,2,8)
    plt.imshow(activation, clim=(0, 1))
    
    
    # Positive and Negative Pulses close
    s = np.zeros((ret_side,ret_side))
    s[5, 4] = 1
    s[5, 6] = -1
    s_tensor = make_S_tensor(s)
    activation_vec = LNL(s_tensor, Wt_tensor)
    activation = tensorVec2mat(activation_vec, ret_side)
    
    plt.subplot(5,2,9)
    plt.imshow(s, clim=(-1, 1))
    plt.subplot(5,2,10)
    plt.imshow(activation, clim=(0, 1))
    
    
#%% Retina output Error
def activationError(retina_output_vec, desired_output_vec): 
   
    retina_output_vec = np.squeeze(retina_output_vec)
    desired_output_vec = np.squeeze(desired_output_vec)
            
    MSEs = []
    
    if retina_output_vec.ndim == 1:    
       MSEs = np.square(np.subtract(retina_output_vec, desired_output_vec)).mean() 
       
    else:
       nimgs = retina_output_vec.shape[0] 
       for i in range(nimgs):        
           MSE = np.square(np.subtract(retina_output_vec[i], desired_output_vec[i])).mean()  
           MSEs.append(MSE)
        
    all_RMSEs = np.sqrt(MSEs)
    mean_RMSE = np.mean(all_RMSEs)
    
    return all_RMSEs, mean_RMSE


#%% Compare Stimulation

# def Compare_stimulation_methods(target_vec, s_nn_vec, ret_side, elec_side, Wt, Wt_tensor, Wt_tensor_noisy = None):
        
#     target = target_vec.reshape(ret_side, ret_side)        
    
#     # Conventional:
#     # ...       
    
    
#     # Linear Inversion:
#     s_lininv_vec = LinearInv(target.reshape(ret_side*ret_side,1), Wt)
#     s_lininv = s_lininv_vec.reshape(elec_side, elec_side)
#     if Wt_tensor_noisy is None:
#         result_lininv_vec = LNL(np.array([s_lininv_vec]), Wt_tensor)
#     else:
#         result_lininv_vec = LNL(np.array([s_lininv_vec]), Wt_tensor_noisy)
        
#     result_lininv = tf.reshape(result_lininv_vec, (ret_side, ret_side))
    
    
#     # Quadratic Programing:
#     # ...    
    
    
#     # Neural Nets:
#     s_nn = s_nn_vec.reshape(elec_side, elec_side)
#     if Wt_tensor_noisy is None:
#         result_nn_vec = LNL(np.array([s_nn_vec]), Wt_tensor)
#     else:
#         result_nn_vec = LNL(np.array([s_nn_vec]), Wt_tensor_noisy)
#     result_nn = np.reshape(result_nn_vec, (ret_side, ret_side)) 
        
#     return s_lininv, s_nn, result_lininv, result_nn
    


#%% 2D Gaussian Kernel:
def gkern(l=5, sig=1.):
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


#%% Measurment from MEA:
# def MEA_measure(ret_activation, measure_radius, ret_side, elec_side):  
        
#     measure_reach_sigma =  measure_radius/4 
#     kernel_side = int(measure_radius*2)
#     h = gkern(kernel_side, measure_reach_sigma)       

#     x_ret_flat, y_ret_flat, x_elec_flat, y_elec_flat, x_ret, y_ret, x_elec, y_elec = ret_and_elec_pos(ret_side, elec_side)
    
#     kernel_side = h.shape[0]
#     elec_side = x_elec.shape[0]
#     ret_side = x_ret.shape[0]
    
#     Ne = elec_side**2
#     measured_activity = np.zeros((Ne, 1))
#     ret_activation = ret_activation.flat
        
#     for rec_elec_i in range(Ne):       
        
#         ret_inds = get_ret_indx_under_elec(x_ret, y_ret, x_elec, y_elec, kernel_side, rec_elec_i)
        
#         # Activity measured in recording electrode:
#         measured_activity[rec_elec_i] = np.sum(np.multiply(ret_activation[ret_inds], h))
    
#     measured_activity = measured_activity.reshape(elec_side, elec_side)
#     return measured_activity


#%% Measurment from MEA:
# def MEA_measure(ret_signal_vec, x_ret, y_ret, x_elec, y_elec, z_elec, elec_side, nSignals):  
    
#     Ne = elec_side**2
#     elec_signal_vec = np.zeros((Ne, nSignals))
    
#     for elec_i in range(Ne):
#         kernel = Elec_measure_kernel(x_ret, y_ret, x_elec[elec_i], y_elec[elec_i], z_elec, rsqaure_norm = True)
#         elec_signal_vec[elec_i, :] = np.sum(np.multiply(ret_signal_vec,kernel.flat));
    
#     return elec_signal_vec


#%% Measurment kernel for given Electrode
# def Elec_measure_kernel(x_ret, y_ret, x_elec, y_elec, z_elec, rsquare_norm = True):
    
#     dists = np.sqrt(np.square(x_ret-x_elec) + np.square(y_ret - y_elec) + np.square(0 - z_elec) )

#     if rsquare_norm:  
#         kernel = 1 / np.square(dists) 
#         kernel = kernel/np.sum(kernel);    
#     else:
#         # Option 2 - Einevoll paper seems more like this    
#         kernel = 1/dists
#         kernel = kernel/np.sum(kernel); # well, actually, his formula doesn't normalize the kernel...but values would explode...should they?

#     # return kernel


#%%  Get indexes of retina cells covered by the recording electrode i
def get_ret_indx_under_elec(x_ret, y_ret, x_elec, y_elec, kernel_side, rec_elec_i):
     
    Nt = x_ret.size
    Ne = x_elec.size
    ret_side = x_ret.shape[0]
    
    x_ret_flat = x_ret.reshape((Nt,1))
    y_ret_flat = y_ret.reshape((Nt,1))
    x_elec_flat = x_elec.reshape((Ne,1))
    y_elec_flat = y_elec.reshape((Ne,1))
       
    x_elec_pos = x_elec_flat[rec_elec_i]
    y_elec_pos = y_elec_flat[rec_elec_i]
       
    elec_reach_x_from = x_elec_pos - kernel_side/2
    elec_reach_x_to = x_elec_pos + kernel_side/2
       
    elec_reach_y_from = y_elec_pos - kernel_side/2
    elec_reach_y_to = y_elec_pos + kernel_side/2
       
    (x_ret[0,:] > elec_reach_x_from) & (x_ret[0,:] < elec_reach_x_to)
    
    ret_cols = np.where((x_ret[0,:] > elec_reach_x_from) & (x_ret[0,:] < elec_reach_x_to))
    ret_rows = np.where((y_ret[:,0] > elec_reach_y_from) & (y_ret[:,0] < elec_reach_y_to))
       
    
    
    ret_cols = np.matlib.repmat(ret_cols, len(ret_rows[0]), 1)
    ret_rows = np.matlib.repmat(ret_rows, len(ret_cols[0]), 1)
       

    # Indexes of retina covered by recording electrode:
    ret_inds = np.ravel_multi_index((ret_rows, ret_cols), [ret_side, ret_side])
    

    return ret_inds


#%%============================================================================
# Measurment ERFs:
# =============================================================================
def estimated_ERFs(W, measure_radius, ret_side, elec_side): 
    
    measure_reach_sigma =  measure_radius/4 
    kernel_side = int(measure_radius*2)
    h = gkern(kernel_side, measure_reach_sigma)

    _,_,_,_, x_ret, y_ret, x_elec, y_elec = ret_and_elec_pos(ret_side, elec_side)
    
    Ne = elec_side**2
    
    W_exper = np.zeros((Ne, Ne))
        
    for rec_elec_i in range(Ne):        
        
        ret_inds = get_ret_indx_under_elec(x_ret, y_ret, x_elec, y_elec, kernel_side, rec_elec_i)
        ret_inds = np.reshape(ret_inds, (kernel_side**2, 1))
        
        for stim_elec_i in range(Ne):
            W_exper[rec_elec_i, stim_elec_i] = np.sum(np.multiply( W[ret_inds, stim_elec_i].flat, h.flat))
    
    return W_exper

#%%=================================================================================
# Mean Activity of retina in pixel positions - To calculate retina activation error
# ==================================================================================
def Ret_Act_in_Pixels(real_activ_ret, ret_side, x_ret, y_ret):
    
    pxl_borders = np.linspace(0,ret_side,ret_side+1)     
    ret_acts_pxl = np.zeros((ret_side, ret_side))    

    scaling = ret_side / 32
    x_ret = x_ret * scaling
    y_ret = y_ret * scaling

    real_activ_ret = real_activ_ret.squeeze()

    for i in range(len(pxl_borders)-1):
        x_borders = (pxl_borders[i], pxl_borders[i+1])
        
        for j in range(len(pxl_borders)-1):            
            y_borders = (pxl_borders[j], pxl_borders[j+1])
            # ret_act_pxl = real_activ_ret[0,np.where((x_ret.squeeze() > x_borders[0]) & (x_ret.squeeze() <= x_borders[1]) & (y_ret.squeeze() > y_borders[0]) & (y_ret.squeeze() <= y_borders[1]))].squeeze()
            
            ret_act_pxl = real_activ_ret[np.where((x_ret.squeeze() > x_borders[0]) & (x_ret.squeeze() <= x_borders[1]) & (y_ret.squeeze() > y_borders[0]) & (y_ret.squeeze() <= y_borders[1]))]
            ret_acts_pxl[j,i] = np.mean(ret_act_pxl)

    return ret_acts_pxl

#%% Downsample target:
    
from scipy import ndimage

def downsample_imgs(imgs, elec_side, normalize = False):

    img_side = imgs.shape[1]
    nFrames = imgs.shape[0]
    scale_factor = int(img_side / elec_side)

    imgs_elec_mat = np.zeros((nFrames, elec_side, elec_side))

    for frame in range(nFrames):
        
        img = imgs[frame]  
        
        # Define the low-pass filter:
        filter_size = 2 * scale_factor + 1
        filter_sigma = scale_factor / 2
        filter_kernel = np.outer(
            np.exp(-(np.arange(filter_size) - scale_factor)**2 / (2 * filter_sigma**2)),
            np.exp(-(np.arange(filter_size) - scale_factor)**2 / (2 * filter_sigma**2))
        ) / (2 * np.pi * filter_sigma**2)
        
        # Apply the filter:
        img_filtered = ndimage.convolve(img, filter_kernel, mode='nearest')
        
        # Downsample the filtered image:
        img_downsampled = img_filtered[::scale_factor, ::scale_factor]
        
        if normalize:
            imgs_elec_mat[frame] = np.divide(img_downsampled , 256)
        else:
            imgs_elec_mat[frame] = img_downsampled
        # # Display the results
        # cv2.imshow('Downsampled image', img_downsampled)
        # cv2.waitKey(30)
        
        
    imgs_elec_vec = imgs_elec_mat.reshape(nFrames, elec_side**2)   
    
    return imgs_elec_mat, imgs_elec_vec
    
    

#%%=============================================================================
# Retina Activation Error
# =============================================================================
def Ret_Act_Error(test_imgs, real_act_ret, ret_side, x_ret, y_ret):
        
    nTest = test_imgs.shape[0]
    RMSEs = np.zeros(nTest)
    
    for i in range(nTest):
        print(i)
        ret_in_pxl = Ret_Act_in_Pixels(real_act_ret[i], ret_side, x_ret, y_ret)
                
        # plt.figure()
        # plt.subplot(1,2,1)
        # plt.imshow(ret_in_pxl)        
        # plt.subplot(1,2,2)
        # plt.imshow(test_imgs[i].reshape((ret_side, ret_side)))        
        # plt.pause(1)
        
        ret_in_pxl = ret_in_pxl.reshape((np.size(ret_in_pxl),1))
        
        # ret_in_pxl_ = ret_in_pxl.reshape((ret_side, ret_side))
        # test_imgs_= test_imgs[i].reshape((ret_side, ret_side))        
        # plt.figure()
        # plt.subplot(1,2,1)
        # plt.imshow(ret_in_pxl_, cmap = 'gray', clim = (0,1))        
        # plt.subplot(1,2,2)
        # plt.imshow(test_imgs_, cmap = 'gray', clim = (0,1))             
        
        RMSEs[i] = np.sqrt(np.mean(np.square(test_imgs[i] - ret_in_pxl)))

    return RMSEs

# # =============================================================================
# # Measurment Non-linearity
# # =============================================================================
# def estimated_NonLin(ret_side, elec_side, measure_radius, b, c):
    
    
#     Ne = elec_side**2
#     Nt = ret_side**2
    
#     measure_reach_sigma =  measure_radius/4 
#     kernel_side = int(measure_radius*2)
#     h = gkern(kernel_side, measure_reach_sigma)
#     h = h.reshape(kernel_side**2, 1)
    
    
#     _,_,_,_, x_ret, y_ret, x_elec, y_elec = ret_and_elec_pos(ret_side, elec_side)
    
               
#     nPoints = 1000
#     xx = np.linspace(-2, 2, nPoints)
    
#     Bs = np.zeros((Ne, 1))
#     Cs = np.zeros((Ne, 1))
    
#     plt.figure()
    
    
#     for rec_elec_i in range(Ne):
        
#         elec_sigmoids = np.zeros((1, nPoints))
        
#         ret_inds = get_ret_indx_under_elec(x_ret, y_ret, x_elec, y_elec, kernel_side, rec_elec_i)
#         ret_inds = np.reshape(ret_inds, (kernel_side**2, 1))
                
#         for ret_i in range(ret_inds.shape[0]):
            
#             sig =  sig_func(xx, b[ret_inds[ret_i]], c[ret_inds[ret_i]])
#             scaled_sig = np.multiply(sig, h[ret_i])
#             elec_sigmoids = elec_sigmoids + scaled_sig   
    
#         plt.plot(xx, elec_sigmoids[0,:])
        
#         popt, _ = curve_fit(sig_func, xx.flatten(), elec_sigmoids.flatten())

#         Bs[rec_elec_i] = popt[0]
#         Cs[rec_elec_i] = popt[1]

#     return Bs, Cs


# def sig_func(xx, b, c):
#     return 1/(1 + np.exp(  np.multiply(-b, abs(xx)) -c) )
