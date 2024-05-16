# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 16:05:36 2022
LNL approximation at the electrode level with NN!
@author: ddiascostale
"""


import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import datasets, layers, models, optimizers
from tensorflow.keras.layers import Lambda
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
import math
import random
import numpy.matlib
from PIL import Image

from Dataset import*
from RetinaFun import*
from PlotFunctions import*

import timeit
from Dataset import*
from RetinaFun import*
from PlotFunctions import*


def Run_NNs_for_LNL(model_pars, elec_side, main_path, nEpochs, elec_rad = 0, close_fig = False, nonLin = 'sigmoid'):
    #INPUT: 
    # Model pars: see below
    # elec_side,: side length of electrode array as a square
    # main_path: main directory of file saving
    # nEpochs: number of epoches of training
    # elec_rad = 0: 
    x_ret = model_pars[0] 
    y_ret = model_pars[1] 
    x_elec = model_pars[2] 
    y_elec = model_pars[3] 
    z_elec = model_pars[4] 
    Nt = model_pars[5] 
    ret_side = model_pars[6] 
    elec_side = model_pars[7] 
    Wt = model_pars[8] 
    Bs = model_pars[9] 
    Cs = model_pars[10] 
    
    
    #%%  Generate Random Stimulus patterns
    Ne = elec_side**2
    
    nTrain = 50000 # 10001
    nVal = 5000 # 2000
    nTest = 1000
    
    train_stims = np.random.normal(0, 0.3, (nTrain, elec_side,elec_side))
    val_stims = np.random.normal(0, 0.3, (nVal, elec_side,elec_side))
    test_stims = np.random.normal(0, 0.3, (nTest, elec_side,elec_side))
    
    single_stim = np.zeros((elec_side, elec_side))
    single_stim[int(elec_side/2), int(elec_side/2)] = 1
    test_stims[0, :,:] = single_stim
            
    train_stims_v = train_stims.reshape(nTrain, Ne)
    val_stims_v = val_stims.reshape(nVal, Ne)
    test_stims_v = test_stims.reshape(nTest, Ne)
    
    # up to here, 3 copies of data is built with iid samples from a normal distribution, with size (nsample,8,8) where 8 is side of electrode array
    #%% Ground Truth:
    print('iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii')
    Gs_ret_train = np.dot(Wt, train_stims_v.transpose())
    real_activ_ret_Train = DoubleSig(Gs_ret_train, Bs, Cs, nonLin = nonLin)
    real_measure_Train_v, real_measure_Train_mat  = Elec_measure(real_activ_ret_Train, x_ret, y_ret, x_elec, y_elec, z_elec, elec_side, nTrain, elec_radius = elec_rad)
    
    Gs_ret_val = np.dot(Wt, val_stims_v.transpose())
    real_activ_ret_Val = DoubleSig(Gs_ret_val, Bs, Cs, nonLin = nonLin)
    real_measure_Val_v, real_measure_Val_mat  = Elec_measure(real_activ_ret_Val, x_ret, y_ret, x_elec, y_elec, z_elec, elec_side, nVal, elec_radius = elec_rad)
    
    Gs_ret_test = np.dot(Wt, test_stims_v.transpose())
    real_activ_ret_Test = DoubleSig(Gs_ret_test, Bs, Cs, nonLin = nonLin)  
    real_measure_Test_v, real_measure_Test_mat = Elec_measure(real_activ_ret_Test, x_ret, y_ret, x_elec, y_elec, z_elec, elec_side, nTest, elec_radius = elec_rad)
    
    
    # fig = plt.figure()
    # plt.subplot(3,3,1)
    # plt.imshow(train_stims[0, :,:], cmap = 'bwr', interpolation = 'none', clim=(-1, 1))
    # plt.colorbar()
    # plt.title('Stimulus')
    
    # ax = fig.add_subplot(3,3,2)
    # plt.scatter( x_ret, y_ret, c = real_activ_ret_Train[0,:], alpha=0.5, edgecolors=None, cmap = 'gray'  ) 
    # ax.set_aspect('equal')          
    # plt.gca().invert_yaxis()
    # plt.colorbar()    
    # plt.axis('off')
    # plt.title('Real Activation') 
    
    # plt.subplot(3,3,3)
    # plt.imshow(real_measure_Train_mat[0], cmap = 'gray', interpolation = 'none', clim=(0, 1)) # clim=(0, 1), 
    # plt.colorbar()
    # plt.title('Measured Activation')
    
    # plt.subplot(3,3,4)
    # plt.imshow(train_stims[10,:,:], cmap = 'bwr', interpolation = 'none', clim=(-1, 1))
    # plt.colorbar()
    
    # ax = fig.add_subplot(3,3,5)
    # plt.scatter( x_ret, y_ret, c = real_activ_ret_Train[10,:], alpha=0.5, edgecolors=None, cmap = 'gray') 
    # plt.colorbar()    
    # plt.gca().invert_yaxis()
    # ax.set_aspect('equal')  
    # plt.axis('off')

    # plt.subplot(3,3,6)
    # plt.imshow(real_measure_Train_mat[10], cmap = 'gray', interpolation = 'none', clim=(0, 1)) # clim=(0, 1), 
    # plt.colorbar()
    
    # plt.subplot(3,3,7)
    # plt.imshow(train_stims[100, :,:], cmap = 'bwr', interpolation = 'none', clim=(-1, 1))
    # plt.colorbar()
    
    # ax = fig.add_subplot(3,3,8)
    # plt.scatter( x_ret, y_ret, c = real_activ_ret_Train[100,:], alpha=0.5, edgecolors=None, cmap = 'gray' )     
    # plt.gca().invert_yaxis()
    # plt.colorbar()
    # ax.set_aspect('equal')  
    # plt.axis('off')
        
    # plt.subplot(3,3,9)
    # plt.imshow(real_measure_Train_mat[100], cmap = 'gray', interpolation = 'none', clim=(0, 1)) # clim=(0, 1), 
    # plt.colorbar()
    
    
    #%% LNL with NNs - Measurement Predictor Network!  
    input_shape = (Ne,)
    input = layers.Input(shape=input_shape, name="input")
    dense_1 = layers.Dense(Ne*2, activation='relu', name='dense_1', input_shape=input_shape)(input)
    dense_2 = layers.Dense(Ne*2, activation='relu', name='dense_2')(dense_1)
    dense_3 = layers.Dense(Ne, activation='relu', name='dense_3')(dense_2)
    

    # Retina Output - Sigmoid
    LNL_model = models.Model(input, dense_3, name="LNL_model")
    LNL_model.compile(optimizer=optimizers.Adam(learning_rate=0.0005), loss='mean_squared_error') # loss= SSIMLoss) #
    LNL_model.summary()
    
    
    #%%============================================================================
    # Compile and train the model
    # =============================================================================
    
    print(real_measure_Val_mat.shape)
    
    # TRAIN
    start = timeit.default_timer()
    history = LNL_model.fit(train_stims_v, real_measure_Train_v, epochs=nEpochs,
                        validation_data=(val_stims_v, real_measure_Val_v), batch_size=32)
    
    stop = timeit.default_timer()
    print('Time [mins]: ' + str((stop - start)/60))
    
    LNL_model.save(main_path + "/LNL_model.keras")
    
    
    # %%===========================================================================
    # Evaluate the model
    # =============================================================================
    
    fig = plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    
    plt.subplot(2, 1, 2)
    min_val = np.min(history.history['val_loss'])
    ylims = (min_val*0.95, min_val*1.5)
    plt.ylim(ylims)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    
    plt.title('Zoom')
    plt.savefig(main_path + '/Stim_to_Measure_NNs_Training.pdf')    
        
    if close_fig:
        plt.close(fig)
    
    
    # %%===========================================================================
    # Prediction of Measured Retina Response
    # =============================================================================
    
    pred_test_rec_v = LNL_model.predict(test_stims_v)
    pred_test_rec = np.reshape(pred_test_rec_v, (nTest, elec_side, elec_side))
        
    total_error = np.mean(np.abs(pred_test_rec - real_measure_Test_mat))
        
    
    
    
    #%% Plot Results
    fig = plt.figure(figsize=(16, 10), dpi=80)
    nplots = 5
    
    ret_order = np.arange(len(x_ret))
    np.random.shuffle(ret_order)  
    
    
    for i in range(nplots):
        
        plt.subplot(nplots, 5, nplots*i + 1 )
        plt.imshow(test_stims[i,:,:], cmap = 'bwr', interpolation = 'none', clim=(-1, 1)), plt.colorbar()    
        if i == 0:
            plt.title('Stimulus')        
            
        ax = fig.add_subplot(nplots, 5, nplots*i + 2 )
        plt.scatter( x_ret[ret_order], y_ret[ret_order], c = real_activ_ret_Test[i, ret_order], s = 10, alpha=0.5, edgecolors=None, cmap = 'gray' )     
        plt.gca().invert_yaxis()
        ax.set_aspect('equal')  
        plt.axis('off')        
        plt.colorbar()  
        
        if i == 0:
            plt.title('Retinal Actication')        
            
        plt.subplot(nplots, 5, nplots*i + 3 )              
        plt.imshow(real_measure_Test_mat[i], cmap = 'gray', interpolation = 'none') #, clim=(0, 1))
        plt.colorbar()
        plt.axis('off')
        if i == 0:
            plt.title('Measured Activation')   
               
        plt.subplot(nplots, 5, nplots*i + 4 )              
        plt.imshow(pred_test_rec[i], cmap = 'gray', interpolation = 'none') #, clim=(0, 1)) 
        plt.colorbar()
        plt.axis('off')
        if i == 0:
            plt.title('Estimated Measured Activation')   
            
        plt.subplot(nplots, 5, nplots*i + 5 )  
        error_img = np.abs(pred_test_rec[i] - real_measure_Test_mat[i])           
        plt.imshow(error_img, cmap = 'gray', interpolation = 'none')  
        plt.colorbar()
        plt.axis('off')
        plt.title('Error: ' + str(np.round(np.mean(error_img)*1000)/1000))  
                
        fig.suptitle('Total error (' + str(nTest) + ' stims): '  + str(np.round(total_error*1000)/1000))
    
    plt.savefig(main_path + '/Stim_to_Measure_NNs.pdf')    
    
    
    if close_fig:
        plt.close(fig)
         

        
        