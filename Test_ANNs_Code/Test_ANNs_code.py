# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 12:45:18 2024

@author: Domingos

Test Code

"""
#%% Import libraries:

import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers,regularizers
from keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import numpy.matlib
import timeit
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os


from Dataset import*
from RetinaFun import*
from PlotFunctions import*


#%% Initializations:
     
test_path = os.getcwd() + '/Test_ANNs_Code/Results'   
print(test_path)
seed = 1 # [1,2,3,4,5] # seed for random generators 
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

train_img_bw = []
with_aug = 'none'

spread =  1.5 # [1, 1.5, 2, 2.5, 3] # ERF (electrical receptive field) spreads
reg_lambda = 1e-3 # [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 0] # regularization
# learnRates = [0.1, 0.01, 0.001, 0.0001]


nEpochs = 1 # 300 # epochs for encoder (stimulus generation)
LNL_epochs = 1 # 200 # epochs for decoder (retina response estimation)

# Size of retina and Multi Electrode Array
elec_side = 8
ret_side = 32

Ne = elec_side**2
Nt = 10000

elec_radius = 1


#%%====================================================================
# Forward Model:
# =====================================================================

# Build retina model:

ret_mode = 'jitter' # retina grid with jittered positions   
x_ret, y_ret, x_elec, y_elec, z_elec, Wt, Wt_tensor, Bs, Cs = BuildRetinaModel(Nt, ret_side, elec_side, spread, mode = ret_mode)
print('------------------------------------------------------------------------------------')   
print('x_ret:',x_ret.shape)
print('x_ret content:',x_ret)                
model_pars = (x_ret, y_ret, x_elec, y_elec, z_elec, Nt, ret_side, elec_side, Wt, Bs, Cs)

# Plot test stimulus:
show_Retina_MEA_rand(x_ret, y_ret, x_elec, y_elec, ret_side, elec_side, Wt, [], Bs, Cs, save_path=test_path,  color_map = 'gray', size = 15, close_fig = False)

           
# Forward model estimation - Create random stimulus and train Measurement Predictor Network
# the model is saved in test_path as "LNL_model.keras"
import LNL_with_NNs
print('ANN for LNL...')
LNL_with_NNs.Run_NNs_for_LNL(model_pars, elec_side, main_path = test_path, nEpochs = LNL_epochs, elec_rad = elec_radius, close_fig = False)
            

#%% =============================================================================
#  Load Natural Images Dataset
# =============================================================================

print('loading dataset...')
train_val_split = 0.20  # 20% of training set is used for validation

# Real Images:
train_img_bw, val_img_bw, test_img_bw, train_img_vec, val_img_vec, test_img_vec = Create_Dataset(train_val_split, ret_side, augmentation=with_aug)

nTrainImg = train_img_bw.shape[0]
nValImg = val_img_bw.shape[0]
nTestImg = test_img_bw.shape[0]

_,_,_,_, x_target, y_target, _, _ = ret_and_elec_pos(ret_side**2, ret_side, elec_side)            

# Downsample without aliasing:
train_img_sub, train_sub_vec = downsample_imgs(train_img_bw.squeeze(), elec_side)  
val_img_sub, val_sub_vec = downsample_imgs(val_img_bw.squeeze(), elec_side)  
test_img_sub, test_sub_vec = downsample_imgs(test_img_bw.squeeze(), elec_side)  



#%% =============================================================================
#  Inverse Model:
# =============================================================================

# Load LNL_model - from stimulus to measurement:
stim_measure_model = load_model(test_path + "/LNL_model.keras")
for layer in stim_measure_model.layers:
        layer.trainable = False # freeze model
        
# From target to Stimulus Model:
input_shape = (Ne,)
input = layers.Input(shape=input_shape, name="input")
dense_1 = layers.Dense(Ne*2, activation='relu', name='dense_1', input_shape=input_shape)(input)
dense_2 = layers.Dense(Ne*2, activation='relu', name='dense_2')(dense_1)
dense_3 = layers.Dense(Ne*2, activation='relu', name='dense_3')(dense_2)

# Stimulus Layer:
stim_layer = layers.Dense(Ne, name='stim_layer', activation='tanh', activity_regularizer=regularizers.l2(reg_lambda))(dense_3)
targ_stim_model = models.Model(input, stim_layer, name="stim_model")
output_stim = targ_stim_model(input)

# Merge two models: targ_stim_model + stim_measure_model(LNL_model)
output_measure = stim_measure_model(output_stim) 
full_model = models.Model(input, output_measure, name = 'full_model')

optimizer = optimizers.Adam()
full_model.compile(optimizer=optimizer,  loss='mean_squared_error') # loss=ssim_loss) #        
full_model.summary()


# %%===========================================================================
# Compile and train the model
# =============================================================================
print('training...')

# TRAIN
start = timeit.default_timer()
history = full_model.fit(train_sub_vec, train_sub_vec, epochs=nEpochs, validation_data=(val_sub_vec, val_sub_vec), batch_size=32, verbose=2) #,  callbacks=[early_stopping, model_checkpoint])
stop = timeit.default_timer()
print('Time [mins]: ' + str((stop - start)/60))
                
# Save Model:                
full_model.save( test_path + '/full_model.keras')
targ_stim_model.save( test_path + '/target_stim_model.keras')
stim_measure_model.save( test_path + '/stim_measure_model.keras')

# Save models architecture in txt file:
def myprint(s):
    with open(test_path + '/models_summary.txt','a') as f:
        print(s, file=f)
        
full_model.summary(print_fn=myprint)
targ_stim_model.summary(print_fn=myprint)
stim_measure_model.summary(print_fn=myprint)


#%% Plot training Evolution:
fig = plt.figure()
plt.subplot(2, 1, 1)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.ylabel('Loss')
plt.legend(loc='upper right')

plt.subplot(2, 1, 2)
min_val = np.min(history.history['val_loss'])
ylims = (min_val*0.95, min_val*1.5)

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.ylim(ylims)

plt.title('Zoom')
plt.savefig(test_path + '/Training.pdf')       


#%%============================================================================
# Testing Stage
# =============================================================================

stimulus_ = targ_stim_model.predict(test_sub_vec)
stimulus = np.reshape(stimulus_, (nTestImg, elec_side, elec_side))


# Estimated Measured Activation
estim_measured_activ_elec = full_model.predict(test_sub_vec)
        
# Real Retina Activation:
Gs_ret = np.dot(Wt, stimulus_.transpose())
real_activ_ret = DoubleSig(Gs_ret, Bs, Cs)
        
# Measured Activation:
real_activ_elec, real_activ_elec_mat = Elec_measure(real_activ_ret, x_ret, y_ret, x_elec, y_elec, z_elec, elec_side, nTestImg, elec_radius = elec_radius)


# Compare Real Activation with Desired Activation:
# all_RMSE_Real_Target_RET = show_prediction_examples(test_img_vec, stimulus, real_activ_ret, ret_side, rand_imgs=False, save_path=test_path, title='Desired Ret Activ VS Real Ret Activ')
show_prediction_examples_rand_ret(x_ret, y_ret, test_img_vec, stimulus, real_activ_ret, ret_side, rand_imgs=False, save_path=test_path, title='Desired Ret Activ VS Real Ret Activ', close_fig = False)

# Compare Real Measurement with Desired Target Measured Activity
all_RMSE_Real_Target_ELEC = show_prediction_examples(test_sub_vec , stimulus, real_activ_elec, elec_side, rand_imgs=False, save_path=test_path, title='Desired Elec Measurement VS Real Elec Measurement', close_fig = False)

# Compare Estimated Measurement with Desired Target Measured Activity
all_RMSE_Real_Estim_ELEC = show_prediction_examples(test_sub_vec , stimulus, estim_measured_activ_elec, elec_side, rand_imgs=False, save_path=test_path, title='Desired Elec Measurement VS Estimated Elec Measurement', close_fig = False)

# test_errors = (all_RMSE_Real_Target_RET, all_RMSE_Real_Target_ELEC, all_RMSE_Real_Estim_ELEC)
test_errors = (all_RMSE_Real_Target_ELEC, all_RMSE_Real_Estim_ELEC)

with open(test_path + "/Test_Erros.pickle", 'wb') as f:
    pickle.dump(test_errors, f)  

