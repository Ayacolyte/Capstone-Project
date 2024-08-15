import time
import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import os


# initialise the grid size, neuron array size, electrode array size. All assumed to be sqaure
grid_side_dim = 10
# elec_side_dim = 10
# neu_side_dim = 16
#==============================================================================================================
# Generate Cartesian Cooordnates for Electrodes array and Neurons 
#==============================================================================================================
def get_neu_and_elec_coord(neu_side_dim,elec_side_dim,grid_side_dim):

    neuspacing = grid_side_dim/ neu_side_dim
    elec_spacing = grid_side_dim/ elec_side_dim
        
    #repmat(np.arrange(start,stop,step), # repetitions along vertical, # neutions along horizontal)    
    x_neu= np.matlib.repmat(np.arange( neuspacing/2, grid_side_dim, neuspacing ), neu_side_dim,1) # x coordinates of N_neu neurons [sqrt(N_neu),sqrt(N_neu)]
    y_neu= np.matlib.repmat(np.reshape(np.arange(neuspacing/2, grid_side_dim, neuspacing ), (neu_side_dim, 1)), 1, neu_side_dim) # y coordinates of N_neu neurons [sqrt(N_neu),sqrt(N_neu)]
    
    x_elec = np.matlib.repmat(np.arange( elec_spacing/2, grid_side_dim, elec_spacing ), elec_side_dim,1) # x coordinates of N_elec neurons [sqrt(N_elec),sqrt(N_elec)]
    y_elec = np.matlib.repmat(np.reshape(np.arange(elec_spacing/2, grid_side_dim, elec_spacing ), (elec_side_dim, 1)), 1, elec_side_dim) # y coordinates of N_neu neurons [sqrt(N_elec),sqrt(N_elec)]

    # avoid distance calculation where (x_elec - x_neu = 0 as denominator
    if any(np.in1d(x_neu, x_elec)):
        x_elec = x_elec + 0.001 
        y_elec = y_elec + 0.001 

    x_neu_flat = np.squeeze(x_neu.reshape((neu_side_dim**2,1)))
    y_neu_flat = np.squeeze(y_neu.reshape((neu_side_dim**2,1)))
    x_elec_flat = np.squeeze(x_elec.reshape((elec_side_dim**2,1)))
    y_elec_flat = np.squeeze(y_elec.reshape((elec_side_dim**2,1)))

    return x_neu_flat,y_neu_flat,x_elec_flat,y_elec_flat,x_neu, y_neu, x_elec, y_elec

#==============================================================================================================
# Get the Linear Transform Matrix (L of the LNL LNL_model)
#==============================================================================================================
def get_linear_ERF_mapping(neu_side_dim, elec_side_dim, grid_side_dim, activ_spread,current_spread, FHWM):

    x_neu_flat, y_neu_flat, x_elec_flat, y_elec_flat,_,_,_,_= get_neu_and_elec_coord(neu_side_dim, elec_side_dim,grid_side_dim)
    #print(x_neu_flat.shape, x_elec_flat.shape, y_neu_flat.shape, y_elec_flat.shape)
    if FHWM:
        current_spread = (x_elec_flat[1] - x_elec_flat[0])/2.235

    W_d = np.zeros((neu_side_dim**2, elec_side_dim**2)) 
    for i in range(neu_side_dim**2):
        for j in range(elec_side_dim**2):
            xsi = np.random.uniform(-1,1)
            W_d[i,j] = (1 + activ_spread*xsi)*np.exp((-(x_neu_flat[i] - x_elec_flat[j])**2 - (y_neu_flat[i] - y_elec_flat[j])**2)/(2*current_spread))
            
    return W_d




    #==============================================================================================================
    # Initialising the randomised electrical stimulus (NL of the LNL LNL_model)
    #==============================================================================================================
    # #%%  Generate Random Stimulus patterns
    # Ne = elec_side_dim**2

    # nTrain = 50000 
    # nVal = 5000
    # nTest = 1000

    # train_stims = np.random.normal(0, 0.3, (nTrain, elec_side_dim,elec_side_dim))
    # val_stims = np.random.normal(0, 0.3, (nVal, elec_side_dim,elec_side_dim))
    # test_stims = np.random.normal(0, 0.3, (nTest, elec_side_dim,elec_side_dim))

    # single_stim = np.zeros((elec_side_dim, elec_side_dim))
    # single_stim[int(elec_side_dim/2), int(elec_side_dim/2)] = 1
    # test_stims[0, :,:] = single_stim
            
    # train_stims_v = train_stims.reshape(nTrain, Ne)
    # val_stims_v = val_stims.reshape(nVal, Ne)
    # test_stims_v = test_stims.reshape(nTest, Ne)

    # #==============================================================================================================
    # # Get the Output from the LNL LNL_model (NL of the LNL LNL_model)
    # #==============================================================================================================

    # Gs_ret_train = np.dot(W_d, train_stims_v.transpose())
    # real_activ_ret_Train = relu_nonlinearity(Gs_ret_train)

    # Gs_ret_val = np.dot(W_d, val_stims_v.transpose())
    # real_activ_ret_Val = relu_nonlinearity(Gs_ret_val)

    # Gs_ret_test = np.dot(W_d, test_stims_v.transpose())
    # real_activ_ret_Test = relu_nonlinearity(Gs_ret_test)  

    # # temporary generation of coordinates
    # _,_,_,_,x_neu, y_neu, x_elec, y_elec= get_neu_and_elec_coord(neu_side_dim, elec_side_dim,grid_side_dim)



def define_save_LNL_model(LNL_model_path,elec_side_dim,neu_side_dim, activ_spread, current_spread,FHWM):
    import torch
    import torch.nn as nn


    W_d = get_linear_ERF_mapping(neu_side_dim, elec_side_dim, grid_side_dim, activ_spread,current_spread, FHWM)

    class SingleLayerNetwork(nn.Module):
        def __init__(self, n_input, n_output):
            super(SingleLayerNetwork, self).__init__()
            # Define a single linear layer
            self.LNL = nn.Linear(n_input, n_output,bias=False)
            # Optionally add an activation function
            #self.activation = nn.ReLU()
            
        def forward(self, x):
            # Forward pass through the linear layer
            x = self.LNL(x)
            # Apply activation function
            #x = self.activation(x)
            return x

    # Model initialization
    n_input = elec_side_dim**2
    n_output = 256
    LNL_model = SingleLayerNetwork(n_input, n_output)


    # Hard coded weights
    with torch.no_grad():  # Disable gradient tracking
        # Set specific weights
        LNL_model.LNL.weight = nn.Parameter(torch.tensor(W_d, dtype=torch.float32))

    torch.save(LNL_model.LNL.state_dict(), LNL_model_path)
    if __name__ == "__main__":
        test_dot_np = np.zeros((elec_side_dim, elec_side_dim))
        test_dot_np[int(elec_side_dim/2), int(elec_side_dim/2)] = 1
        test_dot = torch.tensor(test_dot_np[:], dtype=torch.float32)
        test_dot_flat = test_dot.reshape(1,elec_side_dim**2)

        test_4dot_np = np.zeros((elec_side_dim, elec_side_dim))
        positions = [
        (elec_side_dim // 4, elec_side_dim // 4),
        (elec_side_dim // 4, 3 * elec_side_dim // 4),
        (3 * elec_side_dim // 4, elec_side_dim // 4),
        (3 * elec_side_dim // 4, 3 * elec_side_dim // 4)
        ]
        for (x, y) in positions:
            test_4dot_np[x, y] = 1.0
        test_4dot = torch.tensor(test_4dot_np[:], dtype=torch.float32)
        test_4dot_flat = test_4dot.reshape(1,elec_side_dim**2)
        with torch.no_grad():
            # Get the LNL_model's prediction
            test_4dot_output = LNL_model(test_4dot_flat)
            test_dot_output = LNL_model(test_dot_flat)
        plt.subplot(2, 2, 1)
        plt.title("Input Image")
        plt.imshow(test_dot, cmap='gray')

        plt.subplot(2, 2, 2)
        plt.title("Model Output")
        plt.imshow(test_dot_output.detach().numpy().reshape(neu_side_dim,neu_side_dim), cmap='gray')

        plt.subplot(2, 2, 3)
        plt.title("Input Image")
        plt.imshow(test_4dot_np, cmap='gray')

        plt.subplot(2, 2, 4)
        plt.title("Model Output")
        plt.imshow(test_4dot_output.detach().numpy().reshape(neu_side_dim,neu_side_dim), cmap='gray')

        plt.show()

    return W_d

if __name__ == "__main__":
    cwd = os.getcwd()
    LNL_model_path = cwd+'/data/LNL_model.pth'
    # whether to use curated resolvable gaussian radius of ERF
    FHWM=False
    # if not define current spread arbitrarily here
    current_spread = 1
    # define the electrode grid
    elec_side_dim = 8
    activ_spread= 0.1
    #define neuron grid
    neu_side_dim = 16

    # define the LNL layer with gaussian electrical receptive field
    define_save_LNL_model(LNL_model_path,elec_side_dim, neu_side_dim, activ_spread, current_spread,FHWM)


#########################################################################################################################
# Code block below is for training the neural network with inputs and outputs instead of hard coding the weights
#########################################################################################################################

# # Initialising data for training, divide data into batches
# train_stims_v_tensor = torch.tensor(train_stims_v, dtype=torch.float32)
# real_activ_ret_Train_tensor = torch.tensor(real_activ_ret_Train.transpose(), dtype=torch.float32)
# dataset = TensorDataset(train_stims_v_tensor, real_activ_ret_Train_tensor)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# # Loss Function
# loss_function = nn.MSELoss()

# # Optimizer
# optimizer = torch.optim.Adam(LNL_model.parameters(), lr=0.00005)

# # Number of epochs
# n_epochs = 20

# for epoch in range(n_epochs):
#     for inputs, targets in dataloader:
#         # Reset the gradients
#         optimizer.zero_grad()
        
#         # Generate predictions
#         outputs = LNL_model(inputs)
        
#         # Calculate loss
#         loss = loss_function(outputs, targets)
        
#         # Backpropagation
#         loss.backward()
        
#         # Update LNL_model parameters
#         optimizer.step()
#     print(f'Epoch {epoch+1}, Loss: {loss.item()}')

#########################################################################################################################
# Code block below is for graphical representation of results
#########################################################################################################################

# # Example input data (replace this with actual data)
# test_stims_v_tensor1 = torch.tensor(test_stims_v[0,:], dtype=torch.float32)
# test_stims_v_tensor2 = torch.tensor(test_stims_v[10,:], dtype=torch.float32)
# test_stims_v_tensor3 = torch.tensor(test_stims_v[100,:], dtype=torch.float32)


# # Disable gradient calculation for inference
# with torch.no_grad():
#     # Get the LNL_model's prediction
#     NN_output_test_stims1 = LNL_model(test_stims_v_tensor1)
#     NN_output_test_stims2 = LNL_model(test_stims_v_tensor2)
#     NN_output_test_stims3 = LNL_model(test_stims_v_tensor3)

# # Visualisation
# fig1 = plt.figure()
# plt.subplot(3,3,1)
# plt.imshow(test_stims[0, :,:], cmap = 'bwr', interpolation = 'none', clim=(-1, 1))
# plt.colorbar()
# plt.title('Original Stimulus')

# ax = fig1.add_subplot(3,3,2)
# plt.scatter( x_neu, y_neu, c = real_activ_ret_Test[:,0], alpha=0.5, edgecolors=None, cmap = 'gray'  ) 
# ax.set_aspect('equal')          
# plt.gca().invert_yaxis()
# plt.colorbar()    
# plt.axis('off')
# plt.title('Real Activation from LNL LNL_model') 

# ax = fig1.add_subplot(3,3,3)
# plt.scatter( x_neu, y_neu, c = NN_output_test_stims1, alpha=0.5, edgecolors=None, cmap = 'gray'  ) 
# ax.set_aspect('equal')          
# plt.gca().invert_yaxis()
# plt.colorbar()    
# plt.axis('off')
# plt.title('Activation Pattern from Neural Network') 

# plt.subplot(3,3,4)
# plt.imshow(test_stims[10,:,:], cmap = 'bwr', interpolation = 'none', clim=(-1, 1))
# plt.colorbar()

# ax = fig1.add_subplot(3,3,5)
# plt.scatter( x_neu, y_neu, c = real_activ_ret_Test[:,10], alpha=0.5, edgecolors=None, cmap = 'gray') 
# plt.colorbar()    
# plt.gca().invert_yaxis()
# ax.set_aspect('equal')  
# plt.axis('off')

# ax = fig1.add_subplot(3,3,6)
# plt.scatter( x_neu, y_neu, c = NN_output_test_stims2, alpha=0.5, edgecolors=None, cmap = 'gray') 
# plt.colorbar()    
# plt.gca().invert_yaxis()
# ax.set_aspect('equal')  
# plt.axis('off')



# plt.subplot(3,3,7)
# plt.imshow(test_stims[100, :,:], cmap = 'bwr', interpolation = 'none', clim=(-1, 1))
# plt.colorbar()

# ax = fig1.add_subplot(3,3,8)
# plt.scatter( x_neu, y_neu, c = real_activ_ret_Test[:,100], alpha=0.5, edgecolors=None, cmap = 'gray' )     
# plt.gca().invert_yaxis()
# plt.colorbar()
# ax.set_aspect('equal')  
# plt.axis('off')

# ax = fig1.add_subplot(3,3,9)
# plt.scatter( x_neu, y_neu, c = NN_output_test_stims3, alpha=0.5, edgecolors=None, cmap = 'gray' )     
# plt.gca().invert_yaxis()
# plt.colorbar()
# ax.set_aspect('equal')  
# plt.axis('off')
# plt.show(block=False)

#   # Keep the plot open
# while plt.fignum_exists(1):
#     plt.pause(0.1)
#     time.sleep(0.1)  