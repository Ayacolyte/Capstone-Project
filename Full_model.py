import torch
import torch.nn as nn
import pickle
import os
import time
# import required data from preceding scripts
#from LNL import LNL_model_path,elec_side_dim  # Import LNL_model from LNL script
#from Natural_Images import train_loader,cifar100_train_tsr_flat,cifar100_test_tsr_flat
#import matplotlib.pyplot as plt

#drop_rate = 0.2
# Define Full model, in which the weights from LNL model is inherited
class DoubleSigmoid(nn.Module):
    def __init__(self, shift, magnitude):
        super(DoubleSigmoid, self).__init__()
        self.shift = shift
        self.magnitude = magnitude

    def forward(self, x):
        sigmoid1 = torch.sigmoid(-1*self.magnitude * (x + self.shift))
        sigmoid2 = torch.sigmoid(self.magnitude * (x - self.shift))
        return sigmoid1 + sigmoid2
    
def define_model(elec_side_dim,neu_side_dim, LNL_model_path, drop_rate,af_array,shift,magnitude,noise):
    import numpy as np
    class AutoEncoder(nn.Module):
        def __init__(self):
            super(AutoEncoder, self).__init__()   
            self.layer1 = nn.Linear(1024, elec_side_dim**2, bias=True)
            self.LNL_model = nn.Linear(elec_side_dim**2, neu_side_dim**2, bias=False)
            self.layer3 = nn.Linear(neu_side_dim**2, 1024, bias=True)
            self.relu = nn.ReLU()
            self.LNL_model.load_state_dict(torch.load(LNL_model_path))
            self.dropout = nn.Dropout(p=drop_rate)
            self.double_sigmoid = DoubleSigmoid(shift, magnitude)
            self.noise_model1 = noise*torch.tensor(np.random.uniform(-1, 1, self.LNL_model.weight.shape),dtype=torch.float)
            self.noise_model2 = noise*torch.tensor(np.random.uniform(-1, 1, self.LNL_model.weight.shape),dtype=torch.float)
            
        def forward(self, x):
            def assert_activ(af, x):
                if af == "2sig":
                    x = self.double_sigmoid(x)
                elif af == "sig":
                    x = torch.sigmoid(magnitude * (x - shift))
                elif af == "linear":
                    x = x
                else:
                    x = self.relu(x)
                return x

            x = self.layer1(x) 
            x = assert_activ(af_array[0], x)
            lyr1 = x
            if self.training:
                noise_weight_LNL = self.LNL_model.weight + self.noise_model1
                #print(noise_weight_LNL.shape)
                #print(self.LNL_model.bias)
                x = nn.functional.linear(x, noise_weight_LNL, self.LNL_model.bias)
            else:
                noise_weight_LNL = self.LNL_model.weight + self.noise_model2
                x = nn.functional.linear(x, noise_weight_LNL, self.LNL_model.bias)
            #print(self.LNL_model.weight)
            #x = self.LNL_model(x) 
            x = assert_activ(af_array[1], x)
            additive_noise = torch.tensor(np.random.uniform(-1, 1, x.shape),dtype=torch.float)
            x = x + noise*additive_noise
            lyr2 = x
            x = self.dropout(x)  # Apply dropout after hidden layer
            x = self.layer3(x)
            x = assert_activ(af_array[2], x)

            return x,lyr1, lyr2

    return AutoEncoder
# # define a custom double sigmoid activation function using pytorch built in sigmoid for fast gradient computation
# class DoubleSigmoid(nn.Module):
#     def __init__(self, alpha1=1.0, beta1=0.1, alpha2=-1.0, beta2=0.1):
#         super(DoubleSigmoid, self).__init__()
#         self.sigmoid = nn.Sigmoid()
#         self.alpha1 = alpha1
#         self.beta1 = beta1
#         self.alpha2 = alpha2
#         self.beta2 = beta2

#     def forward(self, x):
#         sigmoid1 = self.sigmoid(self.alpha1 * x + self.beta1)
#         sigmoid2 = self.sigmoid(self.alpha2 * x + self.beta2)
#         return sigmoid1 + sigmoid2
    
def train_and_save(n_epochs,AutoEncoder,model_title,mult_lr = True):
    from Natural_Images import train_loader,cifar100_train_tsr_flat,cifar100_test_tsr_flat
    # # Number of epochs
    if mult_lr:
        learning_rates = [0.01, 0.001, 0.0001, 0.00001]
    else:
        learning_rates = [0.0001]
    train_err = torch.zeros(n_epochs + 1,len(learning_rates))
    val_err = torch.zeros(n_epochs + 1,len(learning_rates))
    Autoencoders = []

    for i, learning_rate in enumerate(learning_rates):
            # Define Model
        basic_autoencoder = AutoEncoder()
         
        # Define a loss function
        criterion = nn.MSELoss()  # mean squared loss, eucledian

        # Define an optimizer, specifying only the parameters of fc2 and fc3
        optimizer = torch.optim.Adam([
            {'params': basic_autoencoder.layer1.parameters()},
            {'params': basic_autoencoder.layer3.parameters()}
        ], lr=learning_rate)
        for epoch in range(n_epochs):  # Number of epochs
            basic_autoencoder.train() 
            for input, _ in train_loader:
                
                # flatten
                input = input.view(input.size(0), -1)
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs,_, _ = basic_autoencoder(input)

                # Compute loss
                loss = criterion(outputs, input)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

            # Print loss
            if (epoch) % 10 == 0:  # Print every 10 epochs
                print(f'Epoch [{epoch}/{n_epochs}], Loss: {loss.item():.4f}')
            
            with torch.no_grad():
                # training data
                output_train,_,_ = basic_autoencoder(cifar100_train_tsr_flat)
                train_loss = criterion(output_train, cifar100_train_tsr_flat)
                train_err[epoch + 1,i] = train_loss.item()  

                basic_autoencoder.eval()  # Set model to evaluation mode     
                # validation data
                output_test,_,_ = basic_autoencoder(cifar100_test_tsr_flat)
                val_loss = criterion(output_test, cifar100_test_tsr_flat)
                val_err[epoch + 1,i] = val_loss.item()
        Autoencoders.append(basic_autoencoder)

    train_err[0], val_err [0] = cifar100_train_tsr_flat.mean(),cifar100_test_tsr_flat.mean()
    # save the result in the data folder
    cwd = os.getcwd()
    data_dir = cwd + '/data'

    # Create the directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    file_path = os.path.join(data_dir, f'NN_{model_title}_output.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump((train_err, val_err), f)
    
    labels = ['lr = 0.01', 'lr = 0.001', 'lr = 0.0001', 'lr = 0.00001']
    for idx, model in enumerate(Autoencoders):
        if len(Autoencoders) == 1:
            model_path = cwd+f'/data/model_{model_title}_lr = 0.0001.pth'
        else:
            model_path = cwd+f'/data/model_{model_title}_{labels[idx]}.pth'
        torch.save(model.state_dict(), model_path)


# labels = ['lr=0.1', 'lr = 0.01', 'lr = 0.001', 'lr = 0.0001']
# # Create the bar plot
# fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# # Plot the first vector
# axs[0].bar(labels, train_err, color='blue')
# axs[0].set_title('Training Error Visualization')
# axs[0].set_xlabel('Learning Rates')
# axs[0].set_ylabel('Error')

# # Plot the second vector
# axs[1].bar(labels, val_err, color='green')
# axs[1].set_title('Validation Error Visualization')
# axs[1].set_xlabel('Learning Rates')
# axs[1].set_ylabel('Error')

# # Adjust layout
# plt.tight_layout()

# # Show the plot and keep the plot open
# plt.show(block=False)
# while plt.fignum_exists(1):
#     plt.pause(0.1)
#     time.sleep(0.1)  