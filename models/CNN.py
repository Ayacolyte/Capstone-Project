import torch.nn as nn
import torch
import numpy as np
from .helper import *

class AutoEncoder_CNN(nn.Module):
    def __init__(self,device,elec_side_dim,neu_side_dim, LNL_model_path, drop_rate, af_array,shift,magnitude,noise,kernal_size):
        super(AutoEncoder_CNN, self).__init__()   
        # define layers
        self.layer1 = nn.Linear(1024, elec_side_dim**2, bias=True)
        self.LNL_model = nn.Linear(elec_side_dim**2, neu_side_dim**2, bias=False)
        self.layer3 = nn.Linear(neu_side_dim**2, 1024, bias=True)
        
        self.LNL_model.load_state_dict(torch.load(LNL_model_path,weights_only=True))
        # activations
        self.activations = {
        "2sig": DoubleSigmoid(shift,magnitude),
        "sig": lambda x: torch.sigmoid(magnitude * (x - shift)),
        "linear": lambda x: x,
        "relu": nn.ReLU()
        }
        # regularisation
        self.dropout = nn.Dropout(p=drop_rate)
        self.noise_model1 = noise*torch.tensor(np.random.uniform(-1, 1, self.LNL_model.weight.shape),dtype=torch.float).to(device)
        self.noise_model2 = noise*torch.tensor(np.random.uniform(-1, 1, self.LNL_model.weight.shape),dtype=torch.float).to(device)
        self.layer1 = nn.Sequential(
        nn.Conv2d(                     
            in_channels=1,
            out_channels=1,  
            kernel_size=kernal_size,         
            stride=2,                    
            padding=7       
        ),
        nn.MaxPool2d(kernel_size=2)     
        )
        #self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=9, stride=2, padding=3,bias=True)
        #self.layer1 = nn.Linear(1024, elec_side_dim**2, bias=True)
        self.LNL_model = nn.Linear(elec_side_dim**2, neu_side_dim**2, bias=False)
        self.layer3 = nn.Linear(neu_side_dim**2, 1024, bias=True)
        self.LNL_model.load_state_dict(torch.load(LNL_model_path))
        self.dropout = nn.Dropout(p=drop_rate)
    def forward(self, x):
        x_2d = x.view(x.shape[0],1,32,32)
        x_2d = self.layer1(x_2d) 
        x = x_2d.view(x_2d.shape[0],-1)
        #x = self.layer1(x) 
        x = self.activations[self.af_array[0]](x)
        lyr1 = x
        x = self.LNL_model(x)  
        x = self.activations[self.af_array[1]](x)
        additive_noise = torch.tensor(np.random.uniform(-1, 1, x.shape),dtype=torch.float)
        x = x + 0.2*additive_noise
        lyr2 = x
        x = self.dropout(x)  # Apply dropout after hidden layer
        x = self.layer3(x)
        x = self.activations[self.af_array[2]](x)

        return x,lyr1, lyr2