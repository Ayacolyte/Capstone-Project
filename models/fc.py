import torch.nn as nn
import torch
import numpy as np
from .helper import DoubleSigmoid


class AutoEncoder_fc(nn.Module):
    def __init__(self,device,elec_side_dim,neu_side_dim, LNL_model_path, drop_rate,af_array,shift,magnitude,noise):
        super(AutoEncoder_fc, self).__init__() 
        self.sig_magnitude = magnitude
        self.sig_shift = shift
        self.af_array = af_array
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
        "ReLU": nn.ReLU()
        }
        # regularisation
        self.dropout = nn.Dropout(p=drop_rate)
        self.noise_model1 = noise*torch.tensor(np.random.uniform(-1, 1, self.LNL_model.weight.shape),dtype=torch.float).to(device)
        self.noise_model2 = noise*torch.tensor(np.random.uniform(-1, 1, self.LNL_model.weight.shape),dtype=torch.float).to(device)
            
    def forward(self, x):
        #print(f"x after input is on: {x.device}")
        x = self.layer1(x) 
        #print(f"x after layer1 is on: {x.device}")
        x = self.activations[self.af_array[0]](x)
        #print(f"x after actv1 is on: {x.device}")
        lyr1 = x
        if self.training:
            noise_weight_LNL = self.LNL_model.weight + self.noise_model1
            #print(f"noise: {noise_weight_LNL.device}")
            x = nn.functional.linear(x, noise_weight_LNL, self.LNL_model.bias)
            #print(f"x after noise1 is on: {x.device}")
        else:
            noise_weight_LNL = self.LNL_model.weight + self.noise_model2
            x = nn.functional.linear(x, noise_weight_LNL, self.LNL_model.bias)
        #print(self.LNL_model.weight)
        #x = self.LNL_model(x) 
        x = self.activations[self.af_array[1]](x)
        #print(f"x after actv2 is on: {x.device}")
        additive_noise = torch.rand_like(x)
        x = x + 0.25*additive_noise
        #print(f"x after addnoise is on: {x.device}")
        lyr2 = x
        x = self.dropout(x)  # Apply dropout after hidden layer
        x = self.layer3(x)
        #print(f"x after layer3 is on: {x.device}")
        x = self.activations[self.af_array[2]](x)

        return x,lyr1, lyr2