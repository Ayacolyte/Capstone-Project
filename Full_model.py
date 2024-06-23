import torch
import torch.nn as nn
from LNL import LNL_model  # Import LNL_model from LNL script

from Natural_Images import cifar10_train_np,cifar10_test_np

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.model_a = LNL_model()  # Use ModelA as a layer
        self.translate = nn.Linear(1024, 64, bias=False)
        self.recons = nn.Linear(256, 1024, bias=False)

    def forward(self, x):
        x = self.model_a(x)  # Forward pass through ModelA
        x = self.fc2(x)  # Forward pass through the new layer
        return x

