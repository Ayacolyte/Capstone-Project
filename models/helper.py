import torch.nn as nn
import torch

class DoubleSigmoid(nn.Module):
    def __init__(self, shift, magnitude):
        super(DoubleSigmoid, self).__init__()
        self.shift = shift
        self.magnitude = magnitude

    def forward(self, x):
        sigmoid1 = torch.sigmoid(-1*self.magnitude * (x + self.shift))
        sigmoid2 = torch.sigmoid(self.magnitude * (x - self.shift))
        return sigmoid1 + sigmoid2
    