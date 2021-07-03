"""
Define DNN Models for Actor & Critic
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    """
    Initialize weights 
    """
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class ActorModel(nn.Module):
    
    def __init__(self, state_size, action_size, seed=10, fsize1=400, fsize2=300):
        super(ActorModel, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fsize1)
        self.bn1 = nn.BatchNorm1d(fsize1)
        self.fc2 = nn.Linear(fsize1, fsize2)
        self.fc3 = nn.Linear(fsize2, action_size)
        self.init_parameters()

    def init_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(self.bn1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class CriticModel(nn.Module):
    
    def __init__(self, state_size, action_size, seed=10, fsize1=400, fsize2=300):
        super(CriticModel, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fsize1)
        self.bn1 = nn.BatchNorm1d(fsize1)
        self.fc2 = nn.Linear(fsize1 + action_size, fsize2)
        self.fc3 = nn.Linear(fsize2, action_size)
    
    def init_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, x, action):
        
        x = self.fc1(x)
        xs = F.relu(self.bn1(x))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)