
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQModel(nn.Module):
  """
  Network architecture for AI agent
  """
  def __init__(self, state_size, action_size, seed):
    """
    Intialize the network with passed size infommation
    """
    super(DQModel, self).__init__()
    self.seed = torch.manual_seed(seed)
    self.fc1 = nn.Linear(state_size, 128)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, action_size)

  def forward(self, x):
    """
    Computes forward pass of the network
    """
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    return self.fc3(x)
