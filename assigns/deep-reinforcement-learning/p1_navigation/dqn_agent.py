
import random
from collections import namedtuple, deque

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from models import DQModel

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
UPDATE_EVERY = 4        # how often to update the network
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQNAgent:
  """
  RL agent based on DQN
  """
  def __init__(self, state_size, action_size, seed):
    """
    """
    self.qnet_local = DQModel(state_size, action_size, seed).to(device)
    self.qnet_target = DQModel(state_size, action_size, seed).to(device)
    self.optim = optim.Adam(self.qnet_local.parameters(), lr=LR)
    self.seed = random.seed(seed)
    self.state_size, self.action_size = state_size, action_size
    self.mem_buffer = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
    self.t_step = 0
    self.action_choices = np.arange(self.action_size)

  def step(self, state, action, reward, next_state, done):
    """
    Agent's response to a step
    """
    self.mem_buffer.add(state, action, reward, next_state, done)
    self.t_step += 1
    self.t_step %= UPDATE_EVERY

    if self.t_step == 0:
      if len(self.mem_buffer) >= BATCH_SIZE:
        self.learn(GAMMA)

  def act(self, state, eps=0):
    """
    AI agent action corresponding to a state
    """
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    self.qnet_local.eval()
    with torch.no_grad():
      resp = self.qnet_local.forward(state)
    self.qnet_local.train()

    if random.random() > eps:
      return np.argmax(resp.cpu().data.numpy())
    else:
      return random.choice(self.action_choices)
  
  def learn(self, gamma):
    """
    Updates model weights from the RelayBuffer experiences
    """
    experiences = self.mem_buffer.sample()
    states, actions, rewards, next_states, dones = experiences

    # Get Q values of next state's
    Q_target_next = self.qnet_target.forward(next_states).detach().max(dim=1)[0].unsqueeze(1)
    # update current Q from current rewards & next state Qvalue
    Q_target = rewards + gamma * (Q_target_next * (1 - dones))

    # compute expected Q value form local network wrt actions
    Q_expected = self.qnet_local.forward(states).gather(1, actions)

    loss = F.mse_loss(Q_expected, Q_target)
    self.optim.zero_grad()
    loss.backward()
    self.optim.step()
   	# update target network weights with tau factor
    self.soft_update(TAU)

  def soft_update(self, tau):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target
    
    Params
    ======
        tau (float): interpolation parameter
    """
    for target_param, local_param in zip(self.qnet_target.parameters(), self.qnet_local.parameters()):
      target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
  """Fixed-size buffer to store experience tuples."""
  def __init__(self, action_size, buffer_size, batch_size, seed):
    """Initialize a ReplayBuffer object.

    Params
    ======
        action_size (int): dimension of each action
        buffer_size (int): maximum size of buffer
        batch_size (int): size of each training batch
        seed (int): random seed
    """
    self.action_size = action_size
    self.memory = deque(maxlen=buffer_size)
    self.batch_size = batch_size
    self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
    self.seed = random.seed(seed)

  def add(self, state, action, reward, next_state, done):
    """Add a new experience to memory."""
    e = self.experience(state, action, reward, next_state, done)
    self.memory.append(e)

  def sample(self):
    """Randomly sample a batch of experiences from memory."""
    experiences = random.sample(self.memory, k=self.batch_size)

    states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
    actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
    rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
    next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
    dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

    return (states, actions, rewards, next_states, dones)

  def __len__(self):
    """Return the current size of internal memory."""
    return len(self.memory)
