"""
Experiments with pong env with REINFORCE algorithm
improvements
"""

import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import progressbar as pb

from parallelEnv import parallelEnv
import pong_utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):
  """
  CNN model to represent the Policy
  """
  def __init__(self):
    super(Policy, self).__init__()
    # 80x80 to outputsize x outputsize
    # outputsize = (inputsize - kernel_size + stride)/stride
    # (round up if not an integer)

    # output = 4x40x40 here
    self.conv1 = nn.Conv2d(2, 4, kernel_size=2, stride=2)
    # output = 2x20x20
    self.conv2 = nn.Conv2d(4, 2, kernel_size=2, stride=2)
    self.pool1 = nn.MaxPool2d(2, 2)
    self.size = 2*10*10

    # 1 fully connected layer
    self.fc1 = nn.Linear(self.size, 50)
    self.fc2 = nn.Linear(50, 25)
    self.fc3 = nn.Linear(25, 1)
    self.sig = nn.Sigmoid()

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = self.pool1(x)
    # flatten the tensor
    x = x.view(-1,self.size)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    return self.sig(self.fc3(x))


def surrogate(policy, old_probs, states, actions, rewards,
            discount = 0.995, beta=0.01):

  ########
  ##
  ## WRITE YOUR OWN CODE HERE
  ##
  ########

  actions = torch.tensor(actions, dtype=torch.int8, device=device)
  dis_rewards = np.zeros((len(actions), actions[0].shape[0]))
  for n in range(actions.shape[0]):
    rwds = []
    for i in range(rewards[0].shape[0]):
      discounts = np.asarray([discount ** j for j in range(i, len(rewards[0]))])
      rwds.append(np.dot(rewards[n][i:], discounts))
    dis_rewards[n, :] = np.asarray(rwds)

  # mean across single trajectory
  mean = np.mean(dis_rewards, axis=1)
  std = np.std(dis_rewards, axis=1) + 1e-10
  dis_rewards = (dis_rewards - mean[:, np.newaxis]) / std[:, np.newaxis]

  dis_rewards = torch.tensor(dis_rewards, dtype=torch.float32, device=device)
  new_probs = pong_utils.states_to_prob(policy, states)
  new_probs = torch.where(actions == pong_utils.RIGHT, new_probs, 1.0-new_probs)

  old_probs = torch.tensor(old_probs, device=device)
  with torch.no_grad():
    ratio = old_probs / new_probs

  loss = dis_rewards * ratio
  # include a regularization term
  # this steers new_policy towards 0.5
  # which prevents policy to become exactly 0 or 1
  # this helps with exploration
  # add in 1.e-10 to avoid log(0) which gives nan
  entropy = -(new_probs*torch.log(old_probs+1.e-10)+ \
      (1.0-new_probs)*torch.log(1.0-old_probs+1.e-10))

  return torch.mean(beta*entropy + loss)


policy = Policy().to(device)
envs = pong_utils.parallelEnv('PongDeterministic-v4', n=4, seed=12345)
prob, state, action, reward = pong_utils.collect_trajectories(envs, policy, tmax=100)
Lsur = surrogate(policy, prob, state, action, reward)

episode = 500

# widget bar to display progress
widget = ['training loop: ', pb.Percentage(), ' ',
          pb.Bar(), ' ', pb.ETA() ]
timer = pb.ProgressBar(widgets=widget, maxval=episode).start()

# initialize environment
envs = parallelEnv('PongDeterministic-v4', n=8, seed=1234)

discount_rate = .99
beta = .01
tmax = 320

# keep track of progress
mean_rewards = []
optimizer = optim.Adam(policy.parameters(), lr=1e-4)

for e in range(episode):
  # collect trajectories
  old_probs, states, actions, rewards = \
      pong_utils.collect_trajectories(envs, policy, tmax=tmax)

  total_rewards = np.sum(rewards, axis=0)

  # this is the SOLUTION!
  # use your own surrogate function
  L = -surrogate(policy, old_probs, states, actions, rewards, beta=beta)

  #L = -pong_utils.surrogate(policy, old_probs, states, actions, rewards, beta=beta)
  optimizer.zero_grad()
  L.backward()
  optimizer.step()
  del L

  # the regulation term also reduces
  # this reduces exploration in later runs
  beta*=.995

  # get the average reward of the parallel environments
  mean_rewards.append(np.mean(total_rewards))

  # display some progress every 20 iterations
  if (e + 1) % 20 == 0:
    print("Episode: {0:d}, score: {1:f}".format(e + 1, np.mean(total_rewards)))
    print(total_rewards)

  # update progress widget bar
  timer.update(e + 1)
timer.finish()

torch.save(policy, 'REINFORCE.policy')

