"""
Uses DDQG agent for solving the environment. Code reused from Assignment project
"""

import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim


from models import ActorModel, CriticModel

#
BUFFER_SIZE = int(1e6)  # Relay buffer maximum size
BATCH_SIZE = 128        # batch used for training
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_RATE = 1e-3
WEIGHT_DECAY = 1e-6     # L2 weight decay
LEARN_EVERY = 20        # interval for learning
LEARN_NUM = 10          # passes
EPSILON = 1.0           # epislon for exploration rate
EPSILON_DECAY = 1e-6    # decay rate for epsilon

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """
    Interacts with and learns from the environment
    Refer: https://towardsdatascience.com/three-aspects-of-deep-rl-noise-overestimation-and-exploration-122ffb4bb92b
    """

    def __init__(self, state_size, action_size, seed):
        """
        @param: state_size [int]: state space dim
        @param: action_size [int]: action space dim
        @param: seed[int]: seed used for random. For recreating experiments
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.epsilon = EPSILON

        #Actor network for local and target
        self.actor_local = ActorModel(state_size, action_size, seed).to(device)
        self.actor_target = ActorModel(state_size, action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_RATE)

        # Critic Network for Local and Target
        self.critic_local = CriticModel(state_size, action_size, seed).to(device)
        self.critic_target = CriticModel(state_size, action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_RATE, weight_decay=WEIGHT_DECAY)

        # add noise to make it robust
        self.ou_noise = OUNoiseProcess(action_size, seed)

        # Use Relay memory for storing step actions for batch training
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)

    def step(self, state, action, reward, next_state, done, timestep):
        """
        Save state transitions into relay buffer and use it for learning the model
        with specified batch number of batch numbers
        @param: state: current state 
        @param: action: action taken in this state
        @param: reward: reward received
        @param: next_state: state output based on action
        @param: done: flag to check if termination condition is reached
        @param: timestep: step number (used to print output)
        """

        # add step into relay memory for learning
        self.memory.add(state, action, reward, next_state, done)

        # Learn at defined interval, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE and timestep % LEARN_EVERY == 0:
            for _ in range(LEARN_NUM):
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """
        Use DNN model to take action on this state
        @param: state: current state
        @param: add_noise: flag to add noise to the action output
        """
        state = torch.from_numpy(state).float().to(device)
        # set model to eval mode
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.epsilon * self.ou_noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.ou_noise.reset()

    def learn(self, experiences, gamma):
        """
        Uses Relay buffer samples to train the networks
        1. Target & Local Networks of Actor & Critic are learned
        2. Uses soft-update, to avoid fluctuations, for target update from local
        """
        states, actions, rewards, next_states, dones = experiences

        # Critic section
        # Get actions from actor model
        actions_next = self.actor_target(next_states)
        #compute Q values from actor's output
        qtargets_next = self.critic_target(next_states, actions_next)
        
        # target outputs from next target 
        qtargets = rewards + (gamma * qtargets_next * (1 - dones))
        
        qexpected = self.critic_local(states, actions)
        #Compute losses
        critic_loss = F.mse_loss(qexpected, qtargets)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Clip gradients to improve learning
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # Actor Sections
        actions_pred = self.actor_local(states)
        # Take -ve loss to maximize the output (Q-Values)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # update decay
        self.epsilon -= EPSILON_DECAY
        # resets noise after learning step
        self.ou_noise.reset()
        
        # Use 
        self.apply_soft_update_weights(self.critic_local, self.critic_target, TAU)
        self.apply_soft_update_weights(self.actor_local, self.actor_target, TAU)

    def apply_soft_update_weights(self, lmodel, tmodel, tau):
        """
        Use softupdate as per DDQN paper. This helps to avoid fluctuations due to
        parameter changes.
        """
        for tparam, lparam in zip(tmodel.parameters(), lmodel.parameters()):
            tparam.data.copy_(tau * lparam.data + (1.0 - tau) * tparam.data)


class OUNoiseProcess:

    def __init__(self, size, seed, mu=0., theta=0.2, sigma=0.15):
        """
        Use OU Noise as per Continous Control paper
        Refer: https://arxiv.org/pdf/1509.02971.pdf
        """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.cur_mu = None
        self.reset()

    def reset(self):
        """
        Resets internal state
        """
        self.cur_mu = copy.copy(self.mu)

    def sample(self):
        """
        Sample noise from process
        """
        if self.cur_mu is None:
            self.reset()
        noise_x = self.theta * (self.mu - self.cur_mu)
        noise_x + self.sigma * np.array([random.random() for i in range(len(self.cur_mu))])
        self.cur_mu += noise_x
        return self.cur_mu


class ReplayBuffer:
    """
    Copied from Assignment 1.
    Fixed-size buffer to store experience tuples
    """

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
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
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)