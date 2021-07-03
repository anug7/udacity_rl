Algorithm details:
1. The agent uses Multi-Agent Deep Deterministic Policy gradient methold which uses Deep Q-Network (DQN) using Fully connected layers for both Actor and Critic.
2. As per DDPG, A Nueral Network is used for computing optimal Q-value for the state (state-value function approximator) called Actor and the action is
    computed from an another NN model for computing the action value called Critic.
3. The hyperparemeter epsilon is reduced over the episodes so that initially more weightage is given for exploration and later to the knowledge. As per literature recommendations,
   the epsilon is reduced linearly over the episode.
4. A RelayBuffer is used to store the SARSA pair for each step of the agent - for both Actor and Critic Models. And this buffer is shared between the agents used in the training.
5. Actor & Critic networks are trained with the samples from the RelayBuffer which are taken with uniform distribution from it. This kind of disregards the temporal relation between consecutives state transitions.
6. Each of the Actor & Critic models has two NN with same architecture, Local & Target networks. The architecture of the Actor Model is as follows,
    * The Neural network has three Fully connected layers.
      * Layer 1: Size: 24 * 128 i.e state_space_size * hidden_layer1_size
      * Layer 2: Size: 128 * 128 i.e hidden_layer1_size * hidden_layer2_size
      * Layer 3: Size: 128 * 2 i.e hidden_layer2_size * action_space_size
      * The first two layers have ReLU activation unit
      * While the last layer has tanh activation to given value within range -1 & 1
      * The number of nodes in the hidden layers and no of hidden layers are chosen based on experiments.
    * Network architecture of Critic model
      * Layer 1: Size: 24 * 128 i.e state_space_size * hidden_layer1_size
      * Layer 2: Size: 132 * 128 i.e hidden_layer1_size + action_space_size * hidden_layer2_size
      * Layer 3: Size: 128 * 2 i.e hidden_layer2_size * action_space_size
      * The first two layers have ReLU activation unit
      * While the last layer has linear activation to give direct values.
      * The number of nodes in the hidden layers and no of hidden layers are chosen based on experiments.
7. The error between the local and target networks is used to train the network.
8. The parameter of target network is updated from local network at a fraction rate.
9. The parameter of target network is updated at every n (4 in this case) steps.
10. Steps 8 & 9, make sures it avoids fluctuation due to high error in a single SARSA transition.
11. As per the Continuous Control paper,  Ornstein–Uhlenbeck process is added to action for exploration. This helps in random exploration of the agents
12. Some of the hyper parameter used in training are as follows,
    * BUFFER_SIZE - 1e5 - size of the relay buffer(mentioned above) to store the SARSA pairs - chosen based on experiments.
    * BATCH_SIZE - 256 - size of experiences from relay buffers used to train the networks at any time - chosen based on literature.
    * GAMMA - 0.99 - the discount factor used in Expected return calculation - standard value of 0.99 is used giving more weights to future rewards.
    * TAU - 1e-3 - factor is used to soft update the target network weights from local network (described above) - as per the DQN paper.
    * LR - 1e-3 - learning rate used in Nueral network weight updates - chosen from experiments
    * EPSILON - 1.0 - exploration noise (OU Noise) added for the model to train
    * EPSILON_DECAY - 1e-6 - decay factor over time
 13. Many of the parameter above have been chosen from the previous assignment.

Training:
  1. The Environment contains two agents getting trained in parallel.
  2. The Relay buffer used is shared by both the agent for faster learning.
  3. The max of the scores from two agents is considered as the score for each episode for which value should be >= 0.5 for a successful termination of training.
  4. The final weight file is located [here](https://github.com/anug7/udacity_rl/blob/master/assigns/deep-reinforcement-learning/p3_collab-compet/solved_weights.pth).

Score vs episodes:
  1. The evironment is considered solved if the average score is +0.5 for any of the two agents over the last 100 iterations
  2. Plot of score against episodes during training is shown below.
  3. If we look at the plot, we can use that the score is over 0.5 i.e average is 0.5, from episodes 1740 to 1840, hence we can consider the environment is solved at episode 1740 i.e the scores after episode 1740, are over 0.5. Or solved at episode 1840 if we include the episodes which are used to compute the average values
 
 ![Plot](https://github.com/anug7/udacity_rl/blob/master/assigns/deep-reinforcement-learning/p3_collab-compet/scorevsepisodes.png)

Future Ideas:
  1. Use Prioritized Experienced Relay - instead of picking samples with uniform distribution from the Relay buffer, we could pick them with probabilities computed from the error values.
     * This makes the system to learn quickly as agent can learn from high error transition.Large error -> more the system can learn from that step.
  2. Explore other methods such as A3C, PPO, etc..
  3. In MDAPPG, to explore ideas like
     * Shared Actor, Shared Critic Or both
     * Stabilization methods to reduce noice on the score during training
