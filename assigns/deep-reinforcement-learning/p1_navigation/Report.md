

Algorithm details:
1. The agent uses Deep Q-Network (DQN) using Fully connected layers for computing the Q value.
2. As per DQN,
   1. A Nueral Network is used for computing optimal Q-value computation for the state and the action is computed from greedy-epsilon policy.
   2. A RelayBuffer is used to store the SARSA pair for each step of the agent.
   3. The Nueral network is trained with the samples from the RelayBuffer which are taken with uniform distribution from it.
   4. DQN has two NN with same architecture, Local & Target networks
   5. Local network is used to compute the Q-value for any given state whereas the target network is used to compute the Q value which is used to calculate the error.
   6. The parameter of target network is updated from local network at a fraction rate in order to avoid fluctuation due to high error in a single transition.
   7. The paramter of target network is updated at every n (4 in this case) steps.

Score vs episodes:
  

Future ideas:
1. Use Prioritized Experienced Relay - instead of picking samples with uniform distribution from the Relay buffer, we could pick them with probabilities computed from the error values.
   1. This makes the system to learn quickly as agent can learn from high error transition.Large error -> more the system can learn from that step.
2. Other ideas like Dueling Networks, Double DQN, etc.. can be used.
3. Use Raw pixel values rather than state space from the environment.
