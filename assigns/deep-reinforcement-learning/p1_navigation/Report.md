

**Algorithm details:**
1. The agent uses Deep Q-Network (DQN) using Fully connected layers for computing the Q value.
2. As per DQN,
   1. A Nueral Network is used for computing optimal Q-value for the state and the action is computed from epislon-greed policy.
   2. The hyperparemeter epsilon is reduced over the episodes so that initially more weightage is given for exploration and later to the knowledge. As per literature recommendations, the epsilon is reduced linearly over the episode.
   3. A RelayBuffer is used to store the SARSA pair for each step of the agent.
   4. The Nueral network is trained with the samples from the RelayBuffer which are taken with uniform distribution from it. This kind of disregards the temporal relation between consecutives state transitions.
   5. DQN has two NN with same architecture, Local & Target networks.
   6. Local network is used to compute the Q-value for any given state whereas the target network is used to compute the Q value which is used to calculate the error.
   7. The parameter of target network is updated from local network at a fraction rate.
   8. The parameter of target network is updated at every n (4 in this case) steps.
   9. Steps 6 & 7, make sures it avoids fluctuation due to high error in a single SARSA transition.

**Score vs episodes:**
1. Plot of score against episodes during training is as follows and we could see the model converged at 400 episodes for required score of +13 for each episode. X axis is and Y axis is score.
 
 <img src="scores_vs_episodes.jpeg" />

**Future ideas:**
1. Use Prioritized Experienced Relay - instead of picking samples with uniform distribution from the Relay buffer, we could pick them with probabilities computed from the error values.
   1. This makes the system to learn quickly as agent can learn from high error transition.Large error -> more the system can learn from that step.
2. Other ideas like Dueling Networks, Double DQN, etc.. can be used.
3. Use Raw pixel values rather than state space from the environment.
