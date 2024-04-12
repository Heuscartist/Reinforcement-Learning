# Cliff Walking Problem

This problem is part of the open ai gymnasium environments. To learn more go to https://gymnasium.farama.org/
## Description
The game starts with the player at location [3, 0] of the 4x12 grid world with the goal located at [3, 11]. If the player reaches the goal the episode ends. A cliff runs along [3, 1..10]. If the player moves to a cliff location it returns to the start location. The player makes moves until they reach the goal.

## Action Space
The action shape is `(1,)` in the range `{0, 3}` indicating which direction to move the player.
- 0: Move up
- 1: Move right
- 2: Move down
- 3: Move left

## Observation Space
There are 3 x 12 + 1 possible states. The player cannot be at the cliff, nor at the goal as the latter results in the end of the episode. What remains are all the positions of the first 3 rows plus the bottom-left cell.

The observation is a value representing the player’s current position as current_row * nrows + current_col (where both the row and col start at 0).

For example, the stating position can be calculated as follows: 3 * 12 + 0 = 36.
The observation is returned as an `int()`.

## Starting State
The episode starts with the player in state `[36]` (location [3, 0]).

## Reward
Each time step incurs -1 reward, unless the player stepped into the cliff, which incurs -100 reward.

## Episode End
The episode terminates when the player enters state `[47]` (location [3, 11]).

![Image](image.png)

## Agent Algorithms
For this problem we created 2 different agents using different algorithms.

**SARSA**
State–action–reward–state–action is an algorithm for learning a Markov decision process policy, used in the reinforcement learning area of machine learning. A typical SARSA algorithm can be written as the following:
![[Pasted image 20240412231536.png]]

Within SARSA agent we are using an epsilon-greedy policy. Epsilon greedy policy is a way to balance exploration and exploitation where we are choosing a random action with probability epsilon and choosing the best action with probability 1-epsilon. After each action we our Q table with the reward according to the action taken at particular state. 
The code implementation of SARSA agent can be found in the sarsa.py file. After our 500 episodes the SARSA agent is consistently able to reach the end state in under 20 actions where 13 steps are the optimal number of actions it can take to reach end state.


**Q-Learning**
Q-Learning is a model-free  reinforcement learning algorithm to learn the value of an action in a particular state. It does not require a model of the environment (hence "model-free"), and it can handle problems with stochastic transitions and rewards without requiring adaptations.

For any finite Markov decision process, _Q_-learning finds an optimal policy in the sense of maximizing the expected value of the total reward over any and all successive steps, starting from the current state. _Q_-learning can identify an optimal action-selection policy for any given finite Markov decision process, given infinite exploration time and a partly random policy. "Q" refers to the function that the algorithm computes – the expected rewards for an action taken in a given state.
![[Pasted image 20240412232552.png]]

The Q-Learning Model will use the same epsilon-greedy policy and we can see that after the 500 episodes the agent is able to minimize the number of action down to 13 which is the optimal number of actions for the problem.


Both the SARSA and Q-Learning are similar algorithms with the key difference being SARSA updates its Q-values based on the current policy's action selection, while Q-learning updates its Q-values based on the maximum Q-value of the next state, regardless of the policy. This leads to SARSA being an on-policy method and Q-learning being an off-policy method.
