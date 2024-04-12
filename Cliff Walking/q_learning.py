import gym
import numpy as np
import pickle as pkl
import os

cliffEnv = gym.make("CliffWalking-v0", render_mode="human") #create an environment from the available in gym
#cliffEnv = gym.make("CliffWalking-v0")

if os.path.exists("q_learning_q_table.pkl"):
    # Load the q_table from the pickle file
    q_table = pkl.load(open("q_learning_q_table.pkl", "rb"))
else:
    q_table = np.zeros(shape=(48, 4)) # 48 states and 4 actions

# We are defining an epsilon greedy policy here
# Epsilon greedy policy is a way to balance exploration and exploitation
# We are choosing a random action with probability epsilon and 
# choosing the best action with probability 1-epsilon
def policy(state, explore=0.0): # epsilon greedy policy
    
    # choose action based on epsilon greedy policy
    # q value of state-action pairs returning the most optimal action
    action = int(np.argmax(q_table[state]))
    
    # if random number is less than epsilon, choose random action
    if np.random.random() <= explore:
        action = int(np.random.randint(low=0, high=1, size=1))
    return action

# Parameters
EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.9
NUM_EPISODES = 500


for episode in range(NUM_EPISODES):
    total_reward = 0
    episode_length = 0
    done = False

    state = cliffEnv.reset()
    #if episode == 0:
    action = policy(state[0], EPSILON) # choose action based on policy
    #else:
    #    action = policy(state, EPSILON) # choose action based on policy

    while not done:
        # print(cliffEnv.render())
        next_state, reward, done, _, _ = cliffEnv.step(action)
        next_action = policy(next_state) # take optimal action

        # update q table
        if episode_length == 0:
            q_table[state[0]][action] += ALPHA * (reward + GAMMA +q_table[next_state][next_action] - q_table[state[0]][action])
        else:
            q_table[state][action] += ALPHA * (reward + GAMMA +q_table[next_state][next_action] - q_table[state][action])

        state = next_state
        action = next_action
        total_reward += reward
        episode_length += 1
    print("Episode: {}, Total Reward: {}, Episode Length: {}".format(episode, total_reward, episode_length))


cliffEnv.close()

pkl.dump(q_table, open("q_learning_q_table.pkl", "wb"))
print("Training Complete. Q Table Saved")