'''import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from Grid import RussellGrid
from utils import print_policy


# Monte-Carlo

env = RussellGrid()
env.reset()

# Q-values inicialization
Q = np.zeros([env.observation_space.n, env.action_space.n])*np.random.rand(env.observation_space.n, env.action_space.n)
gamma = 0.999

# You can change these values to see how they affect the results
lr=0.01
epsilon = 0.2

G = 0
print('Training...')

for episode in range(1, 10001):

    # TODO: Implement the Monte Carlo algorithm
    # First you have to collect the data from the episode and later update the Q-values
    # Generate an episode and store it
    # Update Q-values for each state in the episode. You can implement one-visit or every-visit MC, as you prefer.
    # End TODO

    # Generate an episode and store it
    episode_data = []
    state = env.reset()
    done = False
    
    # Generate episode using epsilon-greedy policy
    while not done:
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(Q[state])  # Exploit
        
        next_state, reward, done, _, _ = env.step(action)
        episode_data.append((state, action, reward))
        state = next_state
    
    # Calculate returns and update Q-values
    G_episode = 0
    returns = []
    
    # Calculate returns from the end to the beginning
    for t in range(len(episode_data)-1, -1, -1):
        state, action, reward = episode_data[t]
        G_episode = reward + gamma * G_episode
        returns.insert(0, (state, action, G_episode))
    
    # Update Q-values using every-visit Monte Carlo
    visited_pairs = set()
    for state, action, return_t in returns:
        # Every-visit MC: update every time we visit (state, action)
        if (state, action) not in visited_pairs:
            visited_pairs.add((state, action))
            Q[state, action] = Q[state, action] + lr * (return_t - Q[state, action])
    
    # Update policy (implicitly through Q-values)
    # For Monte Carlo, the policy is derived from Q-values using epsilon-greedy
    
    # Store total reward for this episode
    total_reward = sum([reward for _, _, reward in episode_data])
    G += total_reward
    
    # End TODO

    # Every 500 episodes, print the average collected reward during training (stored in variable G)
    if episode % 500 == 0:
        print('Episode {} Total Reward: {}'.format(episode,G/500), 'Epsilon =', epsilon, 'lr =', lr)
        G=0

print('End training...')


print('Visualizing policy...')
# Create and print policy and V for valid states
policy = np.zeros((env.world_row,env.world_col))
V = np.zeros((env.world_row,env.world_col))
policy[0,3]=-1  # Special value for terminal state
policy[1,3]=-1  # Special value for terminal state
policy[1,1]=-1  # Not defined for non-valid states

state = np.nditer(env.map, flags=['multi_index'])
while not state.finished:
    if env.map[state.multi_index]==0:
        policy[state.multi_index] = np.argmax(Q[env.cell_id(state.multi_index)])
        V[state.multi_index]= np.max(Q[env.cell_id(state.multi_index)])
    state.iternext()

print_policy(policy,V)


print('Visualizing episodes...')
# Let's test your policy!
for i in range(10):
    state = env.reset()
    env.render()
    done = None
    while done != True:
        action = np.argmax(Q[state])
        state, reward, done, _, info = env.step(action)
        env.render()


print('Collecting reward of the policy while testing (no exploration)...')
G=0
# Now let's see how much reward your policy can collect in 1000 episodes
for i in range(1000):
    state = env.reset()
    done = None
    while done != True:
        action = np.argmax(Q[state])
        state, reward, done, _, info = env.step(action)
        G = G + reward
print('Average reward of the policy while testing:', G/1000)

'''

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from Grid import RussellGrid
from utils import print_policy


# Monte-Carlo

env = RussellGrid()
env.reset()

# Q-values inicialization
Q = np.zeros([env.observation_space.n, env.action_space.n]) * np.random.rand(env.observation_space.n, env.action_space.n)
gamma = 0.999

# Hyperparameters
lr = 0.01
epsilon_start = 0.2
epsilon_end = 0.01
n_episodes = 10000
decay_rate = (epsilon_start - epsilon_end) / n_episodes  # lineal

G = 0
print('Training...')

epsilon = epsilon_start
for episode in range(1, n_episodes + 1):

    # Generate an episode and store it
    episode_data = []
    state = env.reset()
    done = False
    
    # Generate episode using epsilon-greedy policy
    while not done:
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(Q[state])  # Exploit
        
        next_state, reward, done, _, _ = env.step(action)
        episode_data.append((state, action, reward))
        state = next_state
    
    # Calculate returns and update Q-values
    G_episode = 0
    returns = []
    
    # Calculate returns from the end to the beginning
    for t in range(len(episode_data)-1, -1, -1):
        state, action, reward = episode_data[t]
        G_episode = reward + gamma * G_episode
        returns.insert(0, (state, action, G_episode))
    
    # Update Q-values using every-visit Monte Carlo
    visited_pairs = set()
    for state, action, return_t in returns:
        if (state, action) not in visited_pairs:  # one-visit MC
            visited_pairs.add((state, action))
            Q[state, action] = Q[state, action] + lr * (return_t - Q[state, action])
    
    # Store total reward for this episode
    total_reward = sum([reward for _, _, reward in episode_data])
    G += total_reward

    # Decay epsilon linearly
    epsilon = max(epsilon_end, epsilon_start - decay_rate * episode)
    
    # Every 500 episodes, print the average collected reward during training
    if episode % 500 == 0:
        print(f'Episode {episode} Average Reward: {G/500:.2f} | Epsilon = {epsilon:.3f} | lr = {lr}')
        G = 0

print('End training...')


print('Visualizing policy...')
# Create and print policy and V for valid states
policy = np.zeros((env.world_row, env.world_col))
V = np.zeros((env.world_row, env.world_col))
policy[0,3] = -1  # Special value for terminal state
policy[1,3] = -1  # Special value for terminal state
policy[1,1] = -1  # Not defined for non-valid states

state = np.nditer(env.map, flags=['multi_index'])
while not state.finished:
    if env.map[state.multi_index] == 0:
        policy[state.multi_index] = np.argmax(Q[env.cell_id(state.multi_index)])
        V[state.multi_index] = np.max(Q[env.cell_id(state.multi_index)])
    state.iternext()

print_policy(policy,V)


print('Visualizing episodes...')
# Let's test your policy!
for i in range(10):
    state = env.reset()
    env.render()
    done = False
    while not done:
        action = np.argmax(Q[state])
        state, reward, done, _, info = env.step(action)
        env.render()


print('Collecting reward of the policy while testing (no exploration)...')
G = 0
# Now let's see how much reward your policy can collect in 1000 episodes
for i in range(1000):
    state = env.reset()
    done = False
    while not done:
        action = np.argmax(Q[state])
        state, reward, done, _, info = env.step(action)
        G += reward
print('Average reward of the policy while testing:', G/1000)
