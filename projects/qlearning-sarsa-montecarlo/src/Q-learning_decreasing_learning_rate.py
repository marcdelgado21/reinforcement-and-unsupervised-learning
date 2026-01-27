'''import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from Grid import RussellGrid
from utils import print_policy


# Q-learning

env = RussellGrid()
env.reset()

# Q-values inicialization
Q = np.random.rand(env.observation_space.n, env.action_space.n) * 0.01  # small random init
gamma = 0.999

# You can change these values to see how they affect the results
lr = 0.01
epsilon = 0.2
#epsilon = 1

G = 0
print('Training...')

for episode in range(1, 10001):

    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        # ε-greedy policy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()   # explore
        else:
            action = np.argmax(Q[state])        # exploit

        next_state, reward, done, _, _ = env.step(action)

        # Q-learning update
        best_next_action = np.argmax(Q[next_state])
        td_target = reward + gamma * Q[next_state, best_next_action] * (not done)
        td_error = td_target - Q[state, action]
        Q[state, action] += lr * td_error

        state = next_state
        total_reward += reward

    G += total_reward

    # Every 500 episodes, print the average collected reward during training
    if episode % 500 == 0:
        print('Episode {} Average Reward: {}'.format(episode, G/500), 'Epsilon =', epsilon, 'lr =', lr)
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

print_policy(policy, V)


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

'''

#con epsilon decay

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from Grid import RussellGrid
from utils import print_policy

# Q-learning

env = RussellGrid()
env.reset()

# Q-values inicialization
Q = np.random.rand(env.observation_space.n, env.action_space.n) * 0.01  # small random init
gamma = 0.999

# Hyperparameters
lr = 0.01
epsilon_start = 0.2
epsilon_end = 0.01
n_episodes = 10000
decay_rate = (epsilon_start - epsilon_end) / n_episodes  # linear decay

G = 0
print('Training...')

epsilon = epsilon_start
for episode in range(1, n_episodes + 1):

    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        # ε-greedy policy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()   # explore
        else:
            action = np.argmax(Q[state])        # exploit

        next_state, reward, done, _, _ = env.step(action)

        # Q-learning update
        best_next_action = np.argmax(Q[next_state])
        td_target = reward + gamma * Q[next_state, best_next_action] * (not done)
        td_error = td_target - Q[state, action]
        Q[state, action] += lr * td_error

        state = next_state
        total_reward += reward

    G += total_reward

    # Decay epsilon (linearly, until epsilon_end)
    epsilon = max(epsilon_end, epsilon_start - decay_rate * episode)

    # Every 500 episodes, print the average collected reward during training
    if episode % 500 == 0:
        print(f"Episode {episode} Average Reward: {G/500:.2f}  Epsilon = {epsilon:.3f}  lr = {lr}")
        G = 0

print('End training...')

# -------------------------
# Visualizing policy
# -------------------------
print('Visualizing policy...')
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

print_policy(policy, V)

# -------------------------
# Visualizing episodes
# -------------------------
print('Visualizing episodes...')
for i in range(10):
    state = env.reset()
    env.render()
    done = False
    while not done:
        action = np.argmax(Q[state])
        state, reward, done, _, info = env.step(action)
        env.render()

# -------------------------
# Testing policy (no exploration)
# -------------------------
print('Collecting reward of the policy while testing (no exploration)...')
G = 0
for i in range(1000):
    state = env.reset()
    done = False
    while not done:
        action = np.argmax(Q[state])
        state, reward, done, _, info = env.step(action)
        G += reward
print('Average reward of the policy while testing:', G/1000)
