import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from Grid import RussellGrid
from utils import print_policy


# Q-learning

env = gym.make("Taxi-v3", render_mode="ansi")

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

    state, _ = env.reset()   # reset devuelve (obs, info)
    done = False
    total_reward = 0

    while not done:
        # ε-greedy policy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()   # explore
        else:
            action = np.argmax(Q[state])        # exploit

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

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
        print(f'Episode {episode} Average Reward: {G/500:.2f}  Epsilon = {epsilon}  lr = {lr}')
        G = 0

print('End training...')


print('Visualizing episodes...')
# Let's test your policy!
for i in range(5):  # muestra 5 episodios, puedes poner 10 si quieres
    state, _ = env.reset()
    env.render()
    done = False
    while not done:
        action = np.argmax(Q[state])
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        env.render()
        state = next_state

print('Collecting reward of the policy while testing (no exploration)...')
G = 0
# Now let's see how much reward your policy can collect in 1000 episodes
for i in range(1000):
    state, _ = env.reset()
    done = False
    while not done:
        action = np.argmax(Q[state])
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        G += reward
        state = next_state

print('Average reward of the policy while testing:', G / 1000)
env.close()

