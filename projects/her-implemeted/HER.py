import numpy as np
import random
from bit_flip_env import Env
from agent import Agent


# n_bits is the length of the bit string. You should try different values to reproduce 
# the results of figure 1 of the paper.
n_bits = 50

# Standard hyperparameters
lr = 1e-3
gamma = 0.98
MAX_EPISODE_NUM = 20000
memory_size = 1e+6
batch_size = 128

# HER specific hyperparameters
k_future = 4

# Print the number of bits of the string
print(f"Number of bits:{n_bits}")

# Create the environment and the agent
env = Env(n_bits)
agent = Agent(n_bits=n_bits, lr=lr, memory_size=int(memory_size), batch_size=batch_size, gamma=gamma)

# This variable is used to calculate the number of times the agent reaches the goal using optimal actions
optimal=0
# This variable is used to calculate the number of times the agent achieves the goal state
solved=0

for episode_num in range(1,MAX_EPISODE_NUM+1):
    # state is inital state and goal is the target state for the episode. 
    state, goal = env.reset()
    # mincost is the number of bits that are different between the state and the goal, so the minimum cost to reach the goal
    mincost = state.shape[0]-sum(goal == state) 
    episode_reward = 0 # Should maintain for statistics the sum of the rewards obtained in the episode
    episode = [] # Should maintain the transitions of the episode
    done = False

    ### TODO RED BOX : Implementation of the red box of the figure of the lab dossier: Regular Q-learning for an episode
    # Collect an episode (until done)
    while not done:
        # Elegir acción usando la política epsilon-greedy
        action = agent.choose_action(state, goal)
        
        # Ejecutar acción en el entorno
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        # Guardar la transición en la lista local del episodio
        episode.append((state, action, reward, terminated, truncated, next_state, goal))
        
        episode_reward += reward
        state = next_state
        done = terminated or truncated

    # Check if the episode was solved optimally
    if mincost+episode_reward >=0: 
        optimal=optimal+1

    # Loop to Store the episode in the agent's memory
    for t, transition in enumerate(episode):
        # Desempaquetar la transición original
        s, a, r, term, trunc, s_next, g = transition
        
        # 1. Almacenar la experiencia original (Caja Roja)
        agent.store(s, a, r, term, trunc, s_next, g)

        ## END OF RED BOX TODO

        ## TODO GREEN BOX: Implementation of the green box: Hindsight Experience Replay (HER)
        # Estrategia 'future': seleccionamos k estados alcanzados después del tiempo t
        if t < len(episode) - 1:
            # Seleccionar k indices aleatorios de pasos futuros en este mismo episodio
            future_indices = random.sample(range(t + 1, len(episode)), min(k_future, len(episode) - 1 - t))
            
            for idx in future_indices:
                # El nuevo objetivo es el estado alcanzado en el futuro
                new_goal = episode[idx][5] # Usamos el next_state del paso idx como objetivo
                
                # Recalcular la recompensa: 0 si el estado siguiente es el nuevo objetivo, -1 si no
                new_terminated = np.array_equal(s_next, new_goal)
                new_reward = 0.0 if new_terminated else -1.0
                
                # Almacenar la experiencia con el objetivo ficticio
                agent.store(s, a, new_reward, new_terminated, trunc, s_next, new_goal)

        ## END OF GREEN BOX TODO

    # TODO BLUE BOX: 
    # Learn from the episode. All work is done in the agent.learn() method
    loss = agent.learn()
    # END OF BLUE BOX TODO

    # Update epsilon
    agent.update_epsilon()

    # Update and print the results each 500 episodes
    if episode_num == 1:
        global_running_r = episode_reward
    else:
        global_running_r = 0.99 * global_running_r + 0.01 * episode_reward
        
    if episode_num % 500 == 0:
        print(f"Ep:{episode_num:5d}| "
                f"Ep_r:{episode_reward:3.1f}| "
                f"Ep_running_r:{global_running_r:3.3f}| "
                f"Epsilon:{agent.epsilon:3.3f}| "
                f"Mem_size:{len(agent.memory):6d}| "
                f"Optimal:{100*optimal/500:.2f}%")
        optimal=0