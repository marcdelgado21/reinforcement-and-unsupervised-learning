import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import MSELoss
from torch.optim.adam import Adam
from replay_memory import ExperienceReplayBuffer

class Q_value(nn.Module):
    """ Class for the DQN model aproximating the value function. """
    def __init__(self, n_inputs, n_outputs):
        super(Q_value, self).__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.hidden = nn.Linear(self.n_inputs, 256)
        nn.init.kaiming_normal_(self.hidden.weight)
        self.hidden.bias.data.zero_()

        self.output = nn.Linear(256, self.n_outputs)
        nn.init.xavier_uniform_(self.output.weight)
        self.output.bias.data.zero_()

    def forward(self, states, goals):
        x = torch.cat([states, goals], dim=-1)
        x = F.relu(self.hidden(x))
        return self.output(x)


class Agent:
    """ Class for a standar DQN agent. """
    def __init__(self, n_bits, lr, memory_size, batch_size, gamma):
        """
        Initialize the agent.

        Parameters:
        n_bits (int): Number of bits in the state and goal.
        lr (float): Learning rate for the optimizer.
        memory_size (int): Size of the experience replay buffer.
        batch_size (int): Batch size for training.
        gamma (float): Discount factor for future rewards.
        """
        ## TODO: Initialize the agent with all the elements needed for the DQN algorithm. 
        self.n_bits = n_bits
        self.lr = lr
        self.batch_size = batch_size
        self.gamma = gamma
        
        self.model = Q_value(2 * n_bits, n_bits)
        self.target_model = Q_value(2 * n_bits, n_bits)
        self.target_model.load_state_dict(self.model.state_dict())
        
        self.opt = Adam(self.model.parameters(), lr=self.lr)
        self.memory = ExperienceReplayBuffer(memory_size)
        self.loss_fn = MSELoss()
        ## END TODO

        self.epsilon = 1.0
        self.epsilon_decay = 0.999


    def choose_action(self, states, goals):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.n_bits)
        else:
            state_t = torch.as_tensor(states, dtype=torch.float32).unsqueeze(0)
            goal_t = torch.as_tensor(goals, dtype=torch.float32).unsqueeze(0)
            
            self.model.eval()
            with torch.no_grad():
                q_values = self.model(state_t, goal_t)
            self.model.train()
            
            action = torch.argmax(q_values).item()
        return action


    def update_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, 0)


    def store(self, state, action, reward, terminated, truncated, next_state, goal):
        self.memory.store(state, action, reward, terminated, truncated, next_state, goal)


    def learn(self):
        if len(self.memory) < self.batch_size:
            return 0.0

        transitions = self.memory.sample(self.batch_size)
        states, actions, rewards, terminateds, truncateds, next_states, goals = zip(*transitions)

        states_t = torch.tensor(np.array(states), dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        terminateds_t = torch.tensor(terminateds, dtype=torch.float32).unsqueeze(1)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32)
        goals_t = torch.tensor(np.array(goals), dtype=torch.float32)

        current_q = self.model(states_t, goals_t).gather(1, actions_t)

        with torch.no_grad():
            max_next_q = self.target_model(next_states_t, goals_t).max(1)[0].unsqueeze(1)
            target_q = rewards_t + (self.gamma * max_next_q * (1 - terminateds_t))

        loss = self.loss_fn(current_q, target_q)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.soft_update_of_target_network(self.model, self.target_model)
        return loss.item()


    @staticmethod
    def soft_update_of_target_network(local_model, target_model, tau=0.05):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)