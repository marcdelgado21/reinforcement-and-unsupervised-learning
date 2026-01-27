import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from ActorCriticNetworks import ActorNetwork, CriticNetwork, copy_target, soft_update
from ReplayBuffer import ReplayBuffer
from helper import episode_reward_plot, video_agent
import numpy as np
from Noise import NormalActionNoise
import gymnasium as gym
from gymnasium.wrappers import RecordVideo


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TD3:
    """The TD3 Agent."""

    def __init__(self, env, replay_size=1000000, batch_size=100, gamma=0.99,
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        """ Initializes the TD3 method.
        
        Parameters
        ----------
        env: gym.Environment
            The gym environment the agent should learn in.
        replay_size: int
            The size of the replay buffer.
        batch_size: int
            The number of replay buffer entries an optimization step should be performed on.
        gamma: float
            The discount factor.
        policy_noise: float
            Standard deviation of Gaussian noise added to target policy actions.
        noise_clip: float
            Limit for absolute value of target policy smoothing noise.
        policy_freq: int
            Frequency of delayed policy updates.
        """

        self.obs_dim, self.act_dim = env.observation_space.shape[0], env.action_space.shape[0]
        self.env = env
        self.replay_buffer = ReplayBuffer(replay_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        # TODO (2): Initialize the Actor and Critic networks. 
        # Initialize Critic networks (TWIN) and target networks. 
        self.Critic1 = CriticNetwork(self.obs_dim, self.act_dim).to(device)
        self.Critic1_target = CriticNetwork(self.obs_dim, self.act_dim).to(device)
        copy_target(self.Critic1, self.Critic1_target)

        self.Critic2 = CriticNetwork(self.obs_dim, self.act_dim).to(device)
        self.Critic2_target = CriticNetwork(self.obs_dim, self.act_dim).to(device)
        copy_target(self.Critic2, self.Critic2_target)
        
        # Initialize Actor network and its target network. Should be named self.Actor
        self.Actor = ActorNetwork(self.obs_dim, self.act_dim).to(device)
        self.Actor_target = ActorNetwork(self.obs_dim, self.act_dim).to(device)
        copy_target(self.Actor, self.Actor_target) 

        # END TODO (2)
        
        # Define the optimizers for the actor and critic networks as proposed in the paper
        # Optimized both critics parameters
        self.optim_critic = optim.Adam(list(self.Critic1.parameters()) + list(self.Critic2.parameters()), lr=0.001)
        self.optim_actor = optim.Adam(self.Actor.parameters(), lr=0.001) 


    def learn(self, timesteps):
        """Train the agent for timesteps steps inside self.env.
        After every step taken inside the environment observations, rewards, etc. have to be saved inside the replay buffer.
        If there are enough elements already inside the replay buffer (>batch_size), compute MSBE loss and optimize DQN network.

        Parameters
        ----------
        timesteps: int
            Number of timesteps to optimize the DQN network.
        """
        all_rewards = []
        episode_rewards = []
        all_rewards_eval = []
        timeexit = timesteps

        # We use here Gaussian Noise instead of OU for TD3 exploration as suggested in the lab/paper
        exploration_noise = NormalActionNoise(mean=np.zeros(self.act_dim), sigma=0.1 * np.ones(self.act_dim))

        obs, _ = self.env.reset()
        for timestep in range(1, timesteps + 1):

            action = self.choose_action(obs)

            # Here we sample and add the noise to the action to explore the environment. Notice we clip the action
            # between -1 and 1 because the action space is continuous and bounded between -1 and 1.
            noise = exploration_noise.sample()
            action = np.clip(action + noise, -1, 1)

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            self.replay_buffer.put(obs, action, reward, next_obs, terminated, truncated)
            
            obs = next_obs
            episode_rewards.append(reward)
            
            if terminated or truncated:
                all_rewards_eval.append(self.eval_episodes())
                print('\rTimestep: ', timestep, '/' ,timesteps,' Episode reward: ',np.round(all_rewards_eval[-1]), 'Episode: ', len(all_rewards), 'Mean R', np.mean(all_rewards_eval[-100:]))
                obs, _ = self.env.reset()
                all_rewards.append(sum(episode_rewards))
                episode_rewards = []
                    
            if len(self.replay_buffer) > self.batch_size:
                #TODO (6): if there is enouygh data in the replay buffer, sample a batch and perform an optimization step
                # Batch is sampled from the replay buffer and containes a list of tuples (s, a, r, s', term, trunc)
                batch = self.replay_buffer.get(self.batch_size)
                # Get the batch data

                # Compute the loss for the critic and update the critic network 
                critic_loss = self.compute_critic_loss(batch)
                self.optim_critic.zero_grad()
                critic_loss.backward()
                self.optim_critic.step()

                # TD3 Delayed update: Only update Actor and Targets every policy_freq steps
                if timestep % self.policy_freq == 0:
                    # Compute the loss for the actor and update the actor network 
                    actor_loss = self.compute_actor_loss(batch)
                    self.optim_actor.zero_grad()
                    actor_loss.backward()
                    self.optim_actor.step()
                    
                    # TODO (7): Sync the target networks with soft updates and tau=0.005 according to details of the DDPG/TD3 paper
                    # Note: In TD3 tau is usually higher (0.005) because updates are less frequent
                    tau = 0.005
                    soft_update(self.Critic1_target, self.Critic1, tau)
                    soft_update(self.Critic2_target, self.Critic2, tau)
                    soft_update(self.Actor_target, self.Actor, tau)
                    # END TODO (7)
                
                # END TODO (6)

            if timestep % (timesteps-1) == 0:
                episode_reward_plot(all_rewards, timestep, window_size=7, step_size=1)
                pass
            if len(all_rewards_eval)>10 and np.mean(all_rewards_eval[-5:]) > 220:
                episode_reward_plot(all_rewards, timestep, window_size=7, step_size=1)
                break
        return all_rewards, all_rewards_eval
    

    def choose_action(self, s):
        # TODO (3) Implement the function to choose an action given a state. It is deterministic because exploration is added
        # by the Noise in the main loop.

        state = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            action = self.Actor(state).cpu().numpy()[0]
        # END TODO (3)
        return action


    def compute_critic_loss(self, batch):
        # === TODO (4) ===
        states, actions, rewards, next_states, terminated, truncated = batch

        # Convert to tensors
        states      = torch.tensor(states, dtype=torch.float32).to(device)
        actions     = torch.tensor(actions, dtype=torch.float32).to(device)
        rewards     = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        dones       = torch.tensor(terminated | truncated, dtype=torch.float32).unsqueeze(1).to(device)

        with torch.no_grad():
            # TD3: Target Policy Smoothing
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.Actor_target(next_states) + noise).clamp(-1, 1)

            # TD3: Clipped Double Q-learning (Twin Critics)
            target_Q1 = self.Critic1_target(next_states, next_actions)
            target_Q2 = self.Critic2_target(next_states, next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            
            # Bellman target
            target = rewards + self.gamma * (1 - dones) * target_Q

        # Critic forward
        current_Q1 = self.Critic1(states, actions)
        current_Q2 = self.Critic2(states, actions)

        # MSE Loss for both critics
        loss = nn.MSELoss()(current_Q1, target) + nn.MSELoss()(current_Q2, target)

        # === END TODO (4) ===
        return loss

    

    def compute_actor_loss(self,batch):
        """
        The function `compute_actor_loss` calculates the loss for the actor network 
        
        :param batch: The batch parameter is a tuple containing the data for computing the loss.
        :return: the loss, which is the negative mean of the expected Q-values.
        """
        # TODO (5) implement the actor loss. You have to sample from the replay buffer first a set of states.

        states, _, _, _, _, _ = batch
        states = torch.tensor(states, dtype=torch.float32).to(device)
        
        actions_pred = self.Actor(states)
        # TD3 uses Q1 to optimize the actor
        q_values = self.Critic1(states, actions_pred)
        
        loss = -q_values.mean()
        # END TODO (5) 

        return loss



    def eval_episodes(self,n=3):
        """ Evaluate an agent performing inside a Gym environment. """
        lr=[]
        for episode in range(n):
            tr = 0.0
            obs, _ = self.env.reset()
            while True:
                action = self.choose_action(obs)
                obs, reward, terminated, truncated, _ = self.env.step(action)
                tr += reward
                if terminated or truncated:
                    break
            lr.append(tr)
        return np.mean(lr)


if __name__ == '__main__':
    # Create gym environment
    env = gym.make("LunarLander-v3",continuous=True, render_mode='rgb_array')

    # Initialize TD3 agent
    td3 = TD3(env, replay_size=1000000, batch_size=100, gamma=0.99)

    td3.learn(500000)
    env = RecordVideo(gym.make("LunarLander-v3",continuous=True, render_mode='rgb_array'),'video')    
    video_agent(env, td3, n_episodes=5)  
    pass