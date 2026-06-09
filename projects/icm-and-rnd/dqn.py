import random
import gymnasium as gym
#import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
import pickle


from exploration import DummyIntrinsicRewardModule, RNDNetwork, ICMNetwork

class ReplayBuffer(object):
    """A replay buffer as commonly used for off-policy Q-Learning methods."""

    def __init__(self, capacity):
        """Initializes replay buffer with certain capacity."""
        self.buffer = [None] * capacity

        self.capacity = capacity
        self.size = 0
        self.ptr = 0

    def put(self, obs, action, extrinsic_reward, next_obs, truncated, terminated):
        """Put a tuple of (obs, action, extrinsic_reward, next_obs, truncated, terminated) into the replay buffer.
        NOTE: Only stores EXTRINSIC rewards. Intrinsic rewards will be computed on-the-fly.
        The max length specified by capacity should never be exceeded. 
        The oldest elements inside the replay buffer should be overwritten first.
        """
        self.buffer[self.ptr] = (obs, action, extrinsic_reward, next_obs, truncated, terminated)

        self.size = min(self.size + 1, self.capacity)
        self.ptr = (self.ptr + 1) % self.capacity

    def get(self, batch_size):
        """Gives batch_size samples from the replay buffer.
        Should return 6 lists of, each for every attribute stored (i.e. obs_lst, action_lst, ....)
        """
        return zip(*random.sample(self.buffer[: self.size], batch_size))

    def __len__(self):
        """Returns the number of tuples inside the replay buffer."""
        return self.size


class DQNNetwork(nn.Module):
    """The neural network used to approximate the Q-function. Should output n_actions Q-values per state."""

    def __init__(self, num_obs, num_actions):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(num_obs, 128), nn.ReLU(), nn.Linear(128, num_actions)
        )

    def forward(self, x):
        return self.layers(x)


class DQN:
    """The DQN method."""

    def __init__(
        self,
        env,
        replay_size=20000,
        batch_size=32,
        gamma=0.99,
        sync_after=5,
        lr=0.03,
        verbose=False,
        reward_module=None,
        render=False,
    ):
        if isinstance(env.action_space, gym.spaces.Box):
            raise NotImplementedError("Continuous actions not implemented!")

        # Some variables
        self.obs_dim, self.act_dim = env.observation_space.shape[0], env.action_space.n
        self.env = env
        self.replay_buffer = ReplayBuffer(replay_size)
        self.sync_after = sync_after
        self.batch_size = batch_size
        self.gamma = gamma
        self.verbose = verbose
        self.render = render

        # Initialize DQN network
        self.dqn_net = DQNNetwork(self.obs_dim, self.act_dim)
        # Initialize DQN target network, load parameters from DQN network
        self.dqn_target_net = DQNNetwork(self.obs_dim, self.act_dim)
        self.dqn_target_net.load_state_dict(self.dqn_net.state_dict())
        # Set up optimizer, only needed for DQN network
        self.optim_dqn = optim.RMSprop(self.dqn_net.parameters(), lr=lr)

        # Initialize reward module
        if reward_module == "RND":
            self.intrinsic_reward_module = RNDNetwork(self.obs_dim, 128)
            self.optim_reward = optim.RMSprop(
                self.intrinsic_reward_module.predictor.parameters(), lr=lr,
            )
        elif reward_module == "ICM":
            self.intrinsic_reward_module = ICMNetwork(self.obs_dim, 256, self.act_dim)
            self.optim_reward = optim.RMSprop(
                self.intrinsic_reward_module.parameters(), lr=lr / 50.0
            )
        else:
            # This module only has a 'calculate_reward(...)' method which returns 0.0
            # Used for vanilla DQN
            self.intrinsic_reward_module = DummyIntrinsicRewardModule()

    def learn(self, time_steps):
        # We use them for our reward plots
        lrval = []
        episode_length = 0
        episode_lengths = []

        obs,_ = self.env.reset()
        # Save best episode reward so far here
        min_episode_len = 1337
        for timestep in range(1, time_steps + 1):
            if self.render and timestep % 15 == 0:
                # Render every 15th frame to save resources
                self.env.render()
                #pass

            epsilon = epsilon_by_timestep(timestep)
            action = self.predict(obs, epsilon)

            # Do environment step
            next_obs, extrinsic_reward, terminated, truncated,  _ = self.env.step(action)
            done = truncated or terminated

            # Store ONLY extrinsic reward in replay buffer
            # Intrinsic rewards will be computed on-the-fly during training
            self.replay_buffer.put(obs, action, extrinsic_reward, next_obs, truncated, terminated)
            obs = next_obs

            episode_length += 1
            if done:
                obs, _ = self.env.reset()
                if (
                    self.verbose
                    and min_episode_len > episode_length
                    and episode_length < 200
                ):
                    min_episode_len = episode_length
                    print(f"[t={timestep}]: Solved after {min_episode_len}!")
                episode_lengths.append(episode_length)
                episode_length = 0

            if len(self.replay_buffer) > self.batch_size:
                # Update
                # Get data from replay buffer
                obs_, actions, extrinsic_rewards, next_obs_, truncateds, terminateds = self.replay_buffer.get(
                    self.batch_size
                )
                # Convert to Tensors
                obs_ = torch.stack([torch.Tensor(ob) for ob in obs_])
                next_obs_ = torch.stack(
                    [torch.Tensor(next_ob) for next_ob in next_obs_]
                )
                extrinsic_rewards = torch.Tensor(extrinsic_rewards)
                truncateds = torch.Tensor(truncateds)
                terminateds = torch.Tensor(terminateds)
                # Has to be torch.LongTensor in order to being able to use as index for torch.gather()
                actions = torch.LongTensor(actions)

                # COMPUTE INTRINSIC REWARDS 
                if not isinstance(
                    self.intrinsic_reward_module, DummyIntrinsicRewardModule
                ):
                    # Recompute intrinsic rewards using the CURRENT reward module
                    # This ensures fresher rewards instead of stale ones from the buffer
                    with torch.no_grad():
                        intrinsic_rewards_batch = self.intrinsic_reward_module.calculate_reward(
                            obs_, next_obs_, actions
                        )
                    # Combine extrinsic and intrinsic rewards
                    total_rewards = extrinsic_rewards + intrinsic_rewards_batch
                else:
                    # For vanilla DQN, use only extrinsic rewards
                    total_rewards = extrinsic_rewards

                # Update DQN using COMBINED rewards
                dqn_loss = self.compute_msbe_loss(
                    obs_, actions, total_rewards, next_obs_, truncateds, terminateds
                )
                self.optim_dqn.zero_grad()
                dqn_loss.backward()
                self.optim_dqn.step()

                # Update reward module
                # Note: We don't do this in case of vanilla DQN, thus the isinstance(...) check
                if not isinstance(
                    self.intrinsic_reward_module, DummyIntrinsicRewardModule
                ):
                    # Update reward module
                    intrinsic_loss = self.intrinsic_reward_module.calculate_loss(
                        obs_, next_obs_, actions
                    )
                    self.optim_reward.zero_grad()
                    intrinsic_loss.backward()
                    self.optim_reward.step()

            if timestep % self.sync_after == 0:
                # Update target network
                self.dqn_target_net.load_state_dict(self.dqn_net.state_dict())

            if timestep % 1000 == 0 and len(episode_lengths) >= 7:
                rval=self.test_policy_10()
                lrval.append(rval)
                print(' ',timestep, rval,end='')

        return lrval

    def test_policy_10(self):
        """Tests the policy for 100 episodes."""
        env = deepcopy(self.env)
        time = []
        for i in range(100):
            obs,_ = env.reset()
            done = False
            t=0
            while not done:
                # Note: epsilon is only used for vanilla DQN
                action = self.predict(obs, 0)
                t=t+1
                # Do environment step
                obs, extrinsic_reward, terminated, truncated,  _ = env.step(action)
                done = truncated or terminated
            time.append(t)
        return np.array(time).mean()

    def predict(self, state, epsilon=0.0):
        if random.random() > epsilon:
            state = torch.FloatTensor(state).unsqueeze(0)
            q_value = self.dqn_net.forward(state)
            action = q_value.argmax().item()
        else:
            action = random.randrange(self.act_dim)
        return action

    def compute_msbe_loss(self, obs, actions, rewards, next_obs, truncateds, terminateds):
        # Compute q_values and next_q_values
        q_values = self.dqn_net(obs)
        next_q_values = self.dqn_target_net(next_obs)
        # Select Q-values of actions actually taken
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        # Calculate max over next Q-values
        next_q_values = next_q_values.max(1)[0]
        # The target we want to update our network towards
        dones = truncateds + terminateds - truncateds * terminateds
#        expected_q_values = rewards + self.gamma * next_q_values * (1.0 - terminateds)
        expected_q_values = rewards + self.gamma * next_q_values * (1.0 - dones)
        # Calculate DQN loss
        dqn_loss = F.mse_loss(q_values, expected_q_values)

        return dqn_loss


def render_episodes(dqn, env_name="Agent", num_episodes=3):
    """Render episodes with the learned policy.
    
    Args:
        dqn: Trained DQN agent
        env_name: Name of the environment/method (for display purposes)
        num_episodes: Number of episodes to render
    """
    print(f"\nRendering {num_episodes} episodes with learned {env_name} policy...")
    from env import MountainCarCustomized
    env_render = MountainCarCustomized(render_mode="human")
    
    for episode in range(num_episodes):
        obs, _ = env_render.reset()
        done = False
        episode_reward = 0
        step = 0
        print(f"Episode {episode + 1}:")
        while not done:
            action = dqn.predict(obs, epsilon=0)
            obs, reward, terminated, truncated, _ = env_render.step(action)
            done = terminated or truncated
            episode_reward += reward
            step += 1
        print(f"  Steps: {step}, Total Reward: {episode_reward}")


def epsilon_by_timestep(
    timestep, epsilon_start=1.0, epsilon_final=0.01, frames_decay=10000
):
    """Linearly decays epsilon from epsilon_start to epsilon_final in frames_decay timesteps."""
    return max(
        epsilon_final,
        epsilon_start - (timestep / frames_decay) * (epsilon_start - epsilon_final),
    )



def test_policy_100(env,dqn):
    """Tests the policy for 100 episodes."""
    time = []
    for i in range(100):
        obs,_ = env.reset()
        done = False
        t=0
        while not done:
            # Note: epsilon is only used for vanilla DQN
            action = dqn.predict(obs, 0)
            t=t+1
            # Do environment step
            obs, extrinsic_reward, terminated, truncated,  _ = env.step(action)
            done = truncated or terminated
        time.append(t)
    return np.array(time).mean()


if __name__ == "__main__":
    # Switch this to 'True' for realtime rendering during training
    # Note: This might slow down training a bit
    render = False
    TIMESTEPS= 50000

    # Vanilla DQN on original MountainCar environment with epsilon exploration

    #Uncomment this to use the default MountainCar environment
    #See what happens
    '''
    print('Original MountainCar')
    ll=[]
    for i in range(3):
        env = gym.make('MountainCar-v0', render_mode="human") if render else gym.make('MountainCar-v0')
        dqn = DQN(env, verbose=True, render=render)
        l1 = dqn.learn(TIMESTEPS)
        ll.append(l1)
        print('\n Mean: ',test_policy_100(env,dqn))
    min_len = min(len(r) for r in ll)
    arr = np.array([r[:min_len] for r in ll])
    mean, std = arr.mean(axis=0), arr.std(axis=0)
    xs = np.arange(min_len)
    plt.plot(xs, mean)
    plt.fill_between(xs, mean - std, mean + std, alpha=0.3)
    plt.title('Original MountainCar')
    plt.xlabel('Time steps')
    plt.ylabel('Episode length')
    plt.grid()
    plt.ylim((0, 210)) 
    plt.savefig('original.png')
    plt.show()
    print(ll)
    file = open('original', 'wb')
    pickle.dump(ll, file)
    file.close()
    render_episodes(dqn, "Original DQN", 1)
'''
    # DQN on customized sparse MountainCar with epsilon exploration
    from env import MountainCarCustomized
    env = MountainCarCustomized()
    '''
    print('Customized MountainCar sparsified with epsilon exploration')
    ll2=[]
    for i in range(3):
        env = MountainCarCustomized()
        dqn = DQN(env, verbose=True, render=render)
        l1 = dqn.learn(TIMESTEPS)
        ll2.append(l1)
        print('\n Mean: ',test_policy_100(env,dqn))
    plt.title('Customized MountainCar with epsilon exploration')
    plt.xlabel('Time steps')
    plt.ylabel('Episode length')
    plt.grid()
    min_len = min(len(r) for r in ll2)
    arr2 = np.array([r[:min_len] for r in ll2])
    mean2, std2 = arr2.mean(axis=0), arr2.std(axis=0)
    xs2 = np.arange(min_len)
    plt.plot(xs2, mean2)
    plt.fill_between(xs2, mean2 - std2, mean2 + std2, alpha=0.3)
    plt.ylim((0, 210)) 
    plt.savefig('customized-eps.png')
    plt.show()
    print(ll2)
    file = open('Cust-eps', 'wb')
    pickle.dump(ll2, file)
    file.close()
    render_episodes(dqn, "Customized DQN (Epsilon)", 1)

    # DQN + ICM
    print('Customized MountainCar sparsified with ICM')
    ll4=[]
    for i in range(3):
        env = MountainCarCustomized(render_mode="human" if render else None)
        dqn = DQN(env, verbose=True, reward_module="ICM", render=render)
        l1 = dqn.learn(TIMESTEPS)
        ll4.append(l1)
        print('\n Mean: ',test_policy_100(env,dqn))
    min_len = min(len(r) for r in ll4)
    arr4 = np.array([r[:min_len] for r in ll4])
    mean4, std4 = arr4.mean(axis=0), arr4.std(axis=0)
    xs4 = np.arange(min_len)
    plt.plot(xs4, mean4)
    plt.fill_between(xs4, mean4 - std4, mean4 + std4, alpha=0.3)
    plt.title('Customized MountainCar with ICM')
    plt.xlabel('Time steps')
    plt.ylabel('Episode length')
    plt.grid()
    plt.ylim((0, 210)) 
    plt.savefig('Customized-ICM.png')
    plt.show()
    print(ll4)
    file = open('Cust-ICM', 'wb')
    pickle.dump(ll4, file)
    file.close()
    render_episodes(dqn, "Customized DQN (ICM)", 1)
'''
    # DQN + RND
    print('Customized MountainCar sparsified RND exploration')
    ll3=[]
    for i in range(3):
        env = MountainCarCustomized(render_mode="human" if render else None)
        dqn = DQN(env, verbose=True, reward_module="RND", render=render)
        l1 = dqn.learn(TIMESTEPS)
        ll3.append(l1)
        print('\n Mean: ',test_policy_100(env,dqn))
    min_len = min(len(r) for r in ll3)
    arr3 = np.array([r[:min_len] for r in ll3])
    mean3, std3 = arr3.mean(axis=0), arr3.std(axis=0)
    xs3 = np.arange(min_len)
    plt.plot(xs3, mean3)
    plt.fill_between(xs3, mean3 - std3, mean3 + std3, alpha=0.3)
    plt.title('Customized MountainCar with RND')
    plt.xlabel('Time steps')
    plt.ylabel('Episode length')
    plt.grid()
    plt.ylim((0, 210)) 
    plt.savefig('customized-RND.png')
    plt.show()
    print(ll3)
    file = open('Cust-RND', 'wb')
    pickle.dump(ll3, file)
    file.close()
    render_episodes(dqn, "Customized DQN (RND)", 1)


