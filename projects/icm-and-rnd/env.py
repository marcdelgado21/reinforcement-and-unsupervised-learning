import gymnasium as gym
from gymnasium import Wrapper


class MountainCarCustomized(Wrapper):
    """Gymnasium wrapper for MountainCar that modifies the reward function.
    
    Changes:
    - Reward is 0 for each step (sparse reward)
    - Reward is 200 when goal is reached
    - Episode terminates after 200 steps (TimeLimit wrapper not applied)
    
    Args:
        render_mode: Optional render mode ("human", "rgb_array", or None)
    """
    
    def __init__(self, render_mode=None):
        env = gym.make('MountainCar-v0', render_mode=render_mode)
        super().__init__(env)
        self.time_step = 0
    
    def step(self, action):
        self.time_step += 1
        obs, _, terminated, truncated, info = self.env.step(action)
        
        # Get the unwrapped environment to access .state
        unwrapped_env = self.env.unwrapped
        
        # Check if goal reached
        goal = bool(
            unwrapped_env.state[0] >= unwrapped_env.goal_position and 
            unwrapped_env.state[1] >= unwrapped_env.goal_velocity
        )
        
        # Sparse reward: 0 for each step, 200 when goal reached
        reward = 200.0 if goal else 0.0
        
        # Limit to 200 time steps
        if self.time_step >= 200:
            truncated = True
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.time_step = 0
        return obs, info

