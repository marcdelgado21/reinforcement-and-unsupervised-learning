import numpy as np


class Env(object):
    def __init__(self, num_bits, fixed = False, target = None):
        """
        Initialize the environment.

        Parameters:
        num_bits (int): The number of bits in the state and target.
        """
        self.num_bits = num_bits
        self.truncated = None  # Whether the current episode is truncated. Initialized in method reset
        self.terminated = None  # Whether the current episode is truncated. Initialized in method reset
        self.num_steps = None # Number of steps taken in the current episode. Initialized in method reset
        self.state = None # Current state of the environment. Initialized in method reset.
        self.target = None # Target state of the environment. Initialized in method reset.
        self.fixed= fixed
        if fixed:  # If asked for fixed goal
            if target is None: # If no target is given
                self.target = np.random.randint(2, size=self.num_bits) # Random Fixed goal
            else:
                self.target = target   # Assign target to parameter asked target
            print(f"Target: {self.target}")

    def reset(self):
        """
        Reset the environment to an random initial state and a random goal state.
        Notice it returns not only the initial state but also the target state 
        (differently than gym's API).

        Returns:
        tuple: A tuple containing the initial state and target state.
        """
        self.truncated = False
        self.terminated = False
        self.num_steps = 0
        self.state = np.random.randint(2, size=self.num_bits) # Set the initial state to a random binary vector
        if not self.fixed:
            self.target = np.random.randint(2, size=self.num_bits) # Set the goal state to a random binary vector. Comment for Fixed goal
        return self.state, self.target

    def step(self, action):
        """
        Take an action in the environment.

        Parameters:
        action (int): The index of the bit to flip in the state.

        Returns:
        tuple: A tuple containing the new state, reward, termination status, truncation status, and an empty dictionary.
        """
        assert not self.terminated
        assert not self.truncated

        self.state[action] = 1 - self.state[action]  # Flip the bit at action index

        if self.num_steps > self.num_bits + 1: # Check if the number of steps takes is larger than the number of action needed to solve the problem in the worst case.
            self.truncated = True
        self.num_steps += 1

        if np.sum(self.state == self.target) == self.num_bits: # If the state is equal to the target
            self.terminated = True
            return np.copy(self.state), 0, self.terminated, self.truncated, {}
        else:
            return np.copy(self.state), -1, self.terminated, self.truncated, {}
        
    def render(self):
        """
        Print the current state and target state.
        """
        print(f"State : {self.state}")
        print(f"Target: {self.target}")
        print("")
