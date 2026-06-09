import random

## TODO: Implement the init and methods of the ExperienceReplayBuffer that now have a pass statement

class ExperienceReplayBuffer:
    """ Class for the experience replay buffer. """
    def __init__(self, capacity):
        """
        Initialize the replay buffer.

        Parameters:
        capacity (int): The maximum number of transitions to store in the buffer.
        """
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def store(self, *item):
        """
        Add a transition to the replay buffer.

        Parameters:
        *item: The transition to add, consisting of state, action, reward, terminated, truncated, next_state, and goal.
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        
        self.memory[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Sample a batch of transitions from the replay buffer.

        Parameters:
        batch_size (int): The number of transitions to sample.

        Returns:
        list: A list of sampled transitions.
        """
        return random.sample(self.memory, batch_size)


    def __len__(self):
        """
        Get the current size of the replay buffer.

        Returns:
        int: The number of transitions currently stored in the buffer.
        """
        return len(self.memory)