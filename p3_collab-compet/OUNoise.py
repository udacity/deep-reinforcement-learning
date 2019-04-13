import numpy as np
import torch
import random
import copy

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        random.seed(seed)
        np.random.seed(seed)
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return self.state



# # from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
# class OUNoise:

#     def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
#         self.action_dimension = action_dimension
#         self.scale = scale
#         self.mu = mu
#         self.theta = theta
#         self.sigma = sigma
#         self.state = np.ones(self.action_dimension) * self.mu
#         self.reset()

#     def reset(self):
#         self.state = np.ones(self.action_dimension) * self.mu

#     def noise(self):
#         x = self.state
#         dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
#         self.state = x + dx
#         return torch.tensor(self.state * self.scale).float()
        
        

