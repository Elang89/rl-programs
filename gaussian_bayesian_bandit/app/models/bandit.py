import numpy as np

from typing import List

class Bandit:

    def __init__(self, true_mean: float):
        self.true_mean = true_mean
        self.predicted_mean = 1.0
        self.precision = 1.0
        self.sum_reward = 0.0
        self.tau = 1.0
        self.n_trials = 0

    def sample(self) -> float:
        return np.random.randn() / np.sqrt(self.tau) + self.true_mean

    def pull(self) -> float: 
        reward = np.random.randn() / np.sqrt(self.precision) + self.predicted_mean
        return reward


    def update(self, reward: float) -> None: 
        self.precision += self.tau
        self.sum_reward += reward
        self.predicted_mean =  self.tau * self.sum_reward / self.precision 
        self.n_trials += 1