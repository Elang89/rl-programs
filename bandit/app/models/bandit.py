import matplotlib.pyplot as plt
import numpy as np

class Bandit(object):

    def __init__(self, mean: float):
        self.mean = mean
        self.mean_estimate = 0.0
        self.N_trials = 0.0

    def pull(self) -> int:
        reward = np.random.randn() + self.mean
        return reward


    def update(self, reward: int) -> None:
        self.N_trials += 1 
        self.mean_estimate += ((self.N_trials - 1)*self.mean_estimate + reward) / self.N_trials + reward
