import matplotlib.pyplot as plt
import numpy as np

class Bandit(object):

    def __init__(self, probability: float):
        self.probability = probability
        self.probability_estimate = 0.0
        self.N_trials = 0.0

    def pull(self) -> int:
        result = np.random.random() < self.probability
        reward = 1 if result else 0
        return reward


    def update(self, reward: int) -> None:
        self.N_trials += 1 
        self.probability_estimate += ((self.N_trials - 1)*self.probability_estimate + reward) / self.N_trials
