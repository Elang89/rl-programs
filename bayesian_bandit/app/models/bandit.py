import numpy as np

from typing import List

class Bandit:

    def __init__(self, probability: float):
        self.probability = probability
        self.alpha = 1
        self.beta = 1
        self.n_trials = 0

    def sample(self) -> List[float]:
        return np.random.beta(self.alpha, self.beta)

    def pull(self) -> int: 
        result = np.random.random() < self.probability
        reward = 1 if result else 0
        return reward


    def update(self, reward: float) -> None: 
        self.n_trials += 1
        self.alpha += reward
        self.beta += (1 - reward)