import numpy as np

class Bandit:

    def __init__(self, probability: float):
        self.probability = probability
        self.probability_estimate = 5
        self.n_trials = 1

    def pull(self) -> int: 
        result = np.random.random() < self.probability
        reward = 1 if result else 0
        return reward


    def update(self, reward: float) -> None: 
        self.n_trials += 1
        self.probability_estimate = ((self.n_trials-1)*self.probability_estimate + reward) / self.n_trials