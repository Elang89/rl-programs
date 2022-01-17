import numpy as np 
import matplotlib.pyplot as plt
import datetime

from typing import List
from scipy.stats import beta

from app.models.bandit import Bandit


NUM_TRIALS = 2000
BANDIT_PROBABILITIES = [0.1, 0.5, 0.90]

def plot(bandits: List[Bandit], trial: int): 
    x_axis = np.linspace(0, 1, 200)

    fig = plt.figure(dpi=200, figsize=(12, 6))

    for bandit in bandits: 
        y_axis = beta.pdf(x_axis, bandit.alpha, bandit.beta)
        plt.plot(x_axis, y_axis, label=f"Real Probability: {bandit.probability:4f}, Win Rate = {bandit.alpha - 1}/{bandit.n_trials}")
    plt.ylabel("Probability", fontsize=16, labelpad=20)
    plt.xlabel("Trials", fontsize=16, labelpad=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(f"Bandit distributions after {trial} trials")
    plt.legend()

    fig.savefig(f"plots/trial_{trial}_bayesian_bandits.png", dpi=600, bbox_inches = "tight")

def main():
    bandits = [Bandit(probability) for probability in BANDIT_PROBABILITIES]
    sample_points = [5,10,20,50,100,200,500,1000,1500,1999]
    rewards = np.zeros(NUM_TRIALS)

    bandit_index = 0

    for trial in range(NUM_TRIALS):
        bandit_index = np.argmax([bandit.sample() for bandit in bandits])

        reward = bandits[bandit_index].pull()
        rewards[trial] = reward
        bandits[bandit_index].update(reward)

        if trial in sample_points:
            plot(bandits, trial)


    print(f"Total reward earned: {rewards.sum()}")
    print(f"Overall win rate: {rewards.sum() / NUM_TRIALS}")



if __name__ == "__main__":
    main()