import numpy as np 
import matplotlib.pyplot as plt

from typing import List
from scipy.stats import norm

from app.models.bandit import Bandit


NUM_TRIALS = 2000
BANDIT_MEANS = [1, 2, 3]

def plot(bandits: List[Bandit], trial: int): 
    x_axis = np.linspace(-3, 6, 200)

    fig = plt.figure(dpi=200, figsize=(12, 6))

    for bandit in bandits: 
        y_axis = norm.pdf(x_axis, bandit.predicted_mean, np.sqrt(1.0 / bandit.precision))
        plt.plot(x_axis, y_axis, label=f"True Mean: {bandit.true_mean:4f}, Num Plays: {bandit.n_trials}")
    plt.ylabel("Probability", fontsize=16, labelpad=20)
    plt.xlabel("Trials", fontsize=16, labelpad=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(f"Bandit distributions after {trial} trials")
    plt.legend()

    fig.savefig(f"plots/trial_{trial}_gaussian_bayesian_bandits.png", dpi=600, bbox_inches = "tight")

def main():
    bandits = [Bandit(mean) for mean in BANDIT_MEANS]
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