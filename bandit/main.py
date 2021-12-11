import numpy as np 
import matplotlib.pyplot as plt
import datetime

from typing import List

from app.models.bandit import Bandit


def main(means: List[float], num_trials: int = 100000, decay: float = 1.0, epsilon: float = 0.1, reward_threshold: float = 25000.0):
    initial_epsilon = epsilon
    bandits = [Bandit(mean) for mean in means]

    rewards = np.empty(num_trials)

    means = [m1, m2, m3]
    num_times_explored = 0
    num_times_exploited = 0
    num_optimal = 0
    num_suboptimal = 0
    bandit_index = 0
    optimal_bandit_index = np.argmax(means)
    print(f"Optimal Bandit: {optimal_bandit_index}")

    for trial in range(num_trials):
        random_value = np.random.random()

        if random_value < epsilon:
            num_times_explored += 1
            bandit_index = np.random.randint(0, len(bandits))
        else: 
            num_times_exploited += 1 
            bandit_index = np.argmax([bandit.mean_estimate for bandit in bandits])

        if bandit_index == optimal_bandit_index:
            num_optimal += 1
        else: 
            num_suboptimal += 1
        
        reward = bandits[bandit_index].pull()
        rewards[trial] = reward
        bandits[bandit_index].update(reward)
        total_reward = rewards.sum()

        if  total_reward > reward_threshold:
            epsilon = epsilon * decay

    for index, bandit in enumerate(bandits):
        print(f"Mean estimate for bandit number {index}: {bandit.mean_estimate}")


    print(f"Number of times explored: {num_times_explored}")
    print(f"Number of times exploited: {num_times_exploited}")
    print(f"Number of times when optimal bandit was selected: {num_optimal}")
    print(f"Number of times when suboptimal bandit was selected: {num_suboptimal}")
    print(f"Percentage of optimal bandit selection: {(num_optimal / num_trials) * 100:.2f}% of {num_trials}")
    print(f"Percentage of suboptimal bandit selection: {(num_suboptimal / num_trials) * 100:.2f}% of {num_trials}")
    print(f"Total reward earned: {rewards.sum()}")
    print(f"Overall win rate: {rewards.sum() / num_trials}")

    cumulative_average = np.cumsum(rewards) / (np.arange(num_trials) + 1)
    timestamp = datetime.datetime.now().strftime("%H%M%S")
    color = (np.random.random(), np.random.random(), np.random.random())

    fig = plt.figure(dpi=200, figsize=(12, 6))
    plt.title(f"Multi Armed Bandit Approximation (EPS = {initial_epsilon:.2f}, Decay = {decay:.2f}, Reward Threshold = {reward_threshold:.2f})", fontsize=20, pad=20)
    plt.ylabel("Mean", fontsize=16, labelpad=20)
    plt.xlabel("Trials", fontsize=16, labelpad=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xscale("log")
    plt.plot(cumulative_average, linewidth=4, color=color)
    plt.plot(np.ones(num_trials) * m1, linewidth=4, color="green")
    plt.plot(np.ones(num_trials) * m2, linewidth=4, color="blue")
    plt.plot(np.ones(num_trials) * m3, linewidth=4, color="red")
    plt.legend(["Bandit Approximation", "Mean 1", "Mean 2", "Mean 3"], loc="upper left",  bbox_to_anchor=(-0.40, 1.05))
    fig.savefig(f"plots/{timestamp}_eps_{initial_epsilon:.2f}_bandits.png", dpi=600, bbox_inches = "tight")

    return cumulative_average


if __name__ == "__main__":
    m1, m2, m3 = 1.5, 2.5, 3.5
    c_09 = main([m1, m2, m3], num_trials=100000, decay=0.9, epsilon=0.9, reward_threshold = 0)
    c_05 = main([m1, m2, m3], num_trials=100000, decay=0.7, epsilon=0.5, reward_threshold = 25000.0)
    c_01 = main([m1, m2, m3], num_trials=100000, decay=0.5, epsilon=0.1, reward_threshold = 10000.0)


    fig = plt.figure(dpi=200, figsize=(12, 6))
    plt.title("EPS Comparison)")
    plt.ylabel("Mean", fontsize=16, labelpad=20)
    plt.xlabel("Trials", fontsize=16, labelpad=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xscale("log")
    plt.plot(c_09, label="EPS = 0.9", linewidth=4, color="darkorange")
    plt.plot(c_05, label="EPS = 0.5", linewidth=4)
    plt.plot(c_01, label="EPS = 0.1", linewidth=4, color="green")
    plt.legend(loc="upper left",  bbox_to_anchor=(-0.40, 1.05))
    fig.savefig("plots/EPS.png", dpi=600, bbox_inches = "tight")