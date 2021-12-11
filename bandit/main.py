import numpy as np 
import matplotlib.pyplot as plt
import datetime

from multiprocessing import Process

from app.models.bandit import Bandit


NUM_TRIALS = 100000
EPS = 0.1
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]


def main(epsilon: float = 0.1):
    bandits = [Bandit(probability) for probability in BANDIT_PROBABILITIES]

    rewards = np.zeros(NUM_TRIALS)
    num_times_explored = 0
    num_times_exploited = 0
    num_optimal = 0
    bandit_index = 0
    optimal_bandit_index = np.argmax([bandit.probability for bandit in bandits])
    print(f"Optimal Bandit: {optimal_bandit_index}")

    for trial in range(NUM_TRIALS):
        random_value = np.random.random()

        if random_value < epsilon:
            num_times_explored += 1
            bandit_index = np.random.randint(0, len(bandits))
        else: 
            num_times_exploited += 1 
            bandit_index = np.argmax([bandit.probability_estimate for bandit in bandits])

        if bandit_index == optimal_bandit_index:
            num_optimal += 1
        
        reward = bandits[bandit_index].pull()
        rewards[trial] = reward
        bandits[bandit_index].update(reward)

    for index, bandit in enumerate(bandits):
        print(f"Mean estimate for bandit number {index}: {bandit.probability_estimate}")


    print(f"Number of times explored: {num_times_explored}")
    print(f"Number of times exploited: {num_times_exploited}")
    print(f"Number of times when optimal bandit was selected: {num_optimal}")
    print(f"Total reward earned: {rewards.sum()}")
    print(f"Overall win rate: {rewards.sum() / NUM_TRIALS}")

    cumulative_rewards = np.cumsum(rewards)
    win_rates = cumulative_rewards / (np.arange(NUM_TRIALS) + 1)
    timestamp = datetime.datetime.now().strftime("%H%M%S")


    fig = plt.figure(dpi=200, figsize=(12, 6))
    plt.title(f"Multi Armed Bandit Approximation (EPS = {epsilon:.2f})", fontsize=20, pad=20)
    plt.ylabel("Probability", fontsize=16, labelpad=20)
    plt.xlabel("Trials", fontsize=16, labelpad=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xscale("log")
    plt.plot(win_rates, linewidth=4)
    plt.plot(np.ones(NUM_TRIALS) * np.max(BANDIT_PROBABILITIES), linewidth=4)
    plt.legend(["Bandit Approximation", "Real Probability"])
    fig.savefig(f"plots/{timestamp}_eps_{epsilon:.2f}_bandits.png", dpi=600, bbox_inches = "tight")

if __name__ == "__main__":
    main(epsilon=0.99)

    processes = [Process(target=main, args=(np.random.uniform(),)) for _ in range(20)]

    for process in processes: 
        process.start()