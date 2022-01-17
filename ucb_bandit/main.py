import numpy as np 
import matplotlib.pyplot as plt
import datetime

from app.models.bandit import Bandit


NUM_TRIALS = 100000
EPS = 0.1
BANDIT_PROBABILITIES = [0.1, 0.5, 0.90]
UCB_CONSTANT = 2

def ucb(sampled_mean: float, n_trials: int, n_bandit_trials: int, ucb_constant: float = 2.0):
    result = sampled_mean + np.sqrt(ucb_constant * (np.log(n_trials) / n_bandit_trials))
    return result 

def main(ucb_constant: float = 2.0):
    bandits = [Bandit(probability) for probability in BANDIT_PROBABILITIES]
    rewards = np.zeros(NUM_TRIALS)
    num_times_explored = 0
    num_times_exploited = 0
    num_optimal = 0
    total_plays = 0
    bandit_index = 0
    optimal_bandit_index = np.argmax([bandit.probability_estimate for bandit in bandits])
    print(f"Optimal Bandit: {optimal_bandit_index}")



    for bandit in bandits:
        reward = bandit.pull()
        total_plays += 1
        bandit.update(reward)


    for trial in range(NUM_TRIALS):
        bandit_index = np.argmax(
            [
                ucb(bandit.probability_estimate, total_plays, 
                bandit.n_trials, ucb_constant=ucb_constant) for bandit in bandits
        ])

        reward = bandits[bandit_index].pull()
        rewards[trial] = reward
        bandits[bandit_index].update(reward)


    for index, bandit in enumerate(bandits):
        print(f"Mean estimate for bandit number {index}: {bandit.probability_estimate}")


    # print(f"Number of times explored: {num_times_explored}")
    # print(f"Number of times exploited: {num_times_exploited}")
    # print(f"Number of times when optimal bandit was selected: {num_optimal}")
    print(f"Total reward earned: {rewards.sum()}")
    print(f"Overall win rate: {rewards.sum() / NUM_TRIALS}")

    cumulative_rewards = np.cumsum(rewards)
    win_rates = cumulative_rewards / (np.arange(NUM_TRIALS) + 1)
    timestamp = datetime.datetime.now().strftime("%H%M%S")


    fig = plt.figure(dpi=200, figsize=(12, 6))
    plt.title(f"Multi Armed Bandit Approximation (UCB Constant = {ucb_constant})", fontsize=20, pad=20)
    plt.ylabel("Probability", fontsize=16, labelpad=20)
    plt.xlabel("Trials", fontsize=16, labelpad=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim([0,1])
    plt.xscale("log")
    plt.plot(win_rates, linewidth=4)
    plt.plot(np.ones(NUM_TRIALS) * np.max(BANDIT_PROBABILITIES), linewidth=4)
    plt.legend(["Bandit Approximation", "Real Probability"])
    fig.savefig(f"plots/{timestamp}_ucb_{ucb_constant}_bandits.png", dpi=600, bbox_inches = "tight")

if __name__ == "__main__":
    main(ucb_constant=10.0)
    main(ucb_constant=2.0)
    main(ucb_constant=6.0)
    main(ucb_constant=0.8)