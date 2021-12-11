import random
import matplotlib.pyplot as plt

def main(): 
    bandits = [10, 240, 121, 80, 90, 15, 43, 53, 800, 532, 52, 342, 46, 74, 36]
    selections = {number: 0 for number in bandits}
    epsilon = 0.5 
    min_epsilon = 0.1
    less_than_epsilon_count = 0
    epsilons = []
    timesteps = []
    epochs = 1000

    for _, timestep in enumerate(range(epochs)):
        p = random.uniform(0.0, 1.0)
        decay = random.uniform(0.0, 0.9999)
        timesteps.append(timestep)
        epsilons.append(epsilon)

        if p < epsilon:
            less_than_epsilon_count += 1
            random_bandit = random.choice(bandits)
            selections.update({random_bandit: selections.get(random_bandit) + 1})
        else: 
            bandit = max(bandits)
            selections.update({bandit: selections.get(bandit) + 1})

        epsilon =  max(min_epsilon, epsilon*decay)


    numbers = list(selections.keys())
    x_axis = [str(selection) for selection in numbers]
    y_axis = selections.values()
    
    fig = plt.figure(dpi=100, figsize=(15, 3))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    
    ax1.set_title("Distribution of Bandits", pad=20)
    ax1.set_xlabel("Selected Bandit", labelpad=20)
    ax1.set_ylabel("Frequency", labelpad=20)
    ax1.bar(x_axis, y_axis)

    ax2.set_title("Epsilon Decay Timeseries", pad=20)
    ax2.set_xlabel("Timestep", labelpad=20)
    ax2.set_ylabel("Epsilon Value", labelpad=20)
    ax2.plot(timesteps, epsilons)

    fig.savefig("bandits.png", dpi=200, bbox_inches = "tight")
    print(f"Times in which p was less than epsilon: {less_than_epsilon_count}, percentage: {(less_than_epsilon_count / epochs) * 100}%, total: {epochs}")

if __name__ == "__main__":
    main()