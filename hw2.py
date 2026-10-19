
import os
import csv
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import imageio.v2 as imageio

os.makedirs("results", exist_ok=True)

EPISODES = 3000
SEED = 42


def train(alpha=0.5, gamma=0.99, epsilon=1.0,
          decay=0.997, epsilon_min=0.05, episodes=EPISODES):
    env = gym.make("Taxi-v4")
    q = np.zeros((env.observation_space.n, env.action_space.n))
    rewards = []

    rng = np.random.default_rng(SEED)

    for episode in range(episodes):
        state, _ = env.reset(seed=SEED + episode)
        done = False
        total = 0

        while not done:
            if rng.random() < epsilon:
                action = int(rng.integers(env.action_space.n))
            else:
                action = int(np.argmax(q[state]))

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            target = reward
            if not done:
                target += gamma * np.max(q[next_state])

            q[state, action] += alpha * (target - q[state, action])

            state = next_state
            total += reward

        rewards.append(total)
        epsilon = max(epsilon_min, epsilon * decay)

    env.close()
    return q, np.array(rewards)


def evaluate(q, episodes=100):
    env = gym.make("Taxi-v4")
    rewards = []
    successes = 0
    steps_list = []

    for episode in range(episodes):
        state, _ = env.reset(seed=10000 + episode)
        total = 0
        steps = 0

        for step in range(200):
            action = int(np.argmax(q[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            steps += 1

            if terminated:
                successes += 1
            if terminated or truncated:
                break

        rewards.append(total)
        steps_list.append(steps)

    env.close()
    return {
        "mean_reward": float(np.mean(rewards)),
        "success_rate": successes / episodes,
        "mean_steps": float(np.mean(steps_list))
    }


def smooth(values, window=100):
    if len(values) < window:
        return values
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_curves(curves, title, filename):
    plt.figure(figsize=(10, 5))
    for label, rewards in curves.items():
        values = smooth(rewards)
        plt.plot(np.arange(len(values)) + 99, values, label=label)

    plt.xlabel("Training Episode")
    plt.ylabel("Average Reward (100 Episodes)")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/" + filename, dpi=160)
    plt.close()


def record_video(q):
    env = gym.make("Taxi-v4", render_mode="rgb_array")
    state, _ = env.reset(seed=12345)
    frames = [env.render()]

    for _ in range(200):
        action = int(np.argmax(q[state]))
        state, _, terminated, truncated, _ = env.step(action)
        frames.append(env.render())
        if terminated or truncated:
            break

    env.close()

    with imageio.get_writer("results/trained_agent.mp4", fps=4) as writer:
        for frame in frames:
            writer.append_data(frame)


def main():
    print("Training baseline...")
    baseline_q, baseline_rewards = train()
    baseline_metrics = evaluate(baseline_q)

    plot_curves(
        {"Baseline": baseline_rewards},
        "Baseline Q-Learning Training",
        "baseline.png"
    )

    experiments = [
        ("alpha", [0.1, 0.5, 0.9], "Learning Rate", "learning_rate.png"),
        ("gamma", [0.50, 0.90, 0.99], "Discount Factor", "discount_factor.png"),
        ("decay", [0.990, 0.995, 0.999], "Epsilon Decay", "epsilon_decay.png")
    ]

    records = []
    candidates = [("Baseline", baseline_q, baseline_metrics)]

    records.append({
        "experiment": "Baseline",
        "value": "default",
        **baseline_metrics
    })

    for parameter, values, title, filename in experiments:
        curves = {}

        for value in values:
            print(f"Training {parameter}={value}...")
            q, rewards = train(**{parameter: value})
            metrics = evaluate(q)

            label = f"{parameter}={value}"
            curves[label] = rewards
            candidates.append((label, q, metrics))

            records.append({
                "experiment": parameter,
                "value": value,
                **metrics
            })

        plot_curves(curves, title + " Comparison", filename)

    best_name, best_q, best_metrics = max(
        candidates,
        key=lambda x: (x[2]["mean_reward"], x[2]["success_rate"])
    )

    np.save("results/best_q_table.npy", best_q)

    with open("results/metrics.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    with open("results/summary.txt", "w") as file:
        file.write("Environment: Taxi-v4\n")
        file.write("Episodes per training run: 3000\n")
        file.write("Baseline: alpha=0.5, gamma=0.99, epsilon=1.0\n")
        file.write("Epsilon decay=0.997, minimum epsilon=0.05\n")
        file.write("Evaluation episodes: 100\n\n")
        file.write(f"Baseline results: {baseline_metrics}\n")
        file.write(f"Best configuration: {best_name}\n")
        file.write(f"Best results: {best_metrics}\n")

    print("Recording MP4...")
    record_video(best_q)

    print("\nFINISHED!")
    print("Baseline:", baseline_metrics)
    print("Best:", best_name, best_metrics)
    print("Files saved in results folder.")


if __name__ == "__main__":
    main()
