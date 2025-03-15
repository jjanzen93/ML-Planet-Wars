from stable_baselines3 import PPO
from planet_wars_env import PlanetWarsEnv
import matplotlib.pyplot as plt
import time

def plot_metrics(timesteps_list, loss_list, episode_count_list, winrate_list):
    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(timesteps_list, loss_list, label="Loss")
    plt.xlabel("Timesteps")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(episode_count_list, winrate_list, label="Win Rate")
    plt.xlabel("Episodes (per 20)")
    plt.ylabel("Win Rate")
    plt.legend()

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(30)
    plt.close()

def main():
    env = PlanetWarsEnv(max_turns=1000, opponent_model=None, visualize=True)
    model = PPO("MlpPolicy", env, verbose=1)
    opponent_model = PPO.load("ppo_planet_wars2", env=env)
    env.opponent_model = opponent_model
    total_timesteps = 100000
    timesteps_per_iter = 10000
    timesteps = 0
    episode_count = 0

    # Tracking metrics
    timesteps_list = []
    loss_list = []
    episode_count_list = []
    winrate_list = []

    while timesteps < total_timesteps:
        print(f"Training for {timesteps_per_iter} timesteps. Total so far: {timesteps}.")
        model.learn(total_timesteps=timesteps_per_iter, reset_num_timesteps=False)
        timesteps += timesteps_per_iter

        # Access the PPO loss
        loss = model.logger.name_to_value.get("train/loss", 0)

        # Track loss over timesteps
        timesteps_list.append(timesteps)
        loss_list.append(loss)

        # Track episodes
        episode_count += len(env.episode_results)

        if len(env.episode_results) >= 20:
            recent_win_rate = sum(env.episode_results[-20:]) / 20.0
            winrate_list.append(recent_win_rate)
            episode_count_list.append(episode_count)  # Track the episode count
            print(f"Recent win rate over last 20 episodes: {recent_win_rate*100:.1f}%")
            env.episode_results.clear()

            # Plot every 20 episodes
            plot_metrics(timesteps_list, loss_list, episode_count_list, winrate_list)

            if recent_win_rate >= 0.80:
                snapshot_file = "snapshot_model.zip"
                model.save(snapshot_file)
                opponent_model = PPO.load(snapshot_file, env=env)
                env.opponent_model = opponent_model
                print("Opponent updated to current model snapshot!")

    model.save("ppo_planet_wars5")
    env.close()
    plot_metrics(timesteps_list, loss_list, episode_count_list, winrate_list)

if __name__ == "__main__":
    main()