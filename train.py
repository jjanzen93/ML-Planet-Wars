from stable_baselines3 import PPO
from planet_wars_env import PlanetWarsEnv
import matplotlib.pyplot as plt
import time
import random
from opponent_bots import defensive_bot
from opponent_bots import aggressive_bot
from opponent_bots import easy_bot
from opponent_bots import production_bot
from opponent_bots import spread_bot
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO

"""
def plot_metrics(timesteps_list, loss_list, episode_count_list, winrate_list, reward_mean_list):
    plt.figure(figsize=(10, 5))
    plt.subplot(3, 1, 1)
    plt.plot(timesteps_list, loss_list, label="Loss")
    plt.xlabel("Timesteps")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(episode_count_list, winrate_list, label="Win Rate")
    plt.xlabel("Episodes")
    plt.ylabel("Win Rate")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(episode_count_list, reward_mean_list, label="Mean Reward")
    plt.xlabel("Episodes")
    plt.ylabel("Mean Reward")
    plt.legend()

    plt.tight_layout()"""


def main():
    env = PlanetWarsEnv(max_turns=1000, opponent_model=None, visualize=True)
    model = MaskablePPO("MlpPolicy", env, verbose=1, gamma = 0.9995,device = "cpu", tensorboard_log="./ppo_multiagent_tensorboard/")
    model.save("multi_ppo_untrained")
    #opponent_model = MaskablePPO.load("multi_ppo_untrained", env=env, gamma = 0.9995,device = "cpu")
    opponent_list = [easy_bot.Easy_Bot(), production_bot.Production_Bot(), aggressive_bot.Aggressive_Bot(), defensive_bot.Defensive_Bot(), spread_bot.Spread_Bot()]
    opponent_model = spread_bot.Spread_Bot()
    env.opponent_model = opponent_model
    total_timesteps = 10000000
    timesteps_per_iter = 50000
    timesteps = 0
    episode_count = 0

    # Tracking metrics
    timesteps_list = []
    loss_list = []
    reward_mean_list = []
    episode_count_list = []
    winrate_list = []

    while timesteps < total_timesteps:
        print(f"Training for {timesteps_per_iter} timesteps. Total so far: {timesteps}.")
        model.learn(total_timesteps=timesteps_per_iter, reset_num_timesteps=False)
        timesteps += timesteps_per_iter

        # Access the PPO loss and ep_rew_mean
        loss = model.logger.name_to_value.get("train/loss", 0)

        

        # Track loss and ep_rew_mean over timesteps
        timesteps_list.append(timesteps)
        loss_list.append(loss)
        

        # Track data
        episode_count += len(env.episode_results)
        win_rate = sum(env.episode_results) / len(env.episode_results)
        winrate_list.append(win_rate)
        reward_mean = model.rollout_buffer.rewards.mean()
        reward_mean_list.append(reward_mean)
        episode_count_list.append(episode_count)  # Track the episode count

        #plot_metrics(timesteps_list, loss_list, episode_count_list, winrate_list, reward_mean_list)

        if len(env.episode_results) >= 20:
            recent_win_rate = sum(env.episode_results[-20:]) / 20.0
            print(f"Recent win rate over last 20 episodes: {recent_win_rate*100:.1f}%")
            env.episode_results.clear()

            # Display plot every 20 episodes for 30 seconds
            #plt.show(block=False)
            #plt.pause(30)
            #plt.close()

            if recent_win_rate >= 0.80:
                snapshot_file = "snapshot_model.zip"
                model.save(snapshot_file)
                opponent_model = PPO.load(snapshot_file, env=env)
                env.opponent_model = opponent_model
                print("Opponent updated to current model snapshot!")

    model.save("ppo_planet_wars5")
    env.close()


if __name__ == "__main__":
    main()
    plt.savefig("last_fig.png")
    plt.show(block=True)