from math import tanh
from stable_baselines3 import PPO
from planet_wars_env import PlanetWarsEnv
import matplotlib.pyplot as plt
import time
from opponent_bots import defensive_bot
from opponent_bots import aggressive_bot
from opponent_bots import easy_bot
from opponent_bots import production_bot
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
import config

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


from stable_baselines3.common.callbacks import BaseCallback



class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        self.logger.record("Winrate over the last 10 games", sum(config.recent_wr) / 10)
        self.logger.record("Model updates", config.updated)
        config.updated = 0
        return True




def main():
    env = PlanetWarsEnv(max_turns=1000, opponent_model=None, visualize=True)


    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=360),
    )

    model = MaskablePPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, gamma = 0.9995,device = "cpu", tensorboard_log="./ppo_5_tensorboard/")
    model.save("5_ppo_untrained")
    opponent_model = MaskablePPO.load("5_ppo_untrained", env=env, gamma = 0.9995,device = "cpu")
    #opponent_model = aggressive_bot.Aggressive_Bot()
    env.opponent_model = opponent_model

    """snapshot_file = "snapshot_model.zip"
    model = MaskablePPO.load(snapshot_file, env=env, gamma = 0.9995,device = "cpu")
    opponent_model = MaskablePPO.load(snapshot_file, env=env, gamma = 0.9995,device = "cpu")
    env.opponent_model = opponent_model"""

    total_timesteps = 1000000000000
    timesteps_per_iter = 25000
    timesteps = 0
    episode_count = 0

    # Tracking metrics
    timesteps_list = []
    loss_list = []
    reward_mean_list = []
    episode_count_list = []
    winrate_list = []

    i = 0
    while timesteps < total_timesteps:
        print(f"Training for {timesteps_per_iter} timesteps. Total so far: {timesteps}.")
        model.learn(total_timesteps=timesteps_per_iter, reset_num_timesteps=False, callback=TensorboardCallback())
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
            print(f"Recent win rate over last 50 episodes: {recent_win_rate*100:.1f}%")
            env.episode_results.clear()

            # Display plot every 20 episodes for 30 seconds
            #plt.show(block=False)
            #plt.pause(30)
            #plt.close()
            
            #model.save(f"{1}_ppo5save.zip")
            if recent_win_rate >= 0.80:
                i+=1
                snapshot_file = f"snapshot{i}_model.zip"
                model.save(snapshot_file)
                opponent_model = MaskablePPO.load(snapshot_file, env=env, gamma = 0.9995,device = "cpu")
                env.opponent_model = opponent_model
                config.updated = 1
                print("Opponent updated to current model snapshot!")
            elif recent_win_rate < 0.30 and i != 0:
                print("Catastrophic forgetting!\nPlayer updated to last snapshot!")
                model = MaskablePPO.load(snapshot_file, env=env, gamma = 0.9995,device = "cpu")
                env.opponent_model = opponent_model
                config.updated = -1


    model.save("ppo_planet_wars5")
    env.close()



import torch as th
import torch.nn as nn
from gymnasium import spaces

#from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Linear(n_input_channels, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh()
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))



if __name__ == "__main__":
    main()
    plt.savefig("last_fig.png")
    plt.show(block=True)