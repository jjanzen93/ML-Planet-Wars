from math import tanh
from stable_baselines3 import PPO
from planet_wars_env import PlanetWarsEnv
import matplotlib.pyplot as plt
import time
from opponent_bots import defensive_bot, aggressive_bot, easy_bot, production_bot
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
import config
import torch
from datetime import datetime
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
        return True


#TODO :
# 1. implement training reg
# 2. Implement custom map gen: #planets, starting ratios
# 3. maybe port to rllib???
# 4. Start collecting final demos


def main():
    training_reg = [easy_bot.Easy_Bot(), production_bot.Production_Bot(), aggressive_bot.Aggressive_Bot(), "self"]
    env = PlanetWarsEnv(max_turns=1000, opponent_model=training_reg[0], map_size=11, single_map=True, visualize=True)
    
    
    policy_kwargs = dict(activation_fn=torch.nn.Tanh, net_arch=dict(pi=[256]*10, vf=[256]*10))
    model = PPO("MlpPolicy", env, verbose=0, gamma=0.9995, ent_coef=0.01, policy_kwargs=policy_kwargs, device="cuda", tensorboard_log=f"./ppo_multiagent_tensorboard/P{env.map_size}_{datetime.now().strftime("%Y-%m-%d %H_%M_%S")}")    

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=360),
        )

    #model = MaskablePPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, gamma = 0.9995,device = "cpu", tensorboard_log="./ppo_5_tensorboard/")
    print(model.policy)

    #model.save("5_ppo_untrained")
    #opponent_model = MaskablePPO.load("5_ppo_untrained", env=env, gamma = 0.9995,device = "cpu")
    #opponent_model = aggressive_bot.Aggressive_Bot()
    #env.opponent_model = opponent_model


    total_timesteps = 1_000_000_000
    timesteps_per_iter = 25000
    timesteps = 0


    i = 0
    while timesteps < total_timesteps:
        print(f"Training for {timesteps_per_iter} timesteps. Total so far: {timesteps}.")
        print(f"Training against {env.opponent_model}")
        model.learn(total_timesteps=timesteps_per_iter, reset_num_timesteps=False, callback=TensorboardCallback())
        timesteps += timesteps_per_iter

        if len(env.episode_results) >= 20:
            recent_win_rate = sum(env.episode_results[-20:]) / 20.0
            print(f"Recent win rate over last 20 episodes: {recent_win_rate*100:.1f}%")
            env.episode_results.clear()

            if recent_win_rate >= 0.80:
                i+=1
                snapshot_file = f"snapshot{i}_model.zip"
                model.save(snapshot_file)
                env.opponent_model = training_reg[i]

                config.updated += 1
                print(f"Opponent updated to {env.opponent_model}!")
            elif recent_win_rate < 0.25 and i != 0:
                print("Catastrophic forgetting!\nPlayer updated to last snapshot!")
                model = MaskablePPO.load(snapshot_file, env=env, gamma = 0.9995,device = "cpu")
                env.opponent_model = training_reg[i]


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