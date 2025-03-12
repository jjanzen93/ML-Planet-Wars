from stable_baselines3 import PPO
from planet_wars_env import PlanetWarsEnv

def main():
    # Create the training environment without an opponent initially.
    env = PlanetWarsEnv(max_turns=1000, opponent_model=None, visualize=True) #play around with different max_turns if taking too long
    model = PPO("MlpPolicy", env, verbose=1)
    
    opponent_model = PPO.load("ppo_planet_wars1", env=env)
    env.opponent_model = opponent_model
    total_timesteps = 100000
    timesteps_per_iter = 10000
    timesteps = 0

    while timesteps < total_timesteps:
        print(f"Training for {timesteps_per_iter} timesteps. Total so far: {timesteps}.")
        # Train the model for a chunk of timesteps.
        model.learn(total_timesteps=timesteps_per_iter, reset_num_timesteps=False)
        timesteps += timesteps_per_iter

        # Check if we have at least 20 completed episodes.
        if len(env.episode_results) >= 20:
            recent_win_rate = sum(env.episode_results[-20:]) / 20.0
            print(f"Recent win rate over last 20 episodes: {recent_win_rate*100:.1f}%")
            if recent_win_rate >= 0.80: #maybe try >= ? 
                # Save and then load the current model as a snapshot
                snapshot_file = "snapshot_model.zip"
                model.save(snapshot_file)
                opponent_model = PPO.load(snapshot_file, env=env)
                env.opponent_model = opponent_model
                print("Opponent updated to current model snapshot!")
                env.episode_results.clear()

    model.save("ppo_planet_wars3")
    env.close()

if __name__ == "__main__":
    main()
