from stable_baselines3 import PPO

from apps.envs.base import UAVLandingBaseEnv

env = UAVLandingBaseEnv()

model = PPO.load("results/ppo_uav_landing")

obs, info = env.reset()
terminated = False

while not terminated:
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Reward: {reward}, State: {obs}")
