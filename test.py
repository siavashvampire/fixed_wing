from stable_baselines3 import PPO
from app.envs.fixed_wing import UAVLandingEnv

# Create the custom UAV environment
env = UAVLandingEnv()

# Define the PPO model with a MLP policy (2 layers, 64 nodes per layer)
model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=dict(net_arch=[64, 64]))

# Train the model
model.learn(total_timesteps=100000)

# Save the trained model
model.save("ppo_uav_landing")

# To use the trained model later, you can load it as:
# model = PPO.load("ppo_uav_landing")
