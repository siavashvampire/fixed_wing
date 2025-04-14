import numpy as np
from stable_baselines3 import PPO

from apps.RL_logging.RL_logging import close_tensorboard, open_log_web


class BaseLearning:
    def __init__(self, env):

        # Define the PPO model with a MLP policy (2 layers, 64 nodes per layer)
        self.model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            policy_kwargs=dict(net_arch=[64, 64]),
            tensorboard_log="logs/fixedwing/",
        )

    def learn(self, max_iter: int = np.inf):
        self.driver = open_log_web()

        # Train the model
        self.model.learn(total_timesteps=max_iter)

        # Save the trained model
        self.model.save("results/ppo_uav_landing")

        self.driver.close()
        close_tensorboard()
