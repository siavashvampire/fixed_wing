from math import exp, pi, sqrt

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class UAVLandingBaseEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self):

        # TODO: pitch rate q ??
        # Observation space (5 values): position x, z; pitch angle θ; velocities u, w;
        self.observation_len = 5
        self.observation_space = spaces.Box(
            low=np.array(
                [
                    -np.inf,
                    -np.inf,
                    0,
                    -np.inf,
                    -np.inf,
                ]
            ),
            high=np.array(
                [
                    np.inf,
                    np.inf,
                    100,
                    np.inf,
                    np.inf,
                ]
            ),
            dtype=np.float32,
        )

        # Action space: 49-dimensional (7 values for each of 2 actuators: wing sweep angle Λ, elevator deflection 𝜂)
        self.action_space = spaces.Discrete(49)

    def _get_obs(self):
        return self._agent_state

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Initialize UAV state (position x, z; pitch θ; velocities u, w; pitch rate q ??;)
        self._agent_state = np.array(
            [
                -40,
                -5,
                0,
                0,
                0,
            ]
        )  # Initial state: (𝑥, 𝑧, 𝜃) = (−40, −5, 0)

        self._target_state = np.array(
            [
                0,
                0,
                0,
                0,
                0,
            ]
        )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Action interpretation (convert the action into control inputs for the UAV)
        actuator_input = self._action_to_actuator(action)

        # UAV dynamics simulation (simplified, you should define your own dynamics here)
        observation = self._simulate_dynamics(self._agent_state, actuator_input)

        # Reward function based on final state (landing accuracy, velocity, etc.)
        reward = self._calculate_reward(observation)

        # Check if the episode is done (e.g., landing within target bounds, crash, etc.)
        terminated = self._check_terminated(observation)

        info = self._get_info()

        self._agent_state = observation

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def _action_to_actuator(self, action):
        # Convert action to the control inputs (elevator and wing sweep)
        rate_values = [
            -60,
            -10,
            -5,
            0,
            5,
            10,
            60,
        ]  # The discrete action space as per the paper
        actuator_input = np.array(
            [rate_values[action % 7], rate_values[action // 7]]
        )  # Wing sweep and elevator
        return actuator_input

    def _simulate_dynamics(self, state, actuator_input):
        # Simulate the UAV dynamics
        x, z, theta, u, w = state
        elevator_rate, wing_sweep_rate = actuator_input

        # TODO: how to calculate next state based on the current state and action?

        next_state = np.array(
            [
                x,
                z,
                theta,
                u,
                w,
            ]
        )

        return next_state

    def _calculate_reward(self, observation):
        """
        Calculate the reward based on the final state of the UAV after a landing attempt.
        The reward is computed as the product of Gaussian functions for each relevant state.

        :param observation: The observation of the UAV (x, z, theta, u, w)
        :return: Scalar reward value based on the final state
        """
        # Expected final values for each state (as provided in the paper)
        b = np.array([15, 5, 0.349, 10, 10])  # x, z, θ, u, w

        # Calculate the normalized state vector (c)
        c = observation / b

        # Standard deviation (sigma) for Gaussian function
        sigma = 0.4

        # Gaussian function for each state element
        def gaussian(ci):
            return (1 / (sigma * sqrt(2 * pi))) * exp(-((ci - 0) ** 2) / (2 * sigma**2))

        # Calculate the product of Gaussian functions for all state elements
        reward = 1
        for i in range(self.observation_len):
            reward *= gaussian(c[i])

        return reward

    def _check_terminated(self, observation):
        # Check if the UAV has reached the landing target or if it has crashed
        x, z, theta, _, _ = observation
        terminated = (
            np.abs(x) < 1 and np.abs(z) < 1 and np.abs(theta) < 0.1
        )  # Goal: land within 1m of target with small angle
        return terminated
