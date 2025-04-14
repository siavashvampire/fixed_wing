import gymnasium
import numpy as np

# import PyFlyt.gym_envs

# env = gymnasium.make(
#   "PyFlyt/Fixedwing-Waypoints-v0",
#   sparse_reward: bool = False,
#   num_targets: int = 4,
#   goal_reach_distance: float = 2.0,
#   flight_dome_size: float = 100.0,
#   max_duration_seconds: float = 120.0,
#   angle_representation: str = "quaternion",
#   agent_hz: int = 30,
#   render_mode: None | str = None,
# )
env = gymnasium.make(
    "PyFlyt/Fixedwing-Waypoints-v3",
    render_mode="human",
)
# print(env.agent_hz)
print(f"Action space: {env.action_space}")
print(f"Sample action: {env.action_space.sample()}")

term, trunc = False, False
obs, _ = env.reset()

while not (term or trunc):
    asd = env.action_space.sample()
    obs, rew, term, trunc, _ = env.step(np.array([0, 0, 0, 0.5]))
    # print(obs)
