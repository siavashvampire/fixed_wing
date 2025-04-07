from gymnasium.envs.registration import register

register(
    id="fixedwingenv/GridWorld-v0",
    entry_point="fixedwingenv.envs:GridWorldEnv",
)
