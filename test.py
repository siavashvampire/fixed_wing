from apps.envs.base import UAVLandingBaseEnv
from apps.models.base.learn import BaseLearning

# Create the custom UAV environment
env = UAVLandingBaseEnv()
base_learning = BaseLearning(env)
base_learning.learn(1000)
