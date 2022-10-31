import gym
import collections
import math
import numpy as np

from exca_envVelocity import ExcaBot
from stable_baselines3 import SAC

SIM_ON = 0

if __name__ == "__main__":
    env = ExcaBot(SIM_ON)

    # log_path = os.path.join('Training', 'Logs')
    # env = DummyVecEnv([lambda: ExcaBot(SIM_ON)])
    model = SAC('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=3000)