import gym
import collections
import math
import numpy as np

from exca_envPosition import ExcaBot
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

SIM_ON = 0

if __name__ == "__main__":
    # env = ExcaBot(SIM_ON)

    # log_path = os.path.join('Training', 'Logs')
    env = DummyVecEnv([lambda: ExcaBot(SIM_ON)])
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=30000)

    # episode = 1
    # for episodes in range(1, episode+1):
    #     obs = env.reset()
    #     done = False
    #     score = 0

    #     while not done:
    #         env.render()
    #         action = env.action_space.sample()
    #         obs, reward, done, info = env.step(action)
    #         score += reward

    # print(f'Episode: {episodes} Score: {score}')
 
