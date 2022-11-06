import gym
import collections
import math
import numpy as np
import os

from exca_envVelocity2 import ExcaBot
from stable_baselines3 import PPO

SIM_ON = 1

if __name__ == "__main__":
    env = ExcaBot(SIM_ON)
    model = PPO.load('Training/Saved Models/PPO_5000000(14)', env=env)
    obs = env.reset()
    score = 0
    while True:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        score += reward
    print(score)