import gym
import collections
import math
import numpy as np
import os

from exca_envVelocity3 import ExcaBot
from stable_baselines3 import PPO

SIM_ON = 0

if __name__ == "__main__":
    env = ExcaBot(SIM_ON)
    model = PPO.load('Training/Saved Models/PPO_5000000(16)', env=env)
    obs = env.reset()
    score = 0
    done = False
    step = 0
    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        print(env.norm_error)
        score += reward
        step+=1
    print(f"Score: {score}, with step: {step}")