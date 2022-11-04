import gym
import collections
import math
import numpy as np
import os
import time

from exca_envVelocity2 import ExcaBot


SIM_ON = 1

if __name__ == "__main__":
    env = ExcaBot(SIM_ON)

    episode = 1
    for i in range(1,episode+1):
        obs = env.reset()[0]
        done = False
        score = 0

        for i in range(1000): #while not done:
            env.render()
            action= env.action_space.sample()
            obs, reward, done, _ = env.step(action)
            # print(env._get_joint_state())
            # time.sleep(0.5)
            score += reward
        print(f"Score: {score}")