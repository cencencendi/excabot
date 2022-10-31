from cv2 import norm
import gym
import collections
import math
import time
import numpy as np
import pybullet as p
from gym import spaces
import pybullet_data

class ExcaBot(gym.Env):
    def __init__(self, sim_active) :
        super(ExcaBot, self).__init__()
        self.sim_active = sim_active
        if self.sim_active:
               physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
        else:
            physicsClient = p.connect(p.DIRECT)#or p.DIRECT for non-graphical version

        self.MAX_EPISODE = 10_000_000
        self.theta_now = np.array([0.0,0.0,0.0,0.0], dtype = np.float32)
        self.dt = 1.0/240.0
        # threshold = np.array([13,13,13,0], dtype = np.float32) #max x,y,z the end effector can reach
        self.min_obs = np.array([-3.1, -0.954, -0.1214, -0.32, -0.5, -0.1, -0.1, -0.1, -1000], dtype = np.float32)
        self.max_obs = np.array([3.1, 1.03, 1.51, 3.14, 0.5, 0.1, 0.1, 0.1, 0], dtype = np.float32)
        self.max_velocity = np.array([0.5, 0.1, 0.1, 0.1], dtype = np.float32)

        self.observation_space = spaces.Box(low =self.min_obs, high = self.max_obs, dtype=np.float32)
        self.action_space = spaces.Box(low = -self.max_velocity, high = self.max_velocity, dtype=np.float32)
        self.steps_left = self.MAX_EPISODE
        self.state = [0,0,0,0,0,0,0,0,0] #[theta0, theta1, theta2, theta3, thetadot0, thetadot1, thetadot2, thethadot3, norm_error]
        self.orientation = [0,0,0,0] #qarternion
        self.theta_target = [1.5,-0.628,-0.1214, -0.32] #theta0 = joint1, theta1 = joint2, theta2 = joint3, theta3 = joint4
        self.start_simulation()

    def step(self, action):
        action = np.clip(action, -self.max_velocity, self.max_velocity)
        p.setJointMotorControl2(self.boxId, 1 , p.VELOCITY_CONTROL, targetVelocity = action[0], force= 250_000)
        p.setJointMotorControl2(self.boxId, 2 , p.VELOCITY_CONTROL, targetVelocity = action[1], force= 250_000)
        p.setJointMotorControl2(self.boxId, 3 , p.VELOCITY_CONTROL, targetVelocity = action[2], force= 250_000)
        p.setJointMotorControl2(self.boxId, 4 , p.VELOCITY_CONTROL, targetVelocity = action[3], force= 250_000)

        #Update Simulations
        p.stepSimulation()
        time.sleep(self.dt)

        #Orientation (Coming Soon)

        #Calculate error

        self.theta_now = self.theta_now + np.array(action)*self.dt

        done = bool(self.steps_left<0)
        norm_error = math.inf
        if not done:
            error = self.theta_now - np.array(self.theta_target)
            norm_error = np.linalg.norm(error)**2
            self.reward = - (norm_error + 0.1*action[0]**2 + 0.1*action[1]**2 + 0.1*action[2]**2 + 0.1*action[3]**2)
            self.steps_left -= 1
        # else:
        #     reward = -1000

        if norm_error <= 1e-05:
            done = True            

        #Update State
        self.state = np.concatenate((self.theta_now, np.array(action), np.array([norm_error])), axis=None)
        
        self.act = action
        self.cur_done = done
        return self.state,self.reward,done,{}
    def start_simulation(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally

        ## Setup Physics
        p.setGravity(0,0,-9.8)

        ## Load Plane
        planeId = p.loadURDF("plane.urdf")

        ## Load Robot
        startPos = [self.state[0],self.state[1],1.4054411813121799]
        startOrientation = p.getQuaternionFromEuler([0,0,0])
        self.boxId = p.loadURDF("aba_excavator/excavator.urdf",startPos, startOrientation)

    def reset(self):
        p.resetSimulation()
        self.start_simulation()
        self.state = [0,0,0,0,0,0,0,0,0]
        self.theta_now = np.array([0.0,0.0,0.0,0.0], dtype = np.float32)
        self.steps_left = self.MAX_EPISODE
        self.act = [0,0,0,0]
        self.cur_done = False
        return np.array([self.state])

    def render(self, mode='human'):
        print(f'State {self.state}, action: {self.act}, done: {self.cur_done}')