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
        threshold = np.array([13,13,13,0], dtype = np.float32) #max x,y,z the end effector can reach
        self.observation_space = spaces.Box(low = -threshold, high = threshold, dtype=np.float32)
        self.action_space = spaces.Box(low = np.array([-3.1, -0.954, -0.1214, -0.32]), high = np.array([3.1, 1.03, 1.51, 3.14]), dtype=np.float32)
        self.steps_left = self.MAX_EPISODE
        self.state = [0,0,0,0] #[x_pos, y_pos, z_pos, norm_error]
        self.orientation = [0,0,0,0] #qarternion
        self.pose_target = [8,0,4] #x,y,z
        self.start_simulation()

    def step(self, action):
        p.setJointMotorControl2(self.boxId, 1 , p.POSITION_CONTROL, targetPosition = action[0])
        p.setJointMotorControl2(self.boxId, 2 , p.POSITION_CONTROL, targetPosition = action[1], force= 250_000)
        p.setJointMotorControl2(self.boxId, 3 , p.POSITION_CONTROL, targetPosition = action[2], force= 250_000)
        p.setJointMotorControl2(self.boxId, 4 , p.POSITION_CONTROL, targetPosition = action[3], force= 250_000)

        #Update Simulations
        p.stepSimulation()
        time.sleep(1./240.)

        ## Read Sensors or Link Information
        (linkWorldPosition,
            linkWorldOrientation,
            localInertialFramePosition,
            localInertialFrameOrientation,
            worldLinkFramePosition,
            worldLinkFrameOrientation,
            worldLinkLinearVelocity,
            worldLinkAngularVelocity) = p.getLinkState(self.boxId, 4, computeLinkVelocity=1, computeForwardKinematics=1)

        #Orientation (Coming Soon)

        #Calculate error
        done = bool(self.steps_left<0)
        norm_error = math.inf
        if not done:
            error = np.array(linkWorldPosition) - np.array(self.pose_target)
            norm_error = np.linalg.norm(error)**2
            self.reward = - norm_error
        else:
            reward = -100

        if norm_error <= 1e-05:
            done = True
        else:
            self.steps_left -= 1
        #Update State
        self.state = list(linkWorldPosition) + [norm_error]
        self.act = action
        self.cur_done = done
        return np.array([self.state]),self.reward,done,{}
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
        self.state = [0,0,0,0]
        self.steps_left = self.MAX_EPISODE
        self.act = [0,0,0,0]
        self.cur_done = False
        return np.array([self.state])

    def render(self, mode='human'):
        print(f'State {self.state}, action: {self.act}, done: {self.cur_done}')