import gym
import math
import time
import numpy as np
import pybullet as p
from gym import spaces
import pybullet_data

class ExcaBot(gym.Env):
    def __init__(self, sim_active):
        super(ExcaBot, self).__init__()
        self.sim_active = sim_active
        if self.sim_active:
               physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
        else:
            physicsClient = p.connect(p.DIRECT)#or p.DIRECT for non-graphical version

        self.MAX_EPISODE = 50_000
        self.dt = 1.0/240.0
        self.max_theta = [3.1, 1.03, 1.51, 3.14]    
        self.min_theta = [-3.1, -0.954, -0.1214, -0.32]
        self.max_angularVel = [1.0, 1.0, 1.0, 1.0]
        self.min_angularVel = [-1.0, -1.0, -1.0, -1.0]

        self.min_obs = np.array(   
                self.min_theta          +                       # Theta minimum each joint
                self.min_angularVel     +                       # Theta_dot minimum each joint
                [0, 0]                                          # Norm_error, penalty
        )

        self.max_obs = np.array(
                self.max_theta          +                       # Theta maximum each Joint
                self.max_angularVel     +                       # Theta_dot maximum each joint
                [np.finfo(np.float32).max,
                np.finfo(np.float32).max                        # norm_error, penalty maximum (inf)
                ]
        )

        self.max_velocity = np.array(self.max_angularVel, dtype = np.float32)

        self.observation_space = spaces.Box(low =self.min_obs, high = self.max_obs, dtype=np.float32)
        self.action_space = spaces.Box(low = -self.max_velocity, high = self.max_velocity, dtype=np.float32)
        self.steps_left = self.MAX_EPISODE
        self.state = [0,0,0,0,0,0,0,0,0,0] #[theta0, theta1, theta2, theta3, thetadot0, thetadot1, thetadot2, thethadot3, norm_error, penalty]
        self.orientation = [0,0,0,0] #qarternion
        self.theta_target = [1.5,-0.628,-0.125, 0.3] #theta0 = joint1, theta1 = joint2, theta2 = joint3, theta3 = joint4
        self.start_simulation()

    def step(self, action):
        action = np.clip(action, -self.max_velocity, self.max_velocity)
        p.setJointMotorControl2(self.boxId, 1 , p.VELOCITY_CONTROL, targetVelocity = action[0], force= 50_000)
        p.setJointMotorControl2(self.boxId, 2 , p.VELOCITY_CONTROL, targetVelocity = action[1], force= 250_000)
        p.setJointMotorControl2(self.boxId, 3 , p.VELOCITY_CONTROL, targetVelocity = action[2], force= 250_000)
        p.setJointMotorControl2(self.boxId, 4 , p.VELOCITY_CONTROL, targetVelocity = action[3], force= 250_000)

        #Update Simulations
        p.stepSimulation()
        time.sleep(self.dt)

        #Orientation (Coming Soon)

        #Calculate error
        self.theta_now = self._get_joint_state()
        penalty = 0

        if np.any(self.theta_now > self.max_obs[:4]) or np.any(self.theta_now < self.min_obs[:4]):
            less_idx = np.argwhere(self.theta_now < self.min_obs[:4])[:,0]
            more_idx = np.argwhere(self.theta_now > self.max_obs[:4])[:,0]

            if len(less_idx)!=0:
                diff_less = self.normalize01(self.theta_now[less_idx] - self.min_obs[:4][less_idx]).mean()
            else:
                diff_less = 0
            
            if len(more_idx)!=0:
                diff_more = self.normalize01(self.theta_now[more_idx] - self.max_obs[:4][more_idx]).mean()
            else:
                diff_more = 0
            penalty = (diff_less + diff_more)/2

        error = self.theta_now - self.theta_target
        norm_error = self.normalize01(error).mean()
        reward1 = 1 - norm_error
        reward2 = 1 - penalty

        if (reward2 < 0.75):
            done = True

        else:
            done = bool(self.steps_left<0)
        
        if not done:
            self.reward = reward1 + reward2
            self.steps_left -= 1
        
        else:
            self.reward = -100

        #Update State
        self.state = np.concatenate((self.theta_now, np.array(action), np.array([norm_error, penalty])), axis=None)
        
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
        self.state = [0,0,0,0,0,0,0,0,0,0]
        self.theta_now = self._get_joint_state()
        self.steps_left = self.MAX_EPISODE
        self.act = [0,0,0,0]
        self.cur_done = False
        return np.array(self.state)

    def render(self, mode='human'):
        print(f'State {self.state}, action: {self.act}, done: {self.cur_done}')

    def _get_joint_state(self):
        theta0, theta1, theta2, theta3 = p.getJointStates(self.boxId, [1,2,3,4])
        return self.normalize(np.array([theta0[0], theta1[0], theta2[0], theta3[0]]))

    def normalize(self, x):
        return ((x+np.pi)%(2*np.pi)) - np.pi
    
    def normalize01(self, x):
        try:
            xmax = np.max(x)
            xmin = np.min(x)

            result = (x - xmin)/(xmax - xmin)
        except RuntimeWarning:
            result = np.array([0,0,0,0])
        
        return result
