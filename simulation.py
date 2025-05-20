import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import warnings
from steady import steady

class Model(gym.Env):

    def __init__(self, k=0, var_k=0, gamma=0, delta=0, rhoa=0, alpha=0, T=0, noise=0, version=None):
        super().__init__()

        """ self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        )
        self.state = np.zeros(2)
        self.time = 0 """

        #self.ss = steady()
        #self.c_ss, self.k_ss, self.y_ss, self.u_ss, self.v_ss = self.ss.ss()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2, ), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1, ), dtype=np.float32
        )
        self.state = np.zeros(2)
        self.time = 0

        self.T = T
        self.gamma = gamma #consumption pref
        self.delta = delta #depreciation rate
        self.rhoa = rhoa #AR coff 
        self.alpha = alpha #prduction function
        self.k0 = k
        self.var_k = var_k
        self.noise = noise
        self.version = version

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.time = 0
        self.state[0] = 1

        if options=="steady":
            self.state[1] = self.k0
        else:
            self.state[1] = np.random.uniform(low=self.k0*(1-self.var_k), high=self.k0*(1+self.var_k)) 

        obs = np.array(self.state, dtype=np.float32)
        return obs, {'y': 0}
    

    def step(self, action):

        z = self.state[0]
        k = self.state[1]
        s_ratio = action
        y = z* (k**self.alpha)
        y = np.nan_to_num(y, nan=0.0)
        c = y * (1-s_ratio)

        if  c < 0  or y-c < 0:
            U = self.gamma*np.log(c)
            k1 = (1-self.delta)*k + y - c                      
            values = c/y                                       
            warnings.warn(f"Bounds are not working: {values}") 
        else:
            U = self.gamma*np.log(c) 
            k1 = (1-self.delta)*k + y - c  

        if self.version =="deterministic":
            z1 = (1-self.rhoa) + self.rhoa*z  
        else:
            z1 = (1-self.rhoa) + self.rhoa*z + np.random.normal(0, self.noise)

        self.state = np.stack([z1, k1])
        new_state = np.array(self.state, dtype=np.float32)

        self.time += 1

        done = False
        if self.time >= self.T:
            done = True 

        return new_state, U, done, False, {'y': y, 'c': c}

        
    
    



        




    
        
        

    

