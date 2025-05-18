import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
import warnings
from steady import steady

class Model(gym.Env):

    def __init__(self, k=0, var_k=0, gamma=0, psi=0, delta=0, rhoa=0, alpha=0, T=0, noise=0, u_ss = 0, version=None):
        super().__init__()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        )
        self.state = np.zeros(2)
        self.time = 0

        self.T = T
        self.gamma = gamma #consumption pref
        self.psi = psi
        self.delta = delta #depreciation rate
        self.rhoa = rhoa #AR coff 
        self.alpha = alpha #prduction function
        self.k0 = k
        self.var_k = var_k
        self.noise = noise
        self.u_ss = u_ss
        self.version = version

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.time = 0
        self.state[0] = 1

        if options == "steady":
            self.state[1] = self.k0
        else:
            self.state[1] = np.random.uniform(low=self.k0*(1-self.var_k), high=self.k0*(1+self.var_k)) 

        obs = np.array(self.state, dtype=np.float32)
        return obs, {'y': 0}
    

    def step(self, action):

        z = self.state[0]
        k = self.state[1]
        #c = action[0]
        c_ratio = action[0]                                                                        # added
        n = action[1]
        
        #compute Penalty / reward
        y = z * (k**self.alpha) * (n**(1-self.alpha))
        y = np.nan_to_num(y, nan=0.0)
        c = c_ratio*y                                                                               # added

        """ if (1-n) < 0 or c < 0 or n < 0 or y-c < 0:
            U = self.gamma*np.log(c)+self.psi*np.log(1-n)
            k1 = (1-self.delta)*k + y - c                    #updates Capital level
            values = c/y                                       #debugging
            warnings.warn(f"Bounds are not working: {values}") #debugging
        else:
            U = self.gamma*np.log(c) + self.psi*np.log(1-n)
            k1 = (1-self.delta)*k + y - c  # updates Capital level """
        
        U = (self.gamma*np.log(c) + self.psi*np.log(1-n) - self.u_ss)/np.abs(self.u_ss) #additional welfare created by RL policy. 
        k1 = (1-self.delta)*k + y - c                                                   #updates Capital level

        if self.version =="deterministic":
            z1 = (1-self.rhoa) + self.rhoa*z  # updates tech.lvl
        else:
            z1 = (1-self.rhoa) + self.rhoa*z + np.random.normal(0, self.noise)

        self.state = np.stack([z1, k1])
        new_state = np.array(self.state, dtype=np.float32)

        self.time += 1

        done = False
        if self.time >= self.T:
            done = True

        return new_state, U, done, False, {'y': y, 'c': c}




        
    
    



        




    
        
        

    

