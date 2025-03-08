import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces


class Model(gym.Env):

    def __init__(self, k=0, gamma=0, psi=0, delta=0, rhoa=0, alpha=0, T=0):
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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.time = 0
        self.state[0] = 0
        self.state[1] = self.k0
        obs = np.array(self.state, dtype=np.float32)
        return obs, {'y': 0}
    
    def step(self, action):

        #rescale magnitueds coming from NN
        z = self.state[0]
        k = self.state[1]
        c = action[0]
        n = action[1]

        #compute Penalty / reward
        y = np.exp(z)*(k**self.alpha) * (n**(1-self.alpha))
        y = np.nan_to_num(y, nan=0.0)
        if (1-n) < 0 or c < 0 or n < 0 or y-c < 0:
            # return s, torch.tensor(-0.001).float().to(self.device), y/10, True
            U = - (
                    np.maximum(-c, 0)
                    + np.maximum(-n, 0)
                    + np.maximum(n - 1, 0)
                    + np.maximum(c - y, 0)
            )
            new_capital = (1-self.delta)*k
        else:
            investment = y - c
            new_capital = (1-self.delta)*k+investment  # updates Capital level
            
            #U = self.gamma*torch.sqrt(c)+self.psi*torch.sqrt(1-n)
            U = self.gamma*np.log(c)+self.psi*np.log(1-n)

        new_productivity = self.rhoa*z #+ np.random.normal(0, 0.01)  # updates tech.lvl
        #rescale magnitudes to feed into NN
        # new_state = torch.tensor([new_productivity/10, new_capital/100]).float().to(self.device)

        self.state = np.stack([new_productivity, new_capital])
        new_state = np.array(self.state, dtype=np.float32)

        self.time += 1

        done = False
        if self.time >= self.T:
            done = True

        return new_state, U, done, False, {'y': y}





        
    
    



        




    
        
        

    

