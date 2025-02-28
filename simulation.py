import numpy as np
import matplotlib.pyplot as plt
import torch

#Log Utility Model
class model:

    def __init__(self, gamma, psi, delta, rhoa, alpha, device=None):
        self.gamma = gamma #consumption pref
        self.psi = psi
        self.delta = delta #depreciation rate
        self.rhoa = rhoa #AR coff 
        self.alpha = alpha #prduction function

        self.device = device


    def reset(self):
        return torch.tensor([0.0, 0.14836]).float().to(self.device)  # state at time zero (NN scaled)
    
    def step(self, s, a):

        #rescale magnitueds coming from NN
        z = s[0]*10 
        k = s[1]*100
        c = a[0]*10
        n = a[1]*10

        #compute Penalty / reward
        y = torch.exp(z)*(k**self.alpha) * (n**(1-self.alpha))
        y = torch.nan_to_num(y, nan=0.0)
        if (1-n) < 0 or c < 0 or n < 0 or y-c < 0:
            # return s, torch.tensor(-0.001).float().to(self.device), y/10, True
            U = - (torch.clamp(-c, min=0) + torch.clamp(-n, min=0) + torch.clamp(n-1, min=0) + torch.clamp(c - y, min=0))
            new_capital = (1-self.delta)*k
        else:
            investment = y - c
            new_capital = (1-self.delta)*k+investment  # updates Capital level
            U = self.gamma*torch.sqrt(c)+self.psi*torch.sqrt(1-n)

        new_productivity = self.rhoa*z #+ np.random.normal(0, 0.01)  # updates tech.lvl
        #rescale magnitudes to feed into NN
        # new_state = torch.tensor([new_productivity/10, new_capital/100]).float().to(self.device)
        new_state = torch.stack([new_productivity / 10, new_capital / 100])


        return new_state, U/1000, y/10, False





        
    
    



        




    
        
        

    

