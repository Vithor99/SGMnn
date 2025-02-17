import numpy as np
import matplotlib.pyplot as plt

#Log Utility Model
class model:

    def __init__(self, gamma, psi, delta, rhoa, alpha):
        self.gamma = gamma #consumption pref
        self.psi = psi
        self.delta = delta #depreciation rate
        self.rhoa = rhoa #AR coff 
        self.alpha = alpha #prduction function

    def reset(self):
        return np.array([0.0, 0.14836]) # state at time zero (NN scaled)
    
    def step(self, s, a):
        #rescale magnitueds coming from NN
        z = s[0]*10 
        k = s[1]*100
        c = a[0]*10
        n = a[1]*10
        #compute Penalty / reward
        y = np.exp(z)*(k**self.alpha) * (n**(1-self.alpha))
        y = np.nan_to_num(y, nan=0.0)
        if (1-n) <= 0 or c <= 0 or n <= 0 or 0 >= y-c:
            U = - (max(0, -c) + max(0, -n) + max(0, n-1) + max(0, c-y))
            new_capital = (1-self.delta)*k
        else:
            investment = y - c
            new_capital = (1-self.delta)*k+investment  # updates Capital level
            U = self.gamma*np.sqrt(c)+self.psi*np.sqrt(1-n)

        new_productivity = self.rhoa*z #+ np.random.normal(0, 0.01)  # updates tech.lvl
        #rescale magnitudes to feed into NN
        new_state = np.array([new_productivity/10, new_capital/100])
        return new_state, U/1000, y/10





        
    
    



        




    
        
        

    

