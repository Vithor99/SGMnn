import numpy as np
import matplotlib.pyplot as plt


class model: 
    def __init__(self, sigma, rho, alpha, delta):
        self.sigma = sigma #consumption pref
        self.rho = rho #AR coff 
        self.alpha = alpha #prduction function
        self.delta = delta #depreciation rate

    def reset(self):
        return np.array([0.1, np.random.normal(0.118, 0.01)]) # state at time zero (NN scaled)
    
    def step(self, s, a):
        #rescale magnitueds coming from NN 
        k = s[1]*1000
        c = a[0]*100
        n = a[1]*10
        #compute reward
        production = (k**self.alpha) * (n**(1-self.alpha))
        if 1-n<=0 or c<=0:
            U = - np.abs(c-1e-6)**self.sigma * np.abs(1-n-1e-6)**(1-self.sigma)
            new_capital = (1-self.delta)*k
        elif c>= production and 1-n>0 and c>0 or n<=0:
            U = - np.abs(c-1e-6)**self.sigma * np.abs(1-n-1e-6)**(1-self.sigma) 
            new_capital = (1-self.delta)*k
        else:
            investment = production - c
            new_capital = (1-self.delta)*k+investment  # updates Capital level
            U = (c**self.sigma) * ((1-n)**(1-self.sigma))  
        #new_productivity = (1-self.rho)*0.1 + self.rho*s[0] + np.random.normal(0, 0.01)  # updates tech.lvl
        #rescale magnitudes to feed into NN
        new_state = np.array([0.1, new_capital/1000])
        return new_state, U/1000, production/100





        
    
    



        




    
        
        

    

