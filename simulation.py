import numpy as np
import matplotlib.pyplot as plt


class model: 
    def __init__(self, sigma, rho, alpha, eta, delta, psi):
        self.sigma = sigma #consumption pref
        self.rho = rho #AR coff 
        self.alpha = alpha #prduction function
        self.eta = eta #work pref 
        self.delta = delta #depreciation rate
        self.psi = psi
        self.memory2_list = []

    def reset(self):
        return np.array([0.1, np.random.normal(0.1, 0.01)]) # np.random.normal(0, 0.1) #initializes model with tech.lvl and Capital.lvl
        
        
    def step(self, s, a):
        production = (s[1]**self.alpha) * (a[1]**(1-self.alpha))
        if 1-a[1]<=0 or a[0]<=0:
            U = - np.abs(a[0]-1e-6)**self.sigma * np.abs(1-a[1]-1e-6)**(1-self.sigma)
            new_capital = (1-self.delta)*s[1]
        elif a[0]>= production and 1-a[1]>0 and a[0]>0 or a[1]<=0:
            U = - np.abs(a[0]-1e-6)**self.sigma * np.abs(1-a[1]-1e-6)**(1-self.sigma) 
            new_capital = (1-self.delta)*s[1]
        else:
            investment = production - a[0]
            new_capital = (1-self.delta)*s[1]+investment  # updates Capital level
            U = (a[0]**self.sigma) * ((1-a[1])**(1-self.sigma))  
        #new_productivity = (1-self.rho)*0.1 + self.rho*s[0] + np.random.normal(0, 0.01)  # updates tech.lvl
        new_state = np.array([0.1, new_capital])
        return new_state, U/1000





        
    
    



        




    
        
        

    

