import numpy as np
import matplotlib.pyplot as plt


class model: 
    def __init__(self, sigma, rho, alpha, eta, delta, lamda, psi):
        self.sigma = sigma #consumption pref
        self.rho = rho #AR coff 
        self.alpha = alpha #prduction function
        self.eta = eta #work pref 
        self.delta = delta #depreciation rate
        self.lamda = lamda
        self.psi = psi
        self.memory2_list = []

    def reset(self):
        return np.array([0, np.random.normal(0.1, 0.01)]) # np.random.normal(0, 0.1) #initializes model with tech.lvl and Capital.lvl
        
        
    def step(self, s, a):
        #wage = np.exp(s[0])*(1-self.alpha)*(s[1]/a[1])**self.alpha
        #rent = np.exp(s[0])*(self.alpha)*(s[1]/a[1])**(self.alpha-1)
        production = np.exp(s[0]) * (s[1]**self.alpha) * (a[1]**(1-self.alpha))
        investment = production - a[0]
        if investment <= 0:
            new_capital = (1-self.delta)*s[1]
            U = (production**(1-self.sigma)-1)/(1-self.sigma) - (a[1]**(1+self.eta)-1)/(1+self.eta)
        else:
            new_capital = (1-self.delta)*s[1]+investment  # updates Capital level
            U = (a[0]**(1-self.sigma)-1)/(1-self.sigma) - (a[1]**(1+self.eta)-1)/(1+self.eta)  # reward subject to conditions
        new_productivity = self.rho*s[0] + np.random.normal(0, 0.1)  # updates tech.lvl
        new_state = np.array([new_productivity, new_capital])
        return new_state, U/100





        
    
    



        




    
        
        

    

