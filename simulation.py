import numpy as np
import matplotlib.pyplot as plt


class model: 
    def __init__(self, sigma, rho, alpha, eta, delta):
        self.sigma = sigma #consumption pref
        self.rho = rho #AR coff 
        self.alpha = alpha #prduction function
        self.eta = eta #work pref 
        self.delta = delta #depreciation rate

    def reset(self):
        return np.array([np.random.uniform(0, 0.1), np.random.normal(100,10)]) #initializes model with tech.lvl and Capital.lvl
        
        
    def step(self, s, a):
        wage = s[0]*(1-self.alpha)*(s[1]/a[1])**self.alpha
        rent = s[0]*(self.alpha)*(s[1]/a[1])**(self.alpha-1)
        s[0] = self.rho*s[0]+np.random.uniform(0, 0.1) #updates tech.lvl
        s[1] = (1-self.delta)*s[1]+a[2] #updates Capital level
        U = -1000 if a[0]+a[2]>wage*a[1]+rent*s[1] else  (a[0]**(1-self.sigma))/(1-self.sigma) - (a[1]**(1+self.eta))/(1+self.eta) #reward subject to conditions
        return s, U
        
    
    



        




    
        
        

    

