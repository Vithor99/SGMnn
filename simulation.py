import numpy as np
import matplotlib.pyplot as plt

#Model: Agent decideing to either save or consume. 
#Next period agent receives what he saved plus an interest, determined as an AR(1) process. 

class model: 
    def __init__(self, sigma, rho):
        self.sigma = sigma #fixed parameter
        self.rho = rho #fixed parameter

    def reset(self):
        self.r = np.random.uniform(0, 0.1)
        self.i = np.random.normal(100,10)
        
    def step(self, c):
        self.r = self.rho*self.r+np.random.uniform(0, 0.1) #Transition function for r
        self.i = (self.i - c)*(1+self.r)                   #Transition function for i


        




    
        
        

    

