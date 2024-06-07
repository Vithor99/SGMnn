import numpy as np
import matplotlib.pyplot as plt


class model: 
    def __init__(self, sigma, rho):
        self.sigma = sigma #fixed parameter
        self.rho = rho #fixed parameter

    def reset(self):
        return np.array([np.random.uniform(0, 0.1), np.random.normal(100,10)])
        
        
    def step(self, s, c):
        s[0] = self.rho*s[0]+np.random.uniform(0, 0.1)
        s[1] = (s[1] - c)*(1+s[0])
        U = -1 if s[1] < c else (c / self.sigma) ** (self.sigma) #subject to conditions
        return s, U
        
    
    



        




    
        
        

    

