import numpy as np
import matplotlib.pyplot as plt

#Model: Agent deciding to either save or consume.
#Next period agent receives what he saved plus an interest, determined as an AR(1) process. 

class model: 
    def __init__(self, sigma, rho):
        self.sigma = sigma #fixed parameter
        self.rho = rho #fixed parameter

    def reset(self):
        # self.r = np.random.uniform(0, 0.1)
        # self.i = np.random.normal(100,10)
        return np.array([np.random.uniform(0, 0.1), np.random.normal(100,10)])
        # self.Ulist = []
        # self.ilist = []
        
    def step(self, s, c):
        s[0] = self.rho*s[0]+np.random.uniform(0, 0.1)
        s[1] = (s[1] - c)*(1+s[0])
        U = -1 if s[1] < c else (c / self.sigma) ** (self.sigma)
        return s, U
        # self.r = self.rho*self.r+np.random.uniform(0, 0.1) #Transition function for r
        # self.i = (self.i - c)*(1+self.r)                   #Transition function for i
    
    # def save(self, c):
    #     self.U = (c/self.sigma)**(self.sigma)
    #     self.Ulist.append(self.U)
    #     self.ilist.append(self.i)

    # def plot(self):
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(self.Ulist, marker='o')
    #     plt.title('Utility')
    #     plt.xlabel('time')
    #     plt.ylabel('Value')
    #     plt.grid(True)
    #     plt.show()
    #
    #     plt.figure(figsize=(10, 5))
    #     plt.plot(self.ilist, marker='o')
    #     plt.title('Income')
    #     plt.xlabel('time')
    #     plt.ylabel('Value')
    #     plt.grid(True)
    #     plt.show()



        




    
        
        

    

