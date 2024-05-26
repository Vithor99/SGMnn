import numpy as np
import matplotlib.pyplot as plt
from simulation import model

simul = model(0.33,0.5) #creating istance of model

for iter in range(10):     #iterating over 10 simulations
    simul.reset()          #reset simulation
    for t in range(100):   #iterating over 100 steps
        c = 0.8*simul.i    #consuming 80% of the income
        simul.step(c)      #updates income and interest rate
        simul.save(c)      #saving income and utility in a list 
        


simul.plot()



