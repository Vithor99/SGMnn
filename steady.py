# This script computes the steady state of the model 
import numpy as np
from scipy.optimize import fsolve

class steady:
    def __init__(self):
        self.beta = 0.99
        self.gamma = 1 #consumption pref
        self.psi = 1.6
        self.delta = 0.025 #depreciation rate
        self.rhoa = 0.9 #AR coff 
        self.alpha = 0.35 #prduction function

        self.c = 0
        self.n = 0
        self.k = 0
        self.y = 0
        self.u = 0
        self.v = 0
    
    def equations(self, vars):
        c, n, k = vars
        ls = (1-self.alpha)*(k**self.alpha)*(n**(-self.alpha)) - (self.psi/self.gamma)*(np.sqrt(c)/np.sqrt(1-n))
        ee = 1 - self.beta*((1-self.delta)+self.alpha*k**(self.alpha-1)*n**(1-self.alpha))  
        kt = self.delta*k - k**self.alpha * n**(1-self.alpha) + c 
        return [ls, ee, kt]
    
    def ss(self):
        initial_guess = [0.5, 0.5, 0.5]  
        solution = fsolve(self.equations, initial_guess)
        c_ss, n_ss, k_ss = solution
        y_ss = (k_ss)**self.alpha * (n_ss)**(1-self.alpha)
        u_ss = self.gamma*np.sqrt(c_ss)+self.psi*np.sqrt(1-n_ss)
        v_ss = 0
        for t in range(1000):
            v_ss += self.beta**t * u_ss

        self.c = c_ss
        self.n = n_ss
        self.k = k_ss 
        self.y = y_ss
        self.u = u_ss
        self.v = v_ss
        return c_ss, n_ss, k_ss, y_ss, u_ss, v_ss
    
    def foc(self, c, c1, n, n1, k, k1):
        ls = (1-self.alpha)*(k**self.alpha)*(n**(-self.alpha)) - (self.psi/self.gamma)*(np.sqrt(c)/np.sqrt(1-n))
        ee = (self.gamma/np.sqrt(c)) - self.beta*(self.gamma/np.sqrt(c1))*((1-self.delta)+self.alpha*k1**(self.alpha-1)*n1**(1-self.alpha)) 
        return ls, ee
    


