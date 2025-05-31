# This script computes the steady state of the model 
import numpy as np
from scipy.optimize import fsolve
# %%
class steady:
    def __init__(self):
        self.beta = 0.97   #0.99 #0.97
        self.delta =  0.01 #0.03 #0.01 #depreciation rate
        self.states = 2    #4

        self.gamma = 1        #consumption pref
        self.rhoa = 0.1       #AR coff 
        self.alpha = 0.35     #prduction function
        self.dev_eps_z = 0.007 #std dev of TFP shock 
        self.actions = 1
        self.nbz = 11 #dimension of the quadrature
    
    def equations(self, vars):
        c, k = vars
        ee = 1 - self.beta*((1-self.delta)+self.alpha*k**(self.alpha-1))  
        kt = self.delta*k - k**self.alpha + c 
        return [ee, kt]
    
    def ss(self):
        initial_guess = [0.5, 0.5]
        solution = fsolve(self.equations, initial_guess)
        c_ss, k_ss = solution
        y_ss = (k_ss)**self.alpha
        u_ss = self.gamma*np.log(c_ss)
        v_ss = (1/(1-self.beta))*u_ss
        
        return c_ss, k_ss, y_ss, u_ss, v_ss 
    
    def ss_value(self, T):
        c_ss, k_ss, y_ss, u_ss, v_ss = self.ss()
        v_ss = 0
        for t in range(T):
            v_ss += (self.beta**t) * u_ss

        return v_ss 
    
    def foc_log(self, c0, c1, z0, k1):
        E_z1 = (1-self.rhoa) + self.rhoa * z0
        c_ratio_star = self.beta*((1 - self.delta) + E_z1 * self.alpha * ((k1)**(self.alpha-1)) )
        c_ratio = c1/c0
        euler_gap = np.abs((c_ratio - c_ratio_star)/c_ratio_star)
        return euler_gap

    def get_random_util(self, z, k):
        rnd_a = np.random.uniform(0.0, 1.0)
        y = z*(k**self.alpha)
        c = (1 - rnd_a)*y
        U = self.gamma*np.log(c) 
        k1 = (1-self.delta)*k + y - c
        return U, k1
    
    def tauchenhussey(self, N):
        x0, w0 = self.gausshermite(N)
        tau = np.zeros(N)
        for i in range(N):
            tau_i = 0
            for j in range(N):
                tau_i += w0[j] * np.exp(x0[j]**2 - (x0[j] - self.rhoa * x0[i])**2 ) 
            tau[i] = (1/np.sqrt(np.pi)) * tau_i 
        
        Pi = np.zeros((N, N)) #transition matrix
        for i in range(N):
            for j in range(N):
                Pi[i, j] = ((1/np.sqrt(np.pi)) * w0[j] * np.exp(x0[j]**2 - (x0[j] - self.rhoa * x0[i])**2 ))/tau[i]

        for i in range(10):
            Pi = Pi / Pi.sum(axis=1, keepdims=True)
        
        Z = np.zeros(N) #grid
        for i in range(N): 
            Z[i] = self.dev_eps_z * np.sqrt(2) * x0[i] + 1 #mean = 1, std = dev_eps_z

        return Z, Pi
    
    def tauchenhussey_local(self, N, z):
        x0, w0 = self.gausshermite(N)
        #tau = np.zeros(N)
        x =  ((z - 1) / (self.dev_eps_z * np.sqrt(2)))
        
        tau_i = 0
        for j in range(N):
            tau_i += w0[j] * np.exp(x0[j]**2 - (x0[j] - self.rhoa * x)**2 ) 
        tau = (1/np.sqrt(np.pi)) * tau_i 
        
        Pi = np.zeros(N) #transition matrix
        
        for j in range(N):
            Pi[j] = ((1/np.sqrt(np.pi)) * w0[j] * np.exp(x0[j]**2 - (x0[j] - self.rhoa * x)**2 ))/tau

        for i in range(10):
            Pi = Pi / Pi.sum(axis=0, keepdims=True)
        
        Z = np.zeros(N) #grid
        for i in range(N): 
            Z[i] = self.dev_eps_z * np.sqrt(2) * x0[i] + 1 #mean = 1, std = dev_eps_z

        return Z, Pi
    
    def gausshermite(self, n):
        x0, w0 = np.polynomial.hermite.hermgauss(n)
        return x0, w0
    

    
#ss = steady()
#c_ss, k_ss, y_ss, u_ss, v_ss = ss.ss()
#r = (1/ss.beta) + ss.delta - 1
#print(r*100)