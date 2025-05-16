# This script computes the steady state of the model 
import numpy as np
from scipy.optimize import fsolve

class steady:
    def __init__(self):
        self.beta = 0.95 #0.99
        self.gamma = 1 #consumption pref
        self.psi = 1.6
        self.delta = 0.1 #0.025 #depreciation rate
        self.rhoa = 0.9 #AR coff 
        self.alpha = 0.35 #prduction function
        self.var_eps_z = 0.001 #variance of TFP shock
        self.nbz = 11
        self.states = 2 
        self.actions = 2
    
    def equations(self, vars):
        c, n, k = vars
        ls = (1-self.alpha)*(k**self.alpha)*(n**(-self.alpha)) - (self.psi/self.gamma)*(c/(1-n))
        ee = 1 - self.beta*((1-self.delta)+self.alpha*k**(self.alpha-1)*n**(1-self.alpha))  
        kt = self.delta*k - k**self.alpha * n**(1-self.alpha) + c 

        return [ls, ee, kt]
    
    def ss(self):
        initial_guess = [0.5, 0.5, 0.5]
        solution = fsolve(self.equations, initial_guess)
        c_ss, n_ss, k_ss = solution
        y_ss = (k_ss)**self.alpha * (n_ss)**(1-self.alpha)
        u_ss = self.gamma*np.log(c_ss)+self.psi*np.log(1-n_ss)
        
        return c_ss, n_ss, k_ss, y_ss, u_ss
    
        #include a function that computes the value with the number of periods as input 
    def ss_value(self):
        u_ss = self.ss()[-1]
        v_ss = u_ss * (1/(1-ss.beta))
        return v_ss 
    
    """ def get_consumption(self, k, z, n):
        c = (self.gamma/self.psi)*(1-n)*z*(1-self.alpha)*((k/n)**self.alpha)
        return c
    
    def get_n_lb(self):
        n_lb = ((self.gamma/self.psi)*(1-self.alpha))/(1+((self.gamma/self.psi)*(1-self.alpha)))
        return n_lb  """

    def foc_log(self, c0, c1, n0, n1, z0, k0, k1):
        #ls = (1-self.alpha)*(k**self.alpha)*(n**(-self.alpha)) - (self.psi/self.gamma)*(c/(1-n))
        #ee = (self.gamma/c) - self.beta*(self.gamma/c1)*((1-self.delta)+self.alpha*k1**(self.alpha-1)*n1**(1-self.alpha))
        E_z1 = (1-ss.rhoa) + ss.rhoa * z0
        c0_star = (ss.gamma/ss.psi)*(1-n0)*z0*(1-ss.alpha)*((k0/n0)**ss.alpha)
        c_ratio_star = ss.beta*((1 - ss.delta) + E_z1 * ss.alpha * ((k1)**(ss.alpha-1)) * ((n1)**(1-ss.alpha)))
        c_ratio = c1/c0
        labor_gap = np.abs((c0 - c0_star)/c0_star)
        euler_gap = np.abs((c_ratio - c_ratio_star)/c_ratio_star)
        return labor_gap, euler_gap
    
    def get_random_util(self, z, k):
        upper_bound_1 = 1.0
        upper_bound_0 = lambda s0, s1, alpha, a1: s0 * (s1**alpha * a1**(1-alpha))

        rnd_a_1 = np.random.uniform(0.0, upper_bound_1)
        rnd_a_0 = np.random.uniform(0.0, upper_bound_0(z, k, self.alpha, rnd_a_1))
        y = z*(k**self.alpha) * (rnd_a_1**(1-self.alpha))

        U = self.gamma*np.log(rnd_a_0)+self.psi*np.log(1-rnd_a_1)
        k1 = (1-self.delta)*k + y - rnd_a_0
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
        
        Z = np.zeros(N) #grid
        for i in range(N): 
            Z[i] = self.var_eps_z * np.sqrt(2) * x0[i] + 1 #mean = 1, std = var_eps_z

        return Z, Pi 
    
    def gausshermite(self, n):
        x0, w0 = np.polynomial.hermite.hermgauss(n)
        return x0, w0
    


    


ss = steady()
c_ss, n_ss, k_ss, y_ss, u_ss = ss.ss()
    