import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

## Solve steady of the model 
sigma = 0.5
alpha = 0.5 
delta = 0.02
beta = 0.99

def equations(vars):
    c, n, k = vars
    ls = (c/(1-n)) - (sigma/(1-sigma))*(1-alpha)*(k/n)**alpha
    ee = 1 - beta*((1-delta)+alpha*k**(alpha-1)*n**(1-alpha))  
    kt = delta*k - k**alpha * n**(1-alpha) + c 
    return [ls, ee, kt]

initial_guess = [0.1, 0.1, 0.1]  
solution = fsolve(equations, initial_guess)
c_ss, n_ss, k_ss = solution
y_ss = (k_ss)**alpha * (n_ss)**(1-alpha)
u_ss = (c_ss)**sigma * (n_ss)**(1-sigma)

#rescale to match memory and NN magnitudes 
c_ss = c_ss/100
n_ss = n_ss/10
k_ss = k_ss/1000
y_ss = y_ss/100
u_ss = u_ss/1000
print(f"Solution: c = {c_ss}, n = {n_ss}, k = {k_ss}, y={y_ss}, u={u_ss}")




#open last simulation
with open("last_sim.pkl", "rb") as f:
    data = pickle.load(f)
 
st = [entry ['st'] for entry in data]
a = [entry ['a'] for entry in data] 
u = [entry ['u'] for entry in data]
y = [entry ['y'] for entry in data]
k = [pair[1] for pair in st]
z = [pair[0] for pair in st]
z = np.exp(z)
c = [pair[0] for pair in a]
n = [pair[1] for pair in a]


var_list = ['k','c','n','y','u']
for var in var_list:
    plt.plot(globals()[var], label=var)
    plt.axhline(np.mean(globals()[var]), color='red')
    plt.axhline(globals()[f"{var}_ss"], color="green")
    plt.title(f"{var}")
    plt.show()


# distance from FOC 
Euler = []
Lab_supply = []
for i in range(998):
    ls = (c[i]/(1-n[i])) - (sigma/(1-sigma))*(1-alpha)*(k[i]/n[i])**alpha
    ee = c[i]**(sigma-1)*(1-n[i])**(1-sigma) - beta*c[i+1]**(sigma-1)*(1-n[i+1])**(1-sigma)*((1-delta)+alpha*k[i+1]**(alpha-1)*n[i+1]**(1-alpha))     
    Euler.append(ee)
    Lab_supply.append(ls)
plt.plot(Euler)
plt.title('delta Euler')
plt.show()
plt.plot(Lab_supply)
plt.title('delta Labour supply')
plt.show()

#sum of discounted utilities 
V=0
for t in range(999):
    V+= beta**t * u[t]

V_ss = 0
for t in range(1000):
    V_ss += beta**t * (u_ss)
print(f"Steady state value = {V_ss}; Vaue reached by las simulation = {V}")
print(f"Son of NaN beats the steady state by:{V-V_ss}")