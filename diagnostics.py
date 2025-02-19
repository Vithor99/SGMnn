import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

## Solve steady of the model 
gamma = 1
psi = 1.6
alpha = 0.35
delta = 0.025
beta = 0.99

def equations(vars):
    c, n, k = vars
    ls = (1-alpha)*(k**alpha)*(n**(-alpha)) - (psi/gamma)*(np.sqrt(c)/np.sqrt(1-n))
    ee = 1 - beta*((1-delta)+alpha*k**(alpha-1)*n**(1-alpha))  
    kt = delta*k - k**alpha * n**(1-alpha) + c 
    return [ls, ee, kt]

initial_guess = [0.5, 0.5, 0.5]  
solution = fsolve(equations, initial_guess)
c_ss, n_ss, k_ss = solution
y_ss = (k_ss)**alpha * (n_ss)**(1-alpha)
u_ss = gamma*np.sqrt(c_ss)+psi*np.sqrt(1-n_ss)

print(f"Steady state solution: c = {c_ss}, n = {n_ss}, k = {k_ss}, y={y_ss}, u={u_ss}")




#open last simulation and rescale variables
with open("last_sim.pkl", "rb") as f:
    data = pickle.load(f)
 
st = [entry['st'] for _, entry in data.items()]
a = [entry['a'] for _, entry in data.items()]
u = [entry['u']*1000 for _, entry in data.items()]
y = [entry['y']*10 for _, entry in data.items()]
k = [pair[1]*100 for pair in st]
z = [pair[0] for pair in st]
z = np.exp(z)
c = [pair[0]*10 for pair in a]
n = [pair[1]*10 for pair in a]

#plotting variable histories 
plt.plot(k)
plt.ylim(13, 15.5)
plt.axhline(k_ss, color="green")
plt.title("k")
plt.show()

plt.plot(c)
plt.ylim(1, 1.3)
plt.axhline(c_ss, color="green")
plt.title("c")
plt.show()

plt.plot(n)
plt.ylim(0, 1)
plt.axhline(n_ss, color="green")
plt.title("n")
plt.show()

plt.plot(y)
plt.ylim(0, 2)
plt.axhline(y_ss, color="green")
plt.title("y")
plt.show()

plt.plot(u)
plt.ylim(2, 2.5)
plt.axhline(u_ss, color="green")
plt.title("u")
plt.show()

'''
var_list = ['k','c','n','y','u']
for var in var_list:
    plt.plot(globals()[var], label=var)
    plt.axhline(globals()[f"{var}_ss"], color="green")
    plt.title(f"{var}")
    plt.show()
'''

# distance from FOC 
Euler = []
Lab_supply = []
for i in range(998):
    ls = (1-alpha)*(k[i]**alpha)*(n[i]**(-alpha)) - (psi/gamma)*(np.sqrt(c[i])/np.sqrt(1-n[i]))
    ee = (gamma/np.sqrt(c[i])) - beta*(gamma/np.sqrt(c[i+1]))*((1-delta)+alpha*k[i+1]**(alpha-1)*n[i+1]**(1-alpha))     
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
    V += beta**t * u[t]

V_ss = 0
for t in range(1000):
    V_ss += beta**t * u_ss
print(f"Steady state value = {V_ss}; Value reached by last simulation = {V}")
print(f"Steady state beats RL by:{V-V_ss}")