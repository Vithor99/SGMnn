import pickle
import numpy as np
import matplotlib.pyplot as plt
from steady import steady

#compute staedy state
ss = steady()
c_ss, n_ss, k_ss, y_ss, u_ss, v_ss = ss.ss()

#open last simulation and rescale variables
with open("last_sim.pkl", "rb") as f:
    data = pickle.load(f)
 
st = [entry['st'] for _, entry in data.items()]
a = [entry['a'] for _, entry in data.items()]
u = [entry['u'] for _, entry in data.items()]
y = [entry['y'] for _, entry in data.items()]
k = [pair[1] for pair in st]
z = [pair[0] for pair in st]
c = [pair[0] for pair in a]
n = [pair[1] for pair in a]

# distance from FOC 
Euler = []
Lab_supply = []
for i in range(998):
    ls, ee = ss.foc_log(c[i], c[i+1], n[i], n[i+1], k[i], k[i+1])
    Euler.append(ee)
    Lab_supply.append(ls)

#sum of discounted utilities 
V=0
for t in range(999):
    V += ss.beta**t * u[t]

#plotting variable histories 
plt.plot(k)
plt.ylim(10, 15)
plt.axhline(k_ss, color="green")
plt.title("k")
plt.show()

plt.plot(c)
plt.ylim(0.7, 1.1)
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
plt.ylim(-1.3, 0)
plt.axhline(u_ss, color="green")
plt.title("u")
plt.show()

plt.plot(Euler)
plt.title('delta Euler')
plt.show()

plt.plot(Lab_supply)
plt.title('delta Labour supply')
plt.show()

print(f"Steady state value = {v_ss}; Value reached by last simulation = {V}")
print(f"Steady state beats RL by:{V-v_ss}")