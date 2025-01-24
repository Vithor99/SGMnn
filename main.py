import numpy as np
import matplotlib.pyplot as plt
from simulation import model
from NN_model import V

sim = model(0.33, 0.5, 0.5, 0.33, 0.9)

value_function = V() 
losses=[]
for iter in range(10000):
    st = sim.reset()
    for t in range(1000):
        n = 1 
        wage = st[0]*(1-sim.alpha)*(st[1]/n)**sim.alpha
        rent = st[0]*(sim.alpha)*(st[1]/n)**(sim.alpha-1)
        c = 0.5*(wage*n+rent*st[1])
        i = 0.5*(wage*n+rent*st[1])
        a = [c, n, i]
        st1, u = sim.step(st, a)
        st = st1
        value_function.replay_buffer.push(st, a, u, st1)

    # qua alleniamo NN
    loss=value_function.update()
    if loss is not None:
        losses.append(loss)
        
print("Training completed")

plt.plot(losses)
plt.xlabel('Training Iterations')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.show()

