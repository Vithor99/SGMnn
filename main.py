import numpy as np
import matplotlib.pyplot as plt
from simulation import model
from NN_model import V

sim = model(0.33, 0.5)

value_function = V() 
losses=[]
for iter in range(1000):
    st = sim.reset()
    for t in range(100):
        c = 0.8 * st[1]
        st1, u = sim.step(st, c)
        st = st1

        value_function.replay_buffer.push(st, c, u, st1)

    # xTODO: qua alleniamo NN
    loss=value_function.update()
    if loss is not None:
        losses.append(loss)
print("Training completed")

plt.plot(losses)
plt.xlabel('Training Iterations')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.show()