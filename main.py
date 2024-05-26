import numpy as np
import matplotlib.pyplot as plt
from simulation import model

sim = model(0.33, 0.5)

for iter in range(10):
    st = sim.reset()
    for t in range(100):
        c = 0.8 * st[1]
        st1, u = sim.step(st, c)
        st = st1

        # TODO: qua ci salveremo la tupla (st, c, u, st1)

    # TODO: qua alleniamo NN

print()



