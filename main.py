import numpy as np
import matplotlib.pyplot as plt
from simulation import model
from NN_model import V

sim = model(0.33, 0.5)

# TODO: init V

for iter in range(10):
    st = sim.reset()
    for t in range(100):
        c = 0.8 * st[1]
        st1, u = sim.step(st, c)
        st = st1

        V.replay_buffer.push(st, c, u, st1)

    # TODO: qua alleniamo NN

print()



