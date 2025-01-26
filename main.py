import numpy as np
import torch
import matplotlib.pyplot as plt
from simulation import model
from NN_model import V, Policy

sim = model(0.33, 0.5, 0.5, 0.33, 0.9)

value_function = V() 
policy_function = Policy()
policy_function.set_value_function(value_function)

value_losses=[]
policy_losses=[]

for iter in range(10000):
    st = sim.reset()
    for t in range(1000):
       st_tensor = torch.FloatTensor(st) 
       with torch.no_grad():
        action_tensor, log_prob = policy_function.policy_net.get_action(st_tensor)
        
        a = action_tensor.numpy()
        st1, u = sim.step(st, a)

        value_function.replay_buffer.push(st, a, u, st1)
        policy_function.replay_buffer.push(st, a, u, st1)
        st = st1
        

    # qua alleniamo NN per V
    v_loss=value_function.update()
    if v_loss is not None:
        value_losses.append(v_loss)
    
    # qua alleniamo NN per Policy
    p_loss = policy_function.update()
    if p_loss is not None:
        policy_losses.append(p_loss)

print("Training completed")

plt.plot(value_losses, label='Value Loss')
plt.plot(policy_losses, label='Policy Loss')
plt.xlabel("Training Iterations")
plt.ylabel("Loss")
plt.title("Actor-Critic Training Loss")
plt.legend()
plt.show()

