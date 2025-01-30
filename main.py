import numpy as np
import torch
import matplotlib.pyplot as plt
from simulation import model
from NN_model import RL_agent
from tqdm import tqdm 

sim = model(0.3, 0.95, 0.4, 2.5, 0.02, 1, 2.6)
agent = RL_agent(input_dim=2, hidden_dim=128, output_dim=2, lr=1e-3, gamma = 0.99 )

value_losses=[]
policy_losses=[]

for iter in tqdm(range(50000)):
    st = sim.reset()
    for t in range(1000):
       st_tensor = torch.FloatTensor(st) 
       with torch.no_grad():
        action_tensor, log_prob = agent.policy_net.get_action(st_tensor)
        a = action_tensor.squeeze().numpy()
        st1, u = sim.step(st, a)
        agent.replay_buffer.push(st, a, u, st1)
        st = st1
        
    # qua alleniamo NN 
    v_loss, p_loss=agent.update()
    if v_loss is not None:
        value_losses.append(v_loss)
        policy_losses.append(p_loss)
    
    

print("Training completed")

plt.plot(value_losses, label='Value Loss')
plt.plot(policy_losses, label='Policy Loss')
plt.xlabel("Training Iterations")
plt.ylabel("Loss")
plt.title("Actor-Critic Training Loss")
plt.legend()
plt.show()




