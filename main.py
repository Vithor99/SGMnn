import numpy as np
import torch
import matplotlib.pyplot as plt
from simulation import model
from NN_model import RL_agent
from tqdm import tqdm 

sim = model(0.5, 0.95, 0.5, 0.3, 0.02, 0.5)
agent = RL_agent(input_dim=2, hidden_dim=128, output_dim=2, lr=1e-3, gamma = 0.99)

value_losses = []
policy_losses = []
all_utilities = []

for iter in tqdm(range(10000)):
    st = sim.reset()
    total_utility = 0

    for t in range(1000):
        st_tensor = torch.FloatTensor(st)
        with torch.no_grad():
            action_tensor, log_prob = agent.policy_net.get_action(st_tensor)
            a = action_tensor.squeeze().numpy()
            st1, u = sim.step(st, a)
            agent.replay_buffer.push(st, a, u, st1)
            st = st1
            total_utility += u
    all_utilities.append(total_utility)



    # qua alleniamo NN 
    v_loss, p_loss=agent.update()
    if v_loss is not None:
        value_losses.append(v_loss)
        policy_losses.append(p_loss)

    if iter % 100 == 99:
        plt.plot(all_utilities)
        plt.show()
    
    

print("Training completed")

plt.plot(value_losses, label='Value Loss')
plt.plot(policy_losses, label='Policy Loss')
plt.xlabel("Training Iterations")
plt.ylabel("Loss")
plt.title("Actor-Critic Training Loss")
plt.legend()
plt.show()




