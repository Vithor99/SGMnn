import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
from simulation import model
from NN_model import RL_agent
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs/boh_var_-3_discount")


sim = model(1, 1.6, 0.025, 0.9, 0.35)
agent = RL_agent(input_dim=2, hidden_dim=128, output_dim=2, lr=1e-3, gamma=0.99)

# value_losses = []
# policy_losses = []
# utilities_train = []
# utilities_test = []

best_utility = -np.inf

for iter in tqdm(range(8000)):

    ''' Qua mi sto allenando (sto cambiando i pesi del network)'''

    st = sim.reset()
    total_utility = 0

    for t in range(1000):
        st_tensor = torch.FloatTensor(st)
        with torch.no_grad():
            action_tensor, log_prob = agent.policy_net.get_action(st_tensor)
            a = action_tensor.squeeze().numpy()
            st1, u, y = sim.step(st, a)
            agent.replay_buffer.push(st, a, u, st1, y)
            st = st1
            total_utility += (agent.gamma ** t) * u

    writer.add_scalar("train utility", total_utility, iter)
    # utilities_train.append(total_utility)


    # qua alleniamo NN 
    v_loss, p_loss=agent.update()
    if v_loss is not None:
        writer.add_scalar("value loss", v_loss, iter)
        writer.add_scalar("policy loss", p_loss, iter)
        # value_losses.append(v_loss)
        # policy_losses.append(p_loss)

    if iter % 100 == 99:

        '''qua sto testando la policy media'''

        last_sim = {}
        all_actions = np.zeros((1000, 2))

        st = sim.reset()
        total_utility = 0
        for t in range(1000):
            st_tensor = torch.FloatTensor(st)
            with torch.no_grad():
                action_tensor, log_prob = agent.policy_net.get_action(st_tensor, test=True)
                a = action_tensor.squeeze().numpy()
                st1, u, y = sim.step(st, a)

                last_sim[t] = {'st': st, 'a': a, 'u': u, 'st1': st1, 'y': y}
                all_actions[t, :] = a

                st = st1
                total_utility += (agent.gamma ** t) * u

        writer.add_scalar("test utility", total_utility, iter)

        writer.add_scalar("var action 0 per sim", np.var(all_actions[:,0]), iter)
        writer.add_scalar("var action 1 per sim", np.var(all_actions[:,1]), iter)

        if best_utility < total_utility:
            best_utility = total_utility

            with open("last_sim.pkl", "wb") as f:
                pickle.dump(last_sim, f)


        # plt.plot(utilities_train)
        # plt.title('Train Utility')
        # plt.show()
        #
        # plt.plot(utilities_test)
        # plt.title('Test Utility')
        # plt.show()
    


print("Training completed")

# plt.plot(value_losses, label='Value Loss')
# plt.plot(policy_losses, label='Policy Loss')
# plt.xlabel("Training Iterations")
# plt.ylabel("Loss")
# plt.title("Actor-Critic Training Loss")
# plt.legend()
# plt.show()




