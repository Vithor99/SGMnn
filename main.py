import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
from simulation import model
from NN_model import RL_agent
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
''' ARCHITECTURE '''
parser.add_argument('--n_layers', default=1, type=int)
parser.add_argument('--n_neurons', default=128, type=int)
''' ALGORITHM '''
parser.add_argument('--policy_var', default=-4.0, type=float)
parser.add_argument('--epsilon_greedy', default=0.0, type=float)
parser.add_argument('--gamma', default=0.99, type=float)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--batch_size', default=2048, type=int)
parser.add_argument('--learn_std', default=0, type=int)
''' SIMULATOR '''
# TODO:
args = parser.parse_args()

seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

name_exp = ''
for k, v in args.__dict__.items():
    name_exp += str(k) + "=" + str(v) + "_"

writer = SummaryWriter("logs/"+name_exp + "")

''' Define Simulator'''
state_dim = 2
action_dim = 2
c_ss = 0.1116978
n_ss = 0.04313
alpha = 0.35
sim = model(1, 1.6, 0.025, 0.9, alpha, device=device)

action_bounds = {
    'order': [1, 0],
    ''
    'min': [lambda: 0,
            lambda: 0],
    'max': [lambda s0, s1, alpha, a1: torch.exp(s0) * (s1**alpha * a1**(1-alpha)),
            lambda s0, s1, alpha, a1: 0.1]
    }


''' Define Model'''
architecture_params = {'n_layers': args.n_layers,
                       'n_neurons': args.n_neurons,
                       'policy_var': args.policy_var,
                       'action_bounds': action_bounds
                       }

agent = RL_agent(input_dim=state_dim,
                 architecture_params=architecture_params,
                 output_dim=action_dim,
                 lr=args.lr,
                 gamma=args.gamma,
                 epsilon=args.epsilon_greedy,
                 batch_size=args.batch_size,
                 alpha=alpha,
                 learn_std=args.learn_std==1).to(device)

T = 1000
EPOCHS = 40000
frq_train = 3
frq_test = 100
best_utility = -np.inf

for iter in tqdm(range(EPOCHS)):

    st = sim.reset()
    total_utility = 0

    for t in range(T):
        # st_tensor = torch.FloatTensor(st)
        with torch.no_grad():
            action_tensor, log_prob = agent.policy_net.get_action(st)
            a = action_tensor.squeeze() #.numpy()
            st1, u, y, done = sim.step(st, a)
            # agent.replay_buffer.push(st, a, u, st1, y)
            agent.batchdata.push(st, a, log_prob, u, st1, y, float(not done))
            st = st1
            total_utility += (agent.gamma ** t) * u
            if done:
                st = sim.reset()
                # total_utility = 0

    writer.add_scalar("train utility", total_utility.detach().cpu().item(), iter)

    # qua alleniamo NN
    if iter % frq_train == 0:
        v_loss, p_loss = agent.update()
        agent.batchdata.clear()

        if v_loss is not None:
            writer.add_scalar("value loss", v_loss, iter)
            writer.add_scalar("policy loss", p_loss, iter)


    if iter % frq_test == (frq_test-1):

        '''qua sto testando la policy media'''

        last_sim = {}
        all_actions = np.zeros((T, 2))

        st = sim.reset()
        total_utility = 0
        for t in range(T):
            # st_tensor = torch.FloatTensor(st)
            with torch.no_grad():
                action_tensor, log_prob = agent.policy_net.get_action(st, test=True)
                a = action_tensor.squeeze() #.numpy()
                st1, u, y, done = sim.step(st, a)

                last_sim[t] = {'st': st.detach().cpu().numpy(),
                               'a': a.detach().cpu().numpy(),
                               'u': u.detach().cpu().numpy(),
                               'st1': st1.detach().cpu().numpy(),
                               'y': y.detach().cpu().numpy()}
                all_actions[t, :] = a.detach().cpu().numpy()

                st = st1
                total_utility += (agent.gamma ** t) * u

                if done:
                    break

        writer.add_scalar("test utility", total_utility, iter)

        writer.add_scalar("var action 0 per sim", np.var(all_actions[:,0]), iter)
        writer.add_scalar("var action 1 per sim", np.var(all_actions[:,1]), iter)
        writer.add_scalar("distance of action 0 from ss", np.abs(all_actions[1,0]-c_ss)+np.abs(all_actions[len(all_actions)-1,0]-c_ss), iter)
        writer.add_scalar("distance of action 1 from ss", np.abs(all_actions[1,1]-n_ss)+np.abs(all_actions[len(all_actions)-1,1]-n_ss), iter)

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
    


# plt.plot(value_losses, label='Value Loss')
# plt.plot(policy_losses, label='Policy Loss')
# plt.xlabel("Training Iterations")
# plt.ylabel("Loss")
# plt.title("Actor-Critic Training Loss")
# plt.legend()
# plt.show()




