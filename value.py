import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import pickle
from simulation import Model
from rl_algos.actor_critic import ActorCritic
from rl_algos.soft_actor_critic import SoftActorCritic
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse
from steady import steady
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator as interp2d
from scipy.interpolate import CubicSpline as interp
import warnings
warnings.filterwarnings("ignore")

# Be careful: steady must be alligned to what we are plotting here. 
'''CONTROLS'''
rl_model = 'SGM_lowvar_steady_stochastic.pt' 
#grid_model = 'Grid_SGM_stochastic_lowvar_global.pkl'
#folder to store plots 
folder = 'SGM_plots/'


'''LOADING RL MODEL'''
'''SETTING PARAMETERS''' 
ss = steady()
parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
''' ARCHITECTURE '''
parser.add_argument('--n_layers', default=1, type=int)
parser.add_argument('--n_neurons', default=128, type=int)
''' ALGORITHM '''
parser.add_argument('--policy_var', default=-3.0, type=float)
parser.add_argument('--epsilon_greedy', default=0.0, type=float)
parser.add_argument('--gamma', default=ss.beta, type=float)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--batch_size', default=2048, type=int)
parser.add_argument('--learn_std', default=0, type=int)
parser.add_argument('--use_hard_bounds', default=1, type=int)
''' SIMULATOR '''
parser.add_argument('--n_workers', default=4, type=int)
args = parser.parse_args()
device = torch.device('cpu')

''' Define Simulator'''
c_ss, k_ss, y_ss, u_ss, v_ss = ss.ss()
s_ratio_ss = 1 - (c_ss/y_ss)
state_dim = ss.states
action_dim = ss.actions
alpha = ss.alpha

''' Define Model'''
architecture_params = {'n_layers': args.n_layers,
                       'n_neurons': args.n_neurons,
                       'policy_var': args.policy_var}

agent = ActorCritic(input_dim=state_dim,
                    architecture_params=architecture_params,
                    output_dim=action_dim,
                    lr=args.lr,
                    gamma=args.gamma,
                    epsilon=args.epsilon_greedy,
                    batch_size=args.batch_size,
                    alpha=alpha,
                    learn_std=args.learn_std==1,
                    device=device).to(device)

rl_model_path = 'saved_models/' + rl_model
agent.load_state_dict(torch.load(rl_model_path, map_location=device))
agent.eval()

k_rl = k_ss
K = np.zeros(1000)
C = np.zeros(1000)
U = np.zeros(1000)
for t in range(1000):
    #RL 
    state_rl = torch.from_numpy(np.array([1, k_rl])).float().to(device)
    with torch.no_grad():
        action_tensor, _ = agent.get_action(state_rl, test=True)
        sratio_rl = action_tensor.squeeze().numpy()
    k1_rl = (1 - ss.delta)*k_rl + (k_rl**ss.alpha) * sratio_rl
    c_rl = (k_rl**ss.alpha) * (1 - sratio_rl)
    u_rl = np.log(c_rl)
    K[t] = k1_rl
    C[t] = c_rl
    U[t] = u_rl
    k_rl = k1_rl
k_ss_rl = np.mean(K[-100:])
c_ss_rl = np.mean(C[-100:])
u_ss_rl = np.mean(U[-100:])
v_ss_rl = (u_ss_rl / (1 - ss.beta))

''' COMPUTE THE TRUE VALUE OF THE POLICY '''
#1 we need 11 levels of productivity and 100 levels of capital around a 20% interval from the steady state 
#3 Monte Carlo simulate from 100 x 11 initial states TxN time, where T = 550 and N = 100.  
#4 we end up with 100 x 11 values, interpolate the 100 points to get a value function for each level of z. 
#0
dev = 0.2
grid = 100
N = 100
#1 
zgrid = ss.tauchenhussey(N=ss.nbz)[0]
kgrid = np.linspace((1-dev) * k_ss_rl, (1+dev) * k_ss_rl, grid)

V = np.zeros((len(kgrid), len(zgrid)))
#3 
for j in range(len(zgrid)): 
    for i in range(len(kgrid)): 
        exp_v = 0
        for n in range(N):
            st = np.array([zgrid[j], kgrid[i]])
            rl_v = 0
            for t in range(550):
                state_rl = torch.from_numpy(st).float().to(device)
                with torch.no_grad():
                    action_tensor, _ = agent.get_action(state_rl, test=True)
                    sratio_rl = action_tensor.squeeze().numpy()
                k_rl = st[1]
                z = st[0]
                c_rl = z * (k_rl**ss.alpha) * (1-sratio_rl)
                u_rl = ss.gamma*np.log(c_rl)
                k1_rl = (1 - ss.delta)*k_rl + sratio_rl * z * (k_rl**ss.alpha)
                rl_v += ss.beta**t * u_rl
                z1 = (1 - ss.rhoa) + ss.rhoa * z + np.random.normal(0, ss.dev_eps_z)
                st = np.array([z1, k1_rl])
            exp_v += (1/N) * rl_v
        V[i,j] = exp_v
    print(j)

with open('V.pkl', 'wb') as f:
    pickle.dump(V, f)
with open('V.pkl', 'rb') as f:
    V = pickle.load(f)
    
optimal_v = interp2d((zgrid, kgrid), V.T)



st = np.column_stack((np.ones_like(kgrid)*zgrid[5], kgrid))
plt.plot(kgrid, optimal_v(st))
plt.show()