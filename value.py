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
rl_model = 'SGM_steady_stochastic.pt' 
#grid_model = 'Grid_SGM_stochastic_lowvar_global.pkl'
#folder to store plots 
folder = 'SGM_plots/'
run_VFI = "yes"

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



'''CONTROLS'''
dev_k= 20      #deviation from steady state in percent

nbk = 101       #number of of data points in state grid
nba = 1001     #number of of data points in control grid

crit = 1       #initial value for the value distance 
epsi = 1e-3 #1e-3

'''MAKING THE GRIDS'''
ss = steady()
c_ss, ks, y_ss, u_ss, v_ss = ss.ss()
sr_ss = 1 - (c_ss/y_ss)
delta = ss.delta
beta = ss.beta
alpha = ss.alpha
#psi = ss.psi
gamma = ss.gamma


nbz = ss.nbz
zgrid = ss.tauchenhussey(N=nbz)[0]   # Discretized z values
Pi = ss.tauchenhussey(N=nbz)[1]      # Transition probabilities

#kmin = 1 
kmin = 20
#kmin = (1 - (dev_k/100)) * k_ss_rl
kmax = (1 + (dev_k/100)) * k_ss_rl 
kgrid = np.linspace(kmin, kmax, nbk)



if run_VFI == "yes":
    '''VALUE FUNCTION ITERATION'''
    v = np.zeros([nbk, nbz])       #initial guess for values linked to the state grid 
    iter = 0
    crit_hist = []

    while crit > epsi:
        tv = np.zeros([nbk, nbz])
        #dr = np.zeros([nbk, nbz]).astype(int)
        for i in range(nbk):
            for j in range(nbz): 
                st = np.array([zgrid[j], kgrid[i]])
                state_rl = torch.from_numpy(st).float().to(device)
                with torch.no_grad():
                    action_tensor, _ = agent.get_action(state_rl, test=True)
                    sratio_rl = action_tensor.squeeze().numpy()
                y = zgrid[j] * (kgrid[i]**ss.alpha)
                control = (1 - sratio_rl) * y
                kp = y + (1-delta)*kgrid[i] - control
                util = gamma * np.log(control) #+ psi * np.log(1 - control[:,1])
                vfunc = interp(kgrid, v)
                vi = vfunc(kp)
                EV = vi @ Pi[j,:].reshape(-1,1)
                val = util + beta * EV
                tv[i,j] = val
        
        crit = max(abs(tv-v).flatten())
        v = tv
        conv = (crit/epsi)
        iter += 1
        if iter % 10 == 0: 
            print(f"Iteration {iter}, crit: {conv}")
    print(f"Final iteration: {iter}, crit: {conv}")


    v_star = vfunc(kgrid)

    data_to_save = {
        'st': kgrid,
        'value_star': v_star
    }


    with open("V.pkl", 'wb') as f:
        pickle.dump(data_to_save, f)

# Loading Grid (vi) policy
with open("V.pkl", 'rb') as f:
    loaded_data = pickle.load(f)

k = loaded_data['st']
value_star = loaded_data['value_star']

optimal_v = interp2d((zgrid, k), value_star.T)

for i in range(len(zgrid)):
    st = np.column_stack((np.ones_like(k)*zgrid[i], k))
    plt.plot(k, optimal_v(st))
plt.show()

v_ss_rl_true = optimal_v(np.array([1, k_ss_rl]))

N = nbk
dev = 10
zgrid_small = np.array([zgrid[1], zgrid[3], zgrid[5], zgrid[7], zgrid[9]])

c_values_grid = np.zeros((N, len(zgrid_small)))
k1_values_grid = np.zeros((N, len(zgrid_small)))
v_values_grid = np.zeros((N, len(zgrid_small)))
c_values_rl = np.zeros((N, len(zgrid_small)))
k1_values_rl = np.zeros((N, len(zgrid_small)))
v_values_rl = np.zeros((N, len(zgrid_small)))
v_values_rl_true = np.zeros((N, len(zgrid_small)))


#k_values = np.linspace(k_ss_rl * (1-(dev/100)), k_ss_rl * (1+(dev/100)), N)
k_values = k
#z_psx = int(np.where(zgrid == 1)[0])
for j in range(len(zgrid_small)): 
    for i in range(len(k_values)):
        #RL 
        st = np.array([zgrid_small[j], k_values[i]])
        state = torch.from_numpy(st).float().to(device)
        with torch.no_grad():
            action_tensor, _ = agent.get_action(state, test=True)
            action_rl = action_tensor.squeeze().numpy()
            value_tensor = agent.get_value(state)
            value_rl = value_tensor.numpy()
        y_rl = zgrid_small[j] * (k_values[i]**ss.alpha)
        c_rl = (1 - action_rl) * y_rl
        k1_rl = (1 - ss.delta)*k_values[i] + action_rl * y_rl
        v_rl = float(value_rl)
        #Grid 
        #true value
        v_rl_true = optimal_v(st)
        #save 
        v_values_rl[i,j] = v_rl
        v_values_rl_true[i, j] = v_rl_true



#value vs true value
fig, ax = plt.subplots(figsize=(5, 6))
palette = ("#e65300", "#ff6600", 	"#ff9440", "#ffb84d", "#ffe0b3")
for i in range(len(v_values_rl[0,:])):
    ax.plot(k_values, (v_values_rl[:, i] - v_values_rl_true[:, i])/v_values_rl_true[:, i] , color = palette[i],  linewidth=1.5, label='RL')
    #ax.plot(k_values, v_values_rl_true[:, i], color = palette[i], linestyle = 'dashed', linewidth=1.5)
    #ax.plot(k_values, v_values_rl[:, i], color = palette[i], linewidth=1.5)
#ax.scatter(k_ss, v_ss, marker='o', facecolors='none', edgecolors= '#003f5c', s=40, linewidths=1.5, zorder = 5)
ax.scatter(k_ss_rl, (v_ss_rl - v_ss_rl_true)/v_ss_rl_true, marker='o', facecolors='#003f5c', edgecolors='#003f5c', s=30, linewidths=1.5, zorder = 5)
ax.axvline(k_ss_rl, color='#003f5c', linewidth=1.2, linestyle='--', label='Steady State')
#ax.set_title("Value function", fontsize=16)
ax.set_xlabel(r'$k_t$', fontstyle='italic')         
ax.set_ylabel(r'$v_t\% \ \ \ diff.$', fontstyle='italic')
#ax.set_ylim(-0.002, 0.002)
#ax.legend()          
ax.grid(axis='both', alpha=0.5)                         
ax.tick_params(axis='x', direction='in')
ax.tick_params(axis='y', direction='in')

fig.autofmt_xdate() 
plt.tight_layout()
plot_path = folder + rl_model.replace('.pt', '_True_value_rule.png')
fig.savefig(plot_path)