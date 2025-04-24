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

'''CONTROLS'''
rl_model = 'RBC_random_deterministic_T=550.pt' 
grid_model = 'grid_data_dev10pct.pkl'

run_simulation = "no" #if yes it runs the simulation
if run_simulation == "yes":
    T = 500
    dev = 5 #Pct initial deviation from steady state
    zoom = "in" # in or out: if out zoom factor is activated 
    zoom_factor = 10 # if zoom == "out": pct band around ss that we want to visualize


run_policy = "yes" # if yes it runs the policy evaluation
if run_policy == "yes":
    dev = 5
    N = 500

    

run_add_analysis = "no" # if yes it runs some other stuff 


'''LOADING MODELS'''
# Loading model steady state
ss = steady()
c_ss, n_ss, k_ss, y_ss, u_ss = ss.ss()

# Loading RL policy
parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
''' ARCHITECTURE '''
parser.add_argument('--n_layers', default=1, type=int)
parser.add_argument('--n_neurons', default=128, type=int)
''' ALGORITHM '''
parser.add_argument('--policy_var', default=-4, type=float)
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
state_dim = ss.states
action_dim = ss.actions
alpha = ss.alpha
action_bounds = {
    'order': [1, 0],
    ''
    'min': [lambda: 0,
            lambda: 0],
    'max': [lambda s0, s1, alpha, a1: s0 * (s1**alpha * a1**(1-alpha)),
            lambda s0, s1, alpha, a1: 1.0]
    }
''' Define Model'''
architecture_params = {'n_layers': args.n_layers,
                       'n_neurons': args.n_neurons,
                       'policy_var': args.policy_var,
                       'action_bounds': action_bounds,
                       'use_hard_bounds': args.use_hard_bounds
                       }

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


# Loading Grid (vi) policy
grid_model_path = 'saved_models/' + grid_model
with open(grid_model_path, 'rb') as f:
    loaded_data = pickle.load(f)

kgrid = loaded_data['kgrid']
kp_star = loaded_data['kp_star']
control_star = loaded_data['control_star']
optimal_kp = interp1d(kgrid, kp_star, kind="slinear", bounds_error=False, fill_value=0)
optimal_c  = interp1d(kgrid, control_star[:, 0], kind="slinear", bounds_error=False, fill_value=0)
optimal_n  = interp1d(kgrid, control_star[:, 1], kind="slinear", bounds_error=False, fill_value=0)




''' SIMULATION '''
if run_simulation == "yes":
    k = np.array([k_ss*(1+(dev/100)), k_ss*(1+(dev/100))])
    grid_sim={}
    rl_sim={}
    grid_v = 0
    rl_v = 0

    for t in range(T):
        #RL 
        st = np.array([1, k[0]])
        state = torch.from_numpy(st).float().to(device)
        with torch.no_grad():
            action_tensor, _ = agent.get_action(state, test=True)
            action_rl = action_tensor.squeeze().numpy()
        st1_rl = (1-ss.delta)*st[1] + st[0]*(action_rl[1]**(1-ss.alpha))*(st[1]**ss.alpha) - action_rl[0]
        u_rl = ss.gamma*np.log(action_rl[0]) + ss.psi*np.log(1-action_rl[1])
        rl_v += ss.beta**t * u_rl
        rl_sim[t] = {'st': st,
                    'a': action_rl,
                    'u': u_rl,
                    'st1': st1_rl}
        
        #Grid 
        a = np.array([optimal_c(k[1]), optimal_n(k[1])])
        u = ss.gamma*np.log(a[0]) + ss.psi*np.log(1-a[1])
        grid_v += ss.beta**t * u
        kp_grid = optimal_kp(k[1])
        grid_sim[t] = {'st': k[1],
                    'a': a,
                    'u': u,
                    'st1': kp_grid}

        k = np.array([st1_rl, kp_grid])

    #Plotting
    #capital
    k_grid = [entry['st'] for entry in grid_sim.values()]
    k_rl = [entry['st'][1] for entry in rl_sim.values()]
    
    fig, ax = plt.subplots(figsize=(5, 6))  
    ax.plot(k_grid, color='blue', linewidth=1.5, label='Grid')
    ax.plot(k_rl, color='crimson', linewidth=1.5, label='RL')
    ax.axhline(k_ss, color="black", linewidth=1.2, linestyle='--',label='Steady State')
    ax.set_title("Capital", fontsize=16)
    ax.set_xlabel("Periods", fontstyle='italic')         
    ax.set_ylabel(r'$k_t$', fontstyle='italic')
    if zoom == "out":
        ax.set_ylim(k_ss*(1-(zoom_factor/100)), k_ss*(1+(zoom_factor/100)))
    ax.legend()          
    ax.grid(axis='both', alpha=0.5)                          
    ax.tick_params(axis='x', direction='in')
    ax.tick_params(axis='y', direction='in')

    fig.autofmt_xdate() 
    plt.tight_layout()
    plot_path = 'plots/' + rl_model.replace('.pt', '_capital.png')
    fig.savefig(plot_path)

    #consumption
    c_grid = [entry['a'][0] for entry in grid_sim.values()]
    c_rl = [entry['a'][0] for entry in rl_sim.values()]

    fig, ax = plt.subplots(figsize=(5, 6))  
    ax.plot(c_grid, color='blue', linewidth=1.5, label='Grid')
    ax.plot(c_rl, color='crimson', linewidth=1.5, label='RL')
    ax.axhline(c_ss, color="black", linewidth=1.2, linestyle='--', label='Steady State')
    ax.set_title("Consumption", fontsize=16)
    ax.set_xlabel("Periods", fontstyle='italic')         
    ax.set_ylabel(r'$c_t$', fontstyle='italic')
    if zoom == "out":
        ax.set_ylim(c_ss*(1-(zoom_factor/100)), c_ss*(1+(zoom_factor/100)))
    ax.legend()          
    ax.grid(axis='both', alpha=0.5)                          
    ax.tick_params(axis='x', direction='in')
    ax.tick_params(axis='y', direction='in')

    fig.autofmt_xdate() 
    plt.tight_layout()
    plot_path = 'plots/' + rl_model.replace('.pt', '_consumption.png')
    fig.savefig(plot_path)

    #labour
    n_grid = [entry['a'][1] for entry in grid_sim.values()]
    n_rl = [entry['a'][1] for entry in rl_sim.values()]

    fig, ax = plt.subplots(figsize=(5, 6))  
    ax.plot(n_grid, color='blue', linewidth=1.5, label='Grid')
    ax.plot(n_rl, color='crimson', linewidth=1.5, label='RL')
    ax.axhline(n_ss, color="black", linewidth=1.2, linestyle = '--', label='Steady State')
    ax.set_title("Labour", fontsize=16)
    ax.set_xlabel("Periods", fontstyle='italic')         
    ax.set_ylabel(r'$n_t$', fontstyle='italic')
    if zoom == "out":
        ax.set_ylim(n_ss*(1-(zoom_factor/100)), n_ss*(1+(zoom_factor/100)))
    ax.legend()          
    ax.grid(axis='both', alpha=0.5)                         
    ax.tick_params(axis='x', direction='in')
    ax.tick_params(axis='y', direction='in')

    fig.autofmt_xdate() 
    plt.tight_layout()
    plot_path = 'plots/' + rl_model.replace('.pt', '_labour.png')
    fig.savefig(plot_path)

    vss = ss.ss_value(T)
    print(f"Steady state value = {vss}; Value reached by Grid = {grid_v}; ; Value reached by RL = {rl_v}")
    print(f"Pct welfare gain of RL to grid:{-(rl_v - grid_v)*100/(grid_v)}") # if positive v_rl > v_grid




''' POLICY EVALUATION '''
if run_policy == "yes":
    c_values = np.zeros((N, 2))
    n_values = np.zeros((N, 2))
    k1_values = np.zeros((N, 2))
    k_values = np.linspace(k_ss * (1-(dev/100)), k_ss * (1+(dev/100)), N)
    for i in range(len(k_values)):
        #RL 
        st = np.array([1, k_values[i]])
        state = torch.from_numpy(st).float().to(device)
        with torch.no_grad():
            action_tensor, _ = agent.get_action(state, test=True)
            action_rl = action_tensor.squeeze().numpy()
        c_rl = action_rl[0]
        n_rl = action_rl[1]
        k1_rl = (1-ss.delta)*st[1] + st[0]*(action_rl[1]**(1-ss.alpha))*(st[1]**ss.alpha) - action_rl[0]
        #Grid 
        c_grid = optimal_c(k_values[i])
        n_grid = optimal_n(k_values[i])
        k1_grid = optimal_kp(k_values[i])
        #save 
        c_values[i] = [c_rl, c_grid]
        n_values[i] = [n_rl, n_grid]
        k1_values[i] = [k1_rl, k1_grid]
    #plotting
    #consumption
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.plot(k_values, c_values[:, 0], color='crimson', linewidth=1.5, label='RL')
    ax.plot(k_values, c_values[:, 1], color='blue', linewidth=1.5, label='Grid')
    ax.set_title("Consumption Rule", fontsize=16)
    ax.set_xlabel(r'$k_t$', fontstyle='italic')         
    ax.set_ylabel(r'$c_t$', fontstyle='italic')
    ax.legend()          
    ax.grid(axis='both', alpha=0.5)                         
    ax.tick_params(axis='x', direction='in')
    ax.tick_params(axis='y', direction='in')

    fig.autofmt_xdate() 
    plt.tight_layout()
    plot_path = 'plots/' + rl_model.replace('.pt', '_cons_rule.png')
    fig.savefig(plot_path)

    #labour
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.plot(k_values, n_values[:, 0], color='crimson', linewidth=1.5, label='RL')
    ax.plot(k_values, n_values[:, 1], color='blue', linewidth=1.5, label='Grid')
    ax.set_title("Labour Rule", fontsize=16)
    ax.set_xlabel(r'$k_t$', fontstyle='italic')         
    ax.set_ylabel(r'$n_t$', fontstyle='italic')
    ax.legend()          
    ax.grid(axis='both', alpha=0.5)                          
    ax.tick_params(axis='x', direction='in')
    ax.tick_params(axis='y', direction='in')

    fig.autofmt_xdate() 
    plt.tight_layout()
    plot_path = 'plots/' + rl_model.replace('.pt', '_lab_rule.png')
    fig.savefig(plot_path)

    #capital
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.plot(k_values, k1_values[:, 0], color='crimson', linewidth=1.5, label='RL')
    ax.plot(k_values, k1_values[:, 1], color='blue', linewidth=1.5, label='Grid')
    ax.set_title("Saving Rule", fontsize=16)
    ax.set_xlabel(r'$k_t$', fontstyle='italic')         
    ax.set_ylabel(r'$k_{t+1}$', fontstyle='italic')
    ax.legend()          
    ax.grid(axis='both', alpha=0.5)                          
    ax.tick_params(axis='x', direction='in')
    ax.tick_params(axis='y', direction='in')

    fig.autofmt_xdate() 
    plt.tight_layout()
    plot_path = 'plots/' + rl_model.replace('.pt', '_saving_rule.png')
    fig.savefig(plot_path)



'''ADD ANALYSIS'''
if run_add_analysis == 1:
    ''' VALUE CONVERGENCE'''
    #I used this to see how many training periods it takes for the value to converge to its long-run
    ss = steady()
    T=1000000
    pct_eps = 0.4#0.4 # pct distance from long run value 
    v_list = np.zeros(T)
    v_target = ss.ss_value(T)
    #v_change = np.zeros(1000-1)
    for i in range(T):
        T_train = i
        v_t = ss.ss_value(T_train)
        if i>0: 
            v_dist =  ((v_t - v_target)/np.abs(v_target))*100
            if v_dist < pct_eps: 
                delt = v_target / v_t
                print(f"Value converged at T={i}")
                break

        v_list[i] = v_t

    # Here Im interested in looking for the delt values (eg c* = c_ss * delt_c) that makes the distance from the steady state equal to 0.4%
    # i.e, give a distance from the long-run value, what distance can we expect from the steady state variables? 
    #Probably this analysis does not make sense but I will keep it for now
    nbl = 100
    delt_t = ss.gamma*np.log(c_ss) * ((1-delt)/delt) + ss.psi*np.log(1-n_ss) * ((1-delt)/delt) 
    DeltL = np.linspace(0.995, 1.005, nbl)
    DeltC = []
    for i in range(nbl):
        delt_c = np.exp((1/ss.gamma)*(delt_t - ss.psi*np.log(DeltL[i])))
        DeltC.append(delt_c)

    #Now we want to see in our RL model where we are on the curve of DeltL vs DeltC
    # first we compute the 'steady state' values of the RL model
    n=20 #last n periods of the rl simulation to compute the steady state
    l_rl_ss = 1 - np.sum(n_rl[-n:])/n 
    c_rl_ss = np.sum(c_rl[-n:])/n

    delt_c = c_ss / c_rl_ss
    delt_l = (1-n_ss) / l_rl_ss

    plt.plot(DeltL, DeltC)
    plt.scatter(delt_l, delt_c, color='red')
    plt.scatter(1, 1, color='green')
    plt.title("DeltL vs DeltC") 
    plt.show() 

''' version for deterministic model'''
'''
#open last simulation and rescale variables
with open("last_sim.pkl", "rb") as f:
    data = pickle.load(f)
 
st = [entry['st'] for _, entry in data.items()]
a = [entry['a'] for _, entry in data.items()]
u = [entry['u'] for _, entry in data.items()]
y = [entry['y'] for _, entry in data.items()]
k = [pair[1] for pair in st]
z = [pair[0] for pair in st]
c = [pair[0] for pair in a]
n = [pair[1] for pair in a]

# distance from FOC 
Euler = []
Lab_supply = []
for i in range(998):
    ls, ee = ss.foc_log(c[i], c[i+1], n[i], n[i+1], k[i], k[i+1])
    Euler.append(ee)
    Lab_supply.append(ls)

#sum of discounted utilities 
V=0
for t in range(999):
    V += ss.beta**t * u[t]

#plotting variable histories 
plt.plot(k)
plt.ylim(10, 15)
plt.axhline(k_ss, color="green")
plt.title("k")
plt.show()

plt.plot(c)
plt.ylim(0.7, 1.1)
plt.axhline(c_ss, color="green")
plt.title("c")
plt.show()

plt.plot(n)
plt.ylim(0, 1)
plt.axhline(n_ss, color="green")
plt.title("n")
plt.show()

plt.plot(y)
plt.ylim(0, 2)
plt.axhline(y_ss, color="green")
plt.title("y")
plt.show()

plt.plot(u)
plt.ylim(-1.3, 0)
plt.axhline(u_ss, color="green")
plt.title("u")
plt.show()

plt.plot(Euler)
plt.title('delta Euler')
plt.show()

plt.plot(Lab_supply)
plt.title('delta Labour supply')
plt.show()

print(f"Steady state value = {v_ss}; Value reached by last simulation = {V}")
print(f"Steady state beats RL by:{V-v_ss}")'
'''

