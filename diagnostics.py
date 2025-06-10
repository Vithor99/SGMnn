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
from scipy.interpolate import CubicSpline as interp
import warnings
warnings.filterwarnings("ignore")

'''CONTROLS'''
rl_model = 'SGM_steady_deterministic.pt' 
grid_model = 'Grid_SGM_deterministic.pkl'
#folder to store plots 
folder = 'SGM_plots/'

run_simulation = "no" #if yes it runs the simulation

run_policy = "yes" # if yes it runs the policy evaluation

run_policy_sto = "no"

global_policy = "no"


'''LOADING MODELS'''
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

state_dim = ss.states
action_dim = ss.actions
alpha = ss.alpha

action_bounds = {
    'order': [1, 0],
    ''
    'min': [lambda: 0,
            lambda: 0],
    'max': [lambda s0, s1, alpha, a1: s0 * (s1**(alpha) * a1**(1-alpha)),
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


''' LOADING GRID MODEL'''
#need to update for stochastic version 
grid_model_path = 'saved_models/' + grid_model
with open(grid_model_path, 'rb') as f:
    loaded_data = pickle.load(f)


k = loaded_data['st']
k1_star = loaded_data['k1_star']
a_star = loaded_data['a_star']
c_star = loaded_data['c_star']
value_star = loaded_data['value_star']

optimal_a = interp(k, a_star)
optimal_k1 = interp(k, k1_star)
optimal_v = interp(k, value_star)
optimal_c = interp(k, c_star) 


''' SIMULATION '''
if run_simulation == "yes":
    dev = 5
    T = 500
    n = 50
    st = np.array([k_ss*(1+(dev/100)), k_ss*(1+(dev/100))])
    k0_rl_i = np.ones(100) *  k_ss*(1+(dev/100))
    z = 1 #we want the same series for productivity in the two economies
    #z_psx = int(np.where(zgrid == 1)[0])
    grid_sim={}
    rl_sim={}
    grid_v = 0
    rl_v = 0
    #labour_gap = np.zeros((T-1, 2))
    euler_gap = np.zeros((T-1, 2))
    irf_ci = np.zeros((T, 100))
    irf_ki = np.zeros((T, 100))
    

    for t in range(T):
        #RL 
        k_rl = st[0]
        st_rl_i = np.column_stack((np.full_like(k0_rl_i , z), k0_rl_i))
        state = torch.from_numpy(np.array([z, k_rl])).float().to(device)
        state_i = torch.from_numpy(st_rl_i).float().to(device)

        with torch.no_grad():
            action_tensor, _ = agent.get_action(state, test=True)
            sratio_rl = action_tensor.squeeze().numpy()

            actions_i = np.zeros(100)
            for i in range(100):
                action_i_tens, _ = agent.get_action(state_i[i], test= False)
                action_i = action_i_tens.squeeze().numpy()
                actions_i[i] = action_i

        y_rl = z* (k_rl**ss.alpha)
        c_rl = y_rl * (1-sratio_rl)
        u_rl = ss.gamma*np.log(c_rl)
        k1_rl = (1 - ss.delta)*k_rl + y_rl - c_rl
        #st1_rl = (1-ss.delta)*st[1] + st[0]*(action_rl[1]**(1-ss.alpha))*(st[1]**ss.alpha) - action_rl[0]
        #u_rl = ss.gamma*np.log(action_rl[0]) + ss.psi*np.log(1-action_rl[1])
        rl_v += ss.beta**t * u_rl
        rl_sim[t] = {'k': k_rl,
                    'c': c_rl,
                    'y': y_rl,
                    'u': u_rl,
                    'st1': k1_rl}
        
        y_rl_i = (k0_rl_i**ss.alpha)
        c_rl_i = (1 - actions_i) * y_rl_i
        k1_rl_i =  (1 - ss.delta)*k0_rl_i + actions_i * y_rl_i
        # Storing the i IRFs 
        irf_ci[t] = c_rl_i
        irf_ki[t] = k1_rl_i

        #Grid 
        # Find the position of the value 1 in zgrid
        #a = np.array([ss.get_consumption(k[1], 1, float(optimal_n(k[1])[z_psx])), float(optimal_n(k[1])[z_psx])])
        k_grid = st[1]
        y_grid = z* (k_grid**ss.alpha)
        c_grid = float(optimal_c(k_grid))
        u_grid = ss.gamma*np.log(c_grid) #+ ss.psi*np.log(1-a[1])
        k1_grid = float(optimal_k1(k_grid))
        grid_v += ss.beta**t * u_grid
        grid_sim[t] = {'k': k_grid,
                    'c': c_grid,
                    'y': y_grid,
                    'u': u_grid,
                    'st1': k1_grid}
        

       
        if t>0:
            e_gap_grid = ss.foc_log(grid_sim[t-1]['c'], grid_sim[t]['c'],  
                                    z, grid_sim[t]['k'])
            
            e_gap_rl = ss.foc_log(rl_sim[t-1]['c'], rl_sim[t]['c'],  
                                    z, rl_sim[t]['k'])
            #labour_gap[t-1, 1] = l_gap
            euler_gap[t-1, 0] = e_gap_rl
            euler_gap[t-1, 1] = e_gap_grid

        st = np.array([k1_rl, k1_grid])
        k0_rl_i = k1_rl_i
        #z1_psx = int(np.random.choice(ss.nbz, p=Pi[z_psx,:]))
        #z_psx = z1_psx
        #z = zgrid[z_psx]

    #Plotting
    #capital
    k_grid = [entry['k'] for entry in grid_sim.values()]
    k_rl = [entry['k'] for entry in rl_sim.values()]

    #distance from steady state
    rl_ss = np.mean(k_rl[-n:])
    grid_ss = np.mean(k_grid[-n:])
    ss_dev = (rl_ss - grid_ss) / np.abs(grid_ss)
    print(f"Capital distance from steady state: {ss_dev*100:.2f}%")
    
    fig, ax = plt.subplots(figsize=(5, 6))  
    ax.plot(k_grid, color='#003f5c', linewidth=1.5, label='Grid', zorder = 4)
    ax.plot(k_rl, color='#ff6600', linewidth=1.5, label='RL', zorder = 5)
    ax.plot(irf_ki, color = "#ff9440", linewidth= 0.5, alpha = 0.05, label='RL')
    ax.axhline(k_ss, color="black", linewidth=1.2, linestyle='--',label='Steady State')
    ax.set_title("Capital", fontsize=16)
    ax.set_xlabel("Periods", fontstyle='italic')         
    ax.set_ylabel(r'$k_t$', fontstyle='italic')
    #ax.legend()          
    ax.grid(axis='both', alpha=0.5)                          
    ax.tick_params(axis='x', direction='in')
    ax.tick_params(axis='y', direction='in')

    fig.autofmt_xdate() 
    plt.tight_layout()
    plot_path = folder + rl_model.replace('.pt', '_capital.png')
    fig.savefig(plot_path)

    #consumption
    c_grid = [entry['c'] for entry in grid_sim.values()]
    c_rl = [entry['c'] for entry in rl_sim.values()]

    #distance from steady state
    rl_ss = np.mean(c_rl[-n:])
    grid_ss = np.mean(c_grid[-n:])
    ss_dev = (rl_ss - grid_ss) / np.abs(grid_ss)
    print(f"Consumption distance from steady state: {ss_dev*100:.2f}%")

    fig, ax = plt.subplots(figsize=(5, 6))  
    ax.plot(c_grid, color='#003f5c', linewidth=1.5, label='Grid', zorder = 4)
    ax.plot(c_rl, color='#ff6600', linewidth=1.5, label='RL', zorder = 5)
    ax.plot(irf_ci, color = "#ff9440", linewidth= 0.5, alpha = 0.05, label='RL')
    ax.axhline(c_ss, color="black", linewidth=1.2, linestyle='--', label='Steady State')
    ax.set_title("Consumption", fontsize=16)
    ax.set_xlabel("Periods", fontstyle='italic')         
    ax.set_ylabel(r'$c_t$', fontstyle='italic')
    #ax.legend()          
    ax.grid(axis='both', alpha=0.5)                          
    ax.tick_params(axis='x', direction='in')
    ax.tick_params(axis='y', direction='in')

    fig.autofmt_xdate() 
    plt.tight_layout()
    plot_path = folder + rl_model.replace('.pt', '_consumption.png')
    fig.savefig(plot_path)

   

    #euler foc
    fig, ax = plt.subplots(figsize=(5, 6))  
    ax.plot(euler_gap[:, 0], color='#003f5c', linewidth=1.0, alpha = 0.3,  label='RL')
    #ax.plot(euler_gap[:, 1], color='blue', linewidth=1.5, label='Grid')

    alpha = 0.1  # Smoothing parameter
    smoothed_euler_gap = np.zeros_like(euler_gap[:, 0])
    smoothed_euler_gap[0] = euler_gap[0, 0]
    for i in range(1, len(euler_gap[:, 0])):
        smoothed_euler_gap[i] = alpha * euler_gap[i, 0] + (1 - alpha) * smoothed_euler_gap[i - 1]
    
    ax.plot(smoothed_euler_gap, color='#003f5c', linewidth=1.5, label='Smoothed RL')

    ax.axhline(0, color="black", linewidth=1.2, linestyle = '--')
    ax.set_title("Distance from Euler", fontsize=16)
    ax.set_xlabel("Periods", fontstyle='italic')         
    #ax.set_ylabel(r'$\% \Delta \ \ \frac{c_{t+1}}{c_t}$', fontstyle='italic')
    ax.set_ylabel("Euler Residuals", fontstyle='italic')
    #ax.legend()          
    ax.grid(axis='both', alpha=0.5)                         
    ax.tick_params(axis='x', direction='in')
    ax.tick_params(axis='y', direction='in')

    fig.autofmt_xdate() 
    plt.tight_layout()
    plot_path = folder + rl_model.replace('.pt', '_euler_foc.png')
    fig.savefig(plot_path)

    #vss = ss.ss_value(T)
    print(f"Steady state value = {v_ss}; Value reached by Grid = {grid_v}; ; Value reached by RL = {rl_v}")
    print(f"Pct welfare gain of RL to grid:{(rl_v - grid_v)*100/(grid_v)}") 




''' POLICY EVALUATION DETERMINISTIC'''
if run_policy == "yes":
    N = 200
    dev = 10
    c_values = np.zeros((N, 2))
    n_values = np.zeros((N, 2))
    k1_values = np.zeros((N, 2))
    v_values = np.zeros((N, 2))
    k_values = np.linspace(k_ss * (1-(dev/100)), k_ss * (1+(dev/100)), N)
    #z_psx = int(np.where(zgrid == 1)[0])
    for i in range(len(k_values)):
        #RL 
        st = np.array([1, k_values[i]])
        state = torch.from_numpy(st).float().to(device)
        with torch.no_grad():
            action_tensor, _ = agent.get_action(state, test=True)
            action_rl = action_tensor.squeeze().numpy()
            value_tensor = agent.get_value(state)
            value_rl = value_tensor.numpy()
        y_rl = (k_values[i]**ss.alpha)
        c_rl = (1 - action_rl) * y_rl
        k1_rl = (1 - ss.delta)*k_values[i] + action_rl * y_rl
        v_rl = float(value_rl)

        #Grid policy
        c_grid = float(optimal_c(k_values[i]))
        k1_grid = float(optimal_k1(k_values[i]))
        v_grid = float(optimal_v(k_values[i]))
        #save 
        c_values[i] = [c_rl, c_grid]
        k1_values[i] = [k1_rl, k1_grid]
        v_values[i] = [v_rl, v_grid]

    # How much the RL policy deviates from Grid for a 5% deviation from steady state
    p = len(k_values)-1
    k_diff = (k_values[p] - k_ss) / np.abs(k_ss)
    c_diff = (c_values[p, 0] - c_values[p, 1]) / np.abs(c_values[p, 1])
    print(f"Consumption deviation from grid policy for a {k_diff *100}% k deviation: {c_diff*100:.2f}%")
    

    #plotting
    #consumption
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.plot(k_values, c_values[:, 0], color='#ff6600', linewidth=1.5, label='RL', zorder = 4)
    ax.plot(k_values, c_values[:, 1], color='#003f5c', linewidth=1.5, label='Grid')
    ax.scatter(k_ss, c_ss, color='black', label='Steady State', s=20, zorder=5)
    ax.axvline(k_ss, color='black', linestyle=':', linewidth=1)
    ax.axhline(c_ss, color='black', linestyle=':', linewidth=1)
    ax.set_title("Consumption Rule", fontsize=16)
    ax.set_xlabel(r'$k_t$', fontstyle='italic')         
    ax.set_ylabel(r'$c_t$', fontstyle='italic')
    #ax.legend()          
    ax.grid(axis='both', alpha=0.5)                         
    ax.tick_params(axis='x', direction='in')
    ax.tick_params(axis='y', direction='in')
    
    fig.autofmt_xdate() 
    plt.tight_layout()
    plot_path = folder + rl_model.replace('.pt', '_cons_rule.png')
    fig.savefig(plot_path)


    #capital
    """ fig, ax = plt.subplots(figsize=(5, 6))
    ax.plot(k_values, k1_values[:, 0], color='crimson', linewidth=1.5, label='RL')
    ax.plot(k_values, k1_values[:, 1], color='blue', linewidth=1.5, label='Grid')
    ax.scatter(k_ss, k_ss, color='black', label='Steady State', s=20, zorder=5)
    ax.axvline(k_ss, color='black', linestyle=':', linewidth=1)
    ax.axhline(k_ss, color='black', linestyle=':', linewidth=1)
    ax.set_title("Saving Rule", fontsize=16)
    ax.set_xlabel(r'$k_t$', fontstyle='italic')         
    ax.set_ylabel(r'$k_{t+1}$', fontstyle='italic')
    #ax.legend()          
    ax.grid(axis='both', alpha=0.5)                          
    ax.tick_params(axis='x', direction='in')
    ax.tick_params(axis='y', direction='in')

    fig.autofmt_xdate() 
    plt.tight_layout()
    plot_path = folder + rl_model.replace('.pt', '_saving_rule.png')
    fig.savefig(plot_path) """

    #value function
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.plot(k_values, v_values[:, 0], color='#ff6600', linewidth=1.5, label='RL')
    ax.plot(k_values, v_values[:, 1], color='#003f5c', linewidth=1.5, label='Grid')
    ax.scatter(k_ss, v_ss, color='black', label='Steady State', s=20, zorder=5)
    ax.axvline(k_ss, color='black', linestyle=':', linewidth=1)
    ax.axhline(v_ss, color='black', linestyle=':', linewidth=1)
    ax.set_title("Value Function", fontsize=16)
    ax.set_xlabel(r'$k_t$', fontstyle='italic')         
    ax.set_ylabel(r'$V(k_t)$', fontstyle='italic')
    #ax.legend()          
    ax.grid(axis='both', alpha=0.5)                          
    ax.tick_params(axis='x', direction='in')
    ax.tick_params(axis='y', direction='in')

    fig.autofmt_xdate() 
    plt.tight_layout()
    plot_path = folder + rl_model.replace('.pt', '_value_function.png')
    fig.savefig(plot_path)



''' GLOBAL POLICY '''
if global_policy == "yes":
    N = 200
    dev = 20
    c_values = np.zeros((N, 2))
    n_values = np.zeros((N, 2))
    k1_values = np.zeros((N, 2))
    v_values = np.zeros((N, 2))
    k_values = np.linspace(1, k_ss * (1+(dev/100)), N)

    for i in range(len(k_values)):
        #RL 
        st = np.array([1, k_values[i]])
        state = torch.from_numpy(st).float().to(device)
        with torch.no_grad():
            action_tensor, _ = agent.get_action(state, test=True)
            action_rl = action_tensor.squeeze().numpy()
            value_tensor = agent.get_value(state)
            value_rl = value_tensor.numpy()
        y_rl = (k_values[i]**ss.alpha)
        c_rl = (1 - action_rl) * y_rl
        k1_rl = (1 - ss.delta)*k_values[i] + action_rl * y_rl
        v_rl = float(value_rl)

        #Grid policy
        c_grid = float(optimal_c(k_values[i]))
        k1_grid = float(optimal_k1(k_values[i]))
        v_grid = float(optimal_v(k_values[i]))
        #save 
        c_values[i] = [c_rl, c_grid]
        k1_values[i] = [k1_rl, k1_grid]
        v_values[i] = [v_rl, v_grid]

    # How much the RL policy deviates from Grid for a 5% deviation from steady state
    p = len(k_values)-1
    k_diff = (k_values[p] - k_ss) / np.abs(k_ss)
    c_diff = (c_values[p, 0] - c_values[p, 1]) / np.abs(c_values[p, 1])
    print(f"Consumption deviation from grid policy for a {k_diff *100}% k deviation: {c_diff*100:.2f}%")
    

    #plotting
    #consumption
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.plot(k_values, c_values[:, 0], color='#ff6600', linewidth=1.5, label='RL', zorder = 4)
    ax.plot(k_values, c_values[:, 1], color='#003f5c', linewidth=1.5, label='Grid')
    ax.scatter(k_ss, c_ss, color='black', label='Steady State', s=20, zorder=5)
    ax.axvline(k_ss, color='black', linestyle=':', linewidth=1)
    ax.axhline(c_ss, color='black', linestyle=':', linewidth=1)
    ax.set_title("Consumption Rule", fontsize=16)
    ax.set_xlabel(r'$k_t$', fontstyle='italic')         
    ax.set_ylabel(r'$c_t$', fontstyle='italic')
    #ax.legend()          
    ax.grid(axis='both', alpha=0.5)                         
    ax.tick_params(axis='x', direction='in')
    ax.tick_params(axis='y', direction='in')
    
    fig.autofmt_xdate() 
    plt.tight_layout()
    plot_path = folder + rl_model.replace('.pt', '_global_rule.png')
    fig.savefig(plot_path)


    #value function
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.plot(k_values, v_values[:, 0], color='#ff6600', linewidth=1.5, label='RL', zorder = 4)
    ax.plot(k_values, v_values[:, 1], color='#003f5c', linewidth=1.5, label='Grid')
    ax.scatter(k_ss, v_ss, color='black', label='Steady State', s=20, zorder=5)
    ax.axvline(k_ss, color='black', linestyle=':', linewidth=1)
    ax.axhline(v_ss, color='black', linestyle=':', linewidth=1)
    ax.set_title("Value Function", fontsize=16)
    ax.set_xlabel(r'$k_t$', fontstyle='italic')         
    ax.set_ylabel(r'$V(k_t)$', fontstyle='italic')
    #ax.legend()          
    ax.grid(axis='both', alpha=0.5)                          
    ax.tick_params(axis='x', direction='in')
    ax.tick_params(axis='y', direction='in')

    fig.autofmt_xdate() 
    plt.tight_layout()
    plot_path = folder + rl_model.replace('.pt', '_global_value.png')
    fig.savefig(plot_path)



