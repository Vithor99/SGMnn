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

# Be careful: steady must be alligned to what we are plotting here. 
'''CONTROLS'''
rl_model = 'SGM_steady_deterministic.pt' 
grid_model = 'Grid_SGM_deterministic.pkl'
#folder to store plots 
folder = 'SGM_plots/'

run_simulation = "yes" #if yes it runs the simulation
if run_simulation == "yes":
    T = 500
    dev = 5 #Pct initial deviation from steady state
    zoom = "in" # in or out: if out zoom factor is activated 
    zoom_factor = 10 # if zoom == "out": pct band around ss that we want to visualize
    run_foc = "yes" # if yes it runs the focs of the model
    n = 200 # number of periods to compute the steady state of the simulation

run_policy = "yes" # if yes it runs the policy evaluation
if run_policy == "yes":
    dev = 5
    N = 500

run_policy_sto = "no"
if run_policy_sto == "yes": 
    dev = 5 
    N = 500


run_add_analysis = "no" # if yes it runs some other stuff 


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


# Loading Grid (vi) policy
#need to update for stochastic version 
grid_model_path = 'saved_models/' + grid_model
with open(grid_model_path, 'rb') as f:
    loaded_data = pickle.load(f)

#zgrid = ss.tauchenhussey(N=ss.nbz)[0]   # Discretized z values
#Pi = ss.tauchenhussey(N=ss.nbz)[1]      # Transition probabilities

k = loaded_data['st']
k1_star = loaded_data['k1_star']
a_star = loaded_data['a_star']
value_star = loaded_data['value_star']
optimal_k1 = interp(k, k1_star)
optimal_a = interp(k, a_star)
optimal_v = interp(k, value_star)
@np.vectorize
def optimal_c(k): 
    return float((1 - optimal_a(k)) * (k**ss.alpha)) 


''' SIMULATION '''
if run_simulation == "yes":
    st = np.array([k_ss*(1+(dev/100)), k_ss*(1+(dev/100))])
    z = 1 #we want the same series for productivity in the two economies
    #z_psx = int(np.where(zgrid == 1)[0])
    grid_sim={}
    rl_sim={}
    grid_v = 0
    rl_v = 0
    #labour_gap = np.zeros((T-1, 2))
    euler_gap = np.zeros((T-1, 2))
    

    for t in range(T):
        #RL 
        k_rl = st[0]
        state = torch.from_numpy(np.array([z, k_rl])).float().to(device)
        with torch.no_grad():
            action_tensor, _ = agent.get_action(state, test=True)
            sratio_rl = action_tensor.squeeze().numpy()
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
        

        if run_foc == "yes":
            if t>0:
                e_gap_grid = ss.foc_log(grid_sim[t-1]['c'], grid_sim[t]['c'],  
                                        z, grid_sim[t]['k'])
                
                e_gap_rl = ss.foc_log(rl_sim[t-1]['c'], rl_sim[t]['c'],  
                                        z, rl_sim[t]['k'])
                #labour_gap[t-1, 1] = l_gap
                euler_gap[t-1, 0] = e_gap_rl
                euler_gap[t-1, 1] = e_gap_grid

        st = np.array([k1_rl, k1_grid])
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
    plot_path = folder + rl_model.replace('.pt', '_consumption.png')
    fig.savefig(plot_path)

    """ #labour
    n_grid = [entry['a'][1] for entry in grid_sim.values()]
    n_rl = [entry['a'][1] for entry in rl_sim.values()]

    #distance from steady state
    rl_ss = np.mean(n_rl[-n:])
    grid_ss = np.mean(n_grid[-n:])
    ss_dev = (rl_ss - grid_ss) / np.abs(grid_ss)
    print(f"Labour distance from steady state: {ss_dev*100:.2f}%")

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
    fig.savefig(plot_path) """

    if run_foc == "yes":
        """ #labour supply foc
        fig, ax = plt.subplots(figsize=(5, 6))  
        ax.plot(labour_gap[:, 0], color='#F08080', linewidth=1.0, alpha = 0.35,  label='RL')
        ax.plot(labour_gap[:, 1], color='blue', linewidth=1.5, label='Grid')

        # Smooth labour_gap[:, 0] with exponential smoothing
        alpha = 0.2  # Smoothing parameter
        smoothed_labour_gap = np.zeros_like(labour_gap[:, 0])
        smoothed_labour_gap[0] = labour_gap[0, 0]
        for i in range(1, len(labour_gap[:, 0])):
            smoothed_labour_gap[i] = alpha * labour_gap[i, 0] + (1 - alpha) * smoothed_labour_gap[i - 1]
        
        ax.plot(smoothed_labour_gap, color='crimson', linewidth=1.5, label='Smoothed RL')

        ax.axhline(0, color="black", linewidth=1.2, linestyle = '--')
        ax.set_title("Distance from Labour Supply", fontsize=16)
        ax.set_xlabel("Periods", fontstyle='italic')         
        ax.set_ylabel(r'$\% \Delta \ \ c_t$', fontstyle='italic')
        ax.legend()          
        ax.grid(axis='both', alpha=0.5)                         
        ax.tick_params(axis='x', direction='in')
        ax.tick_params(axis='y', direction='in')

        fig.autofmt_xdate() 
        plt.tight_layout()
        plot_path = 'plots/' + rl_model.replace('.pt', '_lab_supply_foc.png')
        fig.savefig(plot_path) """

        #euler foc
        fig, ax = plt.subplots(figsize=(5, 6))  
        ax.plot(euler_gap[:, 0], color='#F08080', linewidth=1.0, alpha = 0.35,  label='RL')
        ax.plot(euler_gap[:, 1], color='blue', linewidth=1.5, label='Grid')

        alpha = 0.1  # Smoothing parameter
        smoothed_euler_gap = np.zeros_like(euler_gap[:, 0])
        smoothed_euler_gap[0] = euler_gap[0, 0]
        for i in range(1, len(euler_gap[:, 0])):
            smoothed_euler_gap[i] = alpha * euler_gap[i, 0] + (1 - alpha) * smoothed_euler_gap[i - 1]
        
        ax.plot(smoothed_euler_gap, color='crimson', linewidth=1.5, label='Smoothed RL')

        ax.axhline(0, color="black", linewidth=1.2, linestyle = '--')
        ax.set_title("Distance from Euler", fontsize=16)
        ax.set_xlabel("Periods", fontstyle='italic')         
        ax.set_ylabel(r'$\% \Delta \ \ \frac{c_{t+1}}{c_t}$', fontstyle='italic')
        ax.legend()          
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
        #c_rl = action_rl[0]
        #n_rl = action_rl[1]
        #k1_rl = (1-ss.delta)*st[1] + st[0]*(action_rl[1]**(1-ss.alpha))*(st[1]**ss.alpha) - action_rl[0]
        #v_rl = float(value_rl)

        #Grid 
        """ n_grid = optimal_n(k_values[i])[z_psx]
        c_grid = optimal_c(k_values[i])[z_psx]
        k1_grid = optimal_kp(k_values[i])[z_psx]
        v_grid = optimal_v(k_values[i])[z_psx] """
        c_grid = float(optimal_c(k_values[i]))
        k1_grid = float(optimal_k1(k_values[i]))
        v_grid = float(optimal_v(k_values[i]))
        #save 
        c_values[i] = [c_rl, c_grid]
        #n_values[i] = [n_rl, n_grid]
        k1_values[i] = [k1_rl, k1_grid]
        v_values[i] = [v_rl, v_grid]

    # How much the RL policy deviates from Grid for a 5% deviation from steady state
    p = len(k_values)-1
    k_diff = (k_values[p] - k_ss) / np.abs(k_ss)
    c_diff = (c_values[p, 0] - c_values[p, 1]) / np.abs(c_values[p, 1])
    #n_diff = (n_values[p, 0] - n_values[p, 1]) / np.abs(n_values[p, 1])
    print(f"Consumption deviation from grid policy for a {k_diff *100}% k deviation: {c_diff*100:.2f}%")
    #print(f"Labour deviation from grid policy for a {k_diff *100}% k deviation: {n_diff*100:.2f}%")
    #plotting
    #consumption
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.plot(k_values, c_values[:, 0], color='crimson', linewidth=1.5, label='RL')
    ax.plot(k_values, c_values[:, 1], color='blue', linewidth=1.5, label='Grid')
    ax.scatter(k_ss, c_ss, color='black', label='Steady State', s=20, zorder=5)
    ax.axvline(k_ss, color='black', linestyle=':', linewidth=1)
    ax.axhline(c_ss, color='black', linestyle=':', linewidth=1)
    ax.set_title("Consumption Rule", fontsize=16)
    ax.set_xlabel(r'$k_t$', fontstyle='italic')         
    ax.set_ylabel(r'$c_t$', fontstyle='italic')
    ax.legend()          
    ax.grid(axis='both', alpha=0.5)                         
    ax.tick_params(axis='x', direction='in')
    ax.tick_params(axis='y', direction='in')
    
    fig.autofmt_xdate() 
    plt.tight_layout()
    plot_path = folder + rl_model.replace('.pt', '_cons_rule.png')
    fig.savefig(plot_path)

    """ #labour
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.plot(k_values, n_values[:, 0], color='crimson', linewidth=1.5, label='RL')
    ax.plot(k_values, n_values[:, 1], color='blue', linewidth=1.5, label='Grid')
    ax.scatter(k_ss, n_ss, color='black', label='Steady State', s=20, zorder=5)
    ax.axvline(k_ss, color='black', linestyle=':', linewidth=1)
    ax.axhline(n_ss, color='black', linestyle=':', linewidth=1)
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
    fig.savefig(plot_path) """

    #capital
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.plot(k_values, k1_values[:, 0], color='crimson', linewidth=1.5, label='RL')
    ax.plot(k_values, k1_values[:, 1], color='blue', linewidth=1.5, label='Grid')
    ax.scatter(k_ss, k_ss, color='black', label='Steady State', s=20, zorder=5)
    ax.axvline(k_ss, color='black', linestyle=':', linewidth=1)
    ax.axhline(k_ss, color='black', linestyle=':', linewidth=1)
    ax.set_title("Saving Rule", fontsize=16)
    ax.set_xlabel(r'$k_t$', fontstyle='italic')         
    ax.set_ylabel(r'$k_{t+1}$', fontstyle='italic')
    ax.legend()          
    ax.grid(axis='both', alpha=0.5)                          
    ax.tick_params(axis='x', direction='in')
    ax.tick_params(axis='y', direction='in')

    fig.autofmt_xdate() 
    plt.tight_layout()
    plot_path = folder + rl_model.replace('.pt', '_saving_rule.png')
    fig.savefig(plot_path)

    #value function
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.plot(k_values, v_values[:, 0], color='crimson', linewidth=1.5, label='RL')
    ax.plot(k_values, v_values[:, 1], color='blue', linewidth=1.5, label='Grid')
    ax.scatter(k_ss, v_ss, color='black', label='Steady State', s=20, zorder=5)
    ax.axvline(k_ss, color='black', linestyle=':', linewidth=1)
    ax.axhline(v_ss, color='black', linestyle=':', linewidth=1)
    ax.set_title("Value Function", fontsize=16)
    ax.set_xlabel(r'$k_t$', fontstyle='italic')         
    ax.set_ylabel(r'$V(k_t)$', fontstyle='italic')
    ax.legend()          
    ax.grid(axis='both', alpha=0.5)                          
    ax.tick_params(axis='x', direction='in')
    ax.tick_params(axis='y', direction='in')

    fig.autofmt_xdate() 
    plt.tight_layout()
    plot_path = 'plots/' + rl_model.replace('.pt', '_value_function.png')
    fig.savefig(plot_path)



""" '''POLICY EVALUATION STOCHASTIC'''
if run_policy_sto == "yes":
    c_values_rl = np.zeros((N, ss.nbz))
    c_values_grid = np.zeros((N, ss.nbz))
    n_values_rl = np.zeros((N, ss.nbz))
    n_values_grid = np.zeros((N, ss.nbz))
    k1_values_rl =  np.zeros((N, ss.nbz))
    k1_values_grid = np.zeros((N, ss.nbz))
    v_values_rl = np.zeros((N, ss.nbz))
    v_values_grid = np.zeros((N, ss.nbz))
    k_values = np.linspace(k_ss * (1-(dev/100)), k_ss * (1+(dev/100)), N)
    #z_psx = int(np.where(zgrid == 1)[0])
    for j in range(ss.nbz):
        for i in range(len(k_values)):
            #RL 
            st = np.array([zgrid[j], k_values[i]])
            state = torch.from_numpy(st).float().to(device)
            with torch.no_grad():
                action_tensor, _ = agent.get_action(state, test=True)
                action_rl = action_tensor.squeeze().numpy()
                value_tensor = agent.get_value(state)
                value_rl = value_tensor.numpy()
            c_values_rl[i,j] = action_rl[0]
            n_values_rl[i,j] = action_rl[1]
            k1_values_rl[i,j] = (1-ss.delta)*st[1] + st[0]*(action_rl[1]**(1-ss.alpha))*(st[1]**ss.alpha) - action_rl[0]
            v_values_rl[i,j] = float(value_rl)
            #Grid 
            n_values_grid[i,j] = optimal_n(k_values[i])[j]
            c_values_grid[i,j] = optimal_c(k_values[i])[j]
            k1_values_grid[i,j] = optimal_kp(k_values[i])[j]
            v_values_grid[i,j] = optimal_v(k_values[i])[j]
            #save 
            #c_values_rl[i,j] = c_rl
            #c_values_grid[i,j] = c_grid
            #n_values_rl[i,j] = n_rl
            #n_values_grid[i,j] = n_grid
            #k1_values_rl[i,j] = k1_rl
            #k1_values_grid[i,j] = k1_grid
            #v_values_rl[i,j] = v_rl
            #v_values_grid[i,j] = v_grid


    # How much the RL policy deviates from Grid for a 5% deviation from steady state
    #p = len(k_values)-1
    #k_diff = (k_values[p] - k_ss) / np.abs(k_ss)
    #c_diff = (c_values[p, 0] - c_values[p, 1]) / np.abs(c_values[p, 1])
    #n_diff = (n_values[p, 0] - n_values[p, 1]) / np.abs(n_values[p, 1])
    #print(f"Consumption deviation from grid policy for a {k_diff *100}% k deviation: {c_diff*100:.2f}%")
    #print(f"Labour deviation from grid policy for a {k_diff *100}% k deviation: {n_diff*100:.2f}%")
    #plotting
    #consumption
    fig, ax = plt.subplots(figsize=(5, 6))
    for i in range(ss.nbz): 
        ax.plot(k_values, c_values_rl[:, i], color=f'C{i}', linewidth=1.5) #label=f'RL z={zgrid[i]}'
        ax.plot(k_values, c_values_grid[:, i], color=f'C{i}', linestyle='--', linewidth=1.5)

    ax.scatter(k_ss, c_ss, color='black', label='Steady State', s=20, zorder=5)
    ax.axvline(k_ss, color='black', linestyle=':', linewidth=1)
    ax.axhline(c_ss, color='black', linestyle=':', linewidth=1)
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
    for i in range(ss.nbz): 
        ax.plot(k_values, n_values_rl[:, i], color=f'C{i}', linewidth=1.5)
        ax.plot(k_values, n_values_grid[:, i], color=f'C{i}', linestyle='--', linewidth=1.5)
    ax.scatter(k_ss, n_ss, color='black', label='Steady State', s=20, zorder=5)
    ax.axvline(k_ss, color='black', linestyle=':', linewidth=1)
    ax.axhline(n_ss, color='black', linestyle=':', linewidth=1)
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
    for i in range(ss.nbz): 
        ax.plot(k_values, k1_values_rl[:, i], color=f'C{i}', linewidth=1.5)
        ax.plot(k_values, k1_values_grid[:, i], color=f'C{i}', linestyle='--', linewidth=1.5)
    ax.scatter(k_ss, k_ss, color='black', label='Steady State', s=20, zorder=5)
    ax.axvline(k_ss, color='black', linestyle=':', linewidth=1)
    ax.axhline(k_ss, color='black', linestyle=':', linewidth=1)
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

    #value function
    fig, ax = plt.subplots(figsize=(5, 6))
    for i in range(ss.nbz): 
        ax.plot(k_values, v_values_rl[:, i], color=f'C{i}', linewidth=1.5)
        ax.plot(k_values, v_values_grid[:, i], color=f'C{i}', linestyle='--', linewidth=1.5)
    ax.scatter(k_ss, v_ss, color='black', label='Steady State', s=20, zorder=5)
    ax.axvline(k_ss, color='black', linestyle=':', linewidth=1)
    ax.axhline(v_ss, color='black', linestyle=':', linewidth=1)
    ax.set_title("Value Function", fontsize=16)
    ax.set_xlabel(r'$k_t$', fontstyle='italic')         
    ax.set_ylabel(r'$V(k_t)$', fontstyle='italic')
    ax.legend()          
    ax.grid(axis='both', alpha=0.5)                          
    ax.tick_params(axis='x', direction='in')
    ax.tick_params(axis='y', direction='in')

    fig.autofmt_xdate() 
    plt.tight_layout()
    plot_path = 'plots/' + rl_model.replace('.pt', '_value_function.png')
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

 """