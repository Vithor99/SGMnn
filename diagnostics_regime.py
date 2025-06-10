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
rl_model = 'SGM_prepoc3_steady_regime.pt' 
grid_model = 'Grid_SGM_regime.pkl'
#folder to store plots 
folder = 'SGM_plots/'

#zoom = "in" #this needs to be adjusted

run_simulation = "yes" #if yes it runs the simulation

run_policy = "yes" # if yes it runs the policy evaluation

global_policy = "no" #needs to be run with appropriate grid solution

#run_add_analysis = "no" # if yes it runs some other stuff


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

tau = ss.tau
pi_tau = ss.pi_tau

'''' Define Model'''
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


''' LOADING GRID MODEL'''
#need to update for stochastic version 
grid_model_path = 'saved_models/' + grid_model
with open(grid_model_path, 'rb') as f:
    loaded_data = pickle.load(f)

nbr = ss.nbr
rgrid = ss.regimes()[0]   # Discretized z values
Pi = ss.regimes()[1]      # Transition probabilities

k = loaded_data['st']
k1_star = loaded_data['k1_star']
a_star = loaded_data['a_star']
c_star = loaded_data['c_star']
value_star = loaded_data['value_star']

optimal_a = interp2d((rgrid, k), a_star.T)
optimal_k1 = interp2d((rgrid, k), k1_star.T)
optimal_v = interp2d((rgrid, k), value_star.T)
optimal_c = interp2d((rgrid, k), c_star.T)

''' COMPUTING THE STEADY STATE OF RL'''
k_rl = k_ss
K = np.zeros(1000)
C = np.zeros(1000)
U = np.zeros(1000)
for t in range(1000):
    #RL 
    state_rl = torch.from_numpy(np.array([rgrid[0], k_rl])).float().to(device)
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
k_ss_rl_ante = np.mean(K[-100:])
c_ss_rl_ante = np.mean(C[-100:])
v_ss_rl_ante = (1 / (1 - ss.beta)) * np.mean(U[-100:])

k_rl = k_ss
K = np.zeros(1000)
C = np.zeros(1000)
U = np.zeros(1000)
for t in range(1000):
    #RL 
    state_rl = torch.from_numpy(np.array([rgrid[1], k_rl])).float().to(device)
    with torch.no_grad():
        action_tensor, _ = agent.get_action(state_rl, test=True)
        sratio_rl = action_tensor.squeeze().numpy()
    k1_rl = (1 - ss.delta)*k_rl + rgrid[1] * (k_rl**ss.alpha) * sratio_rl
    c_rl = rgrid[1] * (k_rl**ss.alpha) * (1 - sratio_rl)
    u_rl = np.log(c_rl)
    K[t] = k1_rl
    C[t] = c_rl
    U[t] = u_rl
    k_rl = k1_rl
k_ss_rl = np.mean(K[-100:])
c_ss_rl = np.mean(C[-100:])
v_ss_rl = (1 / (1 - ss.beta)) * np.mean(U[-100:])
''' COMPUTING THE SS OF GRID BEFORE TAX '''
k_grid = k_ss
K = np.zeros(1000)
C = np.zeros(1000)
U = np.zeros(1000)
for t in range(1000):
    #GRID
    state_grid = np.array([rgrid[0], k_grid])
    sratio_grid = float(optimal_a(state_grid))
    k1_grid = (1 - ss.delta)*k_grid + (k_grid**ss.alpha) * sratio_grid
    c_grid = (k_grid**ss.alpha) * (1 - sratio_grid)
    u_grid = np.log(c_grid)
    K[t] = k1_grid
    C[t] = c_grid
    U[t] = u_grid
    k_grid = k1_grid
k_ss_ante = np.mean(K[-100:])
c_ss_ante = np.mean(C[-100:])
v_ss_ante = (1/(1-ss.beta)) * np.mean(U[-100:])


''' SIMULATION '''
if run_simulation == "yes":
    # RL, Grid
    foc = "yes"
    T = 500
    st = np.array([k_ss_rl_ante, k_ss_ante]) # pre tax steady state capital
    k0_rl_i = np.ones(100) *  k_ss_rl_ante
    r = rgrid[0]
    grid_sim={}
    rl_sim={}
    foc_sim={}
    grid_v = 0
    rl_v = 0
    irf_ci = np.zeros((T, 100))
    irf_ki = np.zeros((T, 100))
    #euler_gap = np.zeros((T-1, 2))
    

    for t in range(T):
        #RL 
        k_rl = st[0]
        st_rl_i = np.column_stack((np.full_like(k0_rl_i , r), k0_rl_i))
        state_rl = torch.from_numpy(np.array([r, k_rl])).float().to(device)
        state_i = torch.from_numpy(st_rl_i).float().to(device)
        with torch.no_grad():
            action_tensor, _ = agent.get_action(state_rl, test=True)
            sratio_rl = action_tensor.squeeze().numpy()

            actions_i = np.zeros(100)
            for i in range(100):
                action_i_tens, _ = agent.get_action(state_i[i], test= False)
                action_i = action_i_tens.squeeze().numpy()
                actions_i[i] = action_i

        y_rl = r * (k_rl**ss.alpha)
        c_rl = y_rl * (1-sratio_rl)
        u_rl = ss.gamma*np.log(c_rl)
        k1_rl = (1 - ss.delta)*k_rl + y_rl - c_rl
        rl_v += ss.beta**t * u_rl
        rl_sim[t] = {'k': k_rl,
                    'r': r,
                    'c': c_rl,
                    'y': y_rl,
                    'u': u_rl,
                    'st1': k1_rl}
        
        y_rl_i = r * (k0_rl_i**ss.alpha)
        c_rl_i = (1 - actions_i) * y_rl_i
        k1_rl_i =  (1 - ss.delta)*k0_rl_i + actions_i * y_rl_i
        # Storing the i IRFs 
        irf_ci[t] = c_rl_i
        irf_ki[t] = k1_rl_i

        #Grid 
        # Find the position of the value 1 in zgrid
        k_grid = st[1]
        y_grid = r * (k_grid**ss.alpha)
        state_grid = np.array([r, float(k_grid)])
        #c_grid = float(optimal_c(k_grid))
        c_grid = float(optimal_c(state_grid))
        u_grid = ss.gamma*np.log(c_grid) #+ ss.psi*np.log(1-a[1])
        #k1_grid = float(optimal_k1(state_grid))
        k1_grid = (1 - ss.delta)*k_grid + y_grid - c_grid
        grid_v += ss.beta**t * u_grid
        grid_sim[t] = {'k': k_grid,
                    'r': r,
                    'c': c_grid,
                    'y': y_grid,
                    'u': u_grid,
                    'st1': k1_grid}
        
        if foc == "yes": 
            mu0 = 1/c_rl
            mu1 = np.zeros(ss.nbr)
            pi = ss.next_regime_prob(r)
            for i in range(ss.nbr): 
                with torch.no_grad():
                    st_tensor_foc = torch.from_numpy(np.array([rgrid[i], k1_rl])).float().to(device)
                    action_tensor_foc, _ = agent.get_action(st_tensor_foc, test=True)
                    a_foc = action_tensor_foc.squeeze().numpy()
                    y = rgrid[i]* (k1_rl**ss.alpha)
                    c1 = y * (1-a_foc)
                    mu1[i] = 1/c1
            R1 = (1-ss.delta) + rgrid * ss.alpha * (k1_rl**(ss.alpha-1))
            EPS = np.sum(pi * (mu1*R1))
            euler_gap = (mu0 - ss.beta * EPS)**2
            foc_sim[t] = {'resid': euler_gap}
        
        r1 = ss.next_regime(r)
        r = r1
        st = np.array([k1_rl, k1_grid])
        k0_rl_i = k1_rl_i
    


    #Plotting
    #capital
    k_grid = [entry['k'] for entry in grid_sim.values()]
    k_rl = [entry['k'] for entry in rl_sim.values()]

    """ #distance from steady state
    rl_ss = np.mean(k_rl[-n:])
    grid_ss = np.mean(k_grid[-n:])
    ss_dev = (rl_ss - grid_ss) / np.abs(grid_ss)
    print(f"Capital distance from steady state: {ss_dev*100:.2f}%") """
    
    fig, ax = plt.subplots(figsize=(5, 6))  
    ax.plot(k_grid, color='#003f5c', linewidth=1.5, label='Grid', zorder = 4)
    ax.plot(k_rl, color='#ff6600', linewidth=1.5, label='RL', zorder = 5)
    ax.plot(irf_ki, color = "#ff9440", linewidth= 0.5, alpha = 0.05, label='RL')
    ax.axhline(k_ss, color="#003f5c", linewidth=1.2, linestyle='--',label='Steady State')
    ax.axhline(k_ss_rl, color="#ff6600", linewidth=1.2, linestyle='--',label='Steady State RL')
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
    """ rl_ss = np.mean(c_rl[-n:])
    grid_ss = np.mean(c_grid[-n:])
    ss_dev = (rl_ss - grid_ss) / np.abs(grid_ss)
    print(f"Consumption distance from steady state: {ss_dev*100:.2f}%") """

    fig, ax = plt.subplots(figsize=(5, 6))  
    ax.plot(c_grid, color='#003f5c', linewidth=1.5, label='Grid', zorder = 4)
    ax.plot(c_rl, color='#ff6600', linewidth=1.5, label='RL', zorder = 5)
    ax.plot(irf_ci, color = "#ff9440", linewidth= 0.5, alpha = 0.05, label='RL')
    ax.axhline(c_ss, color="#003f5c", linewidth=1.2, linestyle='--', label='Steady State')
    ax.axhline(c_ss_rl, color="#ff6600", linewidth=1.2, linestyle='--',label='Steady State RL')
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

    # Plotting Euler residuals 
    resids = [entry['resid'] for entry in foc_sim.values()]

    fig, ax = plt.subplots(figsize=(5, 6))  
    ax.plot(resids, color='#003f5c', linewidth=1.0, alpha = 0.3,  label='RL')

    alpha = 0.1  # Smoothing parameter
    smoothed_euler_gap = np.zeros_like(resids)
    smoothed_euler_gap[0] = resids[0]
    for i in range(1, len(resids)):
        smoothed_euler_gap[i] = alpha * resids[i] + (1 - alpha) * smoothed_euler_gap[i - 1]
    
    ax.plot(smoothed_euler_gap, color='#003f5c', linewidth=1.5, label='Smoothed RL')

    ax.set_title("Euler residuals", fontsize=16)
    ax.set_xlabel("Periods", fontstyle='italic')         
    ax.set_ylabel(r'$Euler \ \ Residuals$', fontstyle='italic')
    #ax.legend()          
    ax.grid(axis='both', alpha=0.5)                          
    ax.tick_params(axis='x', direction='in')
    ax.tick_params(axis='y', direction='in')

    fig.autofmt_xdate() 
    plt.tight_layout()
    plot_path = folder + rl_model.replace('.pt', '_euler.png')
    fig.savefig(plot_path)


    #vss = ss.ss_value(T)
    print(f"Steady state value = {v_ss}; Value reached by Grid = {grid_v}; ; Value reached by RL = {rl_v}")
    print(f"Pct welfare gain of RL to grid:{(rl_v - grid_v)*100/(grid_v)}") 



''' POLICY EVALUATION STOCHASTIC'''
if run_policy == "yes":
    N = 200
    dev = 10

    c_values_grid = np.zeros((N, ss.nbr))
    k1_values_grid = np.zeros((N, ss.nbr))
    v_values_grid = np.zeros((N, ss.nbr))
    c_values_rl = np.zeros((N, ss.nbr))
    k1_values_rl = np.zeros((N, ss.nbr))
    v_values_rl = np.zeros((N, ss.nbr))


    k_values = np.linspace(k_ss * (1-(dev/100)), k_ss * (1+(dev/100)), N)
    #z_psx = int(np.where(zgrid == 1)[0])
    for j in range(ss.nbr): 
        for i in range(len(k_values)):
            #RL 
            st = np.array([rgrid[j], k_values[i]])
            state = torch.from_numpy(st).float().to(device)
            with torch.no_grad():
                action_tensor, _ = agent.get_action(state, test=True)
                action_rl = action_tensor.squeeze().numpy()
                value_tensor = agent.get_value(state)
                value_rl = value_tensor.numpy()
            y_rl = rgrid[j] * (k_values[i]**ss.alpha)
            c_rl = (1 - action_rl) * y_rl
            k1_rl = (1 - ss.delta)*k_values[i] + action_rl * y_rl
            v_rl = float(value_rl)

            #Grid 
            c_grid = optimal_c(st)
            k1_grid = optimal_k1(st)
            v_grid = optimal_v(st)

            #save 
            c_values_grid[i,j] = c_grid
            c_values_rl[i,j] = c_rl
            k1_values_grid[i,j] = k1_grid
            k1_values_rl[i,j] = k1_rl 
            v_values_grid[i,j] = v_grid
            v_values_rl[i,j] = v_rl


        # How much the RL policy deviates from Grid for a 5% deviation from steady state
        p = len(k_values)-1
        z_diff = (rgrid[j] - 1)
        k_diff = (k_values[p] - k_ss) / np.abs(k_ss)
        c_diff = (c_values_rl[p, j] - c_values_grid[p, j]) / np.abs(c_values_grid[p, j])
        #n_diff = (n_values[p, 0] - n_values[p, 1]) / np.abs(n_values[p, 1])
        print(f"Consumption deviation from grid policy for a {k_diff *100}% k deviation and a {z_diff *100}% z deviation: {c_diff*100:.2f}%")
        #print(f"Labour deviation from grid policy for a {k_diff *100}% k deviation: {n_diff*100:.2f}%")



    #plotting
    #consumption
    fig, ax = plt.subplots(figsize=(5, 6))
    palette = ("#ff6600", "#ffb84d")
    for i in range(len(c_values_rl[0,:])):
        ax.plot(k_values, c_values_rl[:, i], color = palette[i],  linewidth=1.5, label='RL')
        ax.plot(k_values, c_values_grid[:, i], color = palette[i], linestyle = 'dashed', linewidth=1.5, label='Grid')
    
    """ ax.plot(k_values, c_values_rl[:, :], color = "#ff6600",  linewidth=1.5, label='RL')
    ax.plot(k_values, c_values_grid[:, :], color = "#003f5c", linewidth=1.5, label='Grid') """
    ax.scatter(k_ss, c_ss, marker='o', facecolors='none', edgecolors='#003f5c', s=40, linewidths=1.5, zorder = 5)
    ax.scatter(k_ss_rl, c_ss_rl, marker='o', facecolors='#003f5c', edgecolors='#003f5c', s=40, linewidths=1.5, zorder = 5)
    ax.scatter(k_ss_rl_ante, c_ss_rl_ante, marker='D', facecolors='#003f5c', edgecolors='#003f5c', s=40, linewidths=1.5, zorder = 5)
    ax.scatter(k_ss_ante, c_ss_ante, marker='D', facecolors='none', edgecolors='#003f5c', s=40, linewidths=1.5, zorder = 5)
    #ax.axvline(k_ss, color='#003f5c', linestyle=':', linewidth=1, alpha = 0.5)
    #ax.axhline(c_ss, color='#003f5c', linestyle=':', linewidth=1, alpha = 0.5)
    #ax.axvline(k_ss_rl, color='#ff6600', linestyle=':', linewidth=1, alpha = 0.5)
    #ax.axhline(c_ss_rl, color='#ff6600', linestyle=':', linewidth=1, alpha = 0.5)
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

    #value
    fig, ax = plt.subplots(figsize=(5, 6))
    palette = ("#ff6600", "#ffb84d")
    for i in range(len(v_values_rl[0,:])):
        ax.plot(k_values, v_values_rl[:, i], color = palette[i],  linewidth=1.5, label='RL')
        ax.plot(k_values, v_values_grid[:, i], color = palette[i], linestyle = 'dashed', linewidth=1.5, label='Grid')
    
    """ ax.plot(k_values, c_values_rl[:, :], color = "#ff6600",  linewidth=1.5, label='RL')
    ax.plot(k_values, c_values_grid[:, :], color = "#003f5c", linewidth=1.5, label='Grid') """
    #ax.scatter(k_ss, v_ss, marker='o', facecolors='none', edgecolors='#003f5c', s=40, linewidths=1.5, zorder = 5)
    #ax.scatter(k_ss_rl, v_ss_rl, marker='o', facecolors='#003f5c', edgecolors='#003f5c', s=40, linewidths=1.5, zorder = 5)
    #ax.scatter(k_ss_rl_ante, v_ss_rl_ante, marker='D', facecolors='#003f5c', edgecolors='#003f5c', s=40, linewidths=1.5, zorder = 5)
    #ax.scatter(k_ss_ante, v_ss_ante, marker='D', facecolors='none', edgecolors='#003f5c', s=40, linewidths=1.5, zorder = 5)
    #ax.axvline(k_ss, color='#003f5c', linestyle=':', linewidth=1, alpha = 0.5)
    #ax.axhline(c_ss, color='#003f5c', linestyle=':', linewidth=1, alpha = 0.5)
    #ax.axvline(k_ss_rl, color='#ff6600', linestyle=':', linewidth=1, alpha = 0.5)
    #ax.axhline(c_ss_rl, color='#ff6600', linestyle=':', linewidth=1, alpha = 0.5)
    ax.set_title("Value Function", fontsize=16)
    ax.set_xlabel(r'$k_t$', fontstyle='italic')         
    ax.set_ylabel(r'$v_t$', fontstyle='italic')
    #ax.legend()          
    ax.grid(axis='both', alpha=0.5)                         
    ax.tick_params(axis='x', direction='in')
    ax.tick_params(axis='y', direction='in')
    
    fig.autofmt_xdate() 
    plt.tight_layout()
    plot_path = folder + rl_model.replace('.pt', '_value_rule.png')
    fig.savefig(plot_path)

    """ #Transition 
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.plot(k_values, k1_values_rl[:, :], color = '#ff6600', linewidth=1.5, label='RL')
    ax.plot(k_values, k1_values_grid[:, :], color = '#003f5c',  linewidth=1.5, label='Grid')
    ax.scatter(k_ss, k_ss, color='#003f5c', label='Steady State', s=20, zorder=5)
    ax.scatter(k_ss_rl, k_ss_rl, color='#ff6600', label='Steady State', s=20, zorder=5)
    ax.axvline(k_ss, color='#003f5c', linestyle=':', linewidth=1)
    ax.axhline(k_ss, color='#003f5c', linestyle=':', linewidth=1)
    ax.axvline(k_ss_rl, color='#ff6600', linestyle=':', linewidth=1)
    ax.axhline(k_ss_rl, color='#ff6600', linestyle=':', linewidth=1)
    ax.plot(k_values, k_values, color='black', linestyle=':', linewidth=1)
    ax.set_title("Transition Rule", fontsize=16)
    ax.set_xlabel(r'$k_t$', fontstyle='italic')         
    ax.set_ylabel(r'$k_{t+1}$', fontstyle='italic')
    #ax.legend()          
    ax.grid(axis='both', alpha=0.5)                         
    ax.tick_params(axis='x', direction='in')
    ax.tick_params(axis='y', direction='in')
    
    fig.autofmt_xdate() 
    plt.tight_layout()
    plot_path = folder + rl_model.replace('.pt', '_global_transition.png')
    fig.savefig(plot_path)
 """


    
''' GLOBAL POLICY'''

if global_policy == "yes":
    N = 200
    c_values_grid = np.zeros((N, ss.nbr))
    k1_values_grid = np.zeros((N, ss.nbr))
    v_values_grid = np.zeros((N, ss.nbr))
    c_values_rl = np.zeros((N, ss.nbr))
    k1_values_rl = np.zeros((N, ss.nbr))
    v_values_rl = np.zeros((N, ss.nbr))

    #zgrid_small = np.array([zgrid[2], zgrid[5], zgrid[8]])

    k_values = np.linspace(1 , k_ss * (1.15), N)
    #z_psx = int(np.where(zgrid == 1)[0])
    for j in range(ss.nbr): 
        for i in range(len(k_values)):
            #RL 
            st = np.array([rgrid[j], k_values[i]])
            state = torch.from_numpy(st).float().to(device)
            with torch.no_grad():
                action_tensor, _ = agent.get_action(state, test=True)
                action_rl = action_tensor.squeeze().numpy()
                value_tensor = agent.get_value(state)
                value_rl = value_tensor.numpy()
            y_rl = rgrid[j] * (k_values[i]**ss.alpha)
            c_rl = (1 - action_rl) * y_rl
            k1_rl = (1 - ss.delta)*k_values[i] + action_rl * y_rl
            v_rl = float(value_rl)

            #Grid 
            c_grid = optimal_c(st)
            k1_grid = optimal_k1(st)
            v_grid = optimal_v(st)

            #save 
            c_values_grid[i,j] = c_grid
            c_values_rl[i,j] = c_rl
            k1_values_grid[i,j] = k1_grid
            k1_values_rl[i,j] = k1_rl 
            v_values_grid[i,j] = v_grid
            v_values_rl[i,j] = v_rl


        # How much the RL policy deviates from Grid for a 5% deviation from steady state
        p = len(k_values)-1
        z_diff = (rgrid[j] - 1)
        k_diff = (k_values[p] - k_ss) / np.abs(k_ss)
        c_diff = (c_values_rl[p, j] - c_values_grid[p, j]) / np.abs(c_values_grid[p, j])
        #n_diff = (n_values[p, 0] - n_values[p, 1]) / np.abs(n_values[p, 1])
        print(f"Consumption deviation from grid policy for a {k_diff *100}% k deviation and a {z_diff *100}% z deviation: {c_diff*100:.2f}%")
        #print(f"Labour deviation from grid policy for a {k_diff *100}% k deviation: {n_diff*100:.2f}%")



    #plotting
    #consumption
    fig, ax = plt.subplots(figsize=(5, 6))
    palette = ("#ff6600", "#ffb84d")
    for i in range(len(c_values_rl[0,:])):
        ax.plot(k_values, c_values_rl[:, i], color = palette[i],  linewidth=1.5, label='RL')
        ax.plot(k_values, c_values_grid[:, i], color = palette[i], linestyle = 'dashed', linewidth=1.5, label='Grid')
    ax.scatter(k_ss, c_ss, marker='o', facecolors='none', edgecolors='#003f5c', s=40, linewidths=1.5, zorder = 5)
    ax.scatter(k_ss_rl, c_ss_rl, marker='o', facecolors='#003f5c', edgecolors='#003f5c', s=40, linewidths=1.5, zorder = 5)
    #ax.scatter(k_ss_rl_ante, c_ss_rl_ante, marker='o', facecolors=palette[0], edgecolors=palette[0], s=40, linewidths=1.5)
    #ax.scatter(k_ss_ante, c_ss_ante, marker='o', facecolors='none', edgecolors=palette[0], s=40, linewidths=1.5)
    #ax.axvline(k_ss, color='#003f5c', linestyle=':', linewidth=1, alpha = 0.5)
    #ax.axhline(c_ss, color='#003f5c', linestyle=':', linewidth=1, alpha = 0.5)
    #ax.axvline(k_ss_rl, color='#ff6600', linestyle=':', linewidth=1, alpha = 0.5)
    #ax.axhline(c_ss_rl, color='#ff6600', linestyle=':', linewidth=1, alpha = 0.5)
    ax.set_title("Consumption Rule", fontsize=16)
    ax.set_xlabel(r'$k_t$', fontstyle='italic')         
    ax.set_ylabel(r'$c_t$', fontstyle='italic')
    #ax.legend()          
    ax.grid(axis='both', alpha=0.5)                         
    ax.tick_params(axis='x', direction='in')
    ax.tick_params(axis='y', direction='in')
    
    fig.autofmt_xdate() 
    plt.tight_layout()
    plot_path = folder + rl_model.replace('.pt', '_global_policy.png')
    fig.savefig(plot_path)


    #value
    fig, ax = plt.subplots(figsize=(5, 6))
    palette = ("#ff6600", "#ffb84d")
    for i in range(len(v_values_rl[0,:])):
        ax.plot(k_values, v_values_rl[:, i], color = palette[i],  linewidth=1.5, label='RL')
        ax.plot(k_values, v_values_grid[:, i], color = palette[i], linestyle = 'dashed', linewidth=1.5, label='Grid')
    #ax.scatter(k_ss, v_ss, marker='o', facecolors='none', edgecolors='#003f5c', s=40, linewidths=1.5, zorder = 5)
    #ax.scatter(k_ss_rl, v_ss_rl, marker='o', facecolors='#003f5c', edgecolors='#003f5c', s=40, linewidths=1.5, zorder = 5)
    #ax.scatter(k_ss_rl_ante, c_ss_rl_ante, marker='o', facecolors=palette[0], edgecolors=palette[0], s=40, linewidths=1.5)
    #ax.scatter(k_ss_ante, c_ss_ante, marker='o', facecolors='none', edgecolors=palette[0], s=40, linewidths=1.5)
    #ax.axvline(k_ss, color='#003f5c', linestyle=':', linewidth=1, alpha = 0.5)
    #ax.axhline(c_ss, color='#003f5c', linestyle=':', linewidth=1, alpha = 0.5)
    #ax.axvline(k_ss_rl, color='#ff6600', linestyle=':', linewidth=1, alpha = 0.5)
    #ax.axhline(c_ss_rl, color='#ff6600', linestyle=':', linewidth=1, alpha = 0.5)
    ax.set_title("Value Function", fontsize=16)
    ax.set_xlabel(r'$k_t$', fontstyle='italic')         
    ax.set_ylabel(r'$v_t$', fontstyle='italic')
    #ax.legend()          
    ax.grid(axis='both', alpha=0.5)                         
    ax.tick_params(axis='x', direction='in')
    ax.tick_params(axis='y', direction='in')
    
    fig.autofmt_xdate() 
    plt.tight_layout()
    plot_path = folder + rl_model.replace('.pt', '_global_value.png')
    fig.savefig(plot_path)
