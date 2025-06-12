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
rl_model = 'SGM_lowvarper_steady_stochastic.pt' 
grid_model = 'Grid_SGM_lowvarper_stochastic.pkl'
#folder to store plots 
folder = 'SGM_plots/'

#zoom = "in" #this needs to be adjusted
run_local = "yes"
global_policy = "no" #needs to be run with appropriate grid solution 

if run_local == "yes":
    run_simulation = "yes" #if yes it runs the simulation

    run_policy = "yes" # if yes it runs the policy evaluation

    run_irfs = "yes"
else:
    run_simulation = "no" #if yes it runs the simulation

    run_policy = "no" # if yes it runs the policy evaluation

    run_irfs = "no"



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

''' COMPUTING THE STEADY STATE OF RL'''
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

        

''' LOADING GRID MODEL'''
#need to update for stochastic version 
grid_model_path = 'saved_models/' + grid_model
with open(grid_model_path, 'rb') as f:
    loaded_data = pickle.load(f)

zgrid = ss.tauchenhussey(N=ss.nbz)[0]   # Discretized z values
zmin = np.min(zgrid)
zmax = np.max(zgrid)
Pi = ss.tauchenhussey(N=ss.nbz)[1]      # Transition probabilities

k = loaded_data['st']
k1_star = loaded_data['k1_star']
a_star = loaded_data['a_star']
c_star = loaded_data['c_star']
value_star = loaded_data['value_star']

optimal_a = interp2d((zgrid, k), a_star.T)
optimal_k1 = interp2d((zgrid, k), k1_star.T)
optimal_v = interp2d((zgrid, k), value_star.T)
optimal_c = interp2d((zgrid, k), c_star.T)




''' SIMULATION '''
if run_simulation == "yes":
    # RL, Grid
    foc = "yes"
    dev = 0
    T = 500
    st = np.array([k_ss_rl*(1+(dev/100)), k_ss*(1+(dev/100))])
    z = 1 #we want the same series for productivity in the two economies
    grid_sim={}
    rl_sim={}
    foc_sim={}
    grid_v = 0
    rl_v = 0
    #labour_gap = np.zeros((T-1, 2))
    euler_gap = np.zeros((T-1, 2))
    

    for t in range(T):
        #RL 
        k_rl = st[0]
        state_rl = torch.from_numpy(np.array([z, k_rl])).float().to(device)
        with torch.no_grad():
            action_tensor, _ = agent.get_action(state_rl, test=True)
            sratio_rl = action_tensor.squeeze().numpy()
        y_rl = z* (k_rl**ss.alpha)
        c_rl = y_rl * (1-sratio_rl)
        u_rl = ss.gamma*np.log(c_rl)
        k1_rl = (1 - ss.delta)*k_rl + y_rl - c_rl
        rl_v += ss.beta**t * u_rl
        rl_sim[t] = {'k': k_rl,
                    'z': z,
                    'c': c_rl,
                    'y': y_rl,
                    'u': u_rl,
                    'st1': k1_rl}

        #Grid 
        # Find the position of the value 1 in zgrid
        k_grid = st[1]
        y_grid = z * (k_grid**ss.alpha)
        state_grid = np.array([z, float(k_grid)])
        #c_grid = float(optimal_c(k_grid))
        c_grid = float(optimal_c(state_grid))
        u_grid = ss.gamma*np.log(c_grid) #+ ss.psi*np.log(1-a[1])
        #k1_grid = float(optimal_k1(state_grid))
        k1_grid = (1 - ss.delta)*k_grid + y_grid - c_grid
        grid_v += ss.beta**t * u_grid
        grid_sim[t] = {'k': k_grid,
                    'z': z,
                    'c': c_grid,
                    'y': y_grid,
                    'u': u_grid,
                    'st1': k1_grid}
        
        if foc == "yes": 
            mu0 = 1/c_rl
            mu1 = np.zeros(ss.nbz)
            Z, Pi = ss.tauchenhussey_local(ss.nbz, z)
            for i in range(ss.nbz): 
                with torch.no_grad():
                    st_tensor_foc = torch.from_numpy(np.array([Z[i], k1_rl])).float().to(device)
                    action_tensor_foc, _ = agent.get_action(st_tensor_foc, test=True)
                    a_foc = action_tensor_foc.squeeze().numpy()
                    y = Z[i]* (k1_rl**ss.alpha)
                    c1 = y * (1-a_foc)
                    mu1[i] = 1/c1
            r1 = (1-ss.delta) + Z * ss.alpha * (k1_rl**(ss.alpha-1))
            EPS = np.sum(Pi * (mu1*r1))
            euler_gap = (mu0 - ss.beta * EPS)**2
            foc_sim[t] = {'resid': euler_gap}
        
        while True:
            z1 = (1 - ss.rhoa) + ss.rhoa * z + np.random.normal(0, ss.dev_eps_z)
            if zmin <= z1 <= zmax:
                break
        z = z1
        st = np.array([k1_rl, k1_grid])
    


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
    ax.plot(k_grid, color='#003f5c', linewidth=1.5, label='Grid')
    ax.plot(k_rl, color="#ff6600", linewidth=1.5, label='RL')
    ax.axhline(k_ss, color='#003f5c', linewidth=1.2, linestyle='--',label='Steady State')
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
    ax.plot(c_grid, color='#003f5c', linewidth=1.5, label='Grid')
    ax.plot(c_rl, color="#ff6600", linewidth=1.5, label='RL')
    ax.axhline(c_ss, color='#003f5c', linewidth=1.2, linestyle='--', label='Steady State')
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
    ax.plot(resids, color='#003f5c', linewidth=1.5, label='Grid')
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
    zgrid_small = np.array([zgrid[1], zgrid[3], zgrid[5], zgrid[7], zgrid[9]])

    c_values_grid = np.zeros((N, len(zgrid_small)))
    k1_values_grid = np.zeros((N, len(zgrid_small)))
    v_values_grid = np.zeros((N, len(zgrid_small)))
    c_values_rl = np.zeros((N, len(zgrid_small)))
    k1_values_rl = np.zeros((N, len(zgrid_small)))
    v_values_rl = np.zeros((N, len(zgrid_small)))


    k_values = np.linspace(k_ss * (1-(dev/100)), k_ss * (1+(dev/100)), N)
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
        z_diff = (zgrid_small[j] - 1)
        k_diff = (k_values[p] - k_ss) / np.abs(k_ss)
        c_diff = (c_values_rl[p, j] - c_values_grid[p, j]) / np.abs(c_values_grid[p, j])
        #n_diff = (n_values[p, 0] - n_values[p, 1]) / np.abs(n_values[p, 1])
        print(f"Consumption deviation from grid policy for a {k_diff *100}% k deviation and a {z_diff *100}% z deviation: {c_diff*100:.2f}%")
        #print(f"Labour deviation from grid policy for a {k_diff *100}% k deviation: {n_diff*100:.2f}%")



    #plotting
    #consumption
    fig, ax = plt.subplots(figsize=(5, 6))
    palette = ("#e65300", "#ff6600", 	"#ff9440", "#ffb84d", "#ffe0b3")
    for i in range(len(c_values_rl[0,:])):
        ax.plot(k_values, c_values_rl[:, i], color = palette[i],  linewidth=1.5, label='RL')
        ax.plot(k_values, c_values_grid[:, i], color = palette[i], linestyle = 'dashed', linewidth=1.5, label='Grid')
    ax.scatter(k_ss, c_ss, marker='o', facecolors='none', edgecolors= '#003f5c', s=40, linewidths=1.5, zorder = 5)
    ax.scatter(k_ss_rl, c_ss_rl, marker='o', facecolors='#003f5c', edgecolors='#003f5c', s=40, linewidths=1.5, zorder = 5)
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
    palette = ("#e65300", "#ff6600", 	"#ff9440", "#ffb84d", "#ffe0b3")
    for i in range(len(v_values_rl[0,:])):
        ax.plot(k_values, v_values_rl[:, i], color = palette[i],  linewidth=1.5, label='RL')
        ax.plot(k_values, v_values_grid[:, i], color = palette[i], linestyle = 'dashed', linewidth=1.5, label='Grid')
    ax.scatter(k_ss, v_ss, marker='o', facecolors='none', edgecolors= '#003f5c', s=40, linewidths=1.5, zorder = 5)
    ax.scatter(k_ss_rl, v_ss_rl, marker='o', facecolors='#003f5c', edgecolors='#003f5c', s=40, linewidths=1.5, zorder = 5)
    ax.set_title("Value function", fontsize=16)
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
    ax.plot(k_values, k1_values_rl[:, :], color = 'crimson', linewidth=1.5, label='RL')
    ax.plot(k_values, k1_values_grid[:, :], color = 'blue',  linewidth=1.5, label='Grid')
    ax.scatter(k_ss, k_ss, color='blue', label='Steady State', s=20, zorder=5)
    ax.scatter(k_ss_rl, k_ss_rl, color='crimson', label='Steady State', s=20, zorder=5)
    ax.axvline(k_ss, color='blue', linestyle=':', linewidth=1)
    ax.axhline(k_ss, color='blue', linestyle=':', linewidth=1)
    ax.axvline(k_ss_rl, color='crimson', linestyle=':', linewidth=1)
    ax.axhline(k_ss_rl, color='crimson', linestyle=':', linewidth=1)
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
    fig.savefig(plot_path) """


''' IRFs '''
# IRFs for consumption, capital, and value function
if run_irfs == 'yes':
    irf_length = 500
    irf_c = np.zeros((irf_length, 2))
    irf_y = np.zeros((irf_length, 2))
    irf_k = np.zeros((irf_length, 2))
    irf_v = np.zeros((irf_length, 2))

    irf_ci = np.zeros((irf_length, 100))
    irf_ki = np.zeros((irf_length, 100))

    # z dev 
    # 1% deviation from steady state
    z0 =  zgrid[9]  # Initial z value
    k0_grid = k_ss
    k0_rl = k_ss_rl
    k0_rl_i = np.ones(100) *  k_ss_rl
    for t in range(irf_length):

        st_rl = np.array([z0, k0_rl])
        st_rl_i = np.column_stack((np.full_like(k0_rl_i , z0), k0_rl_i))
        state = torch.from_numpy(st_rl).float().to(device)
        state_i = torch.from_numpy(st_rl_i).float().to(device)

        with torch.no_grad():
            action_tensor, _ = agent.get_action(state, test=True)          
            action_rl = action_tensor.squeeze().numpy()
            value_tensor = agent.get_value(state)
            value_rl = value_tensor.numpy()

            actions_i = np.zeros(100)
            for i in range(100):
                action_i_tens, _ = agent.get_action(state_i[i], test= False)
                action_i = action_i_tens.squeeze().numpy()
                actions_i[i] = action_i

        y_rl = z0 * (k0_rl**ss.alpha)
        c_rl = (1 - action_rl) * y_rl
        k1_rl = (1 - ss.delta)*k0_rl + action_rl * y_rl
        v_rl = float(value_rl)

        y_rl_i =  z0 * (k0_rl_i**ss.alpha)
        c_rl_i = (1 - actions_i) * y_rl_i
        k1_rl_i =  (1 - ss.delta)*k0_rl_i + actions_i * y_rl_i

        st_grid = np.array([z0, k0_grid])
        c_grid = float(optimal_c(st_grid))
        v_grid = float(optimal_v(st_grid))
        k1_grid = float(optimal_k1(st_grid))
        y_grid = z0*k0_grid**ss.alpha


        # Store IRFs
        irf_c[t] = [c_rl, c_grid]
        irf_k[t] = [k1_rl, k1_grid]
        irf_v[t] = [v_rl, v_grid]
        irf_y[t] = [y_rl, y_grid]

        # Storing the i IRFs 
        irf_ci[t] = c_rl_i
        irf_ki[t] = k1_rl_i

        # Update z for next period
        z1 = (1 - ss.rhoa) + ss.rhoa * z0 
        z0 = z1
        k0_grid = k1_grid
        k0_rl = k1_rl
        k0_rl_i = k1_rl_i
        

    # Plotting IRFs
    fig, ax = plt.subplots(figsize=(5, 6))  
    ax.plot(irf_c[:,1], color='#003f5c', linewidth=1.5, label='Grid')
    ax.plot(irf_c[:,0], color="#e65300", linewidth=1.5, label='RL', zorder = 5)
    ax.plot(irf_ci, color = "#ff9440", linewidth= 0.5, alpha = 0.05, label='RL')
    ax.axhline(c_ss, color='#003f5c', linewidth=1.2, linestyle='--', label='Steady State')
    ax.axhline(c_ss_rl, color="#e65300", linewidth=1.2, linestyle='--', label='Steady State')
    ax.set_title("Consumption to z shock", fontsize=16)
    ax.set_xlabel("Periods", fontstyle='italic')         
    ax.set_ylabel(r'$c_t$', fontstyle='italic')
    #ax.legend()          
    ax.grid(axis='both', alpha=0.5)                          
    ax.tick_params(axis='x', direction='in')
    ax.tick_params(axis='y', direction='in')

    fig.autofmt_xdate() 
    plt.tight_layout()
    plot_path = folder + rl_model.replace('.pt', '_IRF_consumption_z.png')
    fig.savefig(plot_path)

    # Plotting IRFs
    fig, ax = plt.subplots(figsize=(5, 6))  
    ax.plot(irf_k[:,1], color='#003f5c', linewidth=1.5, label='Grid')
    ax.plot(irf_k[:,0], color='#e65300', linewidth=1.5, label='RL', zorder = 5)
    ax.plot(irf_ki, color = "#ff9440", linewidth= 0.5, alpha = 0.05, label='RL')
    ax.axhline(k_ss, color="#003f5c", linewidth=1.2, linestyle='--', label='Steady State')
    ax.axhline(k_ss_rl, color="#e65300", linewidth=1.2, linestyle='--', label='Steady State')
    ax.set_title("Capital to z shock", fontsize=16)
    ax.set_xlabel("Periods", fontstyle='italic')         
    ax.set_ylabel(r'$k_t$', fontstyle='italic')
    #ax.legend()          
    ax.grid(axis='both', alpha=0.5)                          
    ax.tick_params(axis='x', direction='in')
    ax.tick_params(axis='y', direction='in')

    fig.autofmt_xdate() 
    plt.tight_layout()
    plot_path = folder + rl_model.replace('.pt', '_IRF_capital_z.png')
    fig.savefig(plot_path)





    # IRF for Capital deviation
    irf_c = np.zeros((irf_length, 2))
    irf_y = np.zeros((irf_length, 2))
    irf_k = np.zeros((irf_length, 2))
    irf_v = np.zeros((irf_length, 2))

    irf_ci = np.zeros((irf_length, 100))
    irf_ki = np.zeros((irf_length, 100))

     
    #z_dev = 0.03  # 1% deviation from steady state
    z0 = 1 #+ z_dev  # Initial z value
    k0_grid = k_ss * (1 + 0.05)
    k0_rl = k_ss_rl * (1 + 0.05)
    k0_rl_i = np.ones(100) *  k0_rl
    for t in range(irf_length):

        st_rl = np.array([z0, k0_rl])
        st_rl_i = np.column_stack((np.full_like(k0_rl_i , z0), k0_rl_i))
        state = torch.from_numpy(st_rl).float().to(device)
        state_i = torch.from_numpy(st_rl_i).float().to(device)

        with torch.no_grad():
            action_tensor, _ = agent.get_action(state, test=True)
            action_rl = action_tensor.squeeze().numpy()
            value_tensor = agent.get_value(state)
            value_rl = value_tensor.numpy()

            actions_i = np.zeros(100)
            for i in range(100):
                action_i_tens, _ = agent.get_action(state_i[i], test= False)
                action_i = action_i_tens.squeeze().numpy()
                actions_i[i] = action_i

        y_rl = z0 * (k0_rl**ss.alpha)
        c_rl = (1 - action_rl) * y_rl
        k1_rl = (1 - ss.delta)*k0_rl + action_rl * y_rl
        v_rl = float(value_rl)

        y_rl_i =  z0 * (k0_rl_i**ss.alpha)
        c_rl_i = (1 - actions_i) * y_rl_i
        k1_rl_i =  (1 - ss.delta)*k0_rl_i + actions_i * y_rl_i

        st_grid = np.array([z0, k0_grid])
        c_grid = float(optimal_c(st_grid))
        v_grid = float(optimal_v(st_grid))
        k1_grid = float(optimal_k1(st_grid))
        y_grid = z0*k0_grid**ss.alpha


        # Store IRFs
        irf_c[t] = [c_rl, c_grid]
        irf_k[t] = [k1_rl, k1_grid]
        irf_v[t] = [v_rl, v_grid]
        irf_y[t] = [y_rl, y_grid]

        # Storing the i IRFs 
        irf_ci[t] = c_rl_i
        irf_ki[t] = k1_rl_i

        # Update z for next period
        z1 = (1 - ss.rhoa) + ss.rhoa * z0 
        z0 = z1
        k0_grid = k1_grid
        k0_rl = k1_rl
        k0_rl_i = k1_rl_i

    # Plotting IRFs
    fig, ax = plt.subplots(figsize=(5, 6))  
    ax.plot(irf_c[:,1], color='#003f5c', linewidth=1.5, label='Grid')
    ax.plot(irf_c[:,0], color="#e65300", linewidth=1.5, label='RL', zorder = 5)
    ax.plot(irf_ci, color = "#ff9440", linewidth= 0.5, alpha = 0.05, label='RL')
    ax.axhline(c_ss, color='#003f5c', linewidth=1.2, linestyle='--', label='Steady State')
    ax.axhline(c_ss_rl, color="#e65300", linewidth=1.2, linestyle='--', label='Steady State')
    ax.set_title("Consumption to k shock", fontsize=16)
    ax.set_xlabel("Periods", fontstyle='italic')         
    ax.set_ylabel(r'$c_t$', fontstyle='italic')
    #ax.legend()          
    ax.grid(axis='both', alpha=0.5)                          
    ax.tick_params(axis='x', direction='in')
    ax.tick_params(axis='y', direction='in')

    fig.autofmt_xdate() 
    plt.tight_layout()
    plot_path = folder + rl_model.replace('.pt', '_IRF_consumption_k.png')
    fig.savefig(plot_path)

    # Plotting IRFs
    fig, ax = plt.subplots(figsize=(5, 6))  
    ax.plot(irf_k[:,1], color='#003f5c', linewidth=1.5, label='Grid')
    ax.plot(irf_k[:,0], color='#e65300', linewidth=1.5, label='RL', zorder = 5)
    ax.plot(irf_ki, color = "#ff9440", linewidth= 0.5, alpha = 0.05, label='RL')
    ax.axhline(k_ss, color="#003f5c", linewidth=1.2, linestyle='--', label='Steady State')
    ax.axhline(k_ss_rl, color="#e65300", linewidth=1.2, linestyle='--', label='Steady State')
    ax.set_title("Capital to k shock", fontsize=16)
    ax.set_xlabel("Periods", fontstyle='italic')         
    ax.set_ylabel(r'$k_t$', fontstyle='italic')
    #ax.legend()          
    ax.grid(axis='both', alpha=0.5)                          
    ax.tick_params(axis='x', direction='in')
    ax.tick_params(axis='y', direction='in')

    fig.autofmt_xdate() 
    plt.tight_layout()
    plot_path = folder + rl_model.replace('.pt', '_IRF_capital_k.png')
    fig.savefig(plot_path)

    
''' GLOBAL POLICY'''

if global_policy == "yes":
    N = 200
    zgrid_small = np.array([zgrid[1], zgrid[3], zgrid[5], zgrid[7], zgrid[9]])
    c_values_grid = np.zeros((N, len(zgrid_small)))
    k1_values_grid = np.zeros((N, len(zgrid_small)))
    v_values_grid = np.zeros((N, len(zgrid_small)))
    c_values_rl = np.zeros((N, len(zgrid_small)))
    k1_values_rl = np.zeros((N, len(zgrid_small)))
    v_values_rl = np.zeros((N, len(zgrid_small)))

    #zgrid_small = np.array([zgrid[2], zgrid[5], zgrid[8]])

    k_values = np.linspace(1 , k_ss * (1.2), N)
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
        z_diff = (zgrid_small[j] - 1)
        k_diff = (k_values[p] - k_ss) / np.abs(k_ss)
        c_diff = (c_values_rl[p, j] - c_values_grid[p, j]) / np.abs(c_values_grid[p, j])
        #n_diff = (n_values[p, 0] - n_values[p, 1]) / np.abs(n_values[p, 1])
        print(f"Consumption deviation from grid policy for a {k_diff *100}% k deviation and a {z_diff *100}% z deviation: {c_diff*100:.2f}%")
        #print(f"Labour deviation from grid policy for a {k_diff *100}% k deviation: {n_diff*100:.2f}%")



    #plotting
    #consumption
    fig, ax = plt.subplots(figsize=(5, 6))
    palette = ("#e65300", "#ff6600", 	"#ff9440", "#ffb84d", "#ffe0b3")
    for i in range(len(c_values_rl[0,:])):
        ax.plot(k_values, c_values_rl[:, i], color = palette[i],  linewidth=1.5, label='RL')
        ax.plot(k_values, c_values_grid[:, i], color = palette[i], linestyle = 'dashed', linewidth=1.5, label='Grid')
    #ax.scatter(k_ss, c_ss, marker='o', facecolors='none', edgecolors= palette[2], s=40, linewidths=1.5, zorder = 5)
    #ax.scatter(k_ss_rl, c_ss_rl, marker='o', facecolors=palette[2], edgecolors=palette[2], s=40, linewidths=1.5, zorder = 5)
    ax.scatter(k_ss, c_ss, marker='o', facecolors='none', edgecolors= '#003f5c', s=40, linewidths=1.5, zorder = 5)
    ax.scatter(k_ss_rl, c_ss_rl, marker='o', facecolors='#003f5c', edgecolors='#003f5c', s=40, linewidths=1.5, zorder = 5)
    ax.set_title("Consumption Rule", fontsize=16)
    ax.set_xlabel(r'$k_t$', fontstyle='italic')         
    ax.set_ylabel(r'$c_t$', fontstyle='italic')        
    ax.grid(axis='both', alpha=0.5)                         
    ax.tick_params(axis='x', direction='in')
    ax.tick_params(axis='y', direction='in')
    
    fig.autofmt_xdate() 
    plt.tight_layout()
    plot_path = folder + rl_model.replace('.pt', '_global_policy.png')
    fig.savefig(plot_path)

     #value
    fig, ax = plt.subplots(figsize=(5, 6))
    palette = ("#e65300", "#ff6600", 	"#ff9440", "#ffb84d", "#ffe0b3")
    for i in range(len(v_values_rl[0,:])):
        ax.plot(k_values, v_values_rl[:, i], color = palette[i],  linewidth=1.5, label='RL')
        ax.plot(k_values, v_values_grid[:, i], color = palette[i], linestyle = 'dashed', linewidth=1.5, label='Grid')
    ax.scatter(k_ss, v_ss, marker='o', facecolors='none', edgecolors= '#003f5c', s=40, linewidths=1.5, zorder = 5)
    ax.scatter(k_ss_rl, v_ss_rl, marker='o', facecolors='#003f5c', edgecolors='#003f5c', s=40, linewidths=1.5, zorder = 5)
    ax.set_title("Value function", fontsize=16)
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

