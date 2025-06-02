import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
import os
from simulation import Model
from rl_algos.actor_critic import ActorCritic
from rl_algos.soft_actor_critic import SoftActorCritic
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse
from steady import steady
import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.vector import SyncVectorEnv
# baseline version is now: 2 states, beta = 0.97 and delta = 0.01
'''CONTROLS'''
comment = 'SGM_prepoc4_'
#working version
version = "stochastic" # deterministic ; stochastic  
initial_k = "steady"      # steady ; random 
var_k0 = 1                #Pct deviation from ss capital

run_EulerResid = "no"

T_test = 550
T_train = 550
frq_test = 50
EPOCHS = 45000

plot_histogram = 0 #1 plots the action dist conditional on steady state 


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
parser.add_argument('--lr', default=1e-3, type=float)          #here we might want to try 1e-4
parser.add_argument('--batch_size', default=2048, type=int)    #try 1024
parser.add_argument('--learn_std', default=0, type=int)        #try 1
''' SIMULATOR '''
parser.add_argument('--n_workers', default=4, type=int)
args = parser.parse_args()

## Saving the seed 
seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)

## device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
name_exp = ''

## string to indicate type in logs
model_type = comment + initial_k + "_" + version 
sim_length = "_train="+ str(T_train) +"_test="+ str(T_test)
for k, v in args.__dict__.items():
    if k == 'policy_var':
        name_exp += str(k) + "=" + str(v) + "_"
        break
#for k, v in args.__dict__.items():
#    name_exp += str(k) + "=" + str(v) + "_"
name_exp += str(model_type)
if initial_k == "random":
    name_exp += "_var="+str(var_k0)
name_exp += str(sim_length)
writer = SummaryWriter("logs/"+ name_exp)

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


register(
    id="model",
    entry_point="simulation:Model",
    kwargs={'k_ss': k_ss,
            'c_ss': c_ss,
            'y_ss': y_ss,
            'n_states': state_dim,
            'var_k': var_k0/100,
            'gamma': ss.gamma,
            'delta': ss.delta,
            'rhoa': ss.rhoa,
            'alpha': ss.alpha,
            'T': 1000,      
            'noise': ss.dev_eps_z,
            'version': version},
)

def make_env():
    return gym.make("model")

test_sim = gym.make("model")
sims = SyncVectorEnv([make_env for _ in range(args.n_workers)])
# sims = gym.make_vec("model", num_envs=args.n_workers, vectorization_mode="async")
# sims = gym.make_vec("model", num_envs=args.n_workers, vectorization_mode="sync")

'''
Start Training the model 
'''
vss_train = v_ss
frq_train = 3

vss_test = v_ss
n_eval = 5
best_utility = -np.inf

for iter in tqdm(range(EPOCHS)):

    st, _ = sims.reset(options = initial_k) 
    total_utility = 0

    for t in range(T_train):
        st_tensor = torch.from_numpy(st).float().to(device)
        with torch.no_grad():
            action_tensor, log_prob = agent.get_action(st_tensor)
            a = action_tensor.numpy()
            st1, u, done, _, rec = sims.step(a)
            y = rec['y']
            c = rec['c']
            for i in range(args.n_workers):
                agent.batchdata.push(st[i], a[i], log_prob[i].detach().cpu().numpy(), u[i], st1[i], y[i], float(not done[i]))

            st = st1
            total_utility += np.mean((agent.gamma ** t) * u)


    writer.add_scalar("pct welfare gain of steady state to current policy (train)", ((vss_train-total_utility)/total_utility)*100 , iter) # % of additional utility in steady state  

    # qua alleniamo NN
    if iter % frq_train == (frq_train-1):
        v_loss, p_loss = agent.update()
        agent.batchdata.clear()

        if v_loss is not None:
            writer.add_scalar("value loss", v_loss, iter)
            writer.add_scalar("policy loss", p_loss, iter)


    '''
    Start Testing the model 
    '''
    if iter % frq_test == (frq_test-1):

        '''qua sto testando la policy media'''

        total_utility = 0
        euler_gap = 0
        last_state = 0
        last_cons = 0 
        avg_cons = 0
        avg_state = 0
        random_util = 0

        for _ in range(n_eval):
            last_sim = {}
            all_actions = np.zeros((T_test, ss.actions))
            st, _ = test_sim.reset(options=initial_k) 
            rnd_state0 = st[1]

            for t in range(T_test):
                st_tensor = torch.from_numpy(st).float().to(device)
                with torch.no_grad():
                    action_tensor, log_prob = agent.get_action(st_tensor, test=True)
                    a = action_tensor.squeeze().numpy()
                    st1, u, done, _, rec = test_sim.step(a)
                    y = rec['y']
                    c = rec['c']

                    last_sim[t] = {'st': st,
                                   'a': a,
                                   'u': u,
                                   'st1': st1,
                                   'y': y,
                                   'c': c}
                    all_actions[t, :] = a
                    total_utility += (agent.gamma ** t) * u

                    
                    #random policy 
                    rnd_util, rnd_state1 = ss.get_random_util(st[0], rnd_state0)
                    random_util += (agent.gamma ** t) * rnd_util
                    rnd_state0 = rnd_state1
                    
                    if run_EulerResid == "yes":
                        #distance from FOC
                        if version == 'deterministic': 
                            if t>0:
                                k1 = last_sim[t]['st'][1]
                                z0 = last_sim[t-1]['st'][0]
                                #E_z1 = (1-ss.rhoa) + ss.rhoa * z0
                                c0 = last_sim[t-1]['c']
                                c1 = last_sim[t]['c']
                                c_ratio_star = ss.beta*((1 - ss.delta) + ss.alpha * ((k1)**(ss.alpha-1)) )
                                c_ratio = c1/c0
                                #euler_gap += np.abs((c_ratio - c_ratio_star)/c_ratio_star)
                                euler_gap += (c_ratio - c_ratio_star)**2
                        else: 
                            k1 = last_sim[t]['st1'][1]
                            z0 = last_sim[t]['st'][0]
                            c0 = last_sim[t]['c']
                            mu0 = 1/c0
                            mu1 = np.zeros(ss.nbz)
                            Z, Pi = ss.tauchenhussey_local(ss.nbz, z0)
                            for i in range(ss.nbz): 
                                with torch.no_grad():
                                    st_tensor_foc = torch.from_numpy(np.array([Z[i], k1])).float().to(device)
                                    action_tensor_foc, _ = agent.get_action(st_tensor_foc, test=True)
                                    a_foc = action_tensor_foc.squeeze().numpy()
                                    y = Z[i]* (k1**ss.alpha)
                                    c1 = y * (1-a_foc)
                                    mu1[i] = 1/c1
                            r1 = (1-ss.delta) + Z * ss.alpha * (k1**(ss.alpha-1))
                            EPS = np.sum(Pi * (mu1*r1))
                            euler_gap += (mu0 - ss.beta * EPS)**2 
                    
                    #average distance from ss
                    if t==T_test-1:
                        last_state += st1[1]
                        last_cons += last_sim[t]['c']
                        avg_cons += np.mean([last_sim[i]['c'] for i in sorted(last_sim)])
                        avg_state += np.mean([last_sim[i]['st'][1] for i in sorted(last_sim)])
                        
                    st = st1

                    if done:
                        break
            

        total_utility /= n_eval
        euler_gap /= n_eval*T_test
        last_state /= n_eval
        last_cons /= n_eval 
        avg_cons /= n_eval
        avg_state /= n_eval
        random_util /= n_eval

        writer.add_scalar("squared distance from opt consumption ratio (euler)", euler_gap, iter) 
        writer.add_scalar("pct welfare gain of steady state to current policy (test)", ((vss_test-total_utility)/total_utility)*100 , iter)
        writer.add_scalar("pct welfare gain of current policy to random policy", ((total_utility-random_util)/random_util)*100 , iter) 
        writer.add_scalar("pct distance of k to k_ss", (np.abs(last_state - k_ss)/k_ss)*100, iter)
        writer.add_scalar("pct distance of c to c_ss", (np.abs(last_cons - c_ss)/c_ss)*100, iter)
        writer.add_scalar("var action 0 per sim", np.var(all_actions[:, 0]), iter)
        
  
        if best_utility < total_utility:
            best_utility = total_utility

            with open("last_sim.pkl", "wb") as f:
                pickle.dump(last_sim, f)

            agent.save(str(model_type))

    if plot_histogram == 1: 
        if iter % 12 == 12-1:
            st, _ = test_sim.reset(options="steady") 
            rnd_state0 = st[1]


            st_tensor = torch.from_numpy(st).float().to(device)
            with torch.no_grad():
                sample0 = agent.get_dist(st_tensor)

                # Create a figure with two subplots
                plt.subplot(2, 1, 1)
                plt.hist(sample0, bins=50, density=True, alpha=0.6, color='blue')
                plt.axvline(c_ss/y_ss, color='r', linestyle='--', label='c')
                plt.title("Histogram of c/y")
                plt.xlabel("Value")
                plt.ylabel("Density")
                plt.xlim(0, 1)

                # Adjust layout and display the plots
                plt.tight_layout()
                plt.draw()
                plt.pause(1)
                if (iter // 12) % 4 == 3:
                    plt.clf() 
       


