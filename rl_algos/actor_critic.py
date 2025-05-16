import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from utils import BatchData
from steady import steady
import matplotlib.pyplot as plt
from steady import steady
from model_architectures import ValueNetwork, StochasticPolicyNetwork


class ActorCritic(nn.Module):

    def __init__(self, input_dim=2, architecture_params=None, output_dim=2, lr=1e-3, gamma=0.95, epsilon=0.0, batch_size=128, alpha=0, learn_std=True, learn_consumption=True, device=None):
        super(ActorCritic, self).__init__()

        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device

        self.batchdata = BatchData()

        # self.replay_buffer = Memory(2000)
        self.value_net = ValueNetwork(input_dim, architecture_params, 1)
        self.policy_net = StochasticPolicyNetwork(input_dim, architecture_params, output_dim, alpha=alpha, learn_std=learn_std)
        self.ss = steady()
        self.v_ss = self.ss.ss_value()
        self.u_ss = self.ss.ss()[-1]

        self.optimizer_v = optim.Adam(self.value_net.parameters(), lr=lr)
        self.optimizer_pi = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
    
    def get_value(self, st):
        v = self.value_net(st).squeeze()
        return v

    def get_action(self, st, test=False):
        a = self.policy_net.get_action(st, test=test)
        # if np.random.rand() < self.epsilon and not test:
        #     a = torch.rand((st.shape[0], 2))*0.15
        return a
    
    def get_dist(self, st): 
        dist, dist_raw = self.policy_net.get_dist(st)
        return dist, dist_raw

    def update(self):

        # if len(self.replay_buffer) < 1000: #training starts after the first model simulation
        #     return None, None

        # states = torch.cat([x.view(1, -1) for x in self.batchdata.st], 0)
        # next_states = torch.cat([x.view(1, -1) for x in self.batchdata.st1], 0)
        # rewards = torch.cat([x.view(1) for x in self.batchdata.u], 0)
        # actions = torch.cat([x.view(1, -1) for x in self.batchdata.a], 0)
        # terminal = torch.tensor([x for x in self.batchdata.terminal])

        states = torch.from_numpy(np.concatenate([np.expand_dims(x, 0) for x in self.batchdata.st], 0)).float().to(self.device)
        next_states = torch.from_numpy(np.concatenate([np.expand_dims(x, 0) for x in self.batchdata.st1], 0)).float().to(self.device)
        rewards = torch.from_numpy(np.concatenate([np.expand_dims(x, 0) for x in self.batchdata.u], 0)).float().to(self.device)
        actions = torch.from_numpy(np.concatenate([np.expand_dims(x, 0) for x in self.batchdata.a], 0)).float().to(self.device)
        #terminal = torch.from_numpy(np.concatenate([np.array([x]) for x in self.batchdata.terminal])).float().to(self.device)

        idx = np.random.choice(np.arange(states.shape[0]), self.batch_size)
        states = states[idx].detach()
        next_states = next_states[idx].detach()
        rewards = rewards[idx].detach()
        actions = actions[idx].detach()
        #terminal = terminal[idx]

    

        # Compute the target values
        with torch.no_grad():
            next_state_values = self.value_net(next_states).squeeze()
            target_values = rewards + self.gamma * next_state_values #terminal * self.gamma * next_state_values

        # Compute the predicted values
        predicted_values = self.value_net(states).squeeze()
        At = (target_values - predicted_values).detach()
        At_norm = At #(At - At.mean()) / (At.std() + 1e-7)

        """ #start debug: Plotting Advantages
        c_ss, n_ss, k_ss, y_ss, u_ss = self.ss.ss()
        plt.clf()
        plt.subplot(2, 1, 1)
        plt.scatter(actions.cpu().numpy()[:,0], At.cpu().numpy(), alpha=0.6, label='cons ratio')
        plt.xlim( 0, 1)
        plt.axvline(c_ss/y_ss, color='r', linestyle='--', label='c/y_ss')

        plt.subplot(2, 1, 2)
        plt.scatter(actions.cpu().numpy()[:,1], At.cpu().numpy(), alpha=0.6, label='n')
        plt.xlim( 0, 1)  # Set x-axis limits based on the sample
        plt.axvline(n_ss, color='r', linestyle='--', label='n_ss')
        
        plt.draw()
        plt.pause(1)
        #end debug """
        

        # Compute the loss
        loss_V = self.loss_fn(predicted_values, target_values)

        new_logprobs = self.policy_net.get_log_prob(states, actions)

        policy_loss = -(new_logprobs * At_norm).mean() 

        self.optimizer_v.zero_grad()
        loss_V.backward()
        self.optimizer_v.step()

        self.optimizer_pi.zero_grad()
        policy_loss.backward()
        self.optimizer_pi.step()

        return loss_V.detach().item(), policy_loss.detach().item()

    def save(self, file_name):
        if not os.path.exists("saved_models"):
            os.mkdir("saved_models")
        torch.save(self.state_dict(), "saved_models/" + file_name + ".pt")

    def load(self, file_name):
        self.load_state_dict(torch.load("saved_models/" + file_name + ".pt"))










