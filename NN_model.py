import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import random
from utils import Memory, BatchData


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, architecture_params, output_dim):
        super(ValueNetwork, self).__init__()

        layers = [nn.Linear(input_dim, architecture_params['n_neurons']), nn.ReLU()]
        for _ in range(architecture_params['n_layers']):
            layers += [nn.Linear(architecture_params['n_neurons'], architecture_params['n_neurons']),
                       nn.ReLU()]
        layers += [nn.Linear(architecture_params['n_neurons'], output_dim)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x) 


class PolicyNetwork(nn.Module):

    def __init__(self, state_dim, architecture_params, action_dim, alpha=0, learn_std=True):
        super(PolicyNetwork, self).__init__()

        self.var_scale = architecture_params['policy_var']
        self.learn_std = learn_std
        self.action_bounds = architecture_params['action_bounds']
        self.alpha = alpha

        layers = [nn.Linear(state_dim, architecture_params['n_neurons']), nn.ReLU()]
        for _ in range(architecture_params['n_layers']):
            layers += [nn.Linear(architecture_params['n_neurons'], architecture_params['n_neurons']),
                       nn.ReLU()]
        layers += [nn.Linear(architecture_params['n_neurons'], architecture_params['n_neurons']),
                   nn.ReLU()]
        self.base = nn.Sequential(*layers)

        self.mean_head = nn.Sequential(
            nn.Linear(architecture_params['n_neurons'], architecture_params['n_neurons']),
            nn.ReLU(),
            nn.Linear(architecture_params['n_neurons'], action_dim),
        )
        if learn_std:
            self.std_head = nn.Sequential(
                nn.Linear(architecture_params['n_neurons'], architecture_params['n_neurons']),
                nn.ReLU(),
                nn.Linear(architecture_params['n_neurons'], action_dim),
            )
        else:
            self.log_std = nn.Parameter(torch.ones(action_dim))
        self.sigmoid = nn.Sigmoid()

    def forward(self, state):
        x = self.base(state)
        mean = self.sigmoid(self.mean_head(x))
        if self.learn_std:
            std = torch.exp(self.std_head(x)*self.var_scale+1e-6)
        else:
            std = torch.exp(self.log_std*self.var_scale+1e-6)

        # TODO: solve inplace operation
        for i in self.action_bounds['order']:
            a_min = self.action_bounds['min'][i]()
            a_max = self.action_bounds['max'][i](state[:, 0], state[:, 1], self.alpha, mean[:, 1])
            mean[:, i] = mean[:, i] * (a_max - a_min)
            mean[:, i] = mean[:, i] + a_min

        dist = Normal(mean, std)
        return dist
    
    def get_action(self, state, test=False):
        dist = self.forward(state.view(1, -1))
        action = dist.sample() if not test else dist.mean

        for i in self.action_bounds['order']:
            a_min = self.action_bounds['min'][i]()
            a_max = self.action_bounds['max'][i](state[0], state[1], self.alpha, action[0, 1])
            action[:, i] = torch.clamp(action[:, i], min=a_min, max=a_max)

        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob


class RL_agent(nn.Module):

    def __init__(self, input_dim=2, architecture_params=None, output_dim=2, lr=1e-3, gamma=0.99, epsilon=0.0, batch_size=128, alpha=0, learn_std=True):
        super(RL_agent, self).__init__()

        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size

        self.batchdata = BatchData()

        # self.replay_buffer = Memory(2000)
        self.value_net = ValueNetwork(input_dim, architecture_params, 1)
        self.policy_net = PolicyNetwork(input_dim, architecture_params, output_dim, alpha=alpha, learn_std=learn_std)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def get_action(self, st, test=False):
        a = self.policy_net.get_action(st, test=test)
        if np.random.rand() < self.epsilon and not test:
            a = torch.rand((1, 2))*0.15
        return a

    def update(self):

        # if len(self.replay_buffer) < 1000: #training starts after the first model simulation
        #     return None, None

        states = torch.cat([x.view(1, -1) for x in self.batchdata.st], 0)
        next_states = torch.cat([x.view(1, -1) for x in self.batchdata.st1], 0)
        rewards = torch.cat([x.view(1) for x in self.batchdata.u], 0)
        actions = torch.cat([x.view(1, -1) for x in self.batchdata.a], 0)

        idx = np.random.choice(np.arange(states.shape[0]), self.batch_size)
        states = states[idx].detach()
        next_states = next_states[idx].detach()
        rewards = rewards[idx].detach()
        actions = actions[idx].detach()

        # Compute the target values
        with torch.no_grad():
            next_state_values = self.value_net(next_states).squeeze()
            target_values = rewards + self.gamma * next_state_values

        # Compute the predicted values
        predicted_values = self.value_net(states).squeeze()
        At = (target_values - predicted_values).detach()

        # Compute the loss
        loss_V = self.loss_fn(predicted_values, target_values)
        
        current_dist = self.policy_net(states)
        new_logprobs = current_dist.log_prob(actions).sum(dim=-1)

        policy_loss = -(new_logprobs * At).mean()
        loss = loss_V + policy_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss_V.detach().item(), policy_loss.detach().item()













