import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.distributions as D
import torch.distributions.transforms as T
from utils import state_preprocessor1
from utils import state_preprocessor2
from utils import state_preprocessor3
from steady import steady

class ValueNetwork(nn.Module):
    def __init__(self, input_dim, architecture_params, output_dim):
        super(ValueNetwork, self).__init__()

        layers = [nn.Linear(input_dim, architecture_params['n_neurons']), nn.ReLU()]
        for _ in range(architecture_params['n_layers']):
            layers += [nn.Linear(architecture_params['n_neurons'], architecture_params['n_neurons']),
                       nn.ReLU()]
        layers += [nn.Linear(architecture_params['n_neurons'], output_dim)]
        self.network = nn.Sequential(*layers)

        self.ss = steady()
        self.c_ss, self.k_ss, self.y_ss, self.u_ss, self.v_ss = self.ss.ss()

    def forward(self, x, a=None):
        #x = state_preprocessor1(x)
        x = state_preprocessor2(x, self.k_ss)
        #x = state_preprocessor3(x)
        x = x if a is None else torch.cat([x, a], -1)
        return self.network(x)


class StochasticPolicyNetwork(nn.Module):

    def __init__(self, state_dim, architecture_params, action_dim, alpha=0, learn_std=True):
        super(StochasticPolicyNetwork, self).__init__()

        self.var_scale = architecture_params['policy_var']
        self.learn_std = learn_std
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
            self.log_std = nn.Parameter(torch.ones((1, action_dim)) * self.var_scale) 
        self.sigmoid = nn.Sigmoid()

        self.ss = steady()
        self.c_ss, self.k_ss, self.y_ss, self.u_ss, self.v_ss = self.ss.ss()


    def forward(self, state):

        #state = state_preprocessor1(state)
        state = state_preprocessor2(state, self.k_ss)
        #state = state_preprocessor3(state)
        x = self.base(state)
        mean = self.mean_head(x)

        if self.learn_std:
            std = torch.exp(self.std_head(x) + 1e-6)
        else:
            std = torch.exp(self.log_std + 1e-6)

        return mean, std

    def get_action(self, state, test=False):

        state = state.view(1, -1) if state.dim() == 1 else state
        mean, std = self.forward(state)

        lower_bound = torch.zeros_like(mean[:, 0])
        upper_bound = torch.ones_like(mean[:, 0]) * 1.0 
        base_dist = D.Normal(mean[:, 0], std[:, 0])
        sigmoid_transform = T.SigmoidTransform()
        affine_transform = T.AffineTransform(loc=lower_bound, scale=(upper_bound - lower_bound))
        transform = T.ComposeTransform([sigmoid_transform, affine_transform])
        dist = D.TransformedDistribution(base_dist, transform)
        action = dist.sample() if not test else dist.sample([1000]).mean(0)
        log_prob = dist.log_prob(action) 
        log_prob = log_prob.sum(-1) if log_prob.dim() > 1 else log_prob
        return action, log_prob

    def get_log_prob(self, state, action):

        state = state.view(1, -1) if state.dim() == 1 else state
        mean, std = self.forward(state)

        lower_bound = torch.zeros_like(mean[:,0])
        upper_bound = torch.ones_like(mean[:, 0]) * 1.0 
        base_dist = D.Normal(mean[:, 0], std[:, 0])
        sigmoid_transform = T.SigmoidTransform()
        affine_transform = T.AffineTransform(loc=lower_bound, scale=(upper_bound - lower_bound))
        transform = T.ComposeTransform([sigmoid_transform, affine_transform])
        dist = D.TransformedDistribution(base_dist, transform)
        log_prob = dist.log_prob(action) 
        log_prob = log_prob.sum(-1) if log_prob.dim() > 1 else log_prob

        return log_prob
    
    def get_dist(self, state):

        state = state.view(1, -1) if state.dim() == 1 else state
        mean, std = self.forward(state)

        lower_bound = torch.zeros_like(mean[:,0])
        upper_bound = torch.ones_like(mean[:, 0]) * 1.0 
        base_dist = D.Normal(mean[:, 0], std[:, 0])
        sigmoid_transform = T.SigmoidTransform()
        affine_transform = T.AffineTransform(loc=lower_bound, scale=(upper_bound - lower_bound))
        transform = T.ComposeTransform([sigmoid_transform, affine_transform])
        dist = D.TransformedDistribution(base_dist, transform)
        sample  =  dist.sample([1000])
        return sample
