import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.distributions as D
import torch.distributions.transforms as T
from utils import state_preprocessor
import matplotlib.pyplot as plt
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

    def forward(self, x, a=None):
        x = state_preprocessor(x)
        x = x if a is None else torch.cat([x, a], -1)
        return self.network(x)


class StochasticPolicyNetwork(nn.Module):

    def __init__(self, state_dim, architecture_params, action_dim, alpha=0, learn_std=True):
        super(StochasticPolicyNetwork, self).__init__()

        self.var_scale = architecture_params['policy_var']
        self.learn_std = learn_std
        #self.action_bounds = architecture_params['action_bounds']
        #elf.use_hard_bounds = architecture_params['use_hard_bounds']
        self.alpha = alpha

        #self.learn_consumption = learn_consumption
        self.ss = steady()

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

        # self.scale_mean = torch.tensor([[0.1, 0.2]]).float()

    def forward(self, state):

        state = state_preprocessor(state)

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
        upper_bound = torch.ones_like(mean[:, 0]) 

        base_dist_0 = D.Normal(mean[:, 0], std[:, 0])
        base_dist_1 = D.Normal(mean[:, 1], std[:, 1])

        sigmoid_transform = T.SigmoidTransform()
        affine_transform = T.AffineTransform(loc=lower_bound, scale=(upper_bound - lower_bound))
        transform = T.ComposeTransform([sigmoid_transform, affine_transform])
        dist_0 = D.TransformedDistribution(base_dist_0, transform)
        dist_1 = D.TransformedDistribution(base_dist_1, transform)

        action_0 = dist_0.sample() if not test else dist_0.sample([1000]).mean(0)
        action_1 = dist_1.sample() if not test else dist_1.sample([1000]).mean(0)

        action = torch.stack([action_0, action_1], -1)
        log_prob = dist_0.log_prob(action_0) + dist_1.log_prob(action_1)
        log_prob = log_prob.sum(-1) if log_prob.dim() > 1 else log_prob

        return action, log_prob

    def get_log_prob(self, state, action):

        state = state.view(1, -1) if state.dim() == 1 else state
        mean, std = self.forward(state)

        lower_bound = torch.zeros_like(mean[:, 0])
        upper_bound = torch.ones_like(mean[:, 0]) 

        base_dist_0 = D.Normal(mean[:, 0], std[:, 0])
        base_dist_1 = D.Normal(mean[:, 1], std[:, 1])

        sigmoid_transform = T.SigmoidTransform()
        affine_transform = T.AffineTransform(loc=lower_bound, scale=(upper_bound - lower_bound))
        transform = T.ComposeTransform([sigmoid_transform, affine_transform])
        dist_0 = D.TransformedDistribution(base_dist_0, transform)
        dist_1 = D.TransformedDistribution(base_dist_1, transform)

        log_prob = dist_0.log_prob(action[:,0]) + dist_1.log_prob(action[:,1])
        log_prob = log_prob.sum(-1) if log_prob.dim() > 1 else log_prob

        return log_prob
    

    def get_dist(self, state):

        state = state.view(1, -1) if state.dim() == 1 else state
        mean, std = self.forward(state)

        lower_bound = torch.zeros_like(mean[:, 0])
        upper_bound = torch.ones_like(mean[:, 0]) 

        base_dist_0 = D.Normal(mean[:, 0], std[:, 0])
        base_dist_1 = D.Normal(mean[:, 1], std[:, 1])

        sigmoid_transform = T.SigmoidTransform()
        affine_transform = T.AffineTransform(loc=lower_bound, scale=(upper_bound - lower_bound))
        transform = T.ComposeTransform([sigmoid_transform, affine_transform])
        dist_0 = D.TransformedDistribution(base_dist_0, transform)
        dist_1 = D.TransformedDistribution(base_dist_1, transform)

        #debug
        # Generate samples from dist_1
        sample0 = dist_0.sample([1000]).detach().cpu().numpy()
        sample1 = dist_1.sample([1000]).detach().cpu().numpy()
        
        return sample0, sample1
