import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import random


class Memory(object):
    
    def __init__(self, size):
        self.size = size
        self.memory = []
        self.position = 0

    def push(self, st, a, u, st1):
        if len(self.memory) < self.size:
            self.memory.append(None)

        element = {'st': st, 'a': a, 'u': u, 'st1': st1}
        self.memory[int(self.position)] = element
        self.position = (self.position + 1) % self.size

    def sample(self , batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ValueNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.network(x)
    
class PolicyNetwork(nn.Module):  
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # extracts mean of the distributions form the final layer of the NN
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.sigmoid = nn.Sigmoid()

        '''
        standard deviations of the distributions, initialized at zero: this parameter does not depend on the state! 
        it's adjustable by the NN but will remain the same one for the same action, independently from the state. 
        ''' 
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        """Forward pass that returns a *distribution* object."""
        x = self.base(state)               # direct output of the NN --> "hidden rep. of the state"
        mean = self.sigmoid(self.mean_head(x))*100         # from hidden rep. we extract the mean of the sigmoid 
        std = torch.exp(self.log_std+1e-6)                 # ensures positivity by taking exponential of the log_std

        # Create a Normal distribution for each action dimension
        dist = Normal(mean, std)
        return dist
    
    def get_action(self, state):
        """
        Sample an action given 'state'. 
        Returns the action and the log probability of that action under the policy.
        """
        # state is typically 1D or 2D [batch_size, state_dim]. Make sure shapes match.
        dist = self.forward(state.view(1,-1))          # A Normal distribution
        action = dist.sample()              # sample a random action
        action = torch.clamp(action, min=0)
        log_prob = dist.log_prob(action).sum(dim=-1)  # sums log probs of each action to get a singlre scalar for the policy training 
        return action, log_prob


class RL_agent(nn.Module):

    def __init__(self, input_dim=2, hidden_dim=128, output_dim=2, lr=1e-3, gamma = 0.99 ):
        super(RL_agent, self).__init__()

        self.replay_buffer = Memory(2000)  # per salvarti i dati
        self.value_net = ValueNetwork(input_dim, hidden_dim, 1)
        self.policy_net = PolicyNetwork(input_dim, hidden_dim, output_dim)
        # Define the Adam optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.gamma = gamma #discount factor
        return

    def update(self):

        if len(self.replay_buffer) < 1000: #training starts after the first model simulation
            return None, None 

        # sampling from bacth and converting to tensors
        batch = self.replay_buffer.sample(100)
        states = torch.tensor([item['st'] for item in batch], dtype=torch.float32)
        next_states = torch.tensor([item['st1'] for item in batch], dtype=torch.float32)
        rewards = torch.tensor([item['u'] for item in batch], dtype=torch.float32)
        actions = torch.tensor([item['a'] for item in batch], dtype=torch.float32)

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

        self.optimizer.zero_grad() #clearing old gradients 
        loss.backward() #computing new gradients --> VERY IMPORTANT  
        self.optimizer.step() #computing new paramenters based on the gradients 

        return loss_V.detach().item(), policy_loss.detach().item()














