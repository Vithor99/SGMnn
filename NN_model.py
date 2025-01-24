import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
    

class V():

    def __init__(self, input_dim=2, hidden_dim=128, output_dim=1, lr=1e-3, gamma = 0.99 ):

        self.replay_buffer = Memory(10000)  # per salvarti i dati
        self.value_net = ValueNetwork(input_dim, hidden_dim, output_dim)

        # Define the Adam optimizer
        self.optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.gamma = gamma #discount factor

        return

    def sample_action(self):
        # TODO: questo lo facciamo dopo quando impariamo anche la policy
        return

    def update(self):

        if len(self.replay_buffer) < 1000: # questo serve a stabilizzare il training, non impara finche non hai abbastanza dati
            return None

        # sampling from bacth and converting to tensors
        batch = self.replay_buffer.sample(100)
        states = torch.tensor([item['st'] for item in batch], dtype=torch.float32)
        next_states = torch.tensor([item['st1'] for item in batch], dtype=torch.float32)
        rewards = torch.tensor([item['u'] for item in batch], dtype=torch.float32)

        #Checking the dimensions
        #print("states shape:", states.shape)
        #print("next states shape:", next_states.shape)
        #print("Rewards shape:", rewards.shape)

        # Compute the target values
        with torch.no_grad():
            next_state_values = self.value_net(next_states).squeeze()
            target_values = rewards + self.gamma * next_state_values

        # Compute the predicted values
        predicted_values = self.value_net(states).squeeze()

        # Compute the loss
        loss = self.loss_fn(predicted_values, target_values) 

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
'''
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        # Define the first fully connected layer that maps state_dim inputs to hidden_dim outputs
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        # Define the second fully connected layer with hidden_dim inputs and outputs
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Define the output layer for the mean of the action distribution
        self.mean = nn.Linear(hidden_dim, action_dim)
        # Define the output layer for the log standard deviation of the action distribution
        self.log_std = nn.Linear(hidden_dim, action_dim)
        # Store the dimension of the action space
        self.action_dim = action_dim

    def forward(self, state):
        """
        Forward pass through the network.
        Args:
            state (torch.Tensor): The input state tensor.
        Returns:
            (torch.Tensor, torch.Tensor): The mean and standard deviation tensors for the action distribution.
        """
        # Apply ReLU activation after the first fully connected layer
        x = F.relu(self.fc1(state))
        # Apply ReLU activation after the second fully connected layer
        x = F.relu(self.fc2(x))
        # Compute the mean of the action distribution
        mean = self.mean(x)
        # Compute the log standard deviation of the action distribution
        log_std = self.log_std(x)
        # Exponentiate the log_std to obtain the standard deviation (must be positive)
        std = torch.exp(log_std)
        return mean, std

    def get_action(self, state):
        """
        Sample an action from the policy's action distribution given a state.
        Args:
            state (torch.Tensor): The input state tensor.
        Returns:
            torch.Tensor: A sampled action tensor.
        """
        # Forward pass to get the mean and standard deviation
        mean, std = self.forward(state)
        # Create a normal distribution parameterized by mean and std
        dist = torch.distributions.Normal(mean, std)
        # Sample an action from the normal distribution
        action = dist.sample()
        return action
    


class PPOAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=3e-4, gamma=0.99, clip_epsilon=0.2):
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.value_net = ValueNetwork(state_dim, hidden_dim)
        self.optimizer = optim.Adam(list(self.policy_net.parameters()) + list(self.value_net.parameters()), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon

    def compute_loss(self, states, actions, old_log_probs, returns, advantages):
        """
        Compute the PPO loss function.
        """
        # Forward pass through the policy network
        means, stds = self.policy_net(states)
        dist = torch.distributions.Normal(means, stds)
        new_log_probs = dist.log_prob(actions).sum(dim=-1)

        # Compute the ratio of new and old probabilities
        ratio = torch.exp(new_log_probs - old_log_probs)

        # Clip the ratio and compute the policy loss
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # Compute the value loss
        values = self.value_net(states)
        value_loss = F.mse_loss(values, returns)

        # Entropy bonus (optional)
        entropy_bonus = dist.entropy().mean()

        # Total loss
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus
        return loss

    def update(self, states, actions, old_log_probs, returns, advantages):
        """
        Update the policy and value networks.
        """
        # Compute the loss
        loss = self.compute_loss(states, actions, old_log_probs, returns, advantages)
        
        # Perform backpropagation and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

'''















