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

    def push(self, st, c, u, st1):
        if len(self.memory) < self.size:
            self.memory.append(None)

        element = {'st': st, 'c': c, 'u': u, 'st1': st1}

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
        # xTODO: definisci NN in pytorch (tip: fai una nuova miniclasse che eredita da nn.Module e dentro fai nn.Sequantial e una funzione di forward)
        self.value_net = ValueNetwork(input_dim, hidden_dim, output_dim)

        # xTODO: definisci optimizier con i parametri dell'NN di sopra (tip: Adam optimizer)
        # Define the Adam optimizer
        self.optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.gamma = gamma

        return

    def sample_action(self):
        # TODO: questo lo facciamo dopo quando impariamo anche la policy
        return

    def update(self):

        if len(self.replay_buffer) < 1000: # questo serve a stabilizzare il training, non impara finche non hai abbastanza dati
            return None

        # xTODO: sempla un batch da reply_buffer, mettilo su pytorch e contralla che le dimensioni siano giuste
        # sampling from bacth and converting to tensors
        batch = self.replay_buffer.sample(100)
        states = torch.tensor([item['st'] for item in batch], dtype=torch.float32)
        next_states = torch.tensor([item['st1'] for item in batch], dtype=torch.float32)
        rewards = torch.tensor([item['u'] for item in batch], dtype=torch.float32)

        #Checking the dimensions
        print("states shape:", states.shape)
        print("next states shape:", next_states.shape)
        print("Rewards shape:", rewards.shape)

        # xTODO: calcola loss con Bellman
        # Compute the target values
        with torch.no_grad():
            next_state_values = self.value_net(next_states).squeeze()
            target_values = rewards + self.gamma * next_state_values

        # Compute the predicted values
        predicted_values = self.value_net(states).squeeze()

        # Compute the loss
        loss = self.loss_fn(predicted_values, target_values) 

        # xTODO: allena parametri con loss e optimizer definito sopra
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

















