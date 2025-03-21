import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from utils import ReplayBuffer, hard_update, soft_update

from model_architectures import ValueNetwork, StochasticPolicyNetwork


class SoftActorCritic(nn.Module):

    def __init__(self, input_dim=2, architecture_params=None, output_dim=2, lr=1e-3, gamma=0.99, epsilon=0.0, batch_size=128, alpha=0, learn_std=True, device=None):
        super(SoftActorCritic, self).__init__()

        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device

        self.replay_buffer = ReplayBuffer(int(1e6))

        self.policy_net = StochasticPolicyNetwork(input_dim, architecture_params, output_dim, alpha=alpha, learn_std=learn_std)
        self.value_net1 = ValueNetwork(input_dim, architecture_params, 1)
        self.value_net2 = ValueNetwork(input_dim, architecture_params, 1)
        self.value_net1_target = ValueNetwork(input_dim, architecture_params, 1)
        self.value_net2_target = ValueNetwork(input_dim, architecture_params, 1)

        hard_update(self.value_net1, self.value_net1_target)
        hard_update(self.value_net2, self.value_net2_target)

        self.log_alpha = nn.Parameter(torch.log(torch.tensor([0.1], dtype=torch.float)), requires_grad=True)

        self.optimizer_v1 = optim.Adam(self.value_net1.parameters(), lr=lr)
        self.optimizer_v2 = optim.Adam(self.value_net2.parameters(), lr=lr)
        self.optimizer_pi = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        self.loss_fn = nn.MSELoss()

        self.target_entropy = -2
        self.target_entropy_init = 1.0
        self.target_entropy_min = 0.1
        self.update_count = 0

    def get_action(self, st, test=False):
        a = self.policy_net.get_action(st, test=test)
        return a

    def update(self):

        states, actions, rewards, next_states, terminal = self.replay_buffer.sample(self.batch_size)

        states = torch.from_numpy(states).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        actions = torch.from_numpy(actions).float().to(self.device)
        terminal = torch.from_numpy(terminal).float().to(self.device)


        with torch.no_grad():

            next_actions, next_logprob = self.policy_net.get_action(next_states)
            q1_next = self.value_net1_target(next_states, next_actions)
            q2_next = self.value_net2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.log_alpha.exp().detach() * next_logprob
            q_target = rewards + self.gamma * (1 - terminal) * q_next
            # TODO: squeeze?

        q1 = self.value_net1(states, actions)
        q2 = self.value_net2(states, actions)
        critic1_loss = self.mse(q1, q_target)
        critic2_loss = self.mse(q2, q_target)

        self.optimizer_v1.zero_grad()
        critic1_loss.backward()
        self.optimizer_v1.step()

        self.optimizer_v2.zero_grad()
        critic2_loss.backward()
        self.optimizer_v2.step()

        new_actions, new_logprobs = self.policy_net.get_action(states)
        q1 = self.value_net1(states, new_actions)
        q2 = self.value_net2(states, new_actions)
        q_pi = torch.min(q1, q2)

        actor_loss = (self.log_alpha.exp().detach() * new_logprobs - q_pi).mean()

        self.optimizer_pi.zero_grad()
        actor_loss.backward()
        self.optimizer_pi.step()

        alpha_loss = (-self.log_alpha.exp() * (new_logprobs + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        with torch.no_grad():
            self.log_alpha.clamp_(-10, 2)

        self.target_entropy = self.target_entropy_min
        self.target_entropy += (self.target_entropy_init - self.target_entropy_min) * np.exp(-self.update_count / int(1e6))

        self.update_count += 1

        soft_update(self.value_net1, self.value_net1_target, 0.005)
        soft_update(self.value_net2, self.cvalue_net2_target, 0.005)

        return (0.5*critic1_loss+0.5*critic2_loss).detach().item(), actor_loss.detach().item(), alpha_loss.detach().item()

    def save(self, file_name):
        if not os.path.exists("saved_models"):
            os.mkdir("saved_models")
        torch.save(self.state_dict(), "saved_models/" + file_name + ".pt")

    def load(self, file_name):
        self.load_state_dict(torch.load("saved_models/" + file_name + ".pt"))










