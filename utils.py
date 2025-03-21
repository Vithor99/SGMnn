import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import random
from collections import deque


def state_preprocessor(s):
    return s/100


def soft_update(nets, nets_target, tau=0.005):
    for param, target_param in zip(nets.parameters(), nets_target.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def hard_update(nets, nets_target):
    for net_i, net_target_i in zip(nets, nets_target):
        net_target_i.load_state_dict(net_i.state_dict())


class ReplayBuffer:
    def __init__(self, max_size=1000000):
        self.buffer = deque(maxlen=max_size)

    def add(self, st, a, u, st1, y):
        self.buffer.append((st,
                            a,
                            np.array(u, dtype=np.float32),
                            st1,
                            np.array(y, dtype=np.float32)))

    def sample(self, batch_size=64):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.float32),
                np.array(rewards, dtype=np.float32).reshape(-1, 1),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32).reshape(-1, 1))

    def __len__(self):
        return len(self.buffer)


class BatchData:

    def __init__(self):
        self.st = []
        self.a = []
        self.logprobs = []
        self.u = []
        self.st1 = []
        self.y = []
        self.terminal = []

    def push(self, st, a, logprob, u, st1, y, terminal):
        self.st.append(st)
        self.a.append(a)
        self.logprobs.append(logprob)
        self.u.append(u)
        self.st1.append(st1)
        self.y.append(y)
        self.terminal.append(terminal)

    def clear(self):
        self.st.clear()
        self.a.clear()
        self.logprobs.clear()
        self.u.clear()
        self.st1.clear()
        self.y.clear()
        self.terminal.clear()
