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

    def push(self, st, a, u, st1, y):
        if len(self.memory) < self.size:
            self.memory.append(None)

        element = {'st': st, 'a': a, 'u': u, 'st1': st1, 'y': y}
        self.memory[int(self.position)] = element
        self.position = (self.position + 1) % self.size

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


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
