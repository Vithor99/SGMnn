import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class V():

    def __init__(self, ):

        self.replay_buffer = Memory(10000)  # per salvarti i dati

        # TODO: definisci NN in pytorch (tip: fai una nuova miniclasse che eredita da nn.Module e dentro fai nn.Sequantial e una funzione di forward)

        # TODO: definisci optimizier con i parametri dell'NN di sopra (tip: Adam optimizer)

        self.gamma = ...

        return

    def sample_action(self):
        # TODO: questo lo facciamo dopo quando impariamo anche la policy
        return

    def update(self):

        if len(self.replay_buffer) < 1000: # questo serve a stabilizzare il training, non impara finche non hai abbastanza dati
            return None

        # TODO: sempla un batch da reply_buffer, mettilo su pytorch e contralla che le dimensioni siano giuste

        # TODO: calcola loss con Bellman

        # TODO: allena parametri con loss e optimizer definito sopra

        return

















