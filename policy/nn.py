import torch
from torch import nn, Tensor, zeros, cat

import os
import pickle

import math

from utils.obstat import ObStat


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        # nn.init.xavier_uniform_(m.weight, gain=1)
        # nn.init.constant_(m.bias, 0)

        nn.init.kaiming_normal_(m.weight)
        # nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)

        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y


class Actor(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Actor, self).__init__()
        self.action_space = action_space
        num_outputs = action_space

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.ln1 = LayerNorm(hidden_size)

        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = LayerNorm(hidden_size)

        self.mu = nn.Linear(hidden_size, num_outputs)

        self.apply(weights_init_)

        self.obstat = ObStat(num_inputs, 1e-2)

    def forward(self, inputs):
        x = inputs
        x = self.linear1(x)
        x = self.ln1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = self.ln2(x)
        x = torch.relu(x)
        mu = self.mu(x)
        mu = torch.tanh(mu)
        return mu


def save_nn(file, folder: str):
    if not os.path.exists(folder):
        os.makedirs(folder)
    pickle.dump(file, open(os.path.join(folder, 'policy.pkl'), 'wb'))


def load_nn(file: str):
    module = pickle.load(open(file, 'rb'))
    return module
