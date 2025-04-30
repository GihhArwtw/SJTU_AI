import math
import random
import time
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.distributions import Normal
from tqdm import tqdm

class DQN_base(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        raise NotImplementedError

    def act(self, state, epsilon, device, discrete_action_n):
        if random.random() > epsilon:
            with torch.no_grad():
                state = torch.FloatTensor(np.float32(state)).unsqueeze(0).to(device)
                q_value = self.forward(state)
                action = q_value.max(1).indices.item()
        else:
            action = random.randrange(discrete_action_n)
        return action
    

class DQN(DQN_base):
    def __init__(self, input_n, num_actions, h_size=24):
        super(DQN, self).__init__()

        self.input_n = input_n
        self.num_actions = num_actions

        self.fc = nn.Sequential(
            nn.Linear(self.input_n, h_size),
            nn.ReLU(),
            nn.Linear(h_size, h_size),
            nn.ReLU(),
            nn.Linear(h_size, self.num_actions)
        )
        
    def forward(self, x):
        x = self.fc(x)
        return x