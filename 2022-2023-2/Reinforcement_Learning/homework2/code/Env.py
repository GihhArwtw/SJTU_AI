import time
import random
import matplotlib.pyplot as plt
import numpy as np


class Env():
    def __init__(self, length, height):
        self.length = length        
        self.height = height        
        self.x = 0                  
        self.y = 0                  

    def render(self, frames=50):
        for i in range(self.height):
            if i == 0: 
                line = ['S'] + ['x']*(self.length - 2) + ['T'] # the lower left corner is the start point, the lower right corner is the end point
            else:
                line = ['.'] * self.length # -1 reward
            if self.x == i:
                line[self.y] = 'o'
            print(''.join(line))
        print('\033['+str(self.height+1)+'A')  
        time.sleep(1.0 / frames)

    def step(self, action):
        """4 legal actions, 0:up, 1:down, 2:left, 3:right"""
        change = [[0, 1], [0, -1], [-1, 0], [1, 0]]

        tx, ty = self.x + change[action][0], self.y + change[action][1]
        self.x = min(self.height - 1, max(0, tx))
        self.y = min(self.length - 1, max(0, ty))

        states = [self.x, self.y]
        reward = -1
        terminal = False
        if self.x == 0: 
            if self.y > 0:
                terminal = True
                if self.y != self.length - 1:
                    reward = -100
        return reward, states, terminal

    def reset(self):
        self.x = 0
        self.y = 0