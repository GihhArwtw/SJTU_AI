import numpy as np
import random


class Base_Q_table():
    def __init__(self, length, height, actions=4, alpha=0.1, gamma=0.9, eps=0.1):
        self.table = [0] * actions * length * height
        self.actions = actions
        self.length = length
        self.height = height
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def _index(self, a, x, y):
        return a * self.height * self.length + x * self.length + y

    def best_action(self, x, y):
        '''
        return: the best action of current position (x,y)
        '''
        mav = -100000
        mapos = -1
        change = [[0, 1], [0, -1], [-1, 0], [1, 0]]
        for i in range(self.actions):            
            tx, ty = x + change[i][0], y + change[i][1]
            if (min(tx, ty)<0 or tx>=self.height or ty>=self.length):
                continue
            if(self.table[self._index(i,x,y)] > mav):
                mav = self.table[self._index(i,x,y)]
                mapos = i
        return mapos

    def _epsilon(self, num_episode):
        return self.eps

    def max_q(self, x, y):
        action = self.best_action(x, y)
        return self.table[self._index(action,x,y)]
        
    def take_action(self, x, y, num_episode, method='eps_greedy'):
        '''
        method: 'eps_greedy' denotes that taking actions using epsilon greedy (used as the policy for SARSA, and the behavior policy for Q-Learning), 
        'full_greedy' denotes that taking actions fully greedy w.r.t. the current estimated Q table (the target policy for Q-Learning)
        '''
        if method == 'eps_greedy':
            """ ------------- Programming 1: implement the epsilon greedy policy ------------- """
            """ YOUR CODE HERE """

            random_num = random.random()
            if random_num < self._epsilon(num_episode):
                action = random.randint(0, self.actions-1)   # exploration
            else:
                action = self.best_action(x,y)               # exploitation
                
            """ ------------- Programming 1 ------------- """
        elif method == 'full_greedy':
            action = self.best_action(x,y)
        return action

    def update(self, direct, next_direct, s0, s1, reward, is_terminated):
        pass
        
    