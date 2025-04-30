from Base import Base_Q_table
import numpy as np
from Env import Env
from tqdm import tqdm


class Q_table_sarsa(Base_Q_table):
    def __init__(self, length, height, actions=4, alpha=0.1, gamma=0.9, eps=0.1):
        super().__init__(length, height, actions, alpha = alpha, gamma = gamma, eps=eps)

    def update(self, direct, next_direct, s0, s1, reward, is_terminated):
        """ ------------- Programming 4: implement the updating of the Q table for SARSA (you may refer to Lecture 4, Page 8) ------------- """
        """ YOUR CODE HERE """

        # if is_terminated:
        #     return
        # else:
        self.table[self._index(direct, s0[0], s0[1])] = \
            self.table[self._index(direct, s0[0], s0[1])] \
            + self.alpha * \
            (reward + self.gamma * self.table[self._index(next_direct, s1[0], s1[1])] \
                - self.table[self._index(direct, s0[0], s0[1])])

        """ ------------- Programming 4 ------------- """

    def cliff_walk(self):
        rewards = []
        env = Env(length=12, height=4)
        for num_episode in tqdm(range(3000)):
            episodic_cumulative_reward = 0
            is_terminated = False
            s0 = [0, 0]
            action = self.take_action(s0[0], s0[1], num_episode)
            while not is_terminated:
                """
                ------------- 
                Programming 5: implement the SARSA algorithm by invoking the update method you implemented in the above Programming 4 
                (you may refer to Lecture 4, Page 8)
                ------------- 
                """
                """ YOUR CODE HERE """

                # take action
                r, s1, is_terminated = env.step(action)
                next_action = self.take_action(s1[0], s1[1], num_episode)

                # update Q table online
                self.update(action, next_action, s0, s1, r, is_terminated)

                # move to the next state
                episodic_cumulative_reward += r
                s0, action = s1, next_action

                """ ------------- Programming 5 ------------- """
            rewards.append(episodic_cumulative_reward)
            env.reset()
        return rewards