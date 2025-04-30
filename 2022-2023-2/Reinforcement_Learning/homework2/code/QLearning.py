from Base import Base_Q_table
import numpy as np
from Env import Env
from tqdm import tqdm


class Q_table_qlearning(Base_Q_table):
    def __init__(self, length, height, gamma, actions=4, alpha=0.005, eps=0.1):
        super().__init__(length, height, actions, alpha = alpha, gamma = gamma, eps=eps)

    def update(self, a, s0, s1, r, is_terminated):
        """ ------------- Programming 2: implement the updating of the Q table for Q-learning (you may refer to the last equation on Lecture 4, Page 17) ------------- """
        """ YOUR CODE HERE """

        # if is_terminated:
        #     return
        # else:
        self.table[self._index(a, s0[0], s0[1])] = \
            self.table[self._index(a, s0[0], s0[1])] \
            + self.alpha * \
            (r + self.gamma * self.max_q(s1[0], s1[1]) - self.table[self._index(a, s0[0], s0[1])])
        
        """ ------------- Programming 2 ------------- """

    def cliff_walk(self):
        reward_eps_greedy, reward_full_greedy = [], []
        env = Env(length=12, height=4)
        for num_episode in tqdm(range(3000)):
            ''' the reward for current behavior policy '''
            episodic_cumulative_reward = 0
            is_terminated = False
            s0 = [0, 0]
            while not is_terminated:
                """ 
                ------------- 
                Programming 3: implement the Q-learning algorithm by invoking the update method you implemented in the above Programming 2 
                (you may refer to Lecture 4, Page 17-18 or Page 131 of the reinforcement learning book by Rich Sutton) 
                ------------- 
                """
                """ YOUR CODE HERE """
                
                # take action
                action = self.take_action(s0[0], s0[1], num_episode)
                r, s1, is_terminated = env.step(action)

                # update Q table offline
                self.update(action, s0, s1, r, is_terminated)

                # move to the next state
                episodic_cumulative_reward += r
                s0 = s1

                """ ------------- Programming 3 ------------- """
            reward_eps_greedy.append(episodic_cumulative_reward)
            env.reset()
            ''' 
            the episodic cumulative reward for using current target policy
            (note that during the execution of the current target policy of Q-learning, there is no updating of the Q table)
            '''
            n_trial = 0
            episodic_cumulative_reward = 0
            is_terminated = False
            s0 = [0, 0]
            while not is_terminated:
                n_trial += 1
                action = self.take_action(s0[0], s0[1], num_episode, method='full_greedy')
                r, s1, is_terminated = env.step(action)
                episodic_cumulative_reward += r
                s0 = s1
                if n_trial>int(1e2):
                    episodic_cumulative_reward=-int(1e2)
                    break
            reward_full_greedy.append(episodic_cumulative_reward)
            env.reset()
        return reward_eps_greedy, reward_full_greedy