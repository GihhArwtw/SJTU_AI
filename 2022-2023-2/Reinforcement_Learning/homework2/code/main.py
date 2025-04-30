import numpy as np
from SARSA import Q_table_sarsa
from QLearning import Q_table_qlearning
from Env import Env
import matplotlib.pyplot as plt


def plot_cumreward(reward_qlearning_behavior_policy, reward_qlearning_target_policy, reward_sarsa, eps=0.2):
    plt.cla() 
    plt.clf()
    plt.close()
    # rewards for q_learning
    cum_rewards_q = []
    n_interval = 10
    count = 0
    for cache in reward_qlearning_behavior_policy:
        count += 1
        if(count % n_interval == 0):
            cum_rewards_q.append(cache)
    # rewards for q_learning (target policy)
    cum_rewards_q_target = []
    count = 0
    for cache in reward_qlearning_target_policy:
        count += 1
        if(count % n_interval == 0):
            cum_rewards_q_target.append(cache)
    # rewards for SARSA
    cum_rewards_SARSA = []
    count = 0 
    for cache in reward_sarsa:
        count += 1
        if(count % n_interval == 0):
            cum_rewards_SARSA.append(cache)
    plt.plot(cum_rewards_q, label = "Q-Learning (behavior)")
    plt.plot(cum_rewards_q_target, label = "Q-Learning (target)")
    plt.plot(cum_rewards_SARSA, label = "SARSA")
    plt.ylabel('Cumulative Rewards')
    plt.xlabel('Index of Episode Batch')
    plt.title("Performance w/ eps={}".format(np.round(eps,2)))
    plt.legend(loc='lower right', ncol=2, mode="expand", borderaxespad=0.)
    plt.ylim(-150, 0)
    plt.savefig('cum_rewards_eps={}.pdf'.format(np.round(eps,2)))


if __name__ == '__main__':
    height, length = 4, 12
    gamma = 0.95
    alpha = 0.1
    eps_list = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for eps in eps_list:
        agent = Q_table_qlearning(height = height, length = length, gamma = gamma, eps = eps, alpha = alpha)
        reward_qlearning_behavior_policy, reward_qlearning_target_policy = agent.cliff_walk()
        agent = Q_table_sarsa(height = height, length = length, gamma = gamma, eps = eps, alpha = alpha)
        reward_sarsa = agent.cliff_walk()
        plot_cumreward(reward_qlearning_behavior_policy, reward_qlearning_target_policy, reward_sarsa, eps)