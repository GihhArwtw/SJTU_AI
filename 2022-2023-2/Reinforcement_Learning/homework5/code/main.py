import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from TRPO import TRPO
from PPO import PPO

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

def plot_result(episodes_list_TRPO,return_list_TRPO,episodes_list_PPO,return_list_PPO):
    plt.figure()
    plt.plot(episodes_list_TRPO,return_list_TRPO, label = "TRPO")
    plt.plot(episodes_list_PPO,return_list_PPO, label = "PPO")

    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.legend(loc='lower right', ncol=2, mode="expand", borderaxespad=0.)
    plt.title('PPO/TRPO Results')
    plt.savefig('PPO_TRPO Results.png')

def plot_mv_result(episodes_list_TRPO,mv_return_TRPO,episodes_list_PPO,mv_return_PPO):
    plt.figure()
    plt.plot(episodes_list_TRPO,mv_return_TRPO, label = "TRPO")
    plt.plot(episodes_list_PPO,mv_return_PPO, label = "PPO")
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.legend(loc='lower right', ncol=2, mode="expand", borderaxespad=0.)
    plt.title('The moving average results of PPO/TRPO'.format(env_name))
    plt.savefig('mv_PPO_TRPO Results.png')

def plot_vary_beta(episodes_list_PPO, return_list_PPO_beta, beta_list):
    plt.figure()
    for i in range(len(beta_list)):
        plt.plot(episodes_list_PPO,return_list_PPO_beta[i], label = "beta="+str(beta_list[i]))
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.legend(loc='lower right', ncol=2, mode="expand", borderaxespad=0.)
    plt.title('The moving average results of PPO varing beta'.format(env_name))
    plt.savefig('PPO_vary_beta.pdf')

def plot_vary_kl_constraint(episodes_list_TRPO, return_list_TRPO_constraint, kl_constraint_list):
    plt.figure()
    for i in range(len(kl_constraint_list)):
        plt.plot(episodes_list_TRPO,return_list_TRPO_constraint[i], label = "constraint="+str(kl_constraint_list[i]))
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.legend(loc='lower right', ncol=2, mode="expand", borderaxespad=0.)
    plt.title('The moving average results of TRPO varing kl constraint'.format(env_name))
    plt.savefig('TRPO_vary_kl.pdf')

if __name__ == '__main__':
    ################### The common parameters *****************************
    critic_lr = 1e-2
    num_episodes = 500
    hidden_dim = 128
    gamma = 0.98  
    lmbda = 0.95 ########## The parameter of Generalized Advantage Estimation (GAE)
    epochs = 10
    eps = 0.2
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
        "cpu")

    ################### The parametets for TRPO *****************************
    kl_constraint = 0.0005 ######### \delta in orginal TRPO paper
    alpha = 0.5

    ################### The parametets for PPO *****************************
    actor_lr = 5e-3  ######## The actor learning rate only used by PPO


    env_name = 'CartPole-v0'
    env = gym.make(env_name)
    env.seed(14985)
    torch.manual_seed(1)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    
    kl_constraint_list = [1e-6,1e-5, 1e-4,1e-3]
    return_list_TRPO_constraint = []
    for kl_constraint in kl_constraint_list:
        print("kl constraint: "+ str(kl_constraint))
        agent = TRPO(hidden_dim, env.observation_space, env.action_space, lmbda,
            kl_constraint, alpha, critic_lr, gamma, device)

        return_list_TRPO = train_on_policy_agent(env, agent, num_episodes)
        episodes_list_TRPO = list(range(len(return_list_TRPO)))
        mv_return_TRPO = moving_average(return_list_TRPO, 9)
        return_list_TRPO_constraint.append(mv_return_TRPO)
    plot_vary_kl_constraint(episodes_list_TRPO, return_list_TRPO_constraint, kl_constraint_list)


    beta_list = [0,10,100, 1000]
    return_list_PPO_beta = []
    for beta in beta_list:
        print("beta: "+str(beta))
        agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda,
                epochs, beta, gamma, device)

        return_list_PPO = train_on_policy_agent(env, agent, num_episodes)
        episodes_list_PPO = list(range(len(return_list_PPO)))
        mv_return_PPO = moving_average(return_list_PPO, 9)
        return_list_PPO_beta.append(mv_return_PPO)
    plot_vary_beta(episodes_list_PPO, return_list_PPO_beta, beta_list)

