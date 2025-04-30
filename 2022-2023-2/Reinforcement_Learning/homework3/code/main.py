import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
import time
from env import Maze
from DynaQ import DynaQ
from DynaQ_Plus import DynaQ_Plus
import optparse


def parseOptions():
    optParser = optparse.OptionParser()
    optParser.add_option('-e', '--episodes',action='store',
                         type='int',dest='episodes',default=50,
                         metavar="K", help='Number of epsiodes of the MDP to run (default %default)')
    optParser.add_option('-m', '--maze',action='store',
                         metavar="M", type='string',dest='maze',default="basic",
                         help='Maze to use (case sensitive; options are basic, blocking, cut default %default)' )
    optParser.add_option('-a', '--agent',action='store', metavar="A",
                         type='string',dest='agent',default="dynaQ",
                         help='Agent type')
    optParser.add_option('-k', '--kappa',action='store',
                         type='float',dest='kappa',default=1e-3,
                         metavar="K", help='Factor in dynaQplus (default %default)')
    optParser.add_option('-p', '--plot',action='store', metavar="P",
                         type='string',dest='plot',default="p1",
                         help='Plot type')
    opts, _ = optParser.parse_args()

    return opts

if __name__=='__main__':
    opts = parseOptions()
    if opts.plot == "p1": #plot the steps per episode
        n_planning_list = [0, 5, 50]
        for n_planning in n_planning_list:
            print('Q-planning步数为：%d' % n_planning)
            time.sleep(0.5)
            ncol = 9
            nrow = 6
            if opts.maze == "basic":
                env = Maze(ncol, nrow, model="basic")
            if opts.maze == "blocking":
                env = Maze(ncol, nrow, model="blocking")
            if opts.maze == "cut":
                env = Maze(ncol, nrow, model="cut")
            epsilon = 0.01
            alpha = 0.1
            gamma = 0.95
            kappa = opts.kappa
            if opts.agent == "dynaQ":
                agent = DynaQ(ncol, nrow, epsilon, alpha, gamma, n_planning)
            elif opts.agent== "dynaQplus":
                agent = DynaQ_Plus(ncol, nrow, epsilon, alpha, gamma, kappa, n_planning)
            num_episodes = 50

            steps_list = [] 
            
            for i in range(10): 
                with tqdm(total=int(num_episodes / 10),
                        desc='Iteration %d' % i) as pbar:
                    for i_episode in range(int(num_episodes / 10)):
                        episode_return = 0
                        state = env.reset()
                        done = False
                        t_count = 0
                        while not done:
                            action = agent.take_action(state)
                            next_state, reward, done = env.step(action)
                            episode_return += reward  
                            t_count += 1
                            agent.update(state, action, reward, next_state)
                            state = next_state
                        steps_list.append(t_count)
                        if (i_episode + 1) % 10 == 0:  # 每10条序列打印一下这10条序列的平均回报
                            pbar.set_postfix({
                                'episode':
                                '%d' % (num_episodes / 10 * i + i_episode + 1),
                                'return':
                                '%.3f' % np.mean(steps_list[-10:])
                            })
                        pbar.update(1)
            episodes_list = list(range(len(steps_list)))
            plt.plot(episodes_list,
                    steps_list,
                    label=str(n_planning) + ' planning steps')
        plt.legend()
        plt.xlabel('Episodes')
        plt.ylabel('Steps per episode')
        plt.title('Dyna-Q on {}'.format(opts.maze + ' maze'))
        plt.show()
    if opts.plot == "p2": #plot cumulative reward
        for agent_type in ["dynaQ", "dynaQ_Plus"]:
            SEED = 666
            random.seed(SEED)
            np.random.seed(SEED)
            ncol = 9
            nrow = 6
            if opts.maze == "basic":
                env = Maze(ncol, nrow, model="basic")
            if opts.maze == "blocking":
                env = Maze(ncol, nrow, model="blocking")
            if opts.maze == "cut":
                env = Maze(ncol, nrow, model="cut")
            epsilon = 0.1
            alpha = 1
            gamma = 0.95
            kappa = opts.kappa
            n_planning = 20
            num_episodes = 20 
            if agent_type=="dynaQ":
                agent = DynaQ(ncol, nrow, epsilon, alpha, gamma, n_planning)
            else:
                agent = DynaQ_Plus(ncol, nrow, epsilon, alpha, gamma, kappa, n_planning)
            time_step = 0
            cumulative_reward = np.zeros(25000)
            while time_step < 25000:
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done = env.step(action)
                    time_step += 1
                    if time_step>=25000:
                        break
                    cumulative_reward[time_step] = cumulative_reward[time_step-1] + reward
                    agent.update(state, action, reward, next_state)
                    state = next_state
            time_list = list(range(len(cumulative_reward)))
            plt.plot(time_list, cumulative_reward, label = agent_type)
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Cumulative Reward')
        plt.title('Dyna-Q and Dyna-Q+ on {}'.format(opts.maze + ' maze'))
        plt.show()