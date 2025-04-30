import gym
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class TRPO:
    """ TRPO算法 """
    def __init__(self, hidden_dim, state_space, action_space, lmbda,
                 kl_constraint, alpha, critic_lr, gamma, device):
        state_dim = state_space.shape[0]
        action_dim = action_space.n

        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda 
        self.kl_constraint = kl_constraint 
        self.alpha = alpha 
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()

    ############# The following function implement the trick in original TRPO paper, Appendix C.1 ###############
    def hessian_matrix_vector_product(self, states, old_action_dists, vector):

        new_action_dists = torch.distributions.Categorical(self.actor(states))
        kl = torch.mean(
            torch.distributions.kl.kl_divergence(old_action_dists,
                                                 new_action_dists))  
        kl_grad = torch.autograd.grad(kl,
                                      self.actor.parameters(),
                                      create_graph=True)
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])

        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        grad2 = torch.autograd.grad(kl_grad_vector_product,
                                    self.actor.parameters())
        grad2_vector = torch.cat([grad.view(-1) for grad in grad2])
        return grad2_vector

    ############# The following function implement onjugate gradient algorithm in original TRPO paper, Appendix C to find search direction $s$ ############

    def conjugate_gradient(self, grad, states, old_action_dists):  
        x = torch.zeros_like(grad)
        r = grad.clone()
        p = grad.clone()
        rdotr = torch.dot(r, r)
        for i in range(10):  # 共轭梯度主循环
            Hp = self.hessian_matrix_vector_product(states, old_action_dists,
                                                    p)
            alpha = rdotr / torch.dot(p, Hp)
            x += alpha * p
            r -= alpha * Hp
            new_rdotr = torch.dot(r, r)
            if new_rdotr < 1e-10:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
        return x

    def compute_surrogate_obj(self, states, actions, advantage, old_log_probs,
                              actor):  
        log_probs = torch.log(actor(states).gather(1, actions))
        ratio = torch.exp(log_probs - old_log_probs)
        return torch.mean(ratio * advantage)

    def line_search(self, states, actions, advantage, old_log_probs,
                    old_action_dists, max_vec): 
        old_para = torch.nn.utils.convert_parameters.parameters_to_vector(
            self.actor.parameters())
        old_obj = self.compute_surrogate_obj(states, actions, advantage,
                                             old_log_probs, self.actor)
        for i in range(15): 
            coef = self.alpha**i
            new_para = old_para + coef * max_vec
            new_actor = copy.deepcopy(self.actor)
            """ ------------- Programming 1: implement the linear search to find best parameter for actor (you may refer to original TRPO paper, Appendix C) ------------- """
            """ YOUR CODE HERE """

            torch.nn.utils.convert_parameters.vector_to_parameters( new_para, new_actor.parameters() )
            new_action_dists = torch.distributions.Categorical(new_actor(states))

            # KL distance for two distributions
            kl = torch.mean(torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists))

            new_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, new_actor)
            if (old_obj < new_obj) and (kl < self.kl_constraint):
                return new_para
            
            return old_para
        
        """ ------------- Programming 1 ------------- """


    def policy_learn(self, states, actions, old_action_dists, old_log_probs,
                     advantage): 
        surrogate_obj = self.compute_surrogate_obj(states, actions, advantage,
                                                   old_log_probs, self.actor)
        grads = torch.autograd.grad(surrogate_obj, self.actor.parameters())
        """ ------------- Programming 2: implement the conjugate_gradient function, the linear search function, to update actor parameter (you may refer to original TRPO paper, Section 6) ------------- """
        """ YOUR CODE HERE """

        obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        descent_direction = self.conjugate_gradient(obj_grad, states, old_action_dists)

        Hessian = self.hessian_matrix_vector_product(states, old_action_dists, descent_direction)
        epsilon = 1e-8              # to avoid zero division
        max_vec = torch.sqrt(2 * self.kl_constraint / ( torch.dot(descent_direction, Hessian)) + epsilon ) * descent_direction
        new_para = self.line_search(states, actions, advantage, old_log_probs, old_action_dists, max_vec)

        torch.nn.utils.convert_parameters.vector_to_parameters(new_para, self.actor.parameters())

        """ ------------- Programming 2 ------------- """

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        """ ------------- Programming 3: Compute GAE and update the parameter of actor and critic ------------- """
        """ YOUR CODE HERE """

        target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        delta = target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, delta.cpu()).to(self.device)

        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()
        old_action_dists = torch.distributions.Categorical(self.actor(states).detach())

        critic_loss = torch.mean(F.mse_loss(target.detach(), self.critic(states)))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        self.critic_optimizer.step()
        self.policy_learn(states, actions, old_action_dists, old_log_probs, advantage)

        """ ------------- Programming 3 ------------- """

