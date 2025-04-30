import util
from abstractAgent import Agent

class PolicyIterationAgent(Agent):
    """An agent that takes a Markov decision process on initialization
    and runs policy iteration for a given number of iterations.

    Hint: Test your code with commands like `python main.py -a policy -i 100 -k 10`.
    """

    def __init__(self, mdp, discount = 0.9, epsilon=0.001, iterations = 100):
        self.mdp = mdp
        self.discount = discount
        self.epsilon = epsilon  # For examing the convergence of policy iteration
        self.iterations = iterations # The policy iteration will run AT MOST these steps
        self.values = util.Counter() # You need to keep the record of all state values here
        self.policy = dict()
        self.runPolicyIteration()

    def runPolicyIteration(self):
        """ YOUR CODE HERE """
        import logging
        logging.basicConfig(level=logging.INFO, filename="./policyIteration.log", filemode="w", format="%(message)s")
        info = ""
        for s in self.mdp.getStates():
            info = info + str(s) + "\t"
        logging.info(info)

        # Initialize the policy
        for s in self.mdp.getStates():
            self.policy[s] = self.mdp.getPossibleActions(s)[0] if self.mdp.getPossibleActions(s) else None
        
        for iter in range(self.iterations):
            # Policy Estimation
            while (True):
                new_values = util.Counter()
                for s in self.mdp.getStates():
                    action = self.policy[s]
                    if action:
                        for next_state, prob in self.mdp.getTransitionStatesAndProbs(s, action):
                            new_values[s] += prob * (self.mdp.getReward(s, action, next_state)
                                                    + self.discount * self.values[next_state])
                diff = max([abs(new_values[s] - self.values[s]) for s in self.mdp.getStates()])
                self.values = new_values
                if (diff<self.epsilon):             # already converge.
                    break
                
            # Policy Improvement
            policy_bkup = self.policy.copy()
            for s in self.mdp.getStates():
                max_value = -float('inf')
                if self.mdp.getPossibleActions(s):
                    for a in self.mdp.getPossibleActions(s):
                        for next_state, prob in self.mdp.getTransitionStatesAndProbs(s, a):
                            if (prob * (self.mdp.getReward(s, a, next_state) + 
                                        self.discount * new_values[next_state]) > max_value):
                                max_value = prob * (self.mdp.getReward(s, a, next_state) + 
                                                    self.discount * new_values[next_state])
                                self.policy[s] = a
                else:
                    self.policy[s] = None
                    
            # Only used to compare the speed of convergence
            info = ""
            for s in self.mdp.getStates():
                info = info + str(self.values[s]) + "\t"
            logging.info(info)
                
            # Check if the policy has converged
            diff = sum([policy_bkup[s] != self.policy[s] for s in self.mdp.getStates()])
            if (diff<1):
                print("[INFO] Policy converged at iteration %d." % iter)
                break

        if (diff>=self.epsilon):
            print("[INFO] Not converged yet.")


    def getValue(self, state):
        """Return the value of the state (computed in __init__)."""
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """Compute the Q-value of action in state from the value function stored in self.values."""

        value = None

        """ YOUR CODE HERE """
        
        value = 0
        for next_state, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            value += prob * (self.mdp.getReward(state, action, next_state)
                             + self.discount * self.values[next_state])

        return value

    def computeActionFromValues(self, state):
        """The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """

        bestaction = None

        """ YOUR CODE HERE """
        bestaction = self.policy[state]

        return bestaction

    def getPolicy(self, state):
        return self.policy[state]

    def getAction(self, state):
        return self.policy[state]

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)