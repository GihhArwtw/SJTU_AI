import util
from abstractAgent import Agent

class ValueIterationAgent(Agent):
    """An agent that takes a Markov decision process on initialization
    and runs value iteration for a given number of iterations.

    Hint: Test your code with commands like `python main.py -a value -i 100 -k 10`.
    """
    def __init__(self, mdp, discount = 0.9, epsilon=0.001, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.epsilon = epsilon  # For examing the convergence of value iteration
        self.iterations = iterations # The value iteration will run AT MOST these steps
        self.values = util.Counter() # You need to keep the record of all state values here
        self.runValueIteration()

    def runValueIteration(self):
        """ YOUR CODE HERE """
        import logging
        logging.basicConfig(level=logging.INFO, filename="./valueIteration.log", filemode="w", format="%(message)s")
        info = ""
        for s in self.mdp.getStates():
            info = info + str(s) + "\t"
        logging.info(info)

        for iter in range(self.iterations):
            new_values = util.Counter()
            for s in self.mdp.getStates():      # current state
                if self.mdp.getPossibleActions(s):
                    new_values[s] = -float('inf')
                    for a in self.mdp.getPossibleActions(s):
                        new_value = 0
                        for next_state, prob in self.mdp.getTransitionStatesAndProbs(s, a):
                            # [(next state, probability), ...]
                            new_value += prob * (self.mdp.getReward(s, a, next_state)
                                                 + self.discount * self.values[next_state])
                        if new_value > new_values[s]:
                            new_values[s] = new_value
                else:
                    new_values[s] = 0
            
            # Only used to compare the speed of convergence
            info = ""
            for s in self.mdp.getStates():
                info = info + str(self.values[s]) + "\t"
            logging.info(info)

            # Check if converge
            diff = max([abs(new_values[s] - self.values[s]) for s in self.mdp.getStates()])
            self.values = new_values
            if (diff<self.epsilon):             # already converge.
                print("[INFO] Converge at iteration %d" % iter)
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
        max_value = float('-inf')
        if self.mdp.isTerminal(state):
            return None
        
        for a in self.mdp.getPossibleActions(state):
            if self.computeQValueFromValues(state, a) > max_value:
                max_value = self.computeQValueFromValues(state, a)
                bestaction = a
            
        return bestaction

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
