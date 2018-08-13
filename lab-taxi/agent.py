import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.eps = 1.0
        self.alpha = 0.1
        self.N = 1

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if np.random.rand() > self.eps:
            return np.argmax(self.Q[state])
        return np.random.choice(self.nA)

    def step(self, state, action, reward, next_state=None, done=False):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.eps = min(0.0005, 1.0/self.N)
        self.N += 1
        if state in self.Q: 
            Qsa = self.Q[state][action]
        else: 
            Qsa = 0.0
            
        policy_s = np.ones(self.nA) / self.nA * self.eps

        if next_state in self.Q:
            policy_s[np.argmax(self.Q[next_state])] = 1 - self.eps + (self.eps / self.nA)
            EQsa_next = np.dot(self.Q[next_state], policy_s)
        else:
            EQsa_next = 0.0
        
        self.Q[state][action] = Qsa + self.alpha * (reward + EQsa_next - Qsa)
        
