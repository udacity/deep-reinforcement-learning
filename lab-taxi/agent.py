import numpy as np
from collections import defaultdict

# Attempting to implement Q-learning for our agent
# This is an offline learning task? Env is static and batched
# 9.35 is the best result I can get

class Agent:

    def __init__(self, nA=6, gamma=0.9, alpha=0.02, eps_start=0.001, eps_decay=0.9999, eps_min=0.0001):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.gamma = gamma
        self.eps = eps_start
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.alpha = alpha

    def eps_greedy_action(self, state):
        ''' Given a state choose action in epsilon greedy manner'''
        # Select greedy action wrt Q
        if np.random.random() > self.eps:
            action = np.argmax(self.Q[state])
        # Select unirandom action
        else:
            action = np.random.choice(self.nA)
        return action

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        # Action selection using epsilon-greedy policy
        action = self.eps_greedy_action(state)
        self.eps = max(self.eps, self.eps_min)
        return action

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        # Update Q table based on sarsamax off-policy updates
        target = np.max(self.Q[next_state]) if next_state is not None else 0
        self.Q[state][action] = (1-self.alpha) * self.Q[state][action] \
                            + self.alpha * (reward + self.gamma * target)
