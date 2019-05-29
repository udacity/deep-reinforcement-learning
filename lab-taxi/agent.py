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
        self.epsilon = .8
        self.gamma = .99
        self.alpha = .7
    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        chance = np.random.rand()
        if self.epsilon < chance:
            action = np.argmax(self.Q[state])
        else:
            action = np.random.randint(low=0,high=6)
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
#        print (state, action, next_state)
        action_next = np.argmax(self.Q[next_state])
        self.Q[state][action] += self.alpha*(reward + self.gamma*self.Q[next_state][action_next] - self.Q[state][action])

    def update_epsilon(self):
        self.epsilon *= .995
        return print(self.epsilon)