import sys
import gym
import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt

import check_test
from plot_utils import plot_values

env = gym.make('CliffWalking-v0')

def sarsa(env, num_episodes, alpha, gamma=1.0, epsilon_start=0.5):
    # decide epsilon
    epsilon = epsilon_start
    epsilon_min = 0.1
    epsilon_decay = 0.99

    # initialize action-value function (empty dictionary of arrays)
    Q = defaultdict(lambda: np.zeros(env.nA))
    # initialize performance monitor
    # loop over episodes
    for i_episode in range(1, num_episodes + 1):

        # monitor progress
        if i_episode % 1 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # set the value of epsilon
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        # generate episode
        episode = generate_episode(env=env, Q=Q, epsilon=epsilon, nA=4)

        Q = update_q(episode, Q, alpha, gamma)
    return Q


def generate_episode(env, Q, epsilon, nA):
    episode = []
    state, _ = env.reset()

    action = np.random.choice(np.arange(nA), p=get_probs(Q[state], epsilon, nA)) if state not in Q else np.argmax(
        Q[state]
    )

    while True:
        next_state, reward, terminated, truncated, _ = env.step(action)  # ✅ New API
        if next_state not in Q:
            Q[next_state] = np.zeros(nA)  # ✅ Ensure Q-value initialization

        next_action = np.random.choice(np.arange(nA), p=get_probs(Q[next_state], epsilon, nA))

        episode.append((state, action, reward))

        if terminated or truncated:
            break

        state = next_state  # ✅ Track next state
        action = next_action  # ✅ Track next action

    return episode

def get_probs(Q_s, epsilon, nA):
    """ obtains the action probabilities corresponding to epsilon-greedy policy """
    policy_states = np.ones(nA) * epsilon / nA
    best_action = np.argmax(Q_s)
    policy_states[best_action] = 1 - epsilon + (epsilon / nA)
    return policy_states

def pick_action(epsilon, Q, next_state):
    if np.random.rand() < epsilon:
        next_action = env.action_space.sample()  # Explore (random action)
    else:
        next_action = np.argmax(Q[next_state])  # Exploit (best action)

    return next_action

def update_q(episode, Q, alpha, gamma):
    """ updates the action-value function estimate using the most recent episode """
    states, actions, rewards = zip(*episode)
    # prepare for discounting
    for i in range(len(states) - 1):  # Ignore last step
        state, action = states[i], actions[i]
        next_state, next_action = states[i + 1], actions[i + 1]  # ✅ Use episode step

        old_Q = Q[state][action]
        next_Q = Q[next_state][next_action]  # ✅ Correct SARSA update
        Q[state][action] = old_Q + alpha * (rewards[i] + gamma * next_Q - old_Q)
    return Q


# obtain the estimated optimal policy and corresponding action-value function
Q_sarsa = sarsa(env, 5000, .01)

# print the estimated optimal policy
policy_sarsa = np.array([np.argmax(Q_sarsa[key]) if key in Q_sarsa else -1 for key in np.arange(48)]).reshape(4,12)
check_test.run_check('td_control_check', policy_sarsa)
print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
print(policy_sarsa)

# plot the estimated optimal state-value function
V_sarsa = ([np.max(Q_sarsa[key]) if key in Q_sarsa else 0 for key in np.arange(48)])
plot_values(V_sarsa)

