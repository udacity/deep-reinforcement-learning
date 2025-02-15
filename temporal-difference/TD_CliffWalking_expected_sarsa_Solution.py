import sys
import gym
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

import check_test
from plot_utils import plot_values

env = gym.make('CliffWalking-v0')

def expected_sarsa(env, num_episodes, alpha, gamma=0.95, epsilon_start=1.0):
    # Initialize epsilon and related parameters
    epsilon = epsilon_start
    epsilon_min = 0.1
    epsilon_decay = 0.995

    nA = 4  # number of actions

    # Initialize the action-value function (empty dictionary of arrays)
    Q = defaultdict(lambda: np.zeros(env.nA))

    # Loop over episodes
    for i_episode in range(1, num_episodes + 1):
        if i_episode % 100 == 0:
            print(f"Episode {i_episode}: Epsilon = {epsilon:.4f}")

        # Decay epsilon for exploration-exploitation balance
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        # Generate episode
        state, _ = env.reset()
        action = np.random.choice(np.arange(nA), p=get_probs(Q[state], epsilon, nA))  # epsilon-greedy policy

        while True:
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Initialize Q-values for the next state if not already initialized
            if next_state not in Q:
                Q[next_state] = np.zeros(nA)

            # Compute the expected Q-value for the next state
            expected_Q = np.dot(get_probs(Q[next_state], epsilon, nA), Q[next_state])

            # SARSA update
            Q[state][action] += alpha * (reward + gamma * expected_Q - Q[state][action])

            # If the episode ends, break out of the loop
            if terminated or truncated:
                break

            # Transition to the next state and choose the next action using epsilon-greedy
            state = next_state
            action = np.random.choice(np.arange(nA), p=get_probs(Q[state], epsilon, nA))

    return Q

def get_probs(Q_s, epsilon, nA):
    """ Obtains the action probabilities corresponding to an epsilon-greedy policy """
    policy_states = np.ones(nA) * epsilon / nA
    best_action = np.argmax(Q_s)
    policy_states[best_action] = 1 - epsilon + (epsilon / nA)
    return policy_states


# Obtain the estimated optimal policy and corresponding action-value function using Expected SARSA
Q_expected_sarsa = expected_sarsa(env, 50000, .01)

# Print the estimated optimal policy
policy_expected_sarsa = np.array([np.argmax(Q_expected_sarsa[key]) if key in Q_expected_sarsa else -1 for key in np.arange(48)]).reshape(4, 12)
check_test.run_check('td_control_check', policy_expected_sarsa)
print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
print(policy_expected_sarsa)

# Plot the estimated optimal state-value function
V_expected_sarsa = [np.max(Q_expected_sarsa[key]) if key in Q_expected_sarsa else 0 for key in np.arange(48)]
plot_values(V_expected_sarsa)
