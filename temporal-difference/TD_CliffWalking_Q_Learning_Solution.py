from collections import defaultdict

import gym
import numpy as np

import check_test
from plot_utils import plot_values

env = gym.make('CliffWalking-v0')

def q_learning(env, num_episodes, alpha, gamma=1.0, epsilon_start=1.0):
    # decide epsilon
    epsilon = epsilon_start
    epsilon_min = 0.1
    epsilon_decay = 0.997

    nA = 4

    # initialize action-value function (empty dictionary of arrays)
    Q = defaultdict(lambda: np.zeros(env.nA))
    # initialize performance monitor
    # loop over episodes
    for i_episode in range(1, num_episodes + 1):

        # monitor progress
        # if i_episode % 1 == 0:
        #     print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
        #
        #     sys.stdout.flush()

        if i_episode % 100 == 0:
            print(f"Episode {i_episode}: Epsilon = {epsilon:.4f}")

        # set the value of epsilon
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        # generate episode
        state, _ = env.reset()

        while True:
            action = np.random.choice(np.arange(nA), p=get_probs(Q[state], epsilon, nA))

            next_state, reward, terminated, truncated, _ = env.step(action)  # ✅ New API
            if next_state not in Q:
                Q[next_state] = np.zeros(nA)  # ✅ Ensure Q-value initialization

            if terminated or truncated:
                break

            next_Q = 0 if terminated else np.max(Q[next_state])
            Q[state][action] += alpha * (reward + gamma * next_Q - Q[state][action])

            state = next_state
    return Q

def get_probs(Q_s, epsilon, nA):
    """ obtains the action probabilities corresponding to epsilon-greedy policy """
    policy_states = np.ones(nA) * epsilon / nA
    best_action = np.argmax(Q_s)
    policy_states[best_action] = 1 - epsilon + (epsilon / nA)
    return policy_states


# obtain the estimated optimal policy and corresponding action-value function
Q_q_learning = q_learning(env, 5000, .01)

# print the estimated optimal policy
policy_q_learning = np.array([np.argmax(Q_q_learning[key]) if key in Q_q_learning else -1 for key in np.arange(48)]).reshape(4,12)
check_test.run_check('td_control_check', policy_q_learning)
print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
print(policy_q_learning)

# plot the estimated optimal state-value function
V_q_learning = ([np.max(Q_q_learning[key]) if key in Q_q_learning else 0 for key in np.arange(48)])
plot_values(V_q_learning)