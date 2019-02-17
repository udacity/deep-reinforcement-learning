import random
import sys
import gym
import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt


import check_test
from plot_utils import plot_values

def epsilon_greedy_action(Q, state, nA, epsilon):
    if random.random() > epsilon: # select greedy action with probability epsilon
        return np.argmax(Q[state])
    else:                     # otherwise, select an action randomly
        return random.choice(np.arange(nA))

def update_Q_sarsa(Qsa, Qsa_next, reward, alpha, gamma):
    return Qsa + (alpha * (reward + (gamma * Qsa_next) - Qsa))


def sarsa(env, num_episodes, alpha, gamma=1.0):
    # initialize action-value function (empty dictionary of arrays)
    Q = defaultdict(lambda: np.zeros(env.nA))
    # initialize performance monitor
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()   
        
        ## TODO: complete the function
        #epsilon = 1.0 / i_episode
        epsilon = 0.01
        state = env.reset()
        action = epsilon_greedy_action(Q, state, env.nA, epsilon)
        t_step=0
        while True:
            t_step +=1
            next_state, reward, done, info = env.step(action)
            if not done:
                next_action = epsilon_greedy_action(Q, next_state, env.nA, epsilon)
                Q[state][action] = update_Q_sarsa(Q[state][action], Q[next_state][next_action], reward, alpha, gamma)
                state = next_state
                action = next_action
            else:
                Q[state][action] = update_Q_sarsa(Q[state][action], 0, reward, alpha, gamma)
                #print("T_steps:",t_step)
                break
    return Q

def evaluate_sarsa():
    env = gym.make('CliffWalking-v0')
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

def plot_optimal_policy():
    # define the optimal state-value function
    V_opt = np.zeros((4,12))
    V_opt[0:13][0] = -np.arange(3, 15)[::-1]
    V_opt[0:13][1] = -np.arange(3, 15)[::-1] + 1
    V_opt[0:13][2] = -np.arange(3, 15)[::-1] + 2
    V_opt[3][0] = -13

    plot_values(V_opt)

#lehrbuchlösung

def update_Q(Qsa, Qsa_next, reward, alpha, gamma):
    """ updates the action-value function estimate using the most recent time step """
    return Qsa + (alpha * (reward + (gamma * Qsa_next) - Qsa))

def epsilon_greedy_probs(env, Q_s, i_episode, eps=None):
    """ obtains the action probabilities corresponding to epsilon-greedy policy """
    epsilon = 1.0 / i_episode
    if eps is not None:
        epsilon = eps
    policy_s = np.ones(env.nA) * epsilon / env.nA
    policy_s[np.argmax(Q_s)] = 1 - epsilon + (epsilon / env.nA)
    return policy_s

def sarsa_2(env, num_episodes, alpha, gamma=1.0):
    # initialize action-value function (empty dictionary of arrays)
    Q = defaultdict(lambda: np.zeros(env.nA))
    # initialize performance monitor
    plot_every = 100
    tmp_scores = deque(maxlen=plot_every)
    scores = deque(maxlen=num_episodes)
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()   
        # initialize score
        score = 0
        # begin an episode, observe S
        state = env.reset()   
        # get epsilon-greedy action probabilities
        policy_s = epsilon_greedy_probs(env, Q[state], i_episode)
        # pick action A
        action = np.random.choice(np.arange(env.nA), p=policy_s)
        # limit number of time steps per episode
        for t_step in np.arange(300):
            # take action A, observe R, S'
            next_state, reward, done, info = env.step(action)
            # add reward to score
            score += reward
            if not done:
                # get epsilon-greedy action probabilities
                policy_s = epsilon_greedy_probs(env, Q[next_state], i_episode)
                # pick next action A'
                next_action = np.random.choice(np.arange(env.nA), p=policy_s)
                # update TD estimate of Q
                Q[state][action] = update_Q(Q[state][action], Q[next_state][next_action], 
                                            reward, alpha, gamma)
                # S <- S'
                state = next_state
                # A <- A'
                action = next_action
            if done:
                # update TD estimate of Q
                Q[state][action] = update_Q(Q[state][action], 0, reward, alpha, gamma)
                # append score
                tmp_scores.append(score)
                break
        if (i_episode % plot_every == 0):
            scores.append(np.mean(tmp_scores))
    # plot performance
    plt.plot(np.linspace(0,num_episodes,len(scores),endpoint=False),np.asarray(scores))
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)
    plt.show()
    # print best 100-episode performance
    print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(scores))
    return Q

evaluate_sarsa()