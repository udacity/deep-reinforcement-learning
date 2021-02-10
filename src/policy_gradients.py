from collections import deque
import numpy as np
import pandas as pd

def train(env, agent, n_episodes=2000, max_t=1000,
        eps_start=1.0, eps_end=0.01, eps_decay=0.995,
        model_save_path='checkpoint.pth', score_solved=200.):
    """
    Train DQ-Learning agent on a given environment, based on epsilon-greedy policy and GLIE evolution of epsilon parameter
    When the game is considered solved, save the  DQ-Net underlaying the agent in a given path
    This train loop is developed to work with OpenAI Gym's LunarLander-v2 environment (dqn/rainbow/Deep_Q_Network_DQN.ipynb),
    other environments may not work
    Solving is supposed to happen at a moving average window of 100 episodes
    Params
    ======
        env: Environment to solve an episodic game. Should behave like:
            state = env.reset()
            next_state, reward, done, _ = env.step(action)
        agent: DQ-Learning Agent, should estimate and optimal policy estimating Q function using a DQN
        n_episodes (int): Number of episodes to simulate
        max_t (int): Max number of time steps (transitions) on each episode
        eps_start (float): Epsilon parameter starting value (at first episode)
        eps_end (float): Epsilon min value
        eps_decay (float): Epsilon decay rate
        model_save_path (str): Path to persist model
        score_solved (float): Score to consider the game solved
    Returns
    ======
    scores (list): Average reward over 100 episodes
    """

    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon
    for i_episode in range(1, n_episodes + 1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = update_epsilon(eps_end, eps_decay, eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= score_solved:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            agent.save_network(model_save_path)
            break

    return pd.Series(index=range(1, len(scores) +1), data=scores, dtype=np.float32, name='score')

def update_epsilon(eps_end, eps_decay, eps_curr):
    """

    """
    return max(eps_end, eps_decay * eps_curr)