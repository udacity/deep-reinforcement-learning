import numpy as np
import pandas as pd
from collections import deque
import time
import os


def train(env, brain_name, agent, n_episodes, max_t=1000, solved=30, *args, **kwargs):
    """
    Train loop for a continous action space agent
    Params
    ======
        env: Unity Continous Control environmnet
        brain_name:
        agent: Agent to solve the environment, like AgentDDPG
        n_episodes (int): Number of episodes
        max_t (int): max number of time steps on each episode
        solved (float): Environmnet is consdiered solved when the 100 episode moving average
        of the mean (across environments) is above this threshold
        kwargs:
            add_noise (bool): Whether to add noise to actions in order to allow exploration
            action_scaler_fn: Scaling/Clipping function to apply to actions
            noise_decay (float): Noise decaying factor, as noise_raw*noise_decay
            min_noise_weight (float): Minimum noise weight after decaying
            model_save_path (str): Path where models are stored every 500 episodes
    Returns
    =======
    a pandas DataFrame with resulting scores per episode
    """
    action_scaler_fn = kwargs.get('action_scaler_fn', lambda x: x)
    add_noise = kwargs.get('add_noise', True)
    noise_decay = kwargs.get('noise_decay', None)
    model_save_path = kwargs.get('model_save_path', None)
    min_noise_weight = kwargs.get('min_noise_weight', 0.1)

    noise_weight = 1.

    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)

    scores_hist = np.zeros((0, num_agents))
    scores_window = deque(maxlen=100)

    for i_episode in range(1, n_episodes + 1):

        # Reset the enviroment
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        states = env_info.vector_observations  # get the current state (for each agent)
        scores = np.zeros(num_agents)  # initialize the score (for each agent)
        agent.reset()
        start_time = time.time()
        for i in range(max_t + 1):
            # Predict the best action for the current state.
            actions = agent.act(states, add_noise=add_noise, noise_weight=noise_weight)
            actions = action_scaler_fn(actions)
            #
            env_info = env.step(actions)[brain_name]  # send all actions to tne environment
            next_states = env_info.vector_observations  # get next state (for each agent)
            rewards = env_info.rewards  # get reward (for each agent)
            dones = env_info.local_done  # see if episode finished
            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                agent.step(state, action, reward, next_state, done, i)
            scores += rewards  # update the score (for each agent)
            states = next_states  # roll over states to next time step
            #

            if np.any(dones):  # exit loop if episode finished
                scores_hist = np.vstack([scores_hist, scores])
                scores_window.append(scores)
                break
        duration = time.time() - start_time
        print('\rEpisode {}\t{}s\tAverage Score: {:.2f}'.format(i_episode, round(duration), np.mean(scores_window)),
              end="")
        # schedule exploration-explotation
        if noise_decay is not None:
            noise_weight = max(noise_decay ** i_episode, min_noise_weight)
        avg_scores_window = np.mean(scores_window)
        # report
        if (i_episode) % 100 == 0:
            print('\rEpisode [{}/{}]\tAverage score: {:,.2f}'.format(i_episode, n_episodes, avg_scores_window))
            if (i_episode) % 500 == 0:
                if hasattr(agent, "save_network") and model_save_path is not None:
                    model_checkpoint_path = os.path.join(model_save_path, 'checkpoint')
                    agent.save_network(model_checkpoint_path)
                    pd.DataFrame(scores_hist).to_csv(os.path.join(model_checkpoint_path, 'scores.csv'))

        if avg_scores_window >= solved:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         avg_scores_window))
            if hasattr(agent, "save_network") and model_save_path is not None:
                agent.save_network(model_save_path)
            break

        res = pd.DataFrame(scores_hist)
        res.index.name = 'idx_episode'
        res['avg_score'] = res.mean(axis=1)
        res['avg_score_mave100'] = res['avg_score'].rolling(100).mean()

    return res