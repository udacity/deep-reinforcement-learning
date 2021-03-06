import numpy as np
import pandas as pd
import os
import time

from collections import deque

def train(env, agent, n_episodes, max_t=300, solved=30, *args, **kwargs):
    """
    lqt: liquidation time
    n_trades: number of trades
    tr: risk aversion
    """
    action_scaler_fn = kwargs.get('action_scaler_fn', lambda x: x)
    noise_decay = kwargs.get('noise_decay', None)
    model_save_path = kwargs.get('model_save_path', None)
    min_noise_weight = kwargs.get('min_noise_weight', 0.1)
    
    noise_weight = 1.
     
    #num_agents = len(env_info.agents)
    num_agents = 1
    
    scores_hist = np.zeros((0, num_agents))
    scores_window = deque(maxlen=100)
    
    for i_episode in range(1, n_episodes+1): 
        
        # Reset the enviroment
        states = env.reset()      # reset the environment and get the current state (for each agent)
        agent.reset()
        scores = np.zeros(num_agents)                          # initialize the score (for each agent)
        start_time = time.time()
        for i in range(max_t + 1):
            # Predict the best action for the current state. 
            actions = agent.act(states, add_noise = True, noise_weight = noise_weight)
            actions = action_scaler_fn(actions)
            # send all actions to tne environment and get next state (for each agent)actions = action_scaler_fn(actions)
            next_states, rewards, dones, _ = env.step(actions)          
            # current state, action, reward, new state are stored in the experience replay
            agent.step(states, actions, rewards, next_states, dones, i)
            scores += rewards  # update the score (for each agent)                
            states = next_states  # roll over new state
            if np.any(dones):  # exit loop if episode finished
                scores_hist = np.vstack([scores_hist, scores])
                scores_window.append(scores)
                break
        duration = time.time() - start_time
        print('\rEpisode {}\t{}s\tAverage Score: {:.2f}'.format(i_episode, round(duration), np.mean(scores_window)), end="")       
        # schedule exploration-explotation
        if noise_decay is not None:
            noise_weight = max(noise_decay**i_episode, min_noise_weight)
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
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, avg_scores_window))
            if hasattr(agent, "save_network") and model_save_path is not None:
                agent.save_network(model_save_path)
            break
        
        res = pd.DataFrame(scores_hist, columns=['score'])
        res.index.name = 'idx_episode'
        res.rename(inplace=True, columns={0: 'score'})
        
        res['avg_score'] = res.mean(axis=1)
        res['avg_score_mave100'] = res['avg_score'].rolling(100).mean()
        
    return res