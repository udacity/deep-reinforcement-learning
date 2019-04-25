# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
import torch
from unityagents import UnityEnvironment

import numpy as np
from collections import deque
from agent import Agent
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device("cpu")

def test_agent(env, brain_name, agent):
    env_info = env.reset(train_mode=True)[brain_name]
    states = env_info.vector_observations
    num_agents = len(env_info.agents)
    scores = np.zeros(num_agents)
    while True:
        states = torch.FloatTensor(states).to(device)
        actions, _, _, _ = agent.act(states)

        env_info = env.step(actions.cpu().detach().numpy())[brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        scores += env_info.rewards
        states = next_states
        if np.any(dones):
            break
    return np.mean(scores)


def main():
    #global env, brain_name, env_info, num_agents, action_size, states, state_size, n_episodes, ROLLOUT_LENGTH, agent
    os.environ['NO_PROXY'] = 'localhost,127.0.0.*'
    env = UnityEnvironment(file_name='./Reacher20/reacher.exe',
                           base_port=64736, no_graphics=True)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])

    # %%
    num_agents = len(env_info.agents)
    max_t = 1e5
    n_episodes = 300
    LR = 3e-4  # learning rate
    EPSILON = 1e-5  # Adam epsilon
    state_size = env_info.vector_observations.shape[1]
    hidden_size = 512
    action_size = brain.vector_action_space_size
    agent = Agent(num_agents, state_size, action_size)
    ppo(agent, env, n_episodes)



# TO CONTINUE TRAINING:
# agent.model.load_state_dict(torch.load('ppo_checkpoint.pth'))
def ppo(agent, env, n_episodes):
    #device = "cuda"
    ROLLOUT_LENGTH = 1024


    brain_name = env.brain_names[0]
    if True:
        env.info = env.reset(train_mode=True)[brain_name]
        scores = []
        scores_window = deque(maxlen=100)

        for i_episode in tqdm(range(1, n_episodes + 1)):
            # Each iteration, N parallel actors collect T time steps of data

            # AGENT: def step(self, rollout, num_agents):
            env_info = env.reset(train_mode=True)[brain_name]
            states = env_info.vector_observations
#o            rollout = []
            episode_rewards = []
            log_probs_list = []
            values_list = []
            states_list = []
            actions_list = []
            masks = []


            for _ in range(ROLLOUT_LENGTH):
                states = torch.FloatTensor(states).to(device)
                actions, log_probs, values, _ = agent.act(states)
                env_info = env.step(actions.cpu().detach().numpy())[brain_name]
                next_states = env_info.vector_observations
                rewards = env_info.rewards
                #dones = np.array([1 if d else 0 for d in env_info.local_done])
                dones = 1*np.array(env_info.local_done)
                masks.append(torch.FloatTensor(1-dones).to(device))

                # append tuple ( s, a, p(a|s), r, dones, V(s) )
                #rollout.append([states, actions.detach(), log_probs.detach(), rewards, 1 - dones, values.detach()])
                actions_list.append(actions)
                log_probs_list.append(log_probs)
                values_list.append(values)
                states_list.append(states)
                episode_rewards.append(torch.FloatTensor(rewards).to(device))

                states = next_states
                if np.any(dones):  # exit loop if episode finished
                    break

            next_state = torch.FloatTensor(states).to(device)
            _, next_value = agent.model(next_state)

            agent.step( states=states_list, actions=actions_list, values=values_list, rewards=episode_rewards, log_probs=log_probs_list, masks=masks, next_value=next_value)




            test_mean_reward = test_agent(env, brain_name, agent)
            scores.append(test_mean_reward)
            scores_window.append(test_mean_reward)

            if np.mean(scores_window) > 30.0:
                torch.save(agent.model.state_dict(), f"ppo_checkpoint.pth")
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode,
                                                                                             np.mean(
                                                                                                 scores_window)))
                break

            print('Episode {}, Total score this episode: {}, Last {} average: {}'.format(i_episode, test_mean_reward,
                                                                                         min(i_episode, 100),
                                                                                         np.mean(scores_window)))
    # %%



# def ppo( :params: )


def plot(scores):
    score = scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(score)), score)
    plt.ylabel('Score')
    plt.xlabel('Episode Number')
    plt.show()


# PLOT THE SCORES



# %% [markdown]
# ## Run a trained agent
def play():
    global agent, score
    agent = Agent(num_agents, state_size, action_size)
    agent.model.load_state_dict(torch.load('ppo_checkpoint.pth'))
    score = test_agent(env, brain_name)
    print(score)


if __name__=="__main__":
    main()

