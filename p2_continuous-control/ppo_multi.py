import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
#from IPython.display import clear_output
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
from collections import deque
import os
from tqdm import tqdm

from unityagents import UnityEnvironment
import numpy as np


os.environ['NO_PROXY'] = 'localhost,127.0.0.*'


def plot(frame_idx, rewards, figure = None):
    #clear_output(True)
    if figure is None:
        figure = plt.figure(figsize=(20,5))
    figure.subplot(131)
    figure.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    figure.plot(rewards)
    figure.show()
    return figure

class ActorCriticPolicy(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=1.141):
        super(ActorCriticPolicy, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_outputs),
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        #self.apply(init_weights)
        
    def forward(self, x):
        value = self.critic(x)
        mu    = self.actor(x)
        log_std = torch.exp(self.log_std).squeeze(0).expand_as(mu)
        #std   = self.log_std.exp().squeeze(0).expand_as(mu)
        dist  = Normal(mu, log_std)
        return dist, value


def test_agent(env, brain_name):
    env_info = env.reset(train_mode = True)[brain_name]
    states = env_info.vector_observations
    scores = np.zeros(num_agents)
    while True:
        actions, _, _= agent.act(states)
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
    use_cuda = torch.cuda.is_available()
#    device   = torch.device("cuda" if use_cuda else "cpu")
    device   = torch.device("cpu")
    print(device)
    scores_window = deque(maxlen=100)

    #Hyper params:
    hidden_size      = 512
    lr               = 3e-4
    num_steps        = 2048
    mini_batch_size  = 32
    ppo_epochs       = 10
    threshold_reward = 10

    epochs = 250#1e5
    current_epoch  = 0
    test_rewards = []


    env = UnityEnvironment(file_name='p2_continuous-control/reacher/reacher', base_port=64739)
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    action_size = brain.vector_action_space_size
    states = env_info.vector_observations
    state_size = states.shape[1]

    num_inputs  = state_size
    num_outputs = action_size
    
    model = ActorCriticPolicy(num_inputs, num_outputs, hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)#,eps= 1e-5)

    

#    while frame_idx < max_frames and not early_stop:
    for current_epoch in tqdm(range(epochs)):
        log_probs = []
        values    = []
        states    = []
        actions   = []
        rewards   = []
        masks     = []
        model.train()
        env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        epoch_scores = 0.0
        for duration in range(num_steps):
            
            state = torch.FloatTensor(state).to(device)
            dist, value = model(state)

            action_t = dist.sample()
            action_np = action_t.cpu().data.numpy()
            env_info = env.step(action_np)[brain_name]           # send all actions to tne environment
            next_state = env_info.vector_observations[0]        # get next state (for each agent)
            reward = env_info.rewards[0]                        # get reward (for each agent)
            done = env_info.local_done[0]                        # see if episode finished
            if reward == None:
                pass

            log_prob = dist.log_prob(action_t)
            
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            epoch_scores += reward
            masks.append(1 - done)#.unsqueeze(1).to(device))

            states.append(state)
            actions.append(action_t)
            
            state = next_state

            #mean_score=np.mean(scores_window)
            #if mean_score > threshold_reward: break

            if done:
                #print("Epoch: ", frame_idx, " length:", duration)
                break

        print("Lastscore: ", epoch_scores)
        next_state = torch.FloatTensor(state).to(device)
        _, next_value = model(next_state)
        returns = compute_gae(next_value, rewards, masks, values)

        returns   = torch.stack(returns).detach().unsqueeze(1)
        log_probs = torch.stack(log_probs).detach()
        values    = torch.stack(values).detach()
        #print(type(states), len(states))#, states.dtype, states.shape)
        states    = torch.stack(states)
        actions   = torch.stack(actions)
        advantage = returns - values

        print("ppo_update:", len(states))
        ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage, model, optimizer)

        model.eval()
        test_mean_reward = test_agent(env, brain_name, model, device)
        test_rewards.append(test_mean_reward)
        scores_window.append(test_mean_reward)
        mean_score = np.mean(scores_window)
        print("Mean Score: ", mean_score, "Frame: ", current_epoch)
        current_epoch += 1

    #%%
    env.close()

if __name__ == "__main__":
    main()