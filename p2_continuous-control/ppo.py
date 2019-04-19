import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from IPython.display import clear_output
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
from collections import deque
import os
from tqdm import tqdm

os.environ['NO_PROXY'] = 'localhost,127.0.0.*'


def plot(frame_idx, rewards):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=1.141)
        nn.init.constant_(m.bias, 0.1)

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
        self.apply(init_weights)
        
    def forward(self, x):
        value = self.critic(x)
        mu    = self.actor(x)
        log_std = torch.exp(self.log_std).squeeze(0).expand_as(mu)
        #std   = self.log_std.exp().squeeze(0).expand_as(mu)
        dist  = Normal(mu, log_std)
        return dist, value


#def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
#    batch_size = states.size(0)
#    for _ in range(batch_size // mini_batch_size):
#        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
#        yield states[rand_ids], actions[rand_ids], log_probs[rand_ids], returns[rand_ids], advantage[rand_ids]
        

def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]        

def ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, model, optimizer, clip_param=0.2):
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            dist, value = model(state)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

#def ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, model, optimizer, clip_param=0.2):
#    for _ in range(ppo_epochs):
#        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
#            #print(type(state), state.dtype, state.shape)
#            dist, value = model(state)
#            entropy = dist.entropy().mean()
#            new_log_probs = dist.log_prob(action)

#            ratio = (new_log_probs - old_log_probs).exp().unsqueeze(0)
#            surr1 = ratio * advantage
#            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

#            actor_loss  = - torch.min(surr1, surr2).mean()
#            critic_loss = (return_ - value).pow(2).mean()

#            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy
    
#            optimizer.zero_grad()
#            loss.backward()
#            optimizer.step()



def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, (gae + values[step]).squeeze(0))
    return returns

def compute_gaes(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns



def normalize(x, mean=0., std=1., epsilon=1e-8):
    x = (x - np.mean(x)) / (np.std(x) + epsilon)
    x = x * std + mean

    return x




from unityagents import UnityEnvironment
import numpy as np

def test_agent(env, brain_name, model, device):
    env_info = env.reset(train_mode = True)[brain_name]
    state = env_info.vector_observations[0]
    scores = 0.0
    while True:
        state = torch.FloatTensor(state).to(device)
        dist, _ = model(state)
        action_t = dist.sample()
        action_np = action_t.cpu().data.numpy()

        env_info = env.step(action_np)[brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        scores += reward
        state = next_state
        if done:
            break
    return scores



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



    max_frames = 250#1e5
    frame_idx  = 0
    test_rewards = []


    env = UnityEnvironment(file_name='reacher/reacher', base_port=64739)
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
    for frame_idx in tqdm(range(max_frames)):
        log_probs = []
        values    = []
        states    = []
        actions   = []
        rewards   = []
        masks     = []
        model.train()
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        epoch_scores = 0.0
        for duration in range(num_steps):
            
            state = torch.FloatTensor(state).to(device)
            dist, value = model(state)

            action_t = dist.sample()
            action_np = action_t.cpu().data.numpy()
            env_info = env.step(action_np)[brain_name]           # send all actions to tne environment
            next_state = env_info.vector_observations        # get next state (for each agent)
            reward = env_info.rewards                        # get reward (for each agent)
            dones = np.array(env_info.local_done)                        # see if episode finished
            if reward == None:
                pass

            log_prob = dist.log_prob(action_t)
            
            log_probs.append(log_prob)
            values.append(value)

            reward_t = torch.FloatTensor(reward)
            rewards.append(reward_t.unsqueeze(1).to(device))
            masks_t = torch.FloatTensor(1 - dones)
            masks.append(masks_t.unsqueeze(1).to(device))
            states.append(state)
            actions.append(action_t)
            
            state = next_state

            if np.any(dones):
                break

        next_state = torch.FloatTensor(state).to(device)
        _, next_value = model(next_state)
#        returns = compute_gae(next_value, rewards, masks, values)
        returns = compute_gaes(next_value, rewards, masks, values)


        returns   = torch.cat(returns).detach()
        log_probs = torch.cat(log_probs).detach()
        values    = torch.cat(values).detach()
        states    = torch.cat(states)
        actions   = torch.cat(actions)
        advantage = returns - values

        #returns2   = torch.stack(returns).detach().unsqueeze(1)
        #log_probs2 = torch.stack(log_probs).detach()
        #values2    = torch.stack(values).detach()
        #print(type(states), len(states))#, states.dtype, states.shape)
        #states2    = torch.stack(states)
        #actions2   = torch.stack(actions)
        #advantage2 = returns - values

        print("ppo_update:", len(states))
        ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage, model, optimizer)

        model.eval()
        test_mean_reward = test_agent(env, brain_name, model, device)
        test_rewards.append(test_mean_reward)
        scores_window.append(test_mean_reward)
        mean_score = np.mean(scores_window)
        print("Mean Score: ", mean_score, "Frame: ", frame_idx)
        frame_idx += 1

    #%%
    env.close()

if __name__ == "__main__":
    main()