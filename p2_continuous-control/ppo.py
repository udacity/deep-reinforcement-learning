import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from IPython.display import clear_output
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
from collections import deque

def plot(frame_idx, rewards):
    clear_output(True)
    plt.figure(figsize=(20,5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)

class ActorCriticPolicy(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCriticPolicy, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        self.apply(init_weights)
        
    def forward(self, x):
        value = self.critic(x)
        mu    = self.actor(x)
#        log_std = torch.exp(self.log_std).unsqueeze(0).expand_as(mu)
        std   = self.log_std.exp().squeeze(0).expand_as(mu)
        dist  = Normal(mu, std)
        return dist, value


def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
#        print(states.shape)
#        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]
        yield states[rand_ids], actions[rand_ids], log_probs[rand_ids], returns[rand_ids], advantage[rand_ids]
        
        

def ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, model, optimizer, clip_param=0.2):
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            #print(type(state), state.dtype, state.shape)
            dist, value = model(state)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp().unsqueeze(0)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, (gae + values[step]).squeeze(0))
    return returns






from unityagents import UnityEnvironment
import numpy as np





def main():
    use_cuda = torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    scores_window = deque(maxlen=100)




    #Hyper params:
    hidden_size      = 512
    lr               = 3e-4
    num_steps        = 1000
    mini_batch_size  = 16
    ppo_epochs       = 16
    threshold_reward = 10



    max_frames = 1e5
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
    optimizer = optim.Adam(model.parameters(), lr=lr)

    state = env_info.vector_observations[0]
    early_stop = False

#    while frame_idx < max_frames and not early_stop:
    while frame_idx < max_frames and not early_stop:

        #print('clonk')
        log_probs = []
        values    = []
        states    = []
        actions   = []
        rewards   = []
        masks     = []
        #entropy = 0

        for _ in range(num_steps):
            
            state = torch.FloatTensor(state).to(device)
            #print(type(state), state.dtype, state.shape)
            dist, value = model(state)

            action_t = dist.sample()
            action_np = action_t.cpu().data.numpy()
    #
            env_info = env.step(action_np)[brain_name]           # send all actions to tne environment
            next_state = env_info.vector_observations[0]        # get next state (for each agent)
            reward = env_info.rewards[0]                        # get reward (for each agent)
            done = env_info.local_done[0]                        # see if episode finished
    #        
            if reward == None:
                pass

            log_prob = dist.log_prob(action_t)
            #entropy += dist.entropy().mean()
            
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            scores_window.append(reward)
            masks.append(1 - done)#.unsqueeze(1).to(device))

            states.append(state)
            actions.append(action_t)
            
            state = next_state
            frame_idx += 1

            if done:
                env.reset(train_mode=True)[brain_name]
                #break

            mean_score=np.mean(scores_window)
            if mean_score > threshold_reward: early_stop = True


            if frame_idx % 1000 == 0:
                #test_reward = np.mean([test_env() for _ in range(10)])
                #test_rewards.append(test_reward)
                mean_score=np.mean(scores_window)
                print("Mean Score: ", mean_score, "Frame: ", frame_idx)
                #plot(frame_idx, scores_window)
                #if test_reward > threshold_reward: early_stop = True
                

        next_state = torch.FloatTensor(next_state).to(device)
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


    #%%
    env.close()

if __name__ == "__main__":
    main()