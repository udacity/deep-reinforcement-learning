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



def plotlosses(losses, fig=None):

    if fig is None:
        fig=plt.figure(figsize=(20,5))
#        fig.subplot(131)
        fig.title('Losses')
        fig.show()
        return fig
    fig.set_xdata(losses)
    fig.draw()
    return fig

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
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCriticPolicy, self).__init__()
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Tanh(),
        )
        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)
        
        #self.apply(init_weights)
        
    def forward(self, x):
        value = self.critic(x)
        mu    = self.actor(x)
        std = self.log_std.exp().expand_as(mu)
        #std   = self.log_std.exp().squeeze(0).expand_as(mu)
        dist  = torch.distributions.Normal(mu, std)
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



def compute_gaes(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    advantage = 0
    returns = []
    for step in reversed(range(len(rewards))):
        td_error = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        advantage = advantage*tau*gamma*masks[step] + td_error
        returns.insert(0, advantage + values[step])
    return returns

def compute_gae_rollout(rollout, gamma=0.99, tau=0.95):
    storage = [None] * (len(rollout) - 1)

    shape = (num_agents, 1)
    advantage = torch.Tensor(np.zeros(shape))

    for i in reversed(range(len(rollout) - 1)):
        # rollout --> tuple ( s, a, p(a|s), r, dones, V(s) ) FOR ALL AGENT
        # rollout --> last row (s, none, none, none, pending_value) FOR ALL AGENT
        state, action, log_prob, reward, done, value = rollout[i]

        # last step - next_return = pending_value
        if i == len(rollout) - 2:
            next_return = rollout[i + 1][-1]

        state = torch.Tensor(state)
        action = torch.Tensor(action)
        reward = torch.Tensor(reward).unsqueeze(1)
        done = torch.Tensor(done).unsqueeze(1)

        next_value = rollout[i + 1][-1]

        # G(t) = r + G(t+1)
        g_return = reward + gamma * next_return * done
        next_return = g_return
        # g_return = reward + GAMMA * g_return*done

        # Compute TD error
        td_error = reward + gamma * next_value - value
        # Compute advantages
        advantage = advantage * tau * gamma * done + td_error

        # Add (s, a, p(a|s), g, advantage)
        storage[i] = [state, action, log_prob, g_return, advantage]
    return storage


def normalize(x, mean=0., std=1., epsilon=1e-8):
    x = (x - np.mean(x)) / (np.std(x) + epsilon)
    x = x * std + mean

    return x




from unityagents import UnityEnvironment
import numpy as np

def toast_agent(env, brain_name, model, device):
    env_info = env.reset(train_mode = True)[brain_name]
    state = env_info.vector_observations[0]
    scores = 0.0
    while True:
        state = torch.FloatTensor(state).to(device)
        dist, _ = model(state)
        action_t = dist.sample()
        action_np = action_t.cpu().data.numpy()

        env_info = env.step(action_np)[brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        scores += reward
        state = next_state
        if done:
            break
    return scores

def test_agent(env, brain_name,  model, device):
    env_info = env.reset(train_mode = True)[brain_name]
    states = env_info.vector_observations
    num_agents=len(env_info.agents)
    scores = np.zeros(num_agents)
    while True:
        states = torch.FloatTensor(states).to(device)
        dist, _= model(states)
        action_t = dist.sample()
        action_np = action_t.cpu().data.numpy()
        env_info = env.step(action_np)[brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        scores += rewards
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
    hidden_size      = 256
    lr               = 3e-4
    num_steps        = 2048
    mini_batch_size  = 32
    ppo_epochs       = 10
    threshold_reward = 10



    max_episodes = 250#1e5
    episode  = 0
    test_rewards = []


    env = UnityEnvironment(file_name='reacher20/reacher', base_port=64739)
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    action_size = brain.vector_action_space_size
    num_agents = len(env_info.agents)
    states = env_info.vector_observations
    state_size = states.shape[1]

    num_inputs  = state_size
    num_outputs = action_size
    
    model = ActorCriticPolicy(num_inputs, num_outputs, hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr,eps=1e-5)

    

#    while episode < max_episodes and not early_stop:
    for episode in tqdm(range(max_episodes)):
        rollout = []
        log_probs = []
        values    = []
        states_list    = []
        actions_list   = []
        rewards   = []
        masks     = []
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
            log_prob = torch.sum(log_prob, dim=1, keepdim=True)
            log_probs.append(log_prob)
            values.append(value)
            reward_t = torch.FloatTensor(reward).unsqueeze(1).to(device)
            masks_t = torch.FloatTensor(1 - dones)
            rewards.append(reward_t)
            masks.append(masks_t)
            states_list.append(state)
            actions_list.append(action_t)

            state = next_state

            if np.any(dones):
                break

        next_state = torch.FloatTensor(state).to(device)
        _, next_value = model(next_state)

#        returns = compute_gae(next_value, rewards, masks, values)
        mean1 = torch.mean(torch.stack(rewards))
        print("Rewards: ", mean1)
        returns = compute_gaes(next_value, rewards, masks, values)
#        return2 = compute_gae_rollout(rollout)


        returns   = torch.cat(returns).detach()
        mean2 = torch.mean(returns)
        print("Returns: ", mean2)
        log_probs = torch.cat(log_probs).detach()
        values    = torch.cat(values).detach()
        states    = torch.cat(states_list)
        actions   = torch.cat(actions_list)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
  
        losses=[]

        clip_param = 0.2
        for _ in range(ppo_epochs):
            print("return: ", returns.mean(), "advantage:", advantages.mean())
            for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions,
                                                                             log_probs, returns, advantages):
                #print("return: ", return_.mean(), "advantage:", advantage.mean())
                dist, value = model(state)
                entropy = dist.entropy().mean()

                new_log_probs = dist.log_prob(action)
                new_log_probs = torch.sum(new_log_probs, dim=1, keepdim=True)

                ratio = (new_log_probs - old_log_probs).exp()
                #surrogate objective
                surr1 = ratio * advantage
                # Clipped Surrogate Objectiv
                surr2 = ratio.clamp(1.0 - clip_param, 1.0 + clip_param) * advantage

                policy_loss = - torch.min(surr1, surr2).mean()  - 0.01 * entropy
                value_loss = (return_ - value).pow(2).mean()

                loss = 0.5 * value_loss + policy_loss
                losses.append(loss)
                #hl.set_xdata(range(len(losses)))
                #hl.set_ydata(np.array(losses))
                #plt.draw()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
#        model.eval()
        test_mean_reward = test_agent(env, brain_name, model, device)
        test_rewards.append(test_mean_reward)
        scores_window.append(test_mean_reward)
        #mean_score = np.mean(scores_window)
        #print("Mean Score: ", mean_score, "Frame: ", episode)
        print('Episode {}, Total score this episode: {}, Last {} average: {}'.format(episode, test_mean_reward,
                                                                                     min(episode, 100),
                                                                                     np.mean(scores_window)))
        episode += 1

    #%%
    env.close()

if __name__ == "__main__":
    main()