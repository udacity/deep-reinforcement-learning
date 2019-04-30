import torch
import torch.nn as nn
import torch.optim as optim
from IPython.display import clear_output
import matplotlib.pyplot as plt
from collections import deque
import os
from tqdm import tqdm
import datetime

from unityagents import UnityEnvironment
import numpy as np


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

    def forward(self, x):
        value = self.critic(x)
        mu = self.actor(x)
        std = self.log_std.exp().expand_as(mu)
        # std   = self.log_std.exp().squeeze(0).expand_as(mu)
        dist = torch.distributions.Normal(mu, std)
        return dist, value


def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[
                                                                                                       rand_ids, :]


def ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, model, optimizer,
               clip_param=0.2):
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs,
                                                                         returns, advantages):
            dist, value = model(state)
            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(action)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def compute_gaes(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    advantage = 0
    returns = []
    for step in reversed(range(len(rewards))):
        td_error = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        advantage = advantage * tau * gamma * masks[step] + td_error
        returns.insert(0, advantage + values[step])
    return returns


def test_agent(env, brain_name, model, device):
    env_info = env.reset(train_mode=True)[brain_name]
    states = env_info.vector_observations
    num_agents = len(env_info.agents)
    scores = np.zeros(num_agents)
    while True:
        states = torch.FloatTensor(states).to(device)
        dist, _ = model(states)
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


def plot(scores=[], ylabel="Scores", xlabel="Episode #", title="", text=""):
    fig, ax = plt.subplots()

    for score in scores:
        ax.plot(np.arange(len(score)), score)
    xlabel = "\n".join([xlabel, text])
    ax.set(xlabel=xlabel, ylabel=ylabel,
           title=title)
    ax.grid()
    #    fig.text(-.2,-.2,text)
    fig.tight_layout()
    fig.savefig(f"plot_{datetime.datetime.now().isoformat().replace(':', '')}.png")
    plt.show()


def main():
    os.environ['NO_PROXY'] = 'localhost,127.0.0.*'

    # Hyper params:
    hidden_size = 256
    lr = 3e-4
    num_steps = 2048
    mini_batch_size = 512
    ppo_epochs = 3
    threshold_reward = 10
    max_episodes = 15  # 1e5
    episode = 0
    nrmlz_adv = True
    test_mean_reward = 1.

# plot([np.sin(x)], "Pi", "Lolololols sldk lskd lis dpsdlösödp ö spdops dösd psod ö")
# Random tries
#**    scores_window1, test_rewards1 = run_experiment(hidden_size=256, lr=3e-4, max_episodes=20, mini_batch_size=32,
#**                                                   nrmlz_adv=True, num_steps=2048, ppo_epochs=3, threshold_reward=10)

#    scores_window2, test_rewards2 = run_experiment(hidden_size=256, lr=3e-4, max_episodes=20, mini_batch_size=512,
#                                                   nrmlz_adv=True, num_steps=2048, ppo_epochs=3, threshold_reward=10)

#    scores_window3, test_rewards3 = run_experiment(hidden_size=256, lr=1e-5, max_episodes=20, mini_batch_size=32,
#                                                   nrmlz_adv=False, num_steps=2048, ppo_epochs=3, threshold_reward=10)


#Batch size, learning rate, no normalization
#    scores_window1, test_rewards1 = run_experiment(hidden_size=256, lr=3e-4, max_episodes=20, mini_batch_size=512,
#                                                   nrmlz_adv=False, num_steps=2048, ppo_epochs=5, threshold_reward=10)

#    scores_window2, test_rewards2 = run_experiment(hidden_size=256, lr=3e-4, max_episodes=20, mini_batch_size=256,
#                                                   nrmlz_adv=False, num_steps=2048, ppo_epochs=5, threshold_reward=10)

#    scores_window3, test_rewards3 = run_experiment(hidden_size=256, lr=6e-4, max_episodes=20, mini_batch_size=256,
#                                                   nrmlz_adv=False, num_steps=2048, ppo_epochs=3, threshold_reward=10)

#Normalization bigger net.
#    scores_window1, test_rewards1 = run_experiment(hidden_size=512, lr=3e-4, max_episodes=20, mini_batch_size=512,
#                                                    nrmlz_adv=True, num_steps=2048, ppo_epochs=3, threshold_reward=10)

#    scores_window2, test_rewards2 = run_experiment(hidden_size=512, lr=3e-4, max_episodes=20, mini_batch_size=1024,
#                                                   nrmlz_adv=True, num_steps=2048, ppo_epochs=3, threshold_reward=10)

#    scores_window3, test_rewards3 = run_experiment(hidden_size=512, lr=3e-4, max_episodes=20, mini_batch_size=32,
#                                                   nrmlz_adv=True, num_steps=2048, ppo_epochs=3, threshold_reward=10)

#smaller net, normalization, smaller batch
#    scores_window1, test_rewards1 = run_experiment(hidden_size=64, lr=3e-4, max_episodes=20, mini_batch_size=16,
#                                                    nrmlz_adv=True, num_steps=2048, ppo_epochs=5, threshold_reward=10)

#    scores_window2, test_rewards2 = run_experiment(hidden_size=64, lr=3e-4, max_episodes=20, mini_batch_size=32,
#                                                   nrmlz_adv=True, num_steps=2048, ppo_epochs=5, threshold_reward=10)

#**    scores_window3, test_rewards3 = run_experiment(hidden_size=64, lr=3e-4, max_episodes=20, mini_batch_size=64,
#                                                   nrmlz_adv=True, num_steps=2048, ppo_epochs=5, threshold_reward=10)

# new contender
#***    scores_window1, test_rewards1 = run_experiment(hidden_size=512, lr=3e-4, max_episodes=40, mini_batch_size=1024,
#***                                                   nrmlz_adv=True, num_steps=2048, ppo_epochs=5, threshold_reward=10)
# adapted winners
#    scores_window2, test_rewards2 = run_experiment(hidden_size=64, lr=3e-4, max_episodes=40, mini_batch_size=64,
#                                                   nrmlz_adv=True, num_steps=2048, ppo_epochs=5, threshold_reward=10)
#***    scores_window3, test_rewards3 = run_experiment(hidden_size=256, lr=3e-4, max_episodes=40, mini_batch_size=32,
#***                                                      nrmlz_adv=True, num_steps=2048, ppo_epochs=3, threshold_reward=10)
    scores = [

#    run_experiment(hidden_size=256, lr=1e-3, max_episodes=30, mini_batch_size=512,
#                                                   nrmlz_adv=False, num_steps=2048, ppo_epochs=4, threshold_reward=20),


    run_experiment(hidden_size=256, lr=1e-3, max_episodes=30, mini_batch_size=128,
                                                      nrmlz_adv=True, num_steps=2048, ppo_epochs=4, threshold_reward=20, clip_gradients=True),

    run_experiment(hidden_size=256, lr=1e-3, max_episodes=30, mini_batch_size=32,
                                                      nrmlz_adv=True, num_steps=2048, ppo_epochs=4, threshold_reward=20, clip_gradients=True),

    run_experiment(hidden_size=256, lr=1e-3, max_episodes=30, mini_batch_size=128,
                                                      nrmlz_adv=True, num_steps=2048, ppo_epochs=4, threshold_reward=20, clip_gradients=False)
    ]
    plot([x[0] for x in scores], "Scores")


def run_experiment(hidden_size, lr, max_episodes, mini_batch_size, nrmlz_adv, num_steps, ppo_epochs, threshold_reward, clip_gradients):
    scores_window, test_rewards = experiment(hidden_size=hidden_size, lr=lr, num_steps=num_steps,
                                             mini_batch_size=mini_batch_size, ppo_epochs=ppo_epochs,
                                             threshold_reward=threshold_reward, max_episodes=max_episodes,
                                             nrmlz_adv=nrmlz_adv, clip_gradients=clip_gradients)
    test_mean_reward = np.mean(test_rewards)
    text = "\n".join([f"HS:{hidden_size} lr:{lr} st:{num_steps} batch:{mini_batch_size} ppo:{ppo_epochs}",
                      f" r:{threshold_reward} e:{max_episodes} adv:{nrmlz_adv} mean {test_mean_reward}"])
    plot([scores_window], "Last # Scores", text=text)
    return scores_window, test_rewards


def experiment(hidden_size=64, lr=3e-4, num_steps=2048, mini_batch_size=32, ppo_epochs=10, threshold_reward=10,
               max_episodes=15, nrmlz_adv=True, clip_gradients=True):
    use_cuda = torch.cuda.is_available()
    #    device   = torch.device("cuda" if use_cuda else "cpu")
    device = torch.device("cpu")
    print(device)
    scores_window = deque(maxlen=100)

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

    num_inputs = state_size
    num_outputs = action_size

    model = ActorCriticPolicy(num_inputs, num_outputs, hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-5)

    #    while episode < max_episodes and not early_stop:
    for episode in tqdm(range(max_episodes)):
        log_probs = []
        values = []
        states_list = []
        actions_list = []
        rewards = []
        masks = []
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations
        for duration in range(num_steps):

            state = torch.FloatTensor(state).to(device)
            dist, value = model(state)
            action_t = dist.sample()
            action_np = action_t.cpu().data.numpy()
            env_info = env.step(action_np)[brain_name]  # send all actions to the environment

            next_state = env_info.vector_observations  # get next state (for each agent)
            reward = env_info.rewards  # get reward (for each agent)
            dones = np.array(env_info.local_done)  # see if episode finished
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

        returns = torch.cat(returns).detach()
        mean2 = torch.mean(returns)
        print("Returns: ", mean2)
        log_probs = torch.cat(log_probs).detach()
        values = torch.cat(values).detach()
        states = torch.cat(states_list)
        actions = torch.cat(actions_list)
        advantages = returns - values
        if nrmlz_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        losses = []

        clip_param = 0.2
        for _ in range(ppo_epochs):
            print("return: ", returns.mean(), "advantage:", advantages.mean())
            for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions,
                                                                             log_probs, returns, advantages):
                # print("return: ", return_.mean(), "advantage:", advantage.mean())
                dist, value = model(state)
                entropy = dist.entropy().mean()

                new_log_probs = dist.log_prob(action)
                new_log_probs = torch.sum(new_log_probs, dim=1, keepdim=True)

                ratio = (new_log_probs - old_log_probs).exp()
                # surrogate objective
                surr1 = ratio * advantage
                # Clipped Surrogate Objectiv
                surr2 = ratio.clamp(1.0 - clip_param, 1.0 + clip_param) * advantage

                policy_loss = - torch.min(surr1, surr2).mean() - 0.01 * entropy
                value_loss = (return_ - value).pow(2).mean()

                loss = 0.5 * value_loss + policy_loss
                losses.append(loss)
                optimizer.zero_grad()
                loss.backward()
                if clip_gradients:
                    nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()

        test_mean_reward = test_agent(env, brain_name, model, device)
        test_rewards.append(test_mean_reward)
        scores_window.append(test_mean_reward)
        # mean_score = np.mean(scores_window)
        # print("Mean Score: ", mean_score, "Frame: ", episode)
        print('Episode {}, Total score this episode: {}, Last {} average: {}'.format(episode, test_mean_reward,
                                                                                     min(episode, 100),
                                                                                     np.mean(scores_window)))
        if np.mean(scores_window) > threshold_reward:
            torch.save(model.state_dict(),
                       f"ppo_checkpoint_{test_mean_reward}_e{episode}_hs{hidden_size}_lr{lr}_st{num_steps}_b{mini_batch_size}_ppo{ppo_epochs}_r{threshold_reward}_e{episode}_adv{nrmlz_adv}_{test_mean_reward}.pth")
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode, test_mean_reward))
            break

        episode += 1

    # %%
    #torch.save(model.state_dict(),
    #          f"ppo_checkpoint_{test_mean_reward}_e{episode}_hs{hidden_size}_lr{lr}_st{num_steps}_b{mini_batch_size}_ppo{ppo_epochs}_r{threshold_reward}_e{episode}_adv{nrmlz_adv}.pth")

    env.close()
    return scores_window, test_rewards


if __name__ == "__main__":
    main()
