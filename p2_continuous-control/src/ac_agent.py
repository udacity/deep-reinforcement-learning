import numpy as np
import random
import os
from collections import namedtuple, deque

from src.model import DDPGCritic, DDPGActor
from src.dqn_agent import ReplayBuffer
import abc
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Agents in Actor-Critic are divided in: DDPG and others (A3C or A2C)
# DDPG is an Actor-Critic generalization from DQNs, while the later are only applicable in environments
# with discrete action spaces, DDPG can only be used when actions are continuous
class AgentDDPG(object):
    """
    Implement Deep Deterministic Policy Gradient Agent with Actor and Critic Networks and Experience Replay
    Actor: mu[s; theta_mu]
    Critic: Q[s, mu; thea_Q]
    Both networks has local (thea_mu, thea_Q) and target (theta_mu', theta_Q') weights. Local weigths are copied to target
    weights every `update_every` learn() steps
    At each step, an experience is store in memory buffer (multiple environments can store their transitions in this buffer).
    This learn happens every `learn_every` steps and each time a number of `learn_num` batches is run
    Params
    ======
        state_size (int): state space size
        action_size (int): action space size
        gamma (float): discount factor
        actor_hidden_layers (tuple): actor number of activation units by layer
        critic_hidden_layers (tuple): critic number of activation units by layer
        actor_add_bn (tuple): actor whether to add a batchnorm by layer
        critic_add_bn (tuple): critic whether to add a batchnorm by layer
        batch_size (tuple)
        grad_clipping (tuple): whether to clip gradient for (actor, critic)
        learning_rates (tuple): learning rates for (actor, critic)
        weight_decay (tuple): weight_decay for (actor, critic)
        soft_upd_param (float): θ_target = τ*θ_local + (1 - τ)*θ_target
        update_every (int): number of steps to wait before updating target weights
        buffer_size (int): Memory replay buffer size
        noise (int): Gaussian or Ornstein-Uhlenbeck noise to enable exploration
        learn_every (int): Number of steps to wait until learn
        learn_num (int): Number of learn batches to perform when learning
        seed (int): random seed
        action_dtype (str): 'float' for continuous and 'long' for discrete action spaces
    """

    def __init__(self, state_size, action_size, gamma,
                 actor_hidden_layers, critic_hidden_layers, actor_add_bn, critic_add_bn,
                 batch_size, grad_clipping, learning_rates, weight_decay, soft_upd_param, update_every, buffer_size,
                 noise, learn_every, learn_num, seed, action_dtype='float'
                 ):

        # Environment and General Params
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.seed = seed
        random.seed(seed)

        # Hparams
        self.actor_hidden_layers = actor_hidden_layers
        self.critic_hidden_layers = critic_hidden_layers
        self.actor_fc1, self.actor_fc2 = actor_hidden_layers
        self.critic_fc1, self.critic_fc2 = critic_hidden_layers
        self.actor_add_bn1 = actor_add_bn
        self.critic_add_bn1 = critic_add_bn
        self.actor_grad_clip, self.critic_grad_clip = grad_clipping
        self.actor_lr, self.critic_lr = learning_rates
        self.actor_wdec, self.critic_wdec = weight_decay

        self.soft_upd_param = soft_upd_param
        self.update_every = update_every
        self.batch_size = batch_size

        # Experience Replay
        self.buffer_size = buffer_size
        # Initialize time step (for updating every UPDATE_EVERY steps)

        # Noise process
        if noise is None:
            self.noise = OUNoise(action_size, seed)
        else:
            self.noise = noise

        # learning
        self.learn_every = learn_every
        self.learn_num = learn_num

        # A-C Network Architecture
        # Actor
        self.actor_local = DDPGActor(
            self.state_size, self.action_size, self.seed, self.actor_fc1, self.actor_fc2, self.actor_add_bn1).to(device)
        self.actor_target = DDPGActor(
            self.state_size, self.action_size, self.seed, self.actor_fc1, self.actor_fc2, self.actor_add_bn1).to(device)
        self.actor_opt = optim.Adam(self.actor_local.parameters(), lr=self.actor_lr, weight_decay=self.actor_wdec)

        # Critic
        self.critic_local = DDPGCritic(
            self.state_size, self.action_size, self.seed, self.critic_fc1, self.critic_fc2, self.critic_add_bn1).to(device)
        self.critic_target = DDPGCritic(
            self.state_size, self.action_size, self.seed, self.critic_fc1, self.critic_fc2, self.critic_add_bn1).to(device)
        self.critic_opt = optim.Adam(self.critic_local.parameters(), lr=self.critic_lr, weight_decay=self.critic_wdec)

        # Experience Replay
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed, action_dtype)

    def step(self, state, action, reward, next_state, done, t_step):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # If enough samples are available in memory, get a batch of experiences and learn
        if len(self.memory) > self.batch_size and t_step % self.learn_every == 0:
            for _ in range(self.learn_num):
                experiences = self.memory.sample()
                self.learn(experiences, t_step)

    def act(self, state, *args, **kwargs):
        """Returns actions for given state as per current actor policy mu
        Actions may be needed to be clipped, depending on environmnet
        Params
        ======
            state (array_like): current state
            kwargs:
                add_noise (bool): add random noise to actions in order to explore
                noise_weight (float): factor (0., 1.) to apply to noise
        Returns
        ======
            (tensor) action vector of size action_size
        """
        add_noise = kwargs.get('add_noise', None)
        if add_noise:
            def_noise_weight = 1.
        else:
            def_noise_weight = 0.

        noise_weigth = kwargs.get('noise_weight', def_noise_weight)

        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self._forward_actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        # add noise
        action += noise_weigth*self.noise.sample()
        # action values may be needed to scale before supplying to environment, due to tanh range or noise addition

        return action

    def reset(self):
        self.noise.reset()

    def _forward_actor_local(self, states):
        """
        Returns
        ======
            ps_local (torch.tensor)
        """
        ps_local = self.actor_local.forward(states)

        return ps_local

    def _forward_actor_targets(self, states):
        """
        Use Fixed TD-Target Algorithm
        Returns
        ======
            ps_target (torch.tensor)
        """
        ps_target = self.actor_target.forward(states)

        return ps_target

    def _forward_critic_local(self, states, actions):
        """
        Returns
        ======
            ps_local (torch.tensor)
        """
        ps_local = self.critic_local.forward(states, actions)

        return ps_local

    def _forward_critic_targets(self, states, actions):
        """
        Use Fixed TD-Target Algorithm
        Returns
        ======
            ps_target (torch.tensor)
        """
        ps_target = self.critic_target.forward(states, actions)

        return ps_target

    def learn(self, experiences, t_step):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # states and next_states (batch_size x num_states)
        # actions and rewards (batch_size x 1)

        # Forward pass
        # Update Critic
        # Actor: mu_prime(s[t+1]; theta_mu')
        actions_pred_next = self._forward_actor_targets(next_states)
        # Critic: Q_prime(s[t+1], mu_prime)
        Q_targets_next = self._forward_critic_targets(next_states, actions_pred_next)
        # y[i] = r[i] + gamma*Q_prime(s[i+1], mu_prime)
        Q_targets_next = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_targets_next = Q_targets_next.detach()

        # Compute critic loss
        # Q(s[i], a[i]; theta_Q)
        Q_expected = self._forward_critic_local(states, actions)
        loss_critic = F.mse_loss(input=Q_expected, target=Q_targets_next)  # y[i] - Q(s[i], a[i]; theta_Q)

        # Backprop critic loss
        self.critic_opt.zero_grad()
        loss_critic.backward()
        if self.critic_grad_clip > 0.:
            torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), self.critic_grad_clip)

        self.critic_opt.step()

        # Update Actor
        # Compute policy gradient
        actions_pred = self._forward_actor_local(states)  # Actor: mu(s[t]; theta_mu) (same as act())
        pol_grad_actor = -self._forward_critic_local(states, actions_pred).mean()

        # Backprop policy gradient
        self.actor_opt.zero_grad()
        pol_grad_actor.backward()
        self.actor_opt.step()

        # ------------------- update target network ------------------- #

        if (t_step % self.update_every) == 0:
            # Actor
            self.soft_update(self.actor_local, self.actor_target, self.soft_upd_param)
            # Critic
            self.soft_update(self.critic_local, self.critic_target, self.soft_upd_param)

    def soft_update(self, local_model, target_model, soft_upd_param):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            soft_upd_param (float): interpolation parameter
        """

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(soft_upd_param * local_param.data + (1.0 - soft_upd_param) * target_param.data)

    def save_network(self, filename):
        """
        Save local actor and critic networks, always as actor and critic .pth
        Params
        ======
            filename (str): Path
        """

        if not os.path.exists(filename):
            os.makedirs(filename)

        actor_filename = os.path.join(filename, 'actor.pth')
        critic_filename = os.path.join(filename, 'critic.pth')

        torch.save(self.actor_local.state_dict(), actor_filename)
        torch.save(self.critic_local.state_dict(), critic_filename)

    def load_network(self, filename):
        """
        Load local actor and critic networks, always as actor and critic .pth
        Params
        ======
            filename (str): Path
        """
        if torch.cuda.is_available():
            map_location = lambda storage, loc: storage.cuda()
        else:
            map_location = 'cpu'

        actor_filename = os.path.join(filename, 'actor.pth')
        critic_filename = os.path.join(filename, 'critic.pth')

        actor_checkpoint = torch.load(actor_filename, map_location=map_location)
        self.actor_local.load_state_dict(actor_checkpoint)

        critic_checkpoint = torch.load(critic_filename, map_location=map_location)
        self.critic_local.load_state_dict(critic_checkpoint)


class AbstractRandomProcess(object):

    @abc.abstractmethod
    def reset(self):
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self):
        raise NotImplementedError


class OUNoise(AbstractRandomProcess):
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


class GaussianProcess(AbstractRandomProcess):
    """Gaussian (uncorrelated white noise) process."""
    def __init__(self, size, seed, sigma=np.array([0.1]), mu=np.array([0.])):
        self.size = size
        self.seed = random.seed(seed)
        self.sigma = sigma
        self.mu = mu
        self.reset()

    def sample(self):
        return np.random.normal(loc=self.mu, scale=self.sigma, size=self.size)

    def reset(self):
        self.state = copy.copy(self.mu)
