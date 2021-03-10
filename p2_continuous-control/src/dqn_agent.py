import numpy as np
import random
from collections import namedtuple, deque

from src.model import QNetwork, DuelQNetwork
import abc
import torch
import torch.nn.functional as F
import torch.optim as optim

#BUFFER_SIZE = int(1e5)  # replay buffer size
#BATCH_SIZE = 64         # minibatch size
#GAMMA = 0.99            # discount factor
#TAU = 1e-3              # for soft update of target parameters
#LR = 5e-4               # learning rate
#UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AgentAbstract(object):
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, gamma,
                 hidden_layers, drop_p,
                 batch_size, learning_rate, soft_upd_param, update_every, buffer_size, seed):
        """Initialize a Deep Q-Learning Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            gamma (float): discount factor
            hidden_layers (list of int):
            drop_p (float): Dropout layer p parameter. https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
            batch_size (int): Number of examples on each batch
            learning_rate (float): Optimizer parameter. https://pytorch.org/docs/stable/optim.html
            soft_upd_param (float): interpolation parameter in target network weight update
            update_every (float): Number of time steps (batches) between each target network weight update
            buffer_size (int): maximum size of Memory Replay buffer
            seed (int): random seed
        """
        # Environment and General Params
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.seed = seed
        random.seed(seed)

        # Hparams
        self.hidden_layers = hidden_layers
        self.drop_p = drop_p
        self.learning_rate = learning_rate
        self.soft_upd_param = soft_upd_param
        self.update_every = update_every
        self.batch_size = batch_size

        # Experience Replay
        self.buffer_size = buffer_size
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        self.t_step += 1
        # If enough samples are available in memory, get a batch of experiences and learn
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current (epsilon-greedy) policy
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        Returns
        ======
            (int) epsilon greedy action
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()).astype(np.int32)
        else:
            return random.choice(np.arange(self.action_size)).astype(np.int32)

    @abc.abstractmethod
    def _forward_local(self, states, actions):
        """
        Compute a local (online) forward pass. Input current state S and select Q(s,a)
        Params
        ======
            states (torch.tensor(), batch_size x state_size)
            actions (torch.tensor(), batch_size x 1)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _forward_targets(self, rewards, next_states, dones):
        """
        Compute a target (offline) forward pass. Input next state S' and R to compute TD-Target
        Params
        ======
            rewards (torch.tensor(), batch_size x 1)
            next_states (torch.tensor(), batch_size x 1)
            dones (torch.tensor(), batch_size x 1)
        """
        raise NotImplementedError

    def learn(self, experiences):
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
        ps_local = self._forward_local(states, actions)
        ps_target = self._forward_targets(rewards, next_states, dones)

        # Compute loss
        loss = F.mse_loss(ps_local, ps_target)

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        if (self.t_step % self.update_every) == 0:
            self._soft_update(self.qnetwork_local, self.qnetwork_target, self.soft_upd_param)

    def _soft_update(self, local_model, target_model, soft_upd_param):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            soft_upd_param (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(soft_upd_param*local_param.data + (1.0-soft_upd_param)*target_param.data)

    def save_network(self, filename):
        torch.save(self.qnetwork_local.state_dict(), filename)

    def load_network(self, filename):

        if torch.cuda.is_available():
            map_location = lambda storage, loc: storage.cuda()
        else:
            map_location = 'cpu'
        checkpoint = torch.load(filename, map_location=map_location)
        self.qnetwork_local.load_state_dict(checkpoint)


class AgentDQ(AgentAbstract):
    """
    Implement Deep Q-Net with Fixed TD-Target computation and Experience Replay
    Fixed TD-Target: TD-Error computed on a target (offline) and local (online) network,
    where local network weights are copied to target network every `update_every` batches
    """
    def __init__(self, state_size, action_size, gamma,
                 hidden_layers, drop_p,
                 batch_size, learning_rate, soft_upd_param, update_every, buffer_size, seed):
        super(AgentDQ, self).__init__(
            state_size, action_size, gamma,
            hidden_layers, drop_p,
            batch_size, learning_rate, soft_upd_param, update_every, buffer_size, seed)

        # Q-Network Architecture
        self.qnetwork_local = QNetwork(
            self.state_size, self.action_size, self.seed, self.hidden_layers, self.drop_p).to(device)
        self.qnetwork_target = QNetwork(
            self.state_size, self.action_size, self.seed, self.hidden_layers, self.drop_p).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)
        # Experience Replay
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)

    def _forward_local(self, states, actions):
        """
        Returns
        ======
            ps_local (torch.tensor)
        """
        ps_local = self.qnetwork_local.forward(states).gather(1, actions)

        return ps_local

    def _forward_targets(self, rewards, next_states, dones):
        """
        Use Fixed TD-Target Algorithm
        Returns
        ======
            ps_target (torch.tensor)
        """
        # Fixed Q-Targets
        # use target network compute r + g*max(q_est[s',a, w-]), this tensor should be detached in backprop
        ps_target = rewards + self.gamma * (1 - dones) * self.qnetwork_target.forward(next_states).detach().\
            max(dim=1)[0].view(-1, 1)

        return ps_target

class AgentDoubleDQ(AgentAbstract):
    """
    Implement Dueling Q-Net with Double QNet (fixed) TD-Target computation and Experience Replay
    Double Q-Net: Split action selection and Q evaluation in two steps
    """

    def __init__(self, state_size, action_size, gamma,
                 hidden_layers, drop_p,
                 batch_size, learning_rate, soft_upd_param, update_every, buffer_size, seed):
        super(AgentDoubleDQ, self).__init__(
            state_size, action_size, gamma,
            hidden_layers, drop_p,
            batch_size, learning_rate, soft_upd_param, update_every, buffer_size, seed)

        # Q-Network Architecture: Dueling Q-Nets
        self.qnetwork_local = QNetwork(
            self.state_size, self.action_size, self.seed, self.hidden_layers, self.drop_p).to(device)
        self.qnetwork_target = QNetwork(
            self.state_size, self.action_size, self.seed, self.hidden_layers, self.drop_p).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)
        # Experience Replay
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)

    def _forward_local(self, states, actions):
        """
        Returns
        ======
            ps_local (torch.tensor)
        """
        ps_local = self.qnetwork_local.forward(states).gather(1, actions)

        return ps_local

    def _forward_targets(self, rewards, next_states, dones):
        """
        Use Double Q-Net Algorithm
        Returns
        ======
            ps_target (torch.tensor)
        """
        ps_actions = self.qnetwork_local.forward(next_states).detach().max(dim=1)[1].view(-1, 1)
        ps_target = rewards + self.gamma * (1 - dones) * self.qnetwork_target.forward(next_states).detach().\
            gather(1, ps_actions)

        return ps_target


class AgentDuelDQ(AgentAbstract):
    """
    Implement Dueling Q-Net with Double QNet (fixed) TD-Target computation and Experience Replay
    Double Q-Net: Split action selection and Q evaluation in two steps
    Dueling Q-Net: Split Q estimation on two streams, Value and Advantage estimation where Q = V + A
    """

    def __init__(self, state_size, action_size, gamma,
                 hidden_layers, drop_p,
                 batch_size, learning_rate, soft_upd_param, update_every, buffer_size, seed):
        super(AgentDuelDQ, self).__init__(
            state_size, action_size, gamma,
            hidden_layers, drop_p,
            batch_size, learning_rate, soft_upd_param, update_every, buffer_size, seed)

        # Q-Network Architecture: Dueling Q-Nets
        self.qnetwork_local = DuelQNetwork(
            self.state_size, self.action_size, self.seed, self.hidden_layers, self.drop_p).to(device)
        self.qnetwork_target = DuelQNetwork(
            self.state_size, self.action_size, self.seed, self.hidden_layers, self.drop_p).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)
        # Experience Replay
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)

    def _forward_local(self, states, actions):
        """
        Returns
        ======
            ps_local (torch.tensor)
        """
        ps_local = self.qnetwork_local.forward(states).gather(1, actions)

        return ps_local

    def _forward_targets(self, rewards, next_states, dones):
        """
        Use Double Q-Net Algorithm
        Returns
        ======
            ps_target (torch.tensor)
        """
        ps_actions = self.qnetwork_local.forward(next_states).detach().max(dim=1)[1].view(-1, 1)
        ps_target = rewards + self.gamma * (1 - dones) * self.qnetwork_target.forward(next_states).detach().\
            gather(1, ps_actions)

        return ps_target


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, action_dtype='long'):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = seed
        random.seed(seed)

        self.action_dtype = action_dtype

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None]))
        if self.action_dtype == 'long':
            actions = actions.long().to(device)
        elif self.action_dtype == 'float':
            actions = actions.float().to(device)
        else:
            actions = actions.to(device)

        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)