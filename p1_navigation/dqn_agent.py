import numpy as np
import random
from collections import namedtuple, deque
import math

from model import QNetwork

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        # use helper calc_loss function & not this directly
        self.criterion = nn.MSELoss()

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        # first unsqueeze to make it a batch with one sample in it
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        # set the mode back to training mode
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
        # output of model needs to be Q values of actions 0 - (n-1) And output[ind_x] needs to correspond to action_x
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
        
    # helper for learn func
    # y_j does not depend on the weight parameters that gradient descent will be training
    def calc_y_j(self, r_j, dones, gamma, target_out):
        # 1 or 0 flag; if episode doesn't terminate at j+1 (aka done == False), y_j product now already includes the gamma multiplication factor
        # use [[x] for x in y] kind of list comprehension because need them to be batch_size by 1 like r_j
        # use .to(device) to move to gpu (not just setting device arg when creating a tensor)
        dones_flags = torch.Tensor([[0] if done == True else [gamma] for done in dones]).float().to(device)
        max_q_target_out = torch.Tensor([[torch.max(q_for_all_actions)] for q_for_all_actions in target_out]).float().to(device)
        
        #  RuntimeError: Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead.
        #dones_flags = torch.from_numpy(np.vstack([0 if done == True else gamma for done in dones])).float().to(device)
        #max_q_target_out = torch.from_numpy(np.vstack([torch.max(q_for_all_actions) for q_for_all_actions in target_out])).float().to(device)
        
        y_j = r_j + dones_flags * max_q_target_out
        return y_j
    
    # helper for learn func
    def calc_loss(self, y_j, pred_out, actions):
        # need pred_out_actions_taken to be a tensor & built by combining (concatenating) other tensors to maintain gradients
        # actions is batch_size by 1- only have to iterate through rows
        for i in range(actions.shape[0]):
            # action taken. is an index for which col to look at in pred_out (pred_out is batch_size by n_actions]
            action_ind = actions[i, 0].item()
            if i == 0:
                # need to take h from 0 dimensional to 2 dimensional
                pred_out_actions_taken = pred_out[i, action_ind].unsqueeze(0).unsqueeze(0)
            else: 
                # concat along dim 0 -> vertically stack rows
                pred_out_actions_taken = torch.cat((pred_out_actions_taken, pred_out[i, action_ind].unsqueeze(0).unsqueeze(0)), dim=0)
            
        # loss is MSE between pred_out_actions_taken (input) and y_j (target) 
        return self.criterion(pred_out_actions_taken, y_j)

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        # torch
        states, actions, rewards, next_states, dones = experiences

        "*** YOUR CODE HERE ***"
        ## TODO: compute and minimize the loss
        # vstack takes one argument (sequence)- stacks the pieces of that argument vertically
        # SELF NOTE: think about what (if anything additional) needs .todevice()
        
        # make sure to zero the gradients
        self.optimizer.zero_grad()
        
        # q_network_local model output from forward pass
        pred_out = self.qnetwork_local(states)
        target_out = self.qnetwork_target(next_states)
        
        # compute the loss for q_network_local vs q_network_target 
        y_j = self.calc_y_j(rewards, dones, gamma, target_out)
        # calc gradient & take step down the gradient
        loss = self.calc_loss(y_j, pred_out, actions)
        loss.backward()
        self.optimizer.step()
                      

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
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
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)