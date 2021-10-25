import random
import numpy as np
import torch
from torch.optim import Adam
import torch.nn.functional as F
from models import Actor, Critic
import queue
from collections import namedtuple, deque

class Agent():

	def __init__(self, n_states, n_actions):
		# constants
		# from paper
		self.Q_DISCOUNT = .99
		self.TAU = 0.001

		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		self.n_agents = 1
		self.n_actions = n_actions

		self.local_actor = Actor(n_states, n_actions, 400, 300, .1).to(self.device)
		self.target_actor = Actor(n_states, n_actions, 400, 300, .1).to(self.device)
		self.local_critic = Critic(n_states, n_actions, 400, 300, .1).to(self.device)
		self.target_critic = Critic(n_states, n_actions, 400, 300, .1).to(self.device)
	
		self.actor_opt = Adam(self.local_actor.parameters(), lr=.001)
		self.critic_opt = Adam(self.local_critic.parameters(), lr=.01)

		self.min_to_sample = 100
		self.replay_buffer = ReplayBuffer(64, 1000, self.min_to_sample)

	def save(self):
		torch.save(self.local_actor.state_dict(), 'local_actor_state_dict.pt')
		torch.save(self.target_actor.state_dict(), 'target_actor_state_dict.pt')
		torch.save(self.local_critic.state_dict(), 'local_critic_state_dict.pt')
		torch.save(self.target_critic.state_dict(), 'target_critic_state_dict.pt')

		torch.save(self.local_actor, 'local_actor.pt')
		torch.save(self.target_actor, 'target_actor.pt')
		torch.save(self.local_critic, 'local_critic.pt')
		torch.save(self.target_critic, 'target_critic.pt')

	def act(self, state):
		# while NNs won't be learning, generate diverse experiences (otherwise it seems are going in a loop of sorts to the same state when the agent acts based on initial weights while accumulating experience tuples)
		if len(self.replay_buffer) < self.min_to_sample:
			random_actions = np.random.randn(self.n_agents, self.n_actions)
			return np.clip(random_actions, -1, 1) 
		return self.local_actor(torch.from_numpy(state).to(self.device))

	def step(self, state, action, reward, next_state, done):
		self.replay_buffer.add(state, action, reward, next_state, done)
		if len(self.replay_buffer) >= self.replay_buffer.min_to_sample:
			experiences = self.replay_buffer.sample()
			self.learn(experiences)
	
	def soft_update(self, target_NN, local_NN):
		for local_param, target_param in zip(local_NN.parameters(), target_NN.parameters()): 
			target_param.data.copy_(self.TAU * local_param.data + (1.0 - self.TAU) * target_param.data)
		return target_NN
		
	# train local NNs & soft update target NNs
	def learn(self, experiences):
		states, actions, rewards, next_states, dones = experiences
	
		states = states.type(torch.FloatTensor).to(self.device)
		actions = actions.type(torch.FloatTensor).to(self.device)
		rewards = rewards.type(torch.FloatTensor).to(self.device)
		next_states = next_states.type(torch.FloatTensor).to(self.device)	
		dones = dones.type(torch.FloatTensor).to(self.device)	

		######
		# train critic
		######
		pred_q = self.local_critic(states, actions)
		
		next_actions = self.target_actor(next_states)
		# Q is the sum of discounted rewards following a particular first action
		target_q = rewards + self.Q_DISCOUNT * self.target_critic(next_states, next_actions)

		loss = F.mse_loss(pred_q.to("cpu"), target_q.to("cpu")) 
		self.critic_opt.zero_grad()
		loss.backward()
		self.critic_opt.step()
		
		######
		# train actor
		######
		pred_actions = self.local_actor(states)
		# critic used to critique actor: larger Q is better so minimize the negative of it
		loss = -self.local_critic(states, pred_actions).to("cpu").mean()
			
		######
		# soft update target NNs
		######
		self.target_critic = self.soft_update(self.target_critic, self.local_critic)
		self.target_actor = self.soft_update(self.target_actor, self.local_actor)
	

class ReplayBuffer():
	def __init__(self, batch_size, buffer_size, min_to_sample):
		self.sample_size = batch_size
		self.min_to_sample = max(min_to_sample, batch_size + 10)

		self.q = deque(maxlen=buffer_size)
		self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
	def add(self, state, action, reward, next_state, done):
		self.q.append(self.experience(state, action, reward, next_state, done))
	def sample(self):
		samples = random.sample(self.q, self.sample_size)

		# re-orgs the splits. samples is a sample of experience tuples "horizontal records" vs the alike characteristics separated out "vertical cols." Learning uses all states for example when calculating pred_actions with local_actor
		states = torch.from_numpy(np.vstack([tuple_t.state for tuple_t in samples if tuple_t is not None]))
		actions = torch.from_numpy(np.vstack([tuple_t.action for tuple_t in samples if tuple_t is not None]))
		rewards = torch.from_numpy(np.vstack([tuple_t.reward for tuple_t in samples if tuple_t is not None]))
		next_states = torch.from_numpy(np.vstack([tuple_t.state for tuple_t in samples if tuple_t is not None]))
		dones = torch.from_numpy(np.vstack([tuple_t.state for tuple_t in samples if tuple_t is not None]))

		return (states, actions, rewards, next_states, dones)
	def __len__(self):
		return len(self.q)	
