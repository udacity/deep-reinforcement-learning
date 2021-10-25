# with reference to https://github.com/udacity/deep-reinforcement-learning/blob/master/ddpg-pendulum/model.py
import numpy as np

import torch 
import torch.nn as nn
import torch.nn.functional as F

# paper initialized this way
def get_normalization_range(layer):
	size_s = layer.weight.data.size()[0]
	lim = 1. / np.sqrt(size_s)
	return (-lim, lim)

# instantiate one for the regular network & one for the target network
# input:  state
# output: action
class Actor(nn.Module):
	# TODO: maybe use a random seed
	def __init__(self, n_states, n_actions, hidden1_size, hidden2_size, dropout_p=0):
		super().__init__()
		self.output_layer_uniform_limit = .003
		self.n_states = n_states
		self.n_actions = n_actions
		self.hidden1_size = hidden1_size
		self.hidden2_size = hidden2_size

		self.dropout_p = dropout_p
		self.dropout_layer = nn.Dropout(dropout_p)

		self.hidden1 = nn.Linear(self.n_states, self.hidden1_size)
		self.hidden2 = nn.Linear(self.hidden1_size, self.hidden2_size)
		self.output_layer = nn.Linear(self.hidden2_size, self.n_actions)

		# dropout layer doesn't have weight attribute which makes sense
		self.layers_dyn_reset = [self.hidden1, self.hidden2]
	
		self.reset_params()

	def reset_params(self):
		for layer in self.layers_dyn_reset:
			layer.weight.data.uniform_(*get_normalization_range(layer))
		self.output_layer.weight.data.uniform_(- self.output_layer_uniform_limit, self.output_layer_uniform_limit)
	def forward(self, state):
		x = self.dropout_layer(F.tanh(self.hidden1(state)))
		x = self.dropout_layer(F.relu(self.hidden2(x)))
		# action range -1, 1
		x = F.tanh(self.output_layer(x))
		return x

# input:  (state, action) pair
# output: Q(s,a) values
class Critic(nn.Module):
	# TODO: maybe use a random seed
	def __init__(self, n_states, n_actions, hidden1_size, hidden2_size, dropout_p=0):
		super().__init__()
		self.output_layer_uniform_limit = .003
		self.n_states = n_states
		self.n_actions = n_actions
		self.hidden1_size = hidden1_size
		self.hidden2_size = hidden2_size

		self.dropout_p = dropout_p
		self.dropout_layer = nn.Dropout(dropout_p)

		self.hidden1 = nn.Linear(self.n_states, self.hidden1_size)
		# paper: Actions were not included until the 2nd hidden layer of Q
		self.hidden2 = nn.Linear(self.hidden1_size + self.n_actions, self.hidden2_size)
		# look at one (state, action) pair at a time so one Q value
		self.output_layer = nn.Linear(self.hidden2_size, 1)

		self.layers_dyn_reset = [self.hidden1, self.hidden2]
	
		self.reset_params()

	def reset_params(self):
		for layer in self.layers_dyn_reset:
			layer.weight.data.uniform_(*get_normalization_range(layer))
		self.output_layer.weight.data.uniform_(- self.output_layer_uniform_limit, self.output_layer_uniform_limit)
	
	def forward(self, state, action):
		x = self.dropout_layer(F.relu(self.hidden1(state)))
		# want the action info to be in the same record (row) running through the NN
		x = torch.cat((x, action), dim=1)
		x = self.dropout_layer(F.relu(self.hidden2(x)))
		x = F.relu(self.output_layer(x))
		# or return index of highest x?
		return x
	

