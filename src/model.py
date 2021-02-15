import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_layers=[64, 32], drop_p=0.2):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_layers (list of int): Number of activation units on each layer
            drop_p (float): dropout probability to apply to each Dropout layer
        """
        super(QNetwork, self).__init__()
        self.seed = seed
        self.drop_p = drop_p
        torch.manual_seed(seed)

        # hidden layer parameters: (hi, ho)
        if len(hidden_layers) == 1:
            layer_sizes = hidden_layers[0]
        else:
            layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])

        # Instanciate hidden_layers with input_layer
        input_layer = nn.Linear(state_size, hidden_layers[0])
        self.hidden_layers = nn.ModuleList([input_layer])
        self.hidden_layers.extend([nn.Linear(hi, ho) for hi, ho in layer_sizes])

        # output layer: regression: a q_est for every action accessible from state s
        self.out = nn.Linear(hidden_layers[-1], action_size)

        if self.drop_p is not None:
            self.dropout = nn.Dropout(p=self.drop_p)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state
        for fc in self.hidden_layers:
            x = fc(x)
            x = F.relu(x)
            if self.drop_p is not None:
                x = self.dropout(x)
        x = self.out(x)
        
        return x


class DuelQNetwork(nn.Module):
    """Actor (Policy) Model.
    Implement Dueling Q-Network
    input layer is 2d (batch_size x  state_size) tensor
    """

    def __init__(self, state_size, action_size, seed, hidden_layers=[64, 32], drop_p=0.2):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_layers (list of int): Number of activation units on each layer
            drop_p (float): dropout probability to apply to each Dropout layer
        """
        super(DuelQNetwork, self).__init__()
        torch.manual_seed(seed)
        self.seed = seed
        self.action_size = action_size
        self.drop_p = drop_p
        # hidden layer parameters: (hi, ho)
        if len(hidden_layers) == 1:
            layer_sizes = [(hidden_layers[0], hidden_layers[0])]
        else:
            layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        # Instanciate hidden_layers with input_layer
        input_layer = nn.Linear(state_size, hidden_layers[0])
        self.hidden_layers = nn.ModuleList([input_layer])
        self.hidden_layers.extend([nn.Linear(hi, ho) for hi, ho in layer_sizes])

        # output layer:
        # value stream
        # advantage stream
        # regression: a q_est for every action accessible from state s
        self.out_val = nn.Linear(hidden_layers[-1], 1)
        self.out_adv = nn.Linear(hidden_layers[-1], action_size)

        if self.drop_p is not None:
            self.dropout = nn.Dropout(p=self.drop_p)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state
        for fc in self.hidden_layers:
            x = fc(x)
            x = F.relu(x)
            if self.drop_p is not None:
                x = self.dropout(x)

        x_val = F.relu(
            self.out_val(x)
        ).expand(x.size(0), self.action_size)

        x_adv = F.relu(
            self.out_adv(x)
        )

        x_adv_mean = x_adv.mean()
        x_out = x_val + (x_adv - x_adv_mean)

        return x_out


class DDPGActor(nn.Module):
    """
    Actor (Policy) Model.
    mu(states; theta_mu), output is a vector of dim action_size, where each element is a float,
    e.g. controlling an elbow, requires a single metric (flexion torque),
        controlling a wrist, requires: (rotation torque, flexion torque)
    states should be normalized prior to input
    output layer is a tanh to be in [-1,+1]
    """

    def __init__(self, state_size, action_size, seed, fc1_units=24, fc2_units=48):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): state space dimension
            action_size (int): action space dimension, e.g. controlling an elbow requires an action of 1 dimension
                (torque), action vector should be normalized
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(DDPGActor, self).__init__()
        self.seed = seed
        torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)

        # output layer: regression, a deterministic policy mu at state s
        self.out = nn.Linear(fc2_units, action_size)

        self.reset_parameters()

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = state
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.out(x))

        return x

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.out.weight.data.uniform_(-3e-3, 3e-3)


class DDPGCritic(nn.Module):
    """
    Critic (Value) Model.
    Q(states, mu(states; theta_mu); theta_Q)
    state is inputted in the first layer, the second layer takes this output and actions
    see DDPGActor to check mu(states; theta_mu)
    states should be normalized prior to input
    """

    def __init__(self, state_size, action_size, seed, fcs1_units=24, fc2_units=48):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): state space dimension
            action_size (int): action space dimension, e.g. controlling an elbow requires an action of 1 dimension
                (torque), action vector should be normalized
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(DDPGCritic, self).__init__()
        self.seed = seed
        torch.manual_seed(seed)

        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units + action_size, fc2_units)

        # output layer: regression: a q_est(s)
        self.out = nn.Linear(fc2_units, 1)

        self.reset_parameters()

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = state
        xs = F.relu(self.fcs1(xs))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.out.weight.data.uniform_(-3e-3, 3e-3)


class DDPG(nn.Module):
    """
    Critic (Value) Model.
    Q(states, mu(states; theta_mu); theta_Q)
    state is inputted in the first layer, the second layer takes this output and actions
    see DDPGActor to check mu(states; theta_mu)
    states should be normalized prior to input
    """

    def __init__(self, state_size, action_size, seed, fc_phi_units=24, fc1_actor_units=48, fc1_critic_units=48):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): state space dimension
            action_size (int): action space dimension, e.g. controlling an elbow requires an action of 1 dimension
                (torque), action vector should be normalized
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(DDPG, self).__init__()
        self.seed = seed
        torch.manual_seed(seed)

        self.phi_body = nn.Linear(state_size, fc_phi_units)
        self.actor_body = nn.Linear(fc_phi_units, fc1_actor_units)
        self.critic_body = nn.Linear(fc_phi_units + action_size, fc1_critic_units)

        self.out_action = nn.Linear(fc1_actor_units, action_size)
        self.out_critic = nn.Linear(fc1_critic_units, 1)

        self.reset_parameters()

    def forward(self, state):
        phi = F.relu(self.feature(state))
        action = F.relu(self.actor(phi))

        return action

    def feature(self, state):

        return F.relu(self.phi_body(state))

    def actor(self, phi):

        x = F.relu(self.actor_body(phi))
        return torch.tanh(self.out_action(x))

    def critic(self, phi, action):
        x = torch.cat([phi, action], dim=1)
        x = F.relu(self.critic_body(x))
        return self.out_critic(x)

    def reset_parameters(self):
        self.phi_body.weight.data.uniform_(*hidden_init(self.phi_body))
        self.actor_body.weight.data.uniform_(*hidden_init(self.actor_body))
        self.critic_body.weight.data.uniform_(*hidden_init(self.critic_body))
        self.out_action.weight.data.uniform_(-3e-3, 3e-3)
        self.out_critic.weight.data.uniform_(-3e-3, 3e-3)


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)