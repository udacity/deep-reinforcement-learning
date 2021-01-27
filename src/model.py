import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_layers=[64, 32], drop_p=0.2):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = seed
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

        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state
        for fc in self.hidden_layers:
            x = fc(x)
            x = F.relu(x)
            #x = self.dropout(x)
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

        self.dropout = nn.Dropout(p=self.drop_p)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state
        for fc in self.hidden_layers:
            x = fc(x)
            x = F.relu(x)
            #x = self.dropout(x)

        x_val = F.relu(
            self.out_val(x)
        ).expand(x.size(0), self.action_size)

        x_adv = F.relu(
            self.out_adv(x)
        )

        x_adv_mean = x_adv.mean()
        x_out = x_val + (x_adv - x_adv_mean)

        return x_out