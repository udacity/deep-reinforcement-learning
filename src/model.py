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


class ReinforcePolicyConv(nn.Module):

    def __init__(self, state_size, action_size, n_channels, seed):
        super(ReinforcePolicyConv, self).__init__()
        torch.manual_seed(seed)
        self.seed = seed
        self.state_size = state_size
        self.action_size = action_size
        self.n_channels = n_channels

        self.conv1 = nn.Conv2d(n_channels, 4, kernel_size=6, stride=2, bias=False)
        self.conv2 = nn.Conv2d(4, 16, kernel_size=6, stride=4)

        s1 = get_out_dims_convs(inputsize=state_size, kernel_size=6, stride=2)
        s2 = get_out_dims_convs(inputsize=s1, kernel_size=6, stride=4)
        self.size = 16 * s2 * s2  # channels x w x l

        # two fully connected layer
        self.fc1 = nn.Linear(self.size, 256)
        self.fc2 = nn.Linear(256, 1)

        # Sigmoid to
        self.sig = nn.Sigmoid()
        self.softm = nn.Softmax()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.size)
        x = F.relu(self.fc1(x))

        if self.action_size == 2:
            return self.sig(self.fc2(x))
        else:
            return self.softm(self.fc2(x))

def get_out_dims_convs(inputsize, kernel_size, stride):
    return round((inputsize - kernel_size + stride) / stride)


class ReinforcePolicy(nn.Module):

    def __init__(self, state_size, action_size, n_channels, seed):
        super(ReinforcePolicyConv, self).__init__()
        torch.manual_seed(seed)
        self.seed = seed
        self.state_size = state_size
        self.action_size = action_size

        # two fully connected layer
        hidden_units = 256
        self.fc1 = nn.Linear(self.state_size, hidden_units)
        if self.action_size == 2:
            self.fc2 = nn.Linear(hidden_units, 1)
        else:
            self.fc2 = nn.Linear(hidden_units, self.action_size)

        # Sigmoid to
        self.sig = nn.Sigmoid()
        self.softm = nn.Softmax()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.size)
        x = F.relu(self.fc1(x))

        if self.action_size == 2:
            return self.sig(self.fc2(x))
        else:
            return self.softm(self.fc2(x))

