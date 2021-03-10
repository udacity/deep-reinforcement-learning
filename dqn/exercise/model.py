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
        self.seed = torch.manual_seed(seed)
        # hidden layer parameters: (hi, ho)
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        # Instanciate hidden_layers with input_layer
        input_layer = nn.Linear(state_size, hidden_layers[0])
        self.hidden_layers = nn.ModuleList([input_layer])
        self.hidden_layers.extend([nn.Linear(hi, ho) for hi, ho in layer_sizes])
        
        self.dropout = nn.Dropout(p=drop_p)
        # output layer: regression: a q_est for every action accessible from state s
        self.out = nn.Linear(hidden_layers[-1], action_size)


    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = state
        for fc in self.hidden_layers:
            x = fc(x)
            x = F.relu(x)
 
        x = self.out(x)
        
        return x
