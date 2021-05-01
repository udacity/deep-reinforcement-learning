import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        self.state_size = state_size
        self.action_size = action_size
        self.start_in_channel = 1
        self.last_conv_out_channel = self.start_in_channel * 32
        # + 1 is because padding scheme will change that dim from size state_size to state_size + 1
        self.hiddenfc_in = self.last_conv_out_channel * (self.state_size + 1)
        self.hiddenfc_out = math.floor(self.hiddenfc_in * 2/3)
        self.n_conv_layers = 3
        # cite: https://discuss.pytorch.org/t/whats-the-difference-between-nn-relu-vs-f-relu/27599
        # "nn.ReLU() creates an nn.Module which you can add e.g. to an nn.Sequential model.nn.functional.relu on the other side is just the functional API call to the relu function, so that you can add it e.g. in your forward method yourself."
        # padding of 0 on conv layers made input's dim that had 8 shrink by 1 each time
        # padding of 1 on conv layers made input's dim that had 8 grow + 1 each time
        self.conv1 = nn.Conv1d(self.start_in_channel, self.start_in_channel * 4, kernel_size=2, padding=1)
        self.conv2 = nn.Conv1d(self.start_in_channel * 4, self.start_in_channel * 16, kernel_size=2, padding=0)
        self.conv3 = nn.Conv1d(self.start_in_channel * 16, self.last_conv_out_channel, kernel_size=2, padding=1)
        self.hiddenfc = nn.Linear(self.hiddenfc_in, self.hiddenfc_out)
        self.finalfc = nn.Linear(self.hiddenfc_out, self.action_size)
      
        
            
        

    def forward(self, state):
        """Build a network that maps state -> action values."""
        # try 1 additional unsqueeze's - to get it to 2 dim (channel/length dim- creating w/ unsqueeze cuz unsqeeze(1)- & height; no width)
        # already had batch_size dim created by unsqueeze(0) -> to keep batch_size at index 0 & create channel dim, unsqueeze(1)
        batch_size = state.shape[0]
        state = state.unsqueeze(1)
        out = F.relu(self.conv1(state))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        # reshape
        # input to linear layer should be batch_size (by any number of additional dimensions) by in_dim of layer
        out = out.view(batch_size, -1)
        out = F.relu(self.hiddenfc(out))
        # currently no activation after last layer -> logits not probabilities
        # picks the argmax so, for agent.act, doesn't matter if converted into probabilities
        # loss compares output of q_network_local & q_network_target so might matter to loss
        #    need q values (or maybe at least proportional to q values)
        out = self.finalfc(out)
        return out
