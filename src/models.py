####################
# Required Modules #
####################

# Generic/Built-in
from typing import *

# Libs
import torch
import torch.nn as nn

class HarLSTM(nn.Module):
    def __init__(
        self, 
        input_size: int = 6, 
        hidden_size: int = 30,
        num_layers: int = 1, 
        dropout_prob: float = 0.0, 
        num_classes = 7,
        device: Optional[torch.device] = None
    ):
        super(HarLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size, hidden_size, \
                           num_layers = num_layers, 
                           dropout = dropout_prob if num_layers > 1 else 0.0, 
                           batch_first = True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
        # Set up device
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.to(self.device)

    def forward(self, input_seq):
        # Map hidden state of final time step to prediction
        # Ignore all intermediate hidden states
        outputs, (hn, cn) = self.rnn(input_seq) # hn is the hidden state of the final time step
        logits = self.fc(hn[-1]) # hn[-1] is the hidden state of the final layer for the final time step
        return logits