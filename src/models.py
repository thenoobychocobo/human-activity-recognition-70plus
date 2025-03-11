####################
# Required Modules #
####################

# Generic/Built-in
from typing import *
from abc import ABC, abstractmethod

# Libs
import torch
import torch.nn as nn

class HarBaseModel(nn.Module, ABC):
    """
    Abstract base class for Human Activity Recognition (HAR) models, designed for HAR70+ dataset.
    """
    def __init__(
        self,
        input_size: int = 6,
        hidden_size: int = 30,
        num_layers: int = 1,
        dropout_prob: float = 0.0,
        num_classes: int = 7,
        device: Optional[torch.device] = None
    ):
        """
        Initializes the base HAR model.

        Args:
            input_size (int): Number of input features per time step.
            hidden_size (int): Dimensionality of the hidden state vector.
            num_layers (int): Number of recurrent layers.
            dropout_prob (float): Dropout probability (ignored if num_layers == 1).
            num_classes (int): Number of output classes.
            device (Optional[torch.device]): Device to run the model on (CPU or GPU).
        """
        super(HarBaseModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob if num_layers > 1 else 0.0
        self.num_classes = num_classes
        
        # Set up device
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @abstractmethod
    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            input_seq (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes).
        """
        pass    

class HarLSTM(HarBaseModel):
    """LSTM-based model for Human Activity Recognition on HAR70+ dataset."""
    def __init__(
        self, 
        input_size: int = 6, 
        hidden_size: int = 30,
        num_layers: int = 1, 
        dropout_prob: float = 0.0, 
        num_classes = 7,
        device: Optional[torch.device] = None
    ):
        super(HarLSTM, self).__init__(input_size, hidden_size, num_layers, dropout_prob, num_classes, device)
        self.rnn = nn.LSTM(
            self.input_size, self.hidden_size,
            num_layers=self.num_layers, 
            dropout=self.dropout_prob, 
            batch_first=True
        )
        self.fc = nn.Linear(self.hidden_size, self.num_classes)
        
        # Move model to device
        self.to(self.device)
        print(f"{type(self).__name__} model loaded on {self.device}.")

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        # Map hidden state of final time step to prediction
        # Ignore all intermediate hidden states
        outputs, (hn, cn) = self.rnn(input_seq) # hn is the hidden states of the final time step
        logits = self.fc(hn[-1]) # hn[-1] is the hidden state of the final layer for the final time step
        return logits
    
class HarGRU(HarBaseModel):
    """GRU-based model for Human Activity Recognition on HAR70+ dataset."""
    def __init__(
        self, 
        input_size: int = 6, 
        hidden_size: int = 30,
        num_layers: int = 1, 
        dropout_prob: float = 0.0, 
        num_classes = 7,
        device: Optional[torch.device] = None
    ):
        super(HarGRU, self).__init__(input_size, hidden_size, num_layers, dropout_prob, num_classes, device)
        self.rnn = nn.GRU(
            self.input_size, self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout_prob,
            batch_first=True
        )
        self.fc = nn.Linear(self.hidden_size, self.num_classes)
        
        # Move model to device
        self.to(self.device)
        print(f"{type(self).__name__} model loaded on {self.device}.")

    def forward(self, input_seq):
        # Map hidden state of final time step to prediction
        # Ignore all intermediate hidden states
        outputs, hn = self.rnn(input_seq) # hn is the hidden state of the final time step
        logits = self.fc(hn[-1]) # hn[-1] is the hidden state of the final layer for the final time step
        return logits