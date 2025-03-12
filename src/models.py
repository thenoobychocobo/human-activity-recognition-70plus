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
   
    
class HarTransformerRyan(HarBaseModel):
    """Transformer-based model for Human Activity Recognition on HAR70+ dataset."""
    def __init__(
        self, 
        input_size: int = 6, 
        hidden_size: int = 30,
        num_layers: int = 2,  # Number of Transformer encoder layers
        num_heads: int = 2,   # Number of attention heads
        dropout_prob: float = 0.1, 
        num_classes: int = 7,
        device: Optional[torch.device] = None
    ):
        super(HarTransformer, self).__init__(input_size, hidden_size, num_layers, dropout_prob, num_classes, device)
        self.num_heads = num_heads
        
        # Embedding layer to project input features to hidden_size
        self.embedding = nn.Linear(self.input_size, self.hidden_size)
        
        # Positional encoding to inject sequence order information
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, self.hidden_size))  # Max sequence length = 1000
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            dropout=self.dropout_prob,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_size, num_classes)
        
        # Move model to device
        self.to(self.device)
        print(f"{type(self).__name__} model loaded on {self.device}.")

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer-based HAR model.

        Args:
            input_seq (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes).
        """
        batch_size, seq_len, _ = input_seq.shape
        
        # 1) Embed the input features
        embedded = self.embedding(input_seq)  # (batch_size, seq_len, hidden_size)
        
        # 2) Add positional encoding
        embedded += self.positional_encoding[:, :seq_len, :]
        
        # 3) Pass through Transformer encoder
        transformer_output = self.transformer_encoder(embedded)  # (batch_size, seq_len, hidden_size)
        
        # 4) Use the output of the final time step for classification
        final_output = transformer_output[:, -1, :]  # (batch_size, hidden_size)
        
        # 5) Pass through fully connected layer
        logits = self.fc(final_output)  # (batch_size, num_classes)
        
        return logits
    
class HarTransformer(nn.Module):
    """Transformer model for Human Activity Recognition on HAR70+ dataset."""
    def __init__(
        self, 
        input_dim: int = 6,
        # TODO: d_model and nhead are hyperparameters
        d_model: int = 30, 
        nhead: int = 3,
        num_layers: int = 1,
        num_classes = 7,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.device = device if device else torch.device("cpu")
        self.input_projection = nn.Linear(input_dim, d_model) 

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=self.nhead
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.num_layers,
        )
        self.fc = nn.Linear(self.d_model, self.num_classes)
        
        # Move model to device
        self.to(self.device)
        print(f"{type(self).__name__} model loaded on {self.device}.")

    def forward(self, input_seq):
        # Projection to the d_model dimensions
        projected_input = self.input_projection(input_seq)
        # Encodes information of hidden states for all timesteps where each hidden state for a timestep is affected by *all* other timesteps.
        encoded = self.transformer_encoder(projected_input) # The output (batch_size,seq_len,d_model) where each timestep has a size d_model of hidden states
        pooled = encoded.mean(dim=1) # We need to aggregate the sequence information into a single vector.
        output = self.fc(pooled)
        return output