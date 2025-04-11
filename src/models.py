####################
# Required Modules #
####################

# Generic/Built-in
from typing import *
from abc import ABC, abstractmethod

# Libs
import torch
import math
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
        num_classes: int = 7
    ):
        """
        Initializes the base HAR model.

        Args:
            input_size (int): Number of input features per time step.
            hidden_size (int): Dimensionality of the hidden state vector.
            num_layers (int): Number of recurrent layers.
            dropout_prob (float): Dropout probability (ignored if num_layers == 1).
            num_classes (int): Number of output classes.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob if num_layers > 1 else 0.0
        self.num_classes = num_classes
    
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
        num_classes = 7
    ):
        super().__init__(input_size, hidden_size, num_layers, dropout_prob, num_classes)
        self.rnn = nn.LSTM(
            self.input_size, self.hidden_size,
            num_layers=self.num_layers, 
            dropout=self.dropout_prob, 
            batch_first=True
        )
        self.fc = nn.Linear(self.hidden_size, self.num_classes)

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
        num_classes = 7
    ):
        super().__init__(input_size, hidden_size, num_layers, dropout_prob, num_classes)
        self.rnn = nn.GRU(
            self.input_size, self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout_prob,
            batch_first=True
        )
        self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, input_seq):
        # Map hidden state of final time step to prediction
        # Ignore all intermediate hidden states
        outputs, hn = self.rnn(input_seq) # hn is the hidden state of the final time step
        logits = self.fc(hn[-1]) # hn[-1] is the hidden state of the final layer for the final time step
        return logits
   
    
class HarTransformer(HarBaseModel):
    """Transformer-based model for Human Activity Recognition on HAR70+ dataset."""
    def __init__(
        self, 
        input_size: int = 6, 
        hidden_size: int = 30, # d_model
        num_layers: int = 2,  # Number of Transformer encoder layers
        num_heads: int = 2,   # Number of attention heads
        dropout_prob: float = 0.1, 
        num_classes: int = 7,
        max_sequence_length: int = 5000
    ):
        super(HarTransformer, self).__init__(input_size, hidden_size, num_layers, dropout_prob, num_classes)
        self.num_heads = num_heads
        
        # Embedding layer to project input features to hidden_size
        self.embedding = nn.Linear(self.input_size, self.hidden_size)
        # Note: nn.Embedding layer is usually used for embedding categorical variables
        
        # Positional encoding to inject sequence order information
        self.positional_encoder = SinusoidalPositionalEncoding(
            d_model=hidden_size, 
            dropout=dropout_prob, 
            max_len=max_sequence_length
        )
        
        # Single encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            dropout=self.dropout_prob,
            batch_first=True
        )
        
        # Stack of N encoders
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # Fully connected layer for classification
        # We use the embedding vector (hidden state) of the final time step to predict the class
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Transformer-based HAR model.

        Args:
            input_seq (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes).
        """
        
        # 1) Embed the input features
        embedded = self.embedding(input_seq) # (batch_size, seq_len, hidden_size)
        
        # 2) Add positional encoding
        embedded = self.positional_encoder(embedded)
        # embedded += self.positional_encoding[:, :seq_len, :]
        
        # 3) Pass through Transformer encoders (we get refined, contextualized time step embeddings)
        transformer_output = self.transformer_encoder(embedded) # (batch_size, seq_len, hidden_size)
        
        # 4) Use the output of the final time step for classification
        final_timestep_hidden_state = transformer_output[:, -1, :]  # (batch_size, hidden_size)
        
        # 5) Pass through fully connected layer
        logits = self.fc(final_timestep_hidden_state)  # (batch_size, num_classes)
        
        return logits


class SinusoidalPositionalEncoding(nn.Module):
    """
    Adds positional encodings to input sequences to inject information about token positions. 
    This class uses sinusoidal functions (sine and cosine) to generate encodings, following the original Transformer 
    architecture. The encodings are added to the input embeddings, and dropout is applied for regularization.
    
    Code taken from an 
    [official PyTorch tutorial](https://pytorch-tutorials-preview.netlify.app/beginner/transformer_tutorial.html)
    and adapted such that batch dimension comes first (batch_first = True).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # Transpose pe to [seq_len, 1, d_model] then add to x
        x = x + self.pe[:x.size(1)].transpose(0, 1)  # [1, seq_len, d_model]
        return self.dropout(x)