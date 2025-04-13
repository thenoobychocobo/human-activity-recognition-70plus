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
        max_sequence_length: int = 5000,
        device: Optional[torch.device] = None
    ):
        super(HarTransformer, self).__init__(input_size, hidden_size, num_layers, dropout_prob, num_classes, device)
        self.num_heads = num_heads
        
        # Embedding layer to project input features to hidden_size
        self.embedding = nn.Linear(self.input_size, self.hidden_size)
        # Note: nn.Embedding layer is usually used for embedding categorical variables
        
        # Positional encoding to inject sequence order information
        self.positional_encoder = PositionalEncoding(
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
    
class PositionalEncoding(nn.Module):
    """
    Adds positional encodings to input sequences to inject information about token positions. 
    This class uses sinusoidal functions (sine and cosine) to generate encodings, following the original Transformer 
    architecture. The encodings are added to the input embeddings, and dropout is applied for regularization.
    
    Code taken from an 
    [official PyTorch tutorial](https://pytorch-tutorials-preview.netlify.app/beginner/transformer_tutorial.html).
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
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class HarCnnTransformer(HarBaseModel):
    """Hybrid CNN-Transformer model for Human Activity Recognition on HAR70+ dataset.
    Uses 1D CNN as a feature extractor before passing data to a Transformer for sequence modeling.
    """
    def __init__(
        self, 
        input_size: int = 6, 
        cnn_hidden_channels: int = 64,
        cnn_kernel_size: int = 15,
        cnn_layers: int = 2,
        transformer_hidden_size: int = 30,  # d_model
        transformer_num_layers: int = 2,    # Number of Transformer encoder layers
        transformer_num_heads: int = 2,     # Number of attention heads
        dropout_prob: float = 0.1, 
        num_classes: int = 7,
        max_sequence_length: int = 5000,
        device: Optional[torch.device] = None
    ):
        super(HarCnnTransformer, self).__init__(input_size, transformer_hidden_size, 
                                               transformer_num_layers, dropout_prob, 
                                               num_classes, device)
        self.transformer_num_heads = transformer_num_heads
        self.cnn_hidden_channels = cnn_hidden_channels
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_layers = cnn_layers
        
        # 1D CNN Feature Extractor/Tokenizer
        self.cnn_layers_list = nn.ModuleList()
        
        # First CNN layer (input channels -> hidden channels)
        self.cnn_layers_list.append(nn.Sequential(
            nn.Conv1d(self.input_size, self.cnn_hidden_channels, kernel_size=self.cnn_kernel_size, padding='same'),
            nn.BatchNorm1d(self.cnn_hidden_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)  # Downsample sequence length by factor of 2
        ))
        
        # Additional CNN layers (hidden channels -> hidden channels)
        for _ in range(1, self.cnn_layers):
            self.cnn_layers_list.append(nn.Sequential(
                nn.Conv1d(self.cnn_hidden_channels, self.cnn_hidden_channels, kernel_size=self.cnn_kernel_size, padding='same'),
                nn.BatchNorm1d(self.cnn_hidden_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2)  # Further downsample sequence length
            ))
        
        # Linear projection from CNN output to transformer hidden size
        self.cnn_to_transformer = nn.Linear(self.cnn_hidden_channels, self.hidden_size)
        
        # Positional encoding to inject sequence order information
        self.positional_encoder = PositionalEncoding(
            d_model=self.hidden_size, 
            dropout=self.dropout_prob, 
            max_len=max_sequence_length
        )
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.transformer_num_heads,
            dropout=self.dropout_prob,
            batch_first=True
        )
        
        # Stack of N transformer encoders
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # Fully connected layer for classification
        self.fc = nn.Linear(self.hidden_size, self.num_classes)
        
        # Move model to device
        self.to(self.device)
        print(f"{type(self).__name__} model loaded on {self.device}.")
        
    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN-Transformer hybrid HAR model.

        Args:
            input_seq (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes).
        """
        batch_size, seq_len, _ = input_seq.shape
        
        # 1) Transpose for CNN: from (batch, seq_len, features) to (batch, features, seq_len)
        x = input_seq.transpose(1, 2)  # Now shape: (batch_size, input_size, seq_len)
        
        # 2) Apply CNN layers for feature extraction
        for cnn_layer in self.cnn_layers_list:
            x = cnn_layer(x)
            
        # Get the new sequence length after CNN pooling operations
        _, _, new_seq_len = x.shape
        
        # 3) Transpose back to (batch, seq_len, channels) for the transformer
        x = x.transpose(1, 2)  # Now shape: (batch_size, new_seq_len, cnn_hidden_channels)
        
        # 4) Project CNN features to transformer dimension
        x = self.cnn_to_transformer(x)  # Now shape: (batch_size, new_seq_len, hidden_size)
        
        # 5) Add positional encoding
        x = self.positional_encoder(x)
        
        # 6) Pass through Transformer encoders
        transformer_output = self.transformer_encoder(x)  # (batch_size, new_seq_len, hidden_size)
        
        # 7) Use the output of the final time step for classification
        final_timestep_hidden_state = transformer_output[:, -1, :]  # (batch_size, hidden_size)
        
        # 8) Pass through fully connected layer
        logits = self.fc(final_timestep_hidden_state)  # (batch_size, num_classes)
        
        return logits