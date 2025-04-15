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
from torchtune.modules import RotaryPositionalEmbeddings # pip install torchao torchtune


##############
# Base Class #
##############

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
            input_size (int, optional): Number of input features per time step. Defaults to 6.
            hidden_size (int, optional): Dimensionality of the hidden state vector. Defaults to 30.
            num_layers (int, optional): Number of recurrent layers. Defaults to 1.
            dropout_prob (float, optional): Dropout probability (ignored if num_layers == 1). Defaults to 0.0.
            num_classes (int, optional): Number of output classes. Defaults to 7.
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


##############
# RNN Models #
##############

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
        """
        Initializes the LSTM HAR model.

        Args:
            input_size (int, optional): Number of input features per time step. Defaults to 6.
            hidden_size (int, optional): Dimensionality of the hidden state vector. Defaults to 30.
            num_layers (int, optional): Number of recurrent layers. Defaults to 1.
            dropout_prob (float, optional): Dropout probability (ignored if num_layers == 1). Defaults to 0.0.
            num_classes (int, optional): Number of output classes. Defaults to 7.
        """
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
        """
        Initializes the GRU HAR model.

        Args:
            input_size (int, optional): Number of input features per time step. Defaults to 6.
            hidden_size (int, optional): Dimensionality of the hidden state vector. Defaults to 30.
            num_layers (int, optional): Number of recurrent layers. Defaults to 1.
            dropout_prob (float, optional): Dropout probability (ignored if num_layers == 1). Defaults to 0.0.
            num_classes (int, optional): Number of output classes. Defaults to 7.
        """
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
   

######################
# Transformer Models #
######################

class HarTransformer(HarBaseModel):
    """Transformer-based model for Human Activity Recognition on HAR70+ dataset."""
    def __init__(
        self, 
        input_size: int = 6, 
        hidden_size: int = 32, 
        dim_feedforward: Optional[int] = None, 
        num_layers: int = 2, 
        num_heads: int = 2, 
        dropout_prob: float = 0.1, 
        num_classes: int = 7,
        max_sequence_length: int = 5000
    ):
        """
        Initializes the Transformer HAR model.

        Args:
            input_size (int, optional): Number of input features per time step. Defaults to 6.
            hidden_size (int, optional): Dimensionality of the hidden state vector i.e. embedding size (d_model). 
                Defaults to 32.
            dim_feedforward (Optional[int], optional): Number of units in hidden layers of feed-forward network (d_ff). 
                Defaults to None, in which case it will initialize to hidden_size * 4.
            num_layers (int, optional): Number of Transformer encoder layers. Defaults to 2.
            num_heads (int, optional): Number of attention heads. Defaults to 2.
            dropout_prob (float, optional): Dropout probability. Defaults to 0.1.
            num_classes (int, optional): Number of output classes. Defaults to 7.
            max_sequence_length (int, optional): Specifies the maximum sequence length for positional encoding. 
                Defaults to 5000.
        """
        super().__init__(input_size, hidden_size, num_layers, dropout_prob, num_classes)
        self.dim_feedforward = dim_feedforward or self.hidden_size * 4
        self.num_heads = num_heads
        self.max_sequence_length = max_sequence_length
        
        # Embedding layer to project input features to hidden_size
        self.embedding = nn.Linear(self.input_size, self.hidden_size)
        # Note: nn.Embedding layer is usually used for embedding categorical variables
        
        # Add learnable CLS token (inspired by BERT): shape (1, 1, hidden_size) will be expanded per batch
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        
        # Positional encoding to inject sequence order information
        self.positional_encoder = SinusoidalPositionalEncoding(
            d_model=self.hidden_size, 
            dropout=self.dropout_prob, 
            max_len=self.max_sequence_length
        )
        
        # Single encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout_prob,
            batch_first=True
        )
        
        # Stack of N encoders
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # Fully connected layer for classification
        # We use the embedding vector (hidden state) of the final time step to predict the class
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        # 1) Embed the input features
        embedded = self.embedding(input_seq) # (batch_size, seq_len, hidden_size)
        
        # 2) Prepend CLS token to each input sequence: trained to capture sequence-level information
        batch_size = input_seq.size(0)
        cls_tokens = self.cls_token.expand(batch_size, 1, -1) # (batch_size, 1, hidden_size)
        embedded = torch.cat([cls_tokens, embedded], dim=1) # (batch_size, seq_len+1, hidden_size)
        
        # 3) Add positional encoding
        embedded = self.positional_encoder(embedded)
        
        # 4) Pass through Transformer encoders (we get refined, contextualized time step embeddings)
        transformer_output = self.transformer_encoder(embedded) # (batch_size, seq_len, hidden_size)
        
        # 5) Use the CLS token as input to classification network
        cls_output = transformer_output[:, 0, :]  # (batch_size, hidden_size)
        
        # 5) Pass through fully connected layer
        logits = self.fc(cls_output)  # (batch_size, num_classes)
        
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


class HarTransformerExperimental(HarBaseModel):
    """
    Transformer-based model for Human Activity Recognition on HAR70+ dataset with 2 experimental features:
    Rotary Positional Encoding (RoPE) and a 1D CNN as a tokenizer. 
    """
    def __init__(
        self, 
        input_size: int = 6, 
        hidden_size: int = 32,
        dim_feedforward: Optional[int] = None,
        num_layers: int = 2,
        num_heads: int = 2,
        dropout_prob: float = 0.1, 
        num_classes: int = 7,
        cnn_kernel_sizes: Tuple[int] = (5, 3),
        cnn_stride: int = 1,
        cnn_padding: int = 1,
        max_sequence_length: int = 5000
    ):
        """
        Initializes the Transformer HAR model (experimental).

        Args:
            input_size (int, optional): Number of input features per time step. Defaults to 6.
            hidden_size (int, optional): Dimensionality of the hidden state vector i.e. embedding size (d_model). 
                Defaults to 32.
            dim_feedforward (Optional[int], optional): Number of units in hidden layers of feed-forward network (d_ff). 
                Defaults to None, in which case it will initialize to hidden_size * 4.
            num_layers (int, optional): Number of Transformer encoder layers. Defaults to 2.
            num_heads (int, optional): Number of attention heads. Defaults to 2.
            dropout_prob (float, optional): Dropout probability. Defaults to 0.1.
            num_classes (int, optional): Number of output classes. Defaults to 7.
            cnn_kernel_sizes (Tuple[int], optional): Kernel sizes for CNN layer. Defaults to (5, 3).
            cnn_stride (int, optional): Kernel stride for CNN layer. Defaults to 1.
            cnn_padding (int, optional): Padding for CNN layer. Defaults to 1.
            max_sequence_length (int, optional): Specifies the maximum sequence length for positional encoding. 
                Defaults to 5000.
        """
        super().__init__(input_size, hidden_size, num_layers, dropout_prob, num_classes)
        self.dim_feedforward = dim_feedforward or self.hidden_size * 4
        self.num_heads = num_heads
        self.cnn_kernel_sizes = cnn_kernel_sizes
        self.cnn_stride = cnn_stride
        self.cnn_padding = cnn_padding
        self.max_sequence_length = max_sequence_length
        
        # 1D CNN as tokenizer (output used as embeddings)
        self.cnn_tokenizer = nn.Sequential(
            # CNN should not downsample too much as the input sequence length is not very long
            nn.Conv1d(
                in_channels=self.input_size, 
                out_channels=self.hidden_size, # Project to hidden_size (d_model)
                kernel_size=self.cnn_kernel_sizes[0],
                stride=self.cnn_stride,
                padding=self.cnn_padding
            ),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_size),
            nn.Conv1d(
                in_channels=self.hidden_size, 
                out_channels=self.hidden_size,
                kernel_size=self.cnn_kernel_sizes[1],
                stride=self.cnn_stride,
                padding=self.cnn_padding
            ),
            nn.ReLU(),
            nn.BatchNorm1d(self.hidden_size),
            nn.Dropout(self.dropout_prob),
        )
        
        # Add learnable CLS token (inspired by BERT): shape (1, 1, hidden_size) will be expanded per batch
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        
        # Stack of N encoders
        # Rotary positional encoding is implemented within the attention layers
        encoder_layers = nn.ModuleList([
            self.RoPETransformerEncoderLayer(
                embed_dim=self.hidden_size,
                num_heads=self.num_heads,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout_prob,
                max_sequence_length=self.max_sequence_length
            )
            for _ in range(self.num_layers)
        ])
        self.transformer_encoder = nn.Sequential(*encoder_layers)
        
        # Fully connected layer for classification
        # We use the embedding vector (hidden state) of the final time step to predict the class
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input_seq: torch.Tensor) -> torch.Tensor:
        # 1) Tokenize using CNN
        cnn_input = input_seq.permute(0, 2, 1) # (batch_size, input_size, seq_len)
        cnn_output = self.cnn_tokenizer(cnn_input) # (batch_size, hidden_size, seq_len)
        embedded = cnn_output.permute(0, 2, 1) # (batch_size, seq_len, hidden_size)
        
        # 2) Prepend CLS token to each input sequence: trained to capture sequence-level information
        batch_size = input_seq.size(0)
        cls_tokens = self.cls_token.expand(batch_size, 1, -1) # (batch_size, 1, hidden_size)
        embedded = torch.cat([cls_tokens, embedded], dim=1) # (batch_size, seq_len+1, hidden_size)
        
        # 3) Pass through Transformer encoders (we get refined, contextualized time step embeddings)
        for encoder_layer in self.transformer_encoder:
            embedded = encoder_layer(embedded)
            
        # 4) Use the CLS token as input to classification network
        cls_output = embedded[:, 0, :]  # (batch_size, hidden_size)
        
        # 5) Pass through fully connected layer
        logits = self.fc(cls_output)  # (batch_size, num_classes)
        
        return logits
    
    class RoPETransformerEncoderLayer(nn.Module):
        def __init__(self, embed_dim, num_heads, dim_feedforward, dropout, max_sequence_length):
            super().__init__()
            self.self_attn = self.RoPEMultiheadAttention(embed_dim, num_heads, max_sequence_length, dropout)

            self.linear1 = nn.Linear(embed_dim, dim_feedforward)
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(dim_feedforward, embed_dim)

            self.norm1 = nn.LayerNorm(embed_dim)
            self.norm2 = nn.LayerNorm(embed_dim)

            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
            self.activation = nn.ReLU()

        def forward(self, src):
            # Self-attention block
            attn_output = self.self_attn(src)
            src = src + self.dropout1(attn_output)
            src = self.norm1(src)

            # Feedforward block
            ff_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(ff_output)
            src = self.norm2(src)

            return src

        class RoPEMultiheadAttention(nn.Module):
            def __init__(self, embed_dim, num_heads, max_sequence_length, dropout=0.0):
                super().__init__()
                self.embed_dim = embed_dim
                self.num_heads = num_heads
                self.head_dim = embed_dim // num_heads
                self.max_sequence_length = max_sequence_length

                assert self.head_dim * num_heads == embed_dim, "embed_dim (i.e. hidden_size) must be divisible by num_heads"
                assert self.head_dim % 2 == 0, "For RoPE, (head_dim = embed_dim / num_heads) must be even"

                self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
                self.out_proj = nn.Linear(embed_dim, embed_dim)
                self.dropout = nn.Dropout(dropout)

                # RoPE initialized per head dimension
                self.rotary_emb = RotaryPositionalEmbeddings(dim=self.head_dim, max_seq_len=self.max_sequence_length)

            def forward(self, x):
                B, T, C = x.size()  # (batch, seq_len, embed_dim)

                qkv = self.qkv_proj(x)  # (B, T, 3 * C)
                qkv = qkv.view(B, T, 3, self.num_heads, self.head_dim)  # (B, T, 3, H, D)
                qkv = qkv.permute(2, 0, 1, 3, 4)  # (3, B, T, H, D)
                q, k, v = qkv[0], qkv[1], qkv[2]  # Each: (B, T, H, D)

                # Apply Rotary Positional Embeddings
                q = self.rotary_emb(q)  # (B, T, H, D)
                k = self.rotary_emb(k)  # (B, T, H, D)

                # Transpose to standard attention shape
                q = q.permute(0, 2, 1, 3)  # (B, H, T, D)
                k = k.permute(0, 2, 1, 3)
                v = v.permute(0, 2, 1, 3)

                # Scaled dot-product attention
                attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, T, T)
                attn_weights = torch.softmax(attn_scores, dim=-1)
                attn_weights = self.dropout(attn_weights)

                context = torch.matmul(attn_weights, v)  # (B, H, T, D)
                context = context.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, embed_dim)

                return self.out_proj(context)