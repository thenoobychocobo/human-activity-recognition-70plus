####################
# Required Modules #
####################

# Libs
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class HarLSTM(nn.Module):
    def __init__(
        self, 
        input_size: int = 6, 
        hidden_size: int = 30,
        num_layers: int = 1, 
        dropout_prob: float = 0.0, 
        num_classes = 7
    ):
        super(HarLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size, hidden_size, \
                           num_layers = num_layers, 
                           dropout = dropout_prob if num_layers > 1 else 0.0, 
                           batch_first = True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input_seq):
        # Map hidden state of final time step to prediction
        # Ignore all intermediate hidden states
        outputs, (hn, cn) = self.rnn(input_seq) # hn is the hidden state of the final time step
        logits = self.fc(hn[-1]) # hn[-1] is the hidden state of the final layer for the final time step
        return logits
    
def train_HAR70_models(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    num_epochs: int = 15,
    verbose: bool = True,
):    
    criterion = nn.CrossEntropyLoss()
    loss_history = []
    
    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        total_loss = 0
        
        for batch in train_dataloader:
            optimizer.zero_grad()
            
            # Unpack the mini-batch data
            sequence_batch, labels_batch = batch
            
            # Forward pass
            pred_logits = model(sequence_batch)
            loss = criterion(pred_logits, labels_batch)
            total_loss += loss.item()
            
            # Backward pass            
            loss.backward()
            optimizer.step()
            
        # After each epoch
        # 1) Save the total loss for the epoch to loss_history
        loss_history.append()