import torch
import torch.nn as nn

class VanillaEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, \
                 num_layers = 1, dropout = 0.0, num_classes = 7):
        super(VanillaEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size, hidden_size, \
                           num_layers = num_layers, 
                           dropout = dropout if num_layers > 1 else 0.0, 
                           batch_first = True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input_seq):
        # For this encoder, we ignore the outputs 
        # We use the final hidden state for the calculation of the logits
        # TODO: some function that will extract actual hidden state, output could only be a function fo hidden state
        outputs, (hn, _) = self.rnn(input_seq)
        logits = self.fc(hn[-1])
        return logits