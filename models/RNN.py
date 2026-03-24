import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x):
        # Input shape : [B, 49, 2048]
        _, (h_n, c_n) = self.lstm(x)
        # h_n shape: [4, B, hidden_dim]

        # h_n[-2] is the forward direction of the last layer
        forward_hidden = h_n[-2, :, :] # Shape: [B, hidden_dim]
        # h_n[-1] is the backward direction of the last layer
        backward_hidden = h_n[-1, :, :] # Shape: [B, hidden_dim]

        final_hidden = torch.cat((forward_hidden, backward_hidden), dim=1) # output shape: [B, hidden_dim * 2]

        return final_hidden