import torch
import torch.nn as nn
from base_models import TrafficPredictionModel

class GRUPredictor(TrafficPredictionModel):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.2):
        super(GRUPredictor, self).__init__(input_dim, hidden_dim, output_dim, num_layers)
        
        # The GRU layer handles the temporal sequence
        self.gru = nn.GRU(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout
        )
        
        # Fully connected layer to map hidden state to traffic volume
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        # Shape: (num_layers, batch_size, hidden_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate GRU
        out, _ = self.gru(x, h0)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out