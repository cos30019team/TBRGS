import torch
import torch.nn as nn
import numpy as np

class GCNLayer(nn.Module):
    """
    A proper Graph Convolution Layer.
    This handles the SPATIAL math (The Map).
    """
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        # The learnable weights for the spatial connections
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight) # Initialize weights to random small numbers

    def forward(self, A, X):
        # A: The 140x140 Adjacency Matrix
        # X: The Traffic Data (Batch, 140, Features)
        
        # Step 1: Multiply Traffic Data by our Learnable Weights (X * W)
        support = torch.matmul(X, self.weight)
        
        # Step 2: Message Passing!
        # Multiply the Map (A) by the Weighted Data to blend neighbor traffic
        out = torch.matmul(A, support)
        return out

class TrafficGNN(nn.Module):
    def __init__(self, num_nodes=140, timesteps=4, hidden_size=64):
        super(TrafficGNN, self).__init__()
        
        # 1. The Temporal Engine (GRU)
        # We use 64 hidden units and batch_first=True 
        # GRU is structurally similar to LSTM but slightly faster for routing calculations.
        self.temporal = nn.GRU(input_size=1, hidden_size=hidden_size, batch_first=True)
        
        # 2. Dropout Guardrail 
        # Randomly turns off 20% of the neurons during training to prevent memorization.
        self.dropout = nn.Dropout(0.2)
        
        # 3. The Spatial Engine (GCN)
        # Takes the 64 memory patterns found by the GRU and shares them across the physical map.
        self.spatial = GCNLayer(in_features=hidden_size, out_features=hidden_size)
        
        # 4. The Output Layer
        # Compresses the final 64 hidden patterns down to 1 specific traffic prediction.
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, X, A):
        # X shape from Phase 3: (Batch_Size, 140 Nodes, 12 Timesteps, 1 Feature)
        Batch_Size, Num_Nodes, Timesteps, Features = X.shape
        
        # --- STEP 1: TIME (The GRU) ---
        # Flatten the grid so the GRU can read the 12-timestep history of each intersection
        x_temp = X.view(Batch_Size * Num_Nodes, Timesteps, Features)
        
        # Pass through GRU. We only want the 'hidden' state (the final memory after 12 steps)
        _, hidden = self.temporal(x_temp)
        
        # Reshape the memories back into our 140-node map format
        x_spatial_in = hidden.view(Batch_Size, Num_Nodes, -1)
        
        # Apply the Dropout guardrail
        x_spatial_in = self.dropout(x_spatial_in)
        
        # --- STEP 2: SPACE (The GCN) ---
        # Pass the GRU's memories through the Graph to share information with physical neighbors
        x_gcn = self.spatial(A, x_spatial_in)
        
        # Apply the ReLU hinge to allow the math to bend around complex traffic spikes
        x_gcn = torch.relu(x_gcn)
        
        # --- STEP 3: PREDICTION ---
        # Compress the data into a single final guess for each of the 140 nodes
        out = self.fc(x_gcn)
        
        # Remove the last empty dimension. Final shape: (Batch_Size, 140)
        out = out.squeeze(-1)

        return out