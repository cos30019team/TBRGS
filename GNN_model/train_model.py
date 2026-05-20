import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# Import the brain we just built!
from model import TrafficGNN

def normalize_adjacency(A):
    """
    THE REVIEWER's FIX: D^-0.5 * A * D^-0.5
    Prevents highly connected intersections from exploding the math.
    """
    A_sym = A + np.eye(A.shape[0]) # Add self-loops (node connected to itself)
    rowsum = np.array(A_sym.sum(1)) # Count how many connections each node has
    
    # Calculate D^-0.5
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0. # Handle divide-by-zero
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    
    # Mathematically balance the matrix
    normalized_A = d_mat_inv_sqrt.dot(A_sym).dot(d_mat_inv_sqrt)
    return normalized_A

def train_model():
    print("1. Loading Tensors and Adjacency Matrix...")
    X_train = np.load('GeneratedFiles/X_train.npy')
    y_train = np.load('GeneratedFiles/y_train.npy')
    X_val = np.load('GeneratedFiles/X_val.npy')
    y_val = np.load('GeneratedFiles/y_val.npy')
    A_raw = np.load('GeneratedFiles/adjacency_matrix.npy')

    print("2. Normalizing the Spatial Map...")
    A_norm = normalize_adjacency(A_raw)

    # Convert to PyTorch Tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)
    A_tensor = torch.tensor(A_norm, dtype=torch.float32)

    batch_size = 256 
    epochs = 150

    print(f"3. Building DataLoaders (Batch Size: {batch_size})...")
    train_dataset = TensorDataset(X_train_t, y_train_t)
    # shuffle=True acts as an extra guardrail against the AI memorizing time
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = TensorDataset(X_val_t, y_val_t)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("4. Initializing STGCN Model and Optimizer...")
    model = TrafficGNN(num_nodes=140, timesteps=4, hidden_size=64)
    criterion = nn.MSELoss()
    
    # We use RMSprop instead of Adam to perfectly match the tutor's environment
    optimizer = optim.RMSprop(model.parameters(), lr=0.001)

    print(f"\n--- STARTING TRAINING ({epochs} EPOCHS) ---")
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        model.train() # Turns ON Dropout (20% of neurons turn off)
        train_loss = 0.0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X, A_tensor)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)

        # --- VALIDATION PHASE ---
        model.eval() # Turns OFF Dropout (Full brain used for testing)
        val_loss = 0.0

        with torch.no_grad(): # Turn off calculus to speed up testing
            for batch_X, batch_y in val_loader:
                predictions = model(batch_X, A_tensor)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item() * batch_X.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)

        #if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1:3d}/{epochs}] | Train MSE: {avg_train_loss:.6f} | Val MSE: {avg_val_loss:.6f}")

        # --- BEST MODEL CHECKPOINTING ---
        # The Reviewer's fix: Only save the weights if the validation score actually improved!
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'GNN_trained.pth')

    print(f"\n--- TRAINING COMPLETE ---")
    print(f"Best Validation MSE: {best_val_loss:.6f}")
    print("The smartest version of the brain has been saved to 'GNN_trained.pth'")

if __name__ == "__main__":
    train_model()