
import torch
import numpy as np
import joblib
import math
import sklearn.metrics as metrics
import random

from model import TrafficGNN

def normalize_adjacency(A):
    """Must apply the same math normalization we used during training."""
    A_sym = A + np.eye(A.shape[0])
    rowsum = np.array(A_sym.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(A_sym).dot(d_mat_inv_sqrt)

def mape_score(y_true, y_pred):
    y = [x for x in y_true if x > 0]
    pred = [y_pred[i] for i in range(len(y_true)) if y_true[i] > 0]
    if len(y) == 0: return 0
    sums = sum(abs(y[i] - pred[i]) / y[i] for i in range(len(y)))
    return sums * (100 / len(pred))

def evaluate():
    print("1. Loading the Unseen Test Data and Map...")
    X_test = np.load('GeneratedFiles/X_test.npy')
    y_test = np.load('GeneratedFiles/y_test.npy')
    A_raw = np.load('GeneratedFiles/adjacency_matrix.npy')
    scaler = joblib.load('GeneratedFiles/traffic_scaler.pkl')

    A_norm = normalize_adjacency(A_raw)

    # Convert to PyTorch Tensors
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    A_tensor = torch.tensor(A_norm, dtype=torch.float32)

    print("2. Waking up the GNN Brain...")
    model = TrafficGNN(num_nodes=140, timesteps=4, hidden_size=64)
    # Load the weights you just trained!
    model.load_state_dict(torch.load('GNN_trained.pth', weights_only=True))
    model.eval() # Turn off Dropout

    print("3. Generating Predictions for the future...")
    with torch.no_grad():
        y_pred_scaled = model(X_tensor, A_tensor).numpy()

    # We enforce a hard floor at 0.0, since you can't have negative traffic
    y_pred_scaled = np.maximum(y_pred_scaled, 0.0)

    # --- ACADEMIC METRICS ---
    print("\n4. Calculating Strict Academic Scores...")
    
    # THE FIX: Inverse transform in the original 2D shape (Samples, 140 Nodes)
    y_true_cars = scaler.inverse_transform(y_test)
    y_pred_cars = scaler.inverse_transform(y_pred_scaled)
    
    # Now flatten them to calculate total city-wide metrics
    y_true_cars_flat = y_true_cars.flatten()
    y_pred_cars_flat = y_pred_cars.flatten()
    
    # Calculate the exact metrics your tutor used
    mae = metrics.mean_absolute_error(y_true_cars_flat, y_pred_cars_flat)
    mse = metrics.mean_squared_error(y_true_cars_flat, y_pred_cars_flat)
    rmse = math.sqrt(mse)
    mape = mape_score(y_true_cars_flat, y_pred_cars_flat)
    r2 = metrics.r2_score(y_true_cars_flat, y_pred_cars_flat)
    vs = metrics.explained_variance_score(y_true_cars_flat, y_pred_cars_flat)

    print("\n=== FINAL TEST SET PERFORMANCE ===")
    print(f"MAE (Mean Absolute Error)      : {mae:.2f} cars")
    print(f"RMSE (Root Mean Squared Error) : {rmse:.2f} cars")
    print(f"MAPE (Mean Absolute Percentage): {mape:.2f}%")
    print(f"R2 Score (Accuracy out of 1.0) : {r2:.4f}")
    print(f"Explained Variance             : {vs:.4f}")

    # --- HUMAN READABLE CHECK ---
    print("\n=== PHYSICAL REAL-WORLD CHECK ===")
    # Pick a random 15-minute window from the test set
    sample_idx = random.randint(0, len(X_test) - 1)
    print(f"Looking at Random Time Window #{sample_idx}:")
    
    # Compare 5 random intersections in the city during this timeframe
    for i in range(140):
        actual = int(y_true_cars[sample_idx, i])
        predicted = int(y_pred_cars[sample_idx, i])
        error = abs(actual - predicted)
        
        print(f"Node {i}: Actual = {actual:3d} cars | AI Predicted = {predicted:3d} cars | Off by: {error} cars")

if __name__ == "__main__":
    evaluate()