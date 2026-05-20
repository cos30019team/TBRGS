"""
Phase 3: Tensor Generation (Time & Memory)
Converts the 140-node traffic grid into 3D Memory Flashcards for the GNN.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

def create_tensors(file_path='GeneratedFiles/cleaned_scats_directional_data.csv', lags=12):
    print("1. Loading physical grid data...")
    df = pd.read_csv(file_path)

    # Pivot to create a grid: Rows = Time, Columns = 140 Directional Nodes
    print("2. Pivoting timeline...")
    pivot_df = df.pivot(index='Timestamp', columns='Node_ID', values='Traffic_Volume')
    pivot_df = pivot_df.ffill().fillna(0) # Forward fill any sensor gaps
    raw_data = pivot_df.values # Shape: (Total_Timesteps, 140)

    # --- THE FIX: CHRONOLOGICAL SPLIT (60 / 20 / 20) ---
    print("3. Splitting Time-Series (60% Train, 20% Val, 20% Test)...")
    total_samples = len(raw_data)
    train_idx = int(total_samples * 0.60)
    val_idx = int(total_samples * 0.80)

    # We physically separate the timeline into 3 unbroken blocks of time
    train_data = raw_data[:train_idx]       # The Past
    val_data = raw_data[train_idx:val_idx]  # The Present
    test_data = raw_data[val_idx:]          # The Future

    # --- THE FIX: PREVENTING DATA LEAKAGE ---
    print("4. Scaling Data (Fitting ONLY on Training timeframe)...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # We fit the scaler strictly on the Training set so it has no idea what the 
    # maximum/minimum traffic spikes will be next week in the Test set.
    train_scaled = scaler.fit_transform(train_data)
    val_scaled = scaler.transform(val_data)
    test_scaled = scaler.transform(test_data)
    
    # Save the scaler so we can decode the AI's 0.0-1.0 predictions back to real cars later
    joblib.dump(scaler, 'GeneratedFiles/traffic_scaler.pkl')

    # --- THE SLIDING WINDOW ---
    print(f"5. Applying Sliding Window (Lags = {lags})...")
    def apply_window(data):
        X, y = [], []
        # The exact sliding window loop your tutor used
        for i in range(lags, len(data)):
            X.append(data[i - lags : i]) # Past 12 timesteps (History)
            y.append(data[i])            # 13th timestep (Future Target)
        return np.array(X), np.array(y)

    X_train, y_train = apply_window(train_scaled)
    X_val, y_val = apply_window(val_scaled)
    X_test, y_test = apply_window(test_scaled)

    # --- SHUFFLING (TRAIN ONLY) ---
    # We shuffle the training flashcards so the AI learns the PATTERNS of traffic, 
    # not just the chronological order of the days. (Do not shuffle Val or Test!)
    print("6. Shuffling Training Flashcards...")
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    X_train = X_train[indices]
    y_train = y_train[indices]

    # --- FORMATTING FOR GNN ---
    # Current shape: (Samples, Timesteps, Nodes)
    # Required GNN shape: (Samples, Nodes, Timesteps, 1 Feature)
    print("7. Reshaping Tensors for Graph Neural Network...")
    X_train = np.transpose(X_train, (0, 2, 1))
    X_val = np.transpose(X_val, (0, 2, 1))
    X_test = np.transpose(X_test, (0, 2, 1))

    X_train = np.expand_dims(X_train, axis=-1)
    X_val = np.expand_dims(X_val, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    print("\n--- PHASE 3 COMPLETE ---")
    print(f"X_train shape: {X_train.shape} -> (Samples, 140 Nodes, {lags} Timesteps, 1 Feature)")
    print(f"X_val shape:   {X_val.shape}")
    print(f"X_test shape:  {X_test.shape}")

    np.save('GeneratedFiles/X_train.npy', X_train)
    np.save('GeneratedFiles/y_train.npy', y_train)
    np.save('GeneratedFiles/X_val.npy', X_val)
    np.save('GeneratedFiles/y_val.npy', y_val)
    np.save('GeneratedFiles/X_test.npy', X_test)
    np.save('GeneratedFiles/y_test.npy', y_test)

if __name__ == "__main__":
    create_tensors()