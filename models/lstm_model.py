import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.layers import LSTM, Dense

# --- STEP 1: DATA PROCESSING ---
def get_processed_data(file_path, target_site=970):
    # Load raw CSV (Note: Header is on the second row)
    df_raw = pd.read_csv(file_path, header=1)
    
    # 1. Convert Excel serial dates to Datetime
    df_raw['Date_DT'] = pd.to_datetime(df_raw['Date'].astype(float), unit='D', origin='1899-12-30')
    
    # 2. Reshape from Wide (V00-V95) to Long format
    vol_cols = [f'V{i:02d}' for i in range(96)]
    df_long = df_raw.melt(
        id_vars=['SCATS Number', 'Date_DT'], 
        value_vars=vol_cols, 
        var_name='Timeslot', 
        value_name='Volume'
    )
    
    # 3. Create continuous timestamp and sort
    df_long['Min_Offset'] = df_long['Timeslot'].str.extract('(\d+)').astype(int) * 15
    df_long['Timestamp'] = df_long['Date_DT'] + pd.to_timedelta(df_long['Min_Offset'], unit='m')
    
    # 4. Filter for a specific intersection (e.g., Site 970) and drop NaNs
    site_df = df_long[df_long['SCATS Number'] == target_site].sort_values('Timestamp')
    site_df = site_df.dropna(subset=['Volume'])
    
    return site_df[['Volume']].values

# --- STEP 2: INTEGRATION WITH YOUR ML CODE ---

# Load and prepare raw data values
raw_volumes = get_processed_data('Scats Data October 2006.xls - Data.csv')

# Scale the data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(raw_volumes)

# Create lagged features (Using Lag 1 as in your sample)
# X is the traffic at time T, y is the traffic at time T+1
X = data_scaled[:-1]
y = data_scaled[1:]

# Split the data (80% Training, 20% Testing)
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape to 3D for LSTM: [batch, time_steps, features] -> [N, 1, 1]
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Define the LSTM model architecture
model = keras.models.Sequential([
    LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# --- STEP 3: EVALUATION & INVERSE TRANSFORMATION ---
loss = model.evaluate(X_test, y_test)
print(f'Test loss (MSE): {loss}')

# Predict and convert back to actual car counts
y_pred = model.predict(X_test)
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test)

