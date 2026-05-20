import pandas as pd

def process_traffic_data():
    traffic_file = 'Files/Scats Data October 2006.xls'
    print("Loading dataset (this might take a few seconds)...")
    
    try:
        # Load the data, using header=1 to skip the junk first row
        df = pd.read_excel(traffic_file, sheet_name='Data', header=1)
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

    # Filter out the junk columns
    essential_cols = ['SCATS Number', 'Date', 'NB_LATITUDE', 'NB_LONGITUDE']
    time_cols = [f'V{str(i).zfill(2)}' for i in range(96)] 
    df = df[essential_cols + time_cols]

    print(f"Original shape: {df.shape} (Wide Format)")

    # MELT the dataframe (Wide to Long format)
    print("Melting data from Wide to Long format...")
    long_df = pd.melt(
        df,
        id_vars=essential_cols,      
        value_vars=time_cols,        
        var_name='Time_Code',        
        value_name='Traffic_Volume'  
    )

    print("\nCleaning Timestamps and Generating Unique Node IDs...")
    
    # Clean the Date column (strip out the random 00:15:00)
    long_df['Date'] = pd.to_datetime(long_df['Date']).dt.date

    # Convert 'Vxx' to actual Hour and Minute
    time_ints = long_df['Time_Code'].str.replace('V', '').astype(int)
    long_df['Hour'] = time_ints // 4 
    long_df['Minute'] = (time_ints % 4) * 15 

    # Create a perfect Pandas Datetime column
    long_df['Timestamp'] = pd.to_datetime(
        long_df['Date'].astype(str) + ' ' + 
        long_df['Hour'].astype(str).str.zfill(2) + ':' + 
        long_df['Minute'].astype(str).str.zfill(2) + ':00'
    )

    # Create a unique Node_ID for every single sensor based on its exact coordinates.
    # ngroup() assigns a unique integer (0, 1, 2, 3...) to every unique Lat/Long combo.
    long_df['Node_ID'] = long_df.groupby(['SCATS Number', 'NB_LATITUDE', 'NB_LONGITUDE']).ngroup()

    # Select only the columns we actually need for Machine Learning
    final_df = long_df[['Node_ID', 'SCATS Number', 'Timestamp', 'Traffic_Volume', 'NB_LATITUDE', 'NB_LONGITUDE']]

    # Sort it to be perfectly chronological for the ML models
    final_df = final_df.sort_values(by=['Node_ID', 'Timestamp']).reset_index(drop=True)

    print(f"\nFinal Shape (Detector-Level): {final_df.shape}")
    print(f"Total Unique Nodes (Sensors) found: {final_df['Node_ID'].nunique()}")
    
    print("\n--- FINAL ML-READY DATA ---")
    print(final_df.head(10))

    # Save the perfectly formatted data for you and your teammates!
    final_df.to_csv('GeneratedFiles/cleaned_scats_directional_data.csv', index=False)
    print("\nData successfully saved to 'cleaned_scats_directional_data.csv'")
    
    return final_df

if __name__ == "__main__":
    ml_data = process_traffic_data()