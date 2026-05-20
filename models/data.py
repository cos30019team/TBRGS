import pandas as pd
import numpy as np

def get_processed_data(file_path):
    
    df_raw = pd.read_excel('Scats Data October 2006.xls', header=1)
    print(df_raw.columns.tolist())
    return df_raw