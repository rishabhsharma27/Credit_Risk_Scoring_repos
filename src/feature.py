import pandas as pd
import numpy as np

def clean_raw_data(df):
    """
    Cleans raw input data, handles missing values, and fixes data types.
    """
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()

    # Drop the first column if it's an index or ID column
    df= df.drop(df.columns[0], axis=1)

    # Lowercase all column names for consistency
    # Force numeric casting for columns that might have messy strings
    for col in df:
        df.columns = [col.lower() for col in df.columns]
    #df.info()
    #df= pd.to_numeric(df, errors='coerce')
              
    # Fill missing values or drop rows as per business logic
    # For simplicity, drop rows with any missing values   
    df = df.dropna()  
    
    return df

def engineer_features(df):
    """
    Creates custom credit risk features (e.g., ratios, aggregations).
    """
    df = df.copy()
    
    # Example credit risk ratios (safeguarding against division by zero)
    total = 0
    for col in df:
        if col.startswith('T_'):
            total = total + df[col]
    
    df['calcul_saving'] =  (total)  # Avoid division by zero
         
    return df

def preprocess_features(df, train_columns=None):
    """
    Applies encoding and aligns columns with the training feature schema.
    """
    df = clean_raw_data(df)
    df = engineer_features(df)
    
    # One-hot encode categorical variables
    df['cat_gambling'] = df['cat_gambling'].replace({'High':2,'Low':1,'No':0}).astype(int)
    
    return df