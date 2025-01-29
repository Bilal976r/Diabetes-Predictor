# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

def preprocess_input(input_data):
    """
    Preprocess a single or batch input for prediction.
    
    Parameters:
        input_data (numpy.ndarray): Array of input features, where each row is an instance.
        
    Returns:
        numpy.ndarray: Scaled input data.
    """
    # Scale the input data
    return scaler.fit_transform(input_data)

def check_missing_values(data):
    """
    Check for missing values in the input data.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing the input data.
        
    Returns:
        dict: A dictionary with column names as keys and missing value counts as values.
    """
    missing_values = data.isnull().sum()
    return missing_values[missing_values > 0].to_dict()

def handle_missing_values(data, strategy="mean"):
    """
    Handle missing values in the input data using a specified strategy.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing the input data.
        strategy (str): Strategy to handle missing values - 'mean', 'median', or 'zero'.
        
    Returns:
        pd.DataFrame: DataFrame with missing values imputed.
    """
    if strategy == "mean":
        return data.fillna(data.mean())
    elif strategy == "median":
        return data.fillna(data.median())
    elif strategy == "zero":
        return data.fillna(0)
    else:
        raise ValueError("Invalid strategy. Use 'mean', 'median', or 'zero'.")

def validate_columns(data, required_columns):
    """
    Validate that the required columns are present in the DataFrame.
    
    Parameters:
        data (pd.DataFrame): DataFrame to validate.
        required_columns (list): List of required column names.
        
    Returns:
        bool: True if all required columns are present, False otherwise.
    """
    return all(col in data.columns for col in required_columns)