# Import required libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations and array handling
from sklearn.preprocessing import MinMaxScaler  # For scaling features to [0,1] range
import os  # For file and directory operations

def load_water_quality_data(file_path='data/water_quality_data.csv'):
    """
    Load the consolidated water quality dataset.
    
    Parameters:
    -----------
    file_path : str
        Path to the water quality CSV file
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the water quality data
    """
    # Read CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    
    # Handle different date column naming conventions
    if 'dateTime' in df.columns:
        # If dateTime column exists, convert it to datetime format
        df['dateTime'] = pd.to_datetime(df['dateTime'])
    elif 'date' in df.columns:
        # If date column exists, rename it to dateTime and convert to datetime format
        df['dateTime'] = pd.to_datetime(df['date'])
        df = df.drop('date', axis=1)  # Remove the original date column
    
    return df

def load_station_data(station_id, data_dir='data/raw_station_data/'):
    """
    Load data for a specific station from the raw data files.
    
    Parameters:
    -----------
    station_id : str
        The ID of the station to load data for
    data_dir : str
        Directory containing the station data files
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the data for the specified station
    """
    # Construct the file path for the specific station
    file_path = os.path.join(data_dir, f"{station_id}_modified.csv")
    
    # Check if the file exists
    if not os.path.exists(file_path):
        # Raise an error if the file doesn't exist
        raise FileNotFoundError(f"No data file found for station {station_id}")
    
    # Read the station data
    df = pd.read_csv(file_path)
    
    # Convert date column to datetime format
    df['dateTime'] = pd.to_datetime(df['dateTime'])
    
    return df

def clean_and_preprocess(df):
    """
    Clean and preprocess the water quality data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing water quality data
        
    Returns:
    --------
    pd.DataFrame
        Cleaned and preprocessed DataFrame
    """
    # Create a copy to avoid modifying the original dataframe
    df_clean = df.copy()
    
    # Define key water quality parameters (turbidity has been removed)
    key_params = ['dissolved_oxygen', 'pH', 'temperature']
    
    # Remove rows where all key parameters are missing
    # 'how='all'' means drop only if ALL of these values are NaN
    df_clean = df_clean.dropna(subset=key_params, how='all')
    
    # Handle missing values in numerical columns
    # First identify all numeric columns (float and integer types)
    numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
    
    # For each numeric column, fill missing values with the median
    for col in numeric_cols:
        if col in df_clean.columns:
            # Replace NaN values with the median of that column
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # Remove any rows that still have missing values in key parameters
    # This ensures no NaN values in important columns after previous steps
    df_clean = df_clean.dropna(subset=key_params)
    
    # Sort the data chronologically if datetime column exists
    if 'dateTime' in df_clean.columns:
        df_clean = df_clean.sort_values('dateTime')
    
    return df_clean

def prepare_time_series_data(df, target_column, sequence_length=5):
    """
    Prepare time series data for LSTM model using a sliding window approach.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing time series data
    target_column : str
        The column to predict
    sequence_length : int
        The number of previous time steps to use for prediction
        
    Returns:
    --------
    tuple
        (X, y, scaler) where X is the input sequences, y is the target values,
        and scaler is the fitted MinMaxScaler for the target column
    """
    # Extract values from the target column and reshape for scaling
    # The reshape(-1, 1) converts the array to a column vector required by the scaler
    values = df[target_column].values.reshape(-1, 1)
    
    # Create and fit a scaler to normalize data to [0,1] range
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(values)
    
    # Create sequences for time series prediction using sliding window
    X, y = [], []
    
    # Loop through the data to create sequences
    # For each position i, take sequence_length values as input (X)
    # and the next value as the target (y)
    for i in range(len(scaled_values) - sequence_length):
        # X contains 'sequence_length' consecutive values starting at position i
        X.append(scaled_values[i:i+sequence_length])
        # y contains the value immediately following the sequence
        y.append(scaled_values[i+sequence_length])
    
    # Convert lists to numpy arrays for model training
    return np.array(X), np.array(y), scaler

def prepare_multivariate_data(df, feature_columns, target_columns, sequence_length=5):
    """
    Prepare multivariate time series data for LSTM model.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing time series data
    feature_columns : list
        List of columns to use as features
    target_columns : list
        List of columns to predict
    sequence_length : int
        The number of previous time steps to use for prediction
        
    Returns:
    --------
    tuple
        (X, y, feature_scalers, target_scalers) where X is the input sequences, 
        y is the target values, and scalers are dictionaries of fitted MinMaxScalers
    """
    # Extract feature and target values from dataframe
    feature_values = df[feature_columns].values
    target_values = df[target_columns].values
    
    # Initialize dictionaries to store scalers for each column
    feature_scalers = {}
    target_scalers = {}
    
    # Scale each feature column separately
    scaled_features = np.zeros_like(feature_values)
    for i, col in enumerate(feature_columns):
        # Create a new scaler for each feature
        scaler = MinMaxScaler(feature_range=(0, 1))
        # Fit and transform the column, then flatten to 1D array
        scaled_features[:, i] = scaler.fit_transform(feature_values[:, i].reshape(-1, 1)).flatten()
        # Store the scaler for later use (inverse transformation)
        feature_scalers[col] = scaler
    
    # Scale each target column separately
    scaled_targets = np.zeros_like(target_values)
    for i, col in enumerate(target_columns):
        # Create a new scaler for each target
        scaler = MinMaxScaler(feature_range=(0, 1))
        # Fit and transform the column, then flatten to 1D array
        scaled_targets[:, i] = scaler.fit_transform(target_values[:, i].reshape(-1, 1)).flatten()
        # Store the scaler for later use (inverse transformation)
        target_scalers[col] = scaler
    
    # Create sequences for multivariate time series prediction
    X, y = [], []
    
    # Loop through the data to create sequences
    for i in range(len(scaled_features) - sequence_length):
        # Extract sequence of feature vectors (creates a 2D array)
        features_seq = scaled_features[i:i+sequence_length]
        X.append(features_seq)
        # Target is the feature vector at the next time step
        y.append(scaled_targets[i+sequence_length])
    
    # Convert lists to numpy arrays
    # X shape: (samples, sequence_length, n_features)
    # y shape: (samples, n_targets)
    return np.array(X), np.array(y), feature_scalers, target_scalers

def split_train_test(X, y, test_size=0.2):
    """
    Split the data into training and testing sets.
    
    Parameters:
    -----------
    X : np.array
        Input features
    y : np.array
        Target values
    test_size : float
        Proportion of data to use for testing
        
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    # Calculate the index to split the data
    # This keeps the time series order (important for time series data)
    split_idx = int(len(X) * (1 - test_size))
    
    # Split X and y arrays at the same point
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test

def inverse_transform_predictions(predictions, scaler):
    """
    Convert scaled predictions back to original scale.
    
    Parameters:
    -----------
    predictions : np.array
        Scaled predictions
    scaler : MinMaxScaler
        Scaler used to scale the original data
        
    Returns:
    --------
    np.array
        Predictions in original scale
    """
    # Apply inverse transformation to convert normalized [0,1] values 
    # back to original scale
    return scaler.inverse_transform(predictions)

def inverse_transform_multivariate(predictions, scalers, column_names):
    """
    Convert scaled multivariate predictions back to original scale.
    
    Parameters:
    -----------
    predictions : np.array
        Scaled predictions with shape (n_samples, n_features)
    scalers : dict
        Dictionary of scalers with column names as keys
    column_names : list
        List of column names corresponding to the prediction columns
        
    Returns:
    --------
    np.array
        Predictions in original scale
    """
    # Create array with same shape as predictions to store unscaled values
    orig_predictions = np.zeros_like(predictions)
    
    # For each column, apply the corresponding inverse transformation
    for i, col in enumerate(column_names):
        # Extract the column and reshape for inverse transform
        col_preds = predictions[:, i].reshape(-1, 1)
        # Apply inverse transform and flatten back to 1D
        orig_predictions[:, i] = scalers[col].inverse_transform(col_preds).flatten()
    
    return orig_predictions 