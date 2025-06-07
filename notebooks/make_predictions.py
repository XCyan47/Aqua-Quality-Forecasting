#!/usr/bin/env python3
"""
Prediction Script for Water Quality Parameters
This script uses the trained LSTM model to make predictions on new data.
It can either:
1. Make predictions on a single data file (default behavior)
2. Process test data files from a directory (using --test-dir flag)
"""

# Import required libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For creating visualizations
import tensorflow as tf  # For loading the trained model
import os  # For file and directory operations
import sys  # For system-specific parameters
import glob  # For file path pattern matching
import argparse  # For parsing command-line arguments
from sklearn.preprocessing import MinMaxScaler  # For scaling features to [0,1] range

# Add the parent directory to the system path
# This allows importing modules from the src directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import custom functions from the data_processing module
from src.data_processing import clean_and_preprocess, load_water_quality_data, prepare_multivariate_data, inverse_transform_multivariate

def load_test_data(file_path):
    """
    Load and preprocess test data for prediction.
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file containing test data
        
    Returns:
    --------
    tuple
        (data, parameter_columns) where data is a preprocessed DataFrame and
        parameter_columns is a list of available water quality parameters
    """
    # Load data from CSV file
    data = pd.read_csv(file_path)
    
    # Ensure the date column is properly formatted and consistent
    if 'date' in data.columns:
        # Convert 'date' to 'dateTime' for consistency
        data['dateTime'] = pd.to_datetime(data['date'])
        data = data.drop('date', axis=1)  # Remove original date column
    elif 'dateTime' in data.columns:
        # Convert existing dateTime to datetime format
        data['dateTime'] = pd.to_datetime(data['dateTime'])
    
    # Define the water quality parameters we need for prediction
    # Note: turbidity has been removed
    parameter_columns = ['temperature', 'dissolved_oxygen', 'pH']
    
    # Identify all columns we want to keep
    columns_to_keep = ['dateTime'] + parameter_columns
    
    # Check if all required parameter columns exist in the data
    for col in parameter_columns:
        if col not in data.columns:
            # Warn the user if a parameter is missing
            print(f"Warning: Column '{col}' not found in data. This may affect prediction quality.")
    
    # Keep only the available columns from the required set
    available_columns = [col for col in columns_to_keep if col in data.columns]
    data = data[available_columns]
    
    # Handle missing values by replacing with median values
    # This prevents NaN errors during model prediction
    parameter_columns = [col for col in parameter_columns if col in data.columns]
    for col in parameter_columns:
        data[col] = data[col].fillna(data[col].median())
    
    # Set datetime as index for time series processing
    data = data.set_index('dateTime')
    
    return data, parameter_columns

def prepare_test_data(data, parameter_columns, sequence_length=10):
    """
    Prepare test data for prediction - custom version for test data with exactly sequence_length days.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing time series data
    parameter_columns : list
        List of columns to use as features
    sequence_length : int
        The number of previous time steps to use for prediction
        
    Returns:
    --------
    tuple
        (X, feature_scalers, target_scalers) where X is the input sequence
    """
    # Extract feature values from the DataFrame
    feature_values = data[parameter_columns].values
    
    # Initialize dictionaries to store scalers for each parameter
    feature_scalers = {}
    target_scalers = {}
    
    # Scale each feature column separately to [0,1] range
    scaled_features = np.zeros_like(feature_values)
    for i, col in enumerate(parameter_columns):
        # Create a new scaler for this parameter
        scaler = MinMaxScaler(feature_range=(0, 1))
        # Fit and transform the values, then flatten to 1D
        scaled_features[:, i] = scaler.fit_transform(feature_values[:, i].reshape(-1, 1)).flatten()
        # Store the scaler for inverse transformation later
        feature_scalers[col] = scaler
        # Use the same scalers for targets (since we're forecasting the same parameters)
        target_scalers[col] = scaler
    
    # Reshape the scaled features to match the LSTM model's input shape:
    # (batch_size=1, sequence_length, n_features)
    X = scaled_features.reshape(1, sequence_length, len(parameter_columns))
    
    return X, feature_scalers, target_scalers

def make_predictions(data, model_path, parameter_columns, sequence_length=10, forecast_steps=1):
    """
    Make predictions using the trained LSTM model.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing input time series data
    model_path : str
        Path to the saved LSTM model file
    parameter_columns : list
        List of water quality parameters to predict
    sequence_length : int
        Number of time steps used as input for prediction
    forecast_steps : int
        Number of future steps to forecast
        
    Returns:
    --------
    pd.DataFrame
        DataFrame containing the predictions with dates as index
    """
    # Load the trained model from disk
    model = tf.keras.models.load_model(model_path)
    
    # Print debug information
    print(f"Data shape: {data.shape}")
    print(f"Parameters: {parameter_columns}")
    
    # Handle two different test data scenarios:
    
    # Scenario 1: Exact sequence length data (e.g., 10 days of data)
    # This is typically used for the test files that have exactly sequence_length days
    if len(data) == sequence_length:
        print("Using custom test data preparation...")
        # Use custom preparation for exact-length test data
        X, feature_scalers, target_scalers = prepare_test_data(data, parameter_columns, sequence_length)
        input_seq = X  # Already in correct shape
    else:
        # Scenario 2: Standard dataset with more than sequence_length records
        # Prepare data using the standard method from data_processing module
        X, _, feature_scalers, target_scalers = prepare_multivariate_data(
            data, 
            feature_columns=parameter_columns,
            target_columns=parameter_columns,
            sequence_length=sequence_length,
            train_test_split=False  # Use all data for prediction
        )
        
        # Print debug information about prepared data
        print(f"X shape after preparation: {X.shape}")
        
        # Ensure we have enough data to make predictions
        if len(X) == 0:
            raise ValueError("No sequences were generated. Ensure your data has at least 'sequence_length' rows.")
        
        # Use the last sequence as input for forecasting
        # This takes the most recent sequence_length time steps
        input_seq = X[-1].reshape(1, sequence_length, len(parameter_columns))
    
    # Initialize list to store predictions
    forecast = []
    
    # Generate multi-step forecast recursively
    for _ in range(forecast_steps):
        # Predict next time step
        next_pred = model.predict(input_seq, verbose=0)
        
        # Add prediction to forecast list
        forecast.append(next_pred[0])
        
        # Update input sequence for next prediction by:
        # 1. Removing the oldest time step
        # 2. Adding the new prediction
        new_seq = np.concatenate([input_seq[0, 1:, :], next_pred.reshape(1, len(parameter_columns))], axis=0)
        input_seq = new_seq.reshape(1, sequence_length, len(parameter_columns))
    
    # Convert forecast list to numpy array
    forecast = np.array(forecast)
    
    # Convert scaled predictions back to original scale
    forecast_inv = inverse_transform_multivariate(forecast, target_scalers, parameter_columns)
    
    # Generate future dates for the forecast period
    last_date = data.index[-1]  # Get the last date in the dataset
    # Create a sequence of dates starting after last_date
    future_dates = pd.date_range(start=last_date, periods=forecast_steps+1)[1:]
    
    # Create a DataFrame with the predictions and future dates as index
    forecast_df = pd.DataFrame(forecast_inv, index=future_dates, columns=parameter_columns)
    
    return forecast_df

def plot_forecast(data, forecast_df, parameter, output_file=None):
    """
    Plot the forecast for a specific parameter.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing historical data
    forecast_df : pd.DataFrame
        DataFrame containing forecast data
    parameter : str
        The parameter to plot
    output_file : str, optional
        Path to save the plot, if None plot is displayed but not saved
    """
    # Create a new figure with specified dimensions
    plt.figure(figsize=(14, 7))
    
    # Plot historical data
    # If we have a lot of data, only plot the last 100 points for better visibility
    if len(data) > 100:
        plt.plot(data.index[-100:], data[parameter][-100:], 'k-', label='Historical Data')
    else:
        plt.plot(data.index, data[parameter], 'k-', label='Historical Data')
    
    # Plot the forecast with dashed red line
    plt.plot(forecast_df.index, forecast_df[parameter], 'r--', label='Forecast')
    
    # Add chart labels and styling
    plt.title(f'LSTM Model: Forecast for {parameter.capitalize()}')
    plt.xlabel('Date')
    plt.ylabel(f'{parameter.capitalize()}')
    plt.legend()
    plt.grid(True)
    
    # Save the plot if an output file is specified
    if output_file:
        plt.savefig(output_file)
        print(f"Forecast plot saved to {output_file}")
    
    # Close the plot to free memory
    plt.close()

def process_single_file(data_file, model_path, output_dir, forecast_steps=30):
    """
    Process a single data file for prediction.
    
    Parameters:
    -----------
    data_file : str
        Path to the input data file
    model_path : str
        Path to the saved model file
    output_dir : str
        Directory to save outputs
    forecast_steps : int
        Number of future steps to forecast
    """
    # Define parameters to predict (turbidity removed)
    parameter_columns = ['temperature', 'dissolved_oxygen', 'pH']
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    data = load_water_quality_data(data_file)
    data = clean_and_preprocess(data)
    data = data.set_index('dateTime')
    
    # Make predictions using the trained model
    print("Making predictions...")
    forecast_df = make_predictions(data, model_path, parameter_columns, forecast_steps=forecast_steps)
    
    # Save predictions to CSV file
    forecast_csv = os.path.join(output_dir, 'forecast_results.csv')
    forecast_df.to_csv(forecast_csv)
    print(f"Forecast results saved to {forecast_csv}")
    
    # Plot forecasts for each parameter
    for param in parameter_columns:
        output_file = os.path.join(output_dir, f'forecast_{param}.png')
        plot_forecast(data, forecast_df, param, output_file)
    
    print("\nPrediction process completed successfully!")

def process_test_directory(test_data_dir, model_path, forecast_steps=1):
    """
    Process all CSV files in the test data directory.
    
    Parameters:
    -----------
    test_data_dir : str
        Path to the directory containing test data files
    model_path : str
        Path to the saved model file
    forecast_steps : int
        Number of future steps to forecast
    """
    # Find all CSV files in the test_data directory using glob pattern matching
    test_files = glob.glob(os.path.join(test_data_dir, '*.csv'))
    
    # Check if any CSV files were found
    if not test_files:
        print(f"No CSV files found in {test_data_dir}")
        return
    
    # Process each test file
    for test_file in test_files:
        # Extract file name and base name without extension
        file_name = os.path.basename(test_file)
        file_base = os.path.splitext(file_name)[0]
        
        # Define output file path for the predictions
        output_file = os.path.join(test_data_dir, f"{file_base}_prediction.csv")
        
        print(f"\nProcessing {file_name}...")
        
        # Load and preprocess test data
        data, parameter_columns = load_test_data(test_file)
        
        # Check if we have enough data points for prediction
        if len(data) < 10:
            print(f"Warning: {file_name} contains fewer than 10 data points. At least 10 are needed for prediction.")
            continue
        
        # Make predictions using the loaded model
        print(f"Making predictions for {file_name}...")
        forecast_df = make_predictions(data, model_path, parameter_columns, forecast_steps=forecast_steps)
        
        # Add a date column as the first column for better readability
        forecast_df.reset_index(inplace=True)
        forecast_df.rename(columns={'index': 'date'}, inplace=True)
        
        # Save predictions to CSV file
        forecast_df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
        
        # Create visualization plots for each parameter
        for param in parameter_columns:
            # Create a new figure
            plt.figure(figsize=(10, 6))
            
            # Plot historical data and prediction
            plt.plot(data.index, data[param], 'k-', label='Historical Data')
            plt.plot(forecast_df['date'], forecast_df[param], 'r--', label='Prediction')
            
            # Add chart labels and styling
            plt.title(f'Prediction for {param.capitalize()}')
            plt.xlabel('Date')
            plt.ylabel(param.capitalize())
            plt.legend()
            plt.grid(True)
            
            # Save the plot to file
            plot_file = os.path.join(test_data_dir, f"{file_base}_{param}_plot.png")
            plt.savefig(plot_file)
            plt.close()
            print(f"Plot for {param} saved to {plot_file}")
    
    print("\nAll test files processed successfully!")

def main():
    """Main function to handle command-line execution."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Water Quality Prediction Script")
    parser.add_argument("--subdir", help="Specific subdirectory name in test_data to process")
    args = parser.parse_args()
    
    # Define paths for model and data
    current_dir = os.getcwd()
    model_path = os.path.join(current_dir, 'models/lstm_model.h5')
    
    # Get the test_data directory path
    test_data_dir = os.path.join(current_dir, 'test_data')
    
    # Find all subdirectories in test_data
    subdirs = [d for d in os.listdir(test_data_dir) 
              if os.path.isdir(os.path.join(test_data_dir, d))]
    
    # Handle case where a specific subdirectory is specified
    if args.subdir:
        # Check if specified subdirectory exists
        if args.subdir in subdirs:
            print(f"Processing specified subdirectory: {args.subdir}")
            subdir = args.subdir
            subdir_path = os.path.join(test_data_dir, subdir)
            
            # Look for a CSV file with the same name as the directory
            test_file = os.path.join(subdir_path, f"{subdir}.csv")
            if not os.path.exists(test_file):
                # Try to find any CSV file if the matching one doesn't exist
                csv_files = glob.glob(os.path.join(subdir_path, "*.csv"))
                if not csv_files:
                    print(f"No CSV files found in {subdir_path}.")
                    return
                test_file = csv_files[0]
            
            file_name = os.path.basename(test_file)
            file_base = os.path.splitext(file_name)[0]
            output_dir = subdir_path
        else:
            print(f"Specified subdirectory '{args.subdir}' not found in test_data.")
            return
    # Handle case where no subdirectories exist
    elif not subdirs:
        print("No subdirectories found in test_data. Looking for CSV files directly.")
        # Look for CSV files directly in test_data
        test_files = glob.glob(os.path.join(test_data_dir, "*.csv"))
        if not test_files:
            print("No CSV files found in test_data directory.")
            return
        
        # Process first CSV file found
        test_file = test_files[0]
        file_name = os.path.basename(test_file)
        file_base = os.path.splitext(file_name)[0]
        output_dir = test_data_dir
    # Default behavior: use first subdirectory
    else:
        print(f"Found {len(subdirs)} subdirectories in test_data.")
        
        # Use the first subdirectory found
        subdir = subdirs[0]
        print(f"Using subdirectory: {subdir}")
        subdir_path = os.path.join(test_data_dir, subdir)
        
        # Look for a CSV file with the same name as the directory
        test_file = os.path.join(subdir_path, f"{subdir}.csv")
        if not os.path.exists(test_file):
            # Try to find any CSV file if the matching one doesn't exist
            csv_files = glob.glob(os.path.join(subdir_path, "*.csv"))
            if not csv_files:
                print(f"No CSV files found in {subdir_path}.")
                return
            test_file = csv_files[0]
        
        file_name = os.path.basename(test_file)
        file_base = os.path.splitext(file_name)[0]
        output_dir = subdir_path
    
    # Verify that the selected test file exists
    if not os.path.exists(test_file):
        print(f"Error: Test file {test_file} not found!")
        return
        
    print(f"Processing test file: {test_file}")
    print(f"Output will be saved to: {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and preprocess test data
    data, parameter_columns = load_test_data(test_file)
    
    # Check if we have enough data points for prediction
    if len(data) < 10:
        print(f"Warning: The test file contains fewer than 10 data points. At least 10 are needed for prediction.")
        return
    
    # Make predictions for one day (11th day)
    print("Making predictions for day 11...")
    forecast_df = make_predictions(data, model_path, parameter_columns, forecast_steps=1)
    
    # Add a date column as the first column for better readability
    forecast_df.reset_index(inplace=True)
    forecast_df.rename(columns={'index': 'date'}, inplace=True)
    
    # Save predictions to CSV file
    output_file = os.path.join(output_dir, f"{file_base}_prediction.csv")
    forecast_df.to_csv(output_file, index=False)
    print(f"Prediction for day 11 saved to {output_file}")
    
    # Generate visualization plots for each parameter
    for param in parameter_columns:
        plt.figure(figsize=(10, 6))
        plt.plot(data.index, data[param], 'k-', label='Historical Data (10 days)')
        plt.plot(forecast_df['date'], forecast_df[param], 'r--', label='Prediction (Day 11)')
        plt.title(f'Prediction for {param.capitalize()}')
        plt.xlabel('Date')
        plt.ylabel(param.capitalize())
        plt.legend()
        plt.grid(True)
        
        plot_file = os.path.join(output_dir, f"{file_base}_{param}_plot.png")
        plt.savefig(plot_file)
        plt.close()
        print(f"Plot for {param} saved to {plot_file}")
    
    print("\nPrediction process completed successfully!")

# Execute the main function when the script is run directly
if __name__ == "__main__":
    main() 