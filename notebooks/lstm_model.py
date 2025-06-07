#!/usr/bin/env python3
"""
Multivariate LSTM Model for Water Quality Prediction
This script implements a Long Short-Term Memory (LSTM) neural network model 
for forecasting multiple water quality parameters.
"""

# Import required libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For creating visualizations
from sklearn.preprocessing import MinMaxScaler  # For scaling features to [0,1] range
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # For model evaluation
import tensorflow as tf  # Deep learning framework
from tensorflow.keras.models import Sequential  # For creating sequential neural network
from tensorflow.keras.layers import LSTM, Dense, Dropout  # Neural network layers
import sys  # For system-specific parameters
import os  # For file and directory operations

# Add the parent directory to the system path
# This allows importing modules from the src directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import custom functions from the data_processing module
from src.data_processing import load_water_quality_data, clean_and_preprocess, prepare_multivariate_data, split_train_test, inverse_transform_multivariate

def create_multivariate_lstm_model(input_shape, output_dim, units=50, dropout_rate=0.2):
    """
    Create LSTM model for multivariate prediction
    
    Args:
        input_shape (tuple): Shape of input data (sequence_length, features)
        output_dim (int): Number of output features to predict
        units (int): Number of LSTM units
        dropout_rate (float): Dropout rate for regularization
        
    Returns:
        model: Keras LSTM model
    """
    # Initialize a sequential model (layers arranged in sequence)
    model = Sequential()
    
    # Add first LSTM layer with return_sequences=True to stack another LSTM layer
    # This layer processes the input sequences and outputs sequences
    model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape))
    
    # Add dropout layer to prevent overfitting by randomly setting inputs to 0
    model.add(Dropout(dropout_rate))
    
    # Add second LSTM layer without return_sequences to output a single vector
    model.add(LSTM(units=units))
    
    # Add another dropout layer
    model.add(Dropout(dropout_rate))
    
    # Add Dense (fully connected) output layer with output_dim neurons
    # (one neuron for each target parameter)
    model.add(Dense(output_dim))
    
    # Compile the model with Adam optimizer and MSE loss function
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

def main():
    # Announce the start of LSTM model training
    print("Starting Multivariate LSTM model training...")
    
    # Get the current working directory for proper path handling
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")
    
    # Create output directories for storing results, plots, and saved models
    output_dir = os.path.join(current_dir, 'outputs')
    os.makedirs(output_dir, exist_ok=True)  # Create if it doesn't exist
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)  # Create if it doesn't exist
    models_dir = os.path.join(current_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)  # Create if it doesn't exist
    
    # Load and preprocess the water quality data
    print("Loading and preprocessing data...")
    data = load_water_quality_data('data/water_quality_data.csv')
    data = clean_and_preprocess(data)
    # Set datetime as index for time series analysis
    data = data.set_index('dateTime')
    
    # Define water quality parameters for modeling
    water_params = ['temperature', 'dissolved_oxygen', 'pH']
    
    print(f"Modeling parameters: {water_params}")
    print(f"Data shape: {data.shape}")
    
    # Prepare data for multivariate time series prediction
    # Use sequence length of 10 (predict based on 10 previous time steps)
    seq_length = 10
    # Create sequences and scale the data to [0,1] range
    X, y, feature_scalers, target_scalers = prepare_multivariate_data(
        data, 
        feature_columns=water_params,  # Use same columns as both features
        target_columns=water_params,   # and targets for forecasting
        sequence_length=seq_length
    )
    print(f"Created {len(X)} sequences with length {seq_length}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Split data into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2)
    print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    
    # Save indices for later plotting against the original time index
    train_size = len(X_train)
    # Calculate indices for the training and test data points in the original series
    train_indices = list(range(seq_length, train_size + seq_length))
    test_indices = list(range(train_size + seq_length, len(X) + seq_length))
    
    # Build and train LSTM model for multivariate prediction
    # Get the input shape from the training data (sequence_length, number of features)
    input_shape = (X_train.shape[1], X_train.shape[2])
    # Number of output parameters to predict (matches water_params length)
    output_dim = y_train.shape[1]
    # Create the LSTM model with the specified architecture
    model = create_multivariate_lstm_model(input_shape, output_dim)
    
    print(f"Model input shape: {input_shape}, output dimension: {output_dim}")
    
    # Set up early stopping to prevent overfitting
    # This stops training when validation loss stops improving
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',         # Monitor validation loss
        patience=10,                # Wait for 10 epochs without improvement
        restore_best_weights=True   # Restore model to best weights after training
    )
    
    # Train the LSTM model on the prepared data
    print("Training LSTM model...")
    history = model.fit(
        X_train, y_train,             # Training data and targets
        epochs=50,                    # Maximum number of training epochs
        batch_size=32,                # Number of samples per gradient update
        validation_split=0.2,         # 20% of training data used for validation
        callbacks=[early_stopping],   # Use early stopping
        verbose=1                     # Show progress bar
    )
    
    # Save the trained model to disk for later use
    model_path = os.path.join(models_dir, 'lstm_model.h5')
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Make predictions on both training and test data
    print("Making predictions...")
    train_predict = model.predict(X_train)  # Predictions on training data
    test_predict = model.predict(X_test)    # Predictions on test data
    
    # Inverse transform scaled predictions back to original scale
    # This converts the [0,1] normalized values back to actual parameter values
    y_train_inv = inverse_transform_multivariate(y_train, target_scalers, water_params)
    train_predict_inv = inverse_transform_multivariate(train_predict, target_scalers, water_params)
    y_test_inv = inverse_transform_multivariate(y_test, target_scalers, water_params)
    test_predict_inv = inverse_transform_multivariate(test_predict, target_scalers, water_params)
    
    # Calculate error metrics for each parameter on the test set
    test_metrics = {}
    print("\nTest Performance Metrics:")
    # Loop through each water quality parameter
    for i, param in enumerate(water_params):
        # Calculate Root Mean Squared Error (RMSE)
        rmse = np.sqrt(mean_squared_error(y_test_inv[:, i], test_predict_inv[:, i]))
        # Calculate Mean Absolute Error (MAE)
        mae = mean_absolute_error(y_test_inv[:, i], test_predict_inv[:, i])
        # Calculate R-squared (coefficient of determination)
        r2 = r2_score(y_test_inv[:, i], test_predict_inv[:, i])
        
        # Store metrics in dictionary for later use
        test_metrics[param] = {
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2
        }
        
        # Display metrics for current parameter
        print(f"{param.capitalize()}:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        
        # Save individual parameter results to text file
        param_results_file = os.path.join(output_dir, f'{param}_lstm_test_results.txt')
        with open(param_results_file, 'w') as f:
            f.write(f"{param.capitalize()} LSTM Model Test Results\n")
            f.write(f"Sequence Length: {seq_length}\n\n")
            f.write(f"RMSE: {rmse:.4f}\n")
            f.write(f"MAE: {mae:.4f}\n")
            f.write(f"R²: {r2:.4f}\n")
        print(f"Test results for {param} saved to {param_results_file}")
    
    # Save consolidated test results for all parameters to a single file
    try:
        test_results_file = os.path.join(output_dir, 'multivariate_lstm_test_results.txt')
        with open(test_results_file, 'w') as f:
            f.write("Multivariate LSTM Model Test Results\n")
            f.write(f"Sequence Length: {seq_length}\n\n")
            # Write results for each parameter
            for param, metrics in test_metrics.items():
                f.write(f"{param.capitalize()}:\n")
                f.write(f"  RMSE: {metrics['RMSE']:.4f}\n")
                f.write(f"  MAE: {metrics['MAE']:.4f}\n")
                f.write(f"  R²: {metrics['R²']:.4f}\n\n")
        print(f"\nConsolidated test results saved to {test_results_file}")
    except Exception as e:
        # Handle any errors during file writing
        print(f"Error saving test results: {e}")
    
    # Plot training and validation loss curves
    plt.figure(figsize=(12, 6))
    # Training loss (on training data)
    plt.plot(history.history['loss'], label='Training Loss')
    # Validation loss (on validation data)
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Multivariate LSTM Model: Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Save the loss plot to disk
    loss_plot_path = os.path.join(plots_dir, 'multivariate_lstm_training_loss.png')
    plt.savefig(loss_plot_path)
    print(f"\nTraining loss plot saved to {loss_plot_path}")
    plt.close()
    
    # Get the original time index for plotting predictions
    # Skip the first seq_length time steps since they were used to create the first prediction
    time_index = data.index[seq_length:]
    # Split time index for training and test periods
    train_time = time_index[:len(train_indices)]
    test_time = time_index[len(train_indices):len(train_indices) + len(test_indices)]
    
    # Plot predictions vs actual values for each parameter
    for i, param in enumerate(water_params):
        plt.figure(figsize=(14, 7))
        
        # Get original data for this parameter
        original_data = data[param].values
        
        # Plot complete original data series in light black
        plt.plot(data.index, original_data, 'k-', alpha=0.3, label='Original Data')
        
        # Plot training predictions and actual values
        plt.plot(train_time, train_predict_inv[:, i], 'b--', label='Training Predictions')
        plt.plot(train_time, y_train_inv[:, i], 'g-', alpha=0.5, label='Training Targets')
        
        # Plot test predictions and actual values
        plt.plot(test_time, test_predict_inv[:, i], 'r--', label='Test Predictions')
        plt.plot(test_time, y_test_inv[:, i], 'm-', alpha=0.5, label='Test Targets')
        
        # Add chart labels and styling
        plt.title(f'LSTM Model: Actual vs Predicted {param.capitalize()}')
        plt.xlabel('Date')
        plt.ylabel(f'{param.capitalize()}')
        plt.legend()
        plt.grid(True)
        
        # Save the prediction plot to disk
        pred_plot_path = os.path.join(plots_dir, f'lstm_prediction_{param}.png')
        plt.savefig(pred_plot_path)
        print(f"\nPrediction plot for {param} saved to {pred_plot_path}")
        plt.close()
    
    # Generate multi-step forecast for future values
    print("\nGenerating multi-step forecast...")
    forecast_steps = 30  # Forecast 30 days into the future
    
    # Use the last available sequence as the starting point for forecasting
    # Reshape to match model's expected input shape (batch_size, seq_length, n_features)
    input_seq = X[-1].reshape(1, seq_length, len(water_params))
    forecast = []  # List to store forecasted values
    
    # Recursive forecasting loop
    for i in range(forecast_steps):
        # Predict the next time step
        next_pred = model.predict(input_seq, verbose=0)
        
        # Add the prediction to the forecast list
        forecast.append(next_pred[0])
        
        # Update input sequence for next prediction by:
        # 1. Removing the oldest time step (at index 0)
        # 2. Adding the new prediction at the end
        new_seq = np.concatenate([input_seq[0, 1:, :], next_pred.reshape(1, len(water_params))], axis=0)
        # Reshape to match model's input format
        input_seq = new_seq.reshape(1, seq_length, len(water_params))
    
    # Convert forecast list to numpy array for easier processing
    forecast = np.array(forecast)
    
    # Inverse transform the scaled forecast values back to original scale
    forecast_inv = inverse_transform_multivariate(forecast, target_scalers, water_params)
    
    # Generate date range for the forecast period
    last_date = data.index[-1]  # Get the last date in the dataset
    # Create a sequence of dates starting from last_date for forecast_steps days
    future_dates = pd.date_range(start=last_date, periods=forecast_steps+1)[1:]
    
    # Plot the forecast for each parameter
    for i, param in enumerate(water_params):
        plt.figure(figsize=(14, 7))
        
        # Get original data for this parameter
        original_data = data[param].values
        
        # Plot recent historical data (last 100 points) for context
        plt.plot(data.index[-100:], original_data[-100:], 'k-', label='Historical Data')
        
        # Plot the forecasted values
        plt.plot(future_dates, forecast_inv[:, i], 'r--', label='Forecast')
        
        # Add chart labels and styling
        plt.title(f'LSTM Model: {forecast_steps}-Day Forecast for {param.capitalize()}')
        plt.xlabel('Date')
        plt.ylabel(f'{param.capitalize()}')
        plt.legend()
        plt.grid(True)
        
        # Save the forecast plot to disk
        forecast_plot_path = os.path.join(plots_dir, f'lstm_forecast_{param}.png')
        plt.savefig(forecast_plot_path)
        print(f"Forecast plot for {param} saved to {forecast_plot_path}")
        plt.close()
    
    # Indicate successful completion of the script
    print("\nMultivariate LSTM model execution completed successfully!")

# Execute the main function when script is run directly
if __name__ == "__main__":
    main() 