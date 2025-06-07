import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from statsmodels.tsa.arima.model import ARIMA
import warnings

# Add parent directory to system path to import local modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import local modules
from src.data_processing import (
    load_water_quality_data, load_station_data, clean_and_preprocess,
    prepare_time_series_data, split_train_test, inverse_transform_predictions,
    prepare_multivariate_data
)
from src.time_series_utils import (
    check_stationarity, difference_series, fit_arima_model,
    forecast_arima, evaluate_forecast, find_optimal_arima_order,
    save_model_comparison
)
from src.visualization import (
    plot_time_series, plot_multiple_series, plot_correlation_heatmap,
    plot_prediction_vs_actual, plot_prediction_error
)

# Suppress warnings
warnings.filterwarnings('ignore')

class WaterQualityForecaster:
    """
    A class to forecast water quality parameters using various models.
    """
    
    def __init__(self, data_dir='data'):
        """
        Initialize the forecaster.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing the data files
        """
        self.data_dir = data_dir
        self.consolidated_data_path = os.path.join(data_dir, 'water_quality_data.csv')
        self.station_data_dir = os.path.join(data_dir, 'raw_station_data')
        self.models = {}
        self.scalers = {}
    
    def load_data(self, station_id=None):
        """
        Load water quality data.
        
        Parameters:
        -----------
        station_id : str or None
            Station ID to load data for. If None, load consolidated data.
            
        Returns:
        --------
        pandas DataFrame
            The loaded data
        """
        if station_id:
            data = load_station_data(station_id, self.station_data_dir)
        else:
            data = load_water_quality_data(self.consolidated_data_path)
        
        return clean_and_preprocess(data)
    
    def train_lstm_model(self, data, target_column, sequence_length=5, epochs=50, batch_size=32, test_size=0.2):
        """
        Train an LSTM model for forecasting.
        
        Parameters:
        -----------
        data : pandas DataFrame
            The data to train on
        target_column : str
            The column to predict
        sequence_length : int
            Number of previous time steps to use for prediction
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        test_size : float
            Proportion of data to use for testing
            
        Returns:
        --------
        dict
            Dictionary containing model, history, and evaluation metrics
        """
        # Prepare the data
        X, y, scaler = prepare_time_series_data(data, target_column, sequence_length)
        X_train, X_test, y_train, y_test = split_train_test(X, y, test_size)
        
        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        
        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=0
        )
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate the model
        y_test_inv = inverse_transform_predictions(y_test, scaler)
        y_pred_inv = inverse_transform_predictions(y_pred, scaler)
        
        metrics = evaluate_forecast(y_test_inv, y_pred_inv)
        
        # Store the model and scaler
        model_key = f"lstm_{target_column}"
        self.models[model_key] = model
        self.scalers[model_key] = scaler
        
        # Save test samples to a file for comparison
        results_file = os.path.join('outputs', f'{target_column}_lstm_test_results.txt')
        try:
            # Create the output directory if it doesn't exist
            os.makedirs(os.path.dirname(results_file), exist_ok=True)
            
            # Save the results to a file
            with open(results_file, 'w') as f:
                f.write(f'LSTM Model Test Results for {target_column}\n')
                f.write('=' * 40 + '\n\n')
                f.write('Test Metrics:\n')
                f.write(f'  RMSE: {metrics["rmse"]:.4f}\n')
                f.write(f'  MAE: {metrics["mae"]:.4f}\n')
                f.write(f'  R²: {metrics["r2"]:.4f}\n\n')
                f.write('Sample Test Predictions (first 20 samples):\n')
                f.write('Index | Actual | Predicted | Error\n')
                f.write('-' * 40 + '\n')
                
                # Write up to 20 sample predictions
                num_samples = min(20, len(y_test_inv))
                for i in range(num_samples):
                    actual = y_test_inv[i][0]
                    pred = y_pred_inv[i][0]
                    error = actual - pred
                    f.write(f'{i:5d} | {actual:7.2f} | {pred:9.2f} | {error:6.2f}\n')
                    
            print(f"LSTM test results saved to {results_file}")
            
        except Exception as e:
            print(f"Error saving LSTM test results: {str(e)}")
        
        return {
            'model': model,
            'history': history,
            'metrics': metrics,
            'predictions': y_pred_inv,
            'actual': y_test_inv,
            'scaler': scaler
        }
    
    def train_arima_model(self, data, target_column, test_size=0.2):
        """
        Train an ARIMA model for forecasting.
        
        Parameters:
        -----------
        data : pandas DataFrame
            The data to train on
        target_column : str
            The column to predict
        test_size : float
            Proportion of data to use for testing
            
        Returns:
        --------
        dict
            Dictionary containing model and evaluation metrics
        """
        # Prepare the data
        data_sorted = data.sort_values('dateTime')
        series = data_sorted[target_column]
        
        # Split into train and test
        split_idx = int(len(series) * (1 - test_size))
        train_data = series[:split_idx]
        test_data = series[split_idx:]
        
        # Check if series is stationary
        is_stationary, _ = check_stationarity(train_data, plot=False)
        
        # If not stationary, difference the series
        if not is_stationary:
            train_data_diff = difference_series(train_data)
            is_stationary_diff, _ = check_stationarity(train_data_diff, plot=False)
            
            if is_stationary_diff:
                # Find optimal ARIMA order
                p, d, q = find_optimal_arima_order(train_data)
            else:
                # Default order
                p, d, q = 1, 1, 1
        else:
            # Find optimal ARIMA order
            p, d, q = find_optimal_arima_order(train_data)
        
        # Fit ARIMA model
        model = fit_arima_model(train_data, order=(p, d, q))
        
        # Forecast for test period
        forecast_steps = len(test_data)
        forecast, conf_int = forecast_arima(model, steps=forecast_steps)
        
        # Evaluate the model
        metrics = evaluate_forecast(test_data.values, forecast.values)
        
        # Store the model
        model_key = f"arima_{target_column}"
        self.models[model_key] = model
        
        # Save test samples to a file for comparison
        results_file = os.path.join('outputs', f'{target_column}_arima_test_results.txt')
        try:
            # Create the output directory if it doesn't exist
            os.makedirs(os.path.dirname(results_file), exist_ok=True)
            
            # Save the results to a file
            with open(results_file, 'w') as f:
                f.write(f'ARIMA Model Test Results for {target_column}\n')
                f.write('=' * 40 + '\n\n')
                f.write(f'Order: ({p}, {d}, {q})\n\n')
                f.write('Test Metrics:\n')
                f.write(f'  RMSE: {metrics["rmse"]:.4f}\n')
                f.write(f'  MAE: {metrics["mae"]:.4f}\n')
                if metrics["r2"] is not None:
                    f.write(f'  R²: {metrics["r2"]:.4f}\n\n')
                
                f.write('Sample Test Predictions (first 20 samples):\n')
                f.write('Index | Actual | Predicted | Error\n')
                f.write('-' * 40 + '\n')
                
                # Write up to 20 sample predictions
                num_samples = min(20, len(test_data))
                for i in range(num_samples):
                    actual = test_data.iloc[i]
                    pred = forecast.iloc[i]
                    error = actual - pred
                    f.write(f'{i:5d} | {actual:7.2f} | {pred:9.2f} | {error:6.2f}\n')
                    
            print(f"ARIMA test results saved to {results_file}")
            
        except Exception as e:
            print(f"Error saving ARIMA test results: {str(e)}")
        
        return {
            'model': model,
            'metrics': metrics,
            'predictions': forecast,
            'actual': test_data,
            'confidence_intervals': conf_int,
            'order': (p, d, q)
        }
    
    def train_multivariate_lstm(self, data, target_columns, sequence_length=5, epochs=50, batch_size=32, test_size=0.2):
        """
        Train a multivariate LSTM model for forecasting multiple water quality parameters.
        
        Parameters:
        -----------
        data : pandas DataFrame
            The data to train on
        target_columns : list
            List of columns to predict
        sequence_length : int
            Number of previous time steps to use for prediction
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        test_size : float
            Proportion of data to use for testing
            
        Returns:
        --------
        dict
            Dictionary containing model, history, and evaluation metrics
        """
        # Prepare the data
        X, y, scalers = prepare_multivariate_data(data, target_columns, sequence_length)
        X_train, X_test, y_train, y_test = split_train_test(X, y, test_size)
        
        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(units=100, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=len(target_columns)))
        
        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=0
        )
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Evaluate the model
        metrics = {}
        predictions = {}
        actuals = {}
        
        for i, col in enumerate(target_columns):
            y_test_col = y_test[:, i].reshape(-1, 1)
            y_pred_col = y_pred[:, i].reshape(-1, 1)
            
            # Inverse transform
            scaler = scalers[col]
            y_test_inv = inverse_transform_predictions(y_test_col, scaler)
            y_pred_inv = inverse_transform_predictions(y_pred_col, scaler)
            
            # Calculate metrics
            metrics[col] = evaluate_forecast(y_test_inv, y_pred_inv)
            
            # Store predictions and actuals
            predictions[col] = y_pred_inv
            actuals[col] = y_test_inv
        
        # Store the model and scalers
        model_key = "multivariate_lstm"
        self.models[model_key] = model
        self.scalers[model_key] = scalers
        
        return {
            'model': model,
            'history': history,
            'metrics': metrics,
            'predictions': predictions,
            'actual': actuals,
            'scalers': scalers
        }
    
    def predict(self, station_id, target_columns, date, model_type='lstm'):
        """
        Make predictions for a specific station, parameters, and date.
        
        Parameters:
        -----------
        station_id : str
            Station ID to make predictions for
        target_columns : list
            List of parameters to predict
        date : str or datetime
            Date to make predictions for
        model_type : str
            Type of model to use ('lstm' or 'arima')
            
        Returns:
        --------
        dict
            Dictionary containing predictions for each parameter
        """
        # Load station data
        data = self.load_data(station_id)
        
        # Check if model exists for all target columns
        missing_models = []
        for col in target_columns:
            model_key = f"{model_type}_{col}"
            if model_key not in self.models:
                missing_models.append(col)
        
        # Train missing models
        for col in missing_models:
            if model_type == 'lstm':
                self.train_lstm_model(data, col)
            elif model_type == 'arima':
                self.train_arima_model(data, col)
        
        # Convert date to datetime if string
        if isinstance(date, str):
            date = pd.to_datetime(date)
        
        # Make predictions
        predictions = {}
        for col in target_columns:
            model_key = f"{model_type}_{col}"
            
            if model_type == 'lstm':
                # Get the most recent data for the sequence
                sequence_length = 5  # Default sequence length
                
                # Find data points before the prediction date
                valid_data = data[data['dateTime'] < date].sort_values('dateTime')
                if len(valid_data) < sequence_length:
                    raise ValueError(f"Not enough historical data for prediction. Need at least {sequence_length} data points.")
                
                # Get the most recent sequence
                recent_data = valid_data.iloc[-sequence_length:][col].values.reshape(-1, 1)
                
                # Scale the data
                scaler = self.scalers[model_key]
                recent_data_scaled = scaler.transform(recent_data)
                
                # Reshape for LSTM input
                X = recent_data_scaled.reshape(1, sequence_length, 1)
                
                # Make prediction
                model = self.models[model_key]
                pred_scaled = model.predict(X)
                
                # Inverse transform
                pred = scaler.inverse_transform(pred_scaled)[0][0]
                
            elif model_type == 'arima':
                # Get the model
                model = self.models[model_key]
                
                # Make prediction for 1 step ahead
                forecast, _ = forecast_arima(model, steps=1)
                pred = forecast.values[0]
            
            predictions[col] = pred
        
        return predictions

def predict_water_quality(station_id, parameters, prediction_date):
    """
    Predict water quality parameters for a specific station and date.
    This is the main function for external use.
    
    Parameters:
    -----------
    station_id : str
        Station ID to make predictions for
    parameters : list
        List of parameters to predict
    prediction_date : str or datetime
        Date to make predictions for
        
    Returns:
    --------
    dict
        Dictionary containing predictions for each parameter
    """
    # Initialize the forecaster
    forecaster = WaterQualityForecaster()
    
    # Load the data
    data = forecaster.load_data(station_id)
    
    # Train models for each parameter
    results = {}
    for param in parameters:
        # Train LSTM model
        lstm_result = forecaster.train_lstm_model(data, param)
        
        # Train ARIMA model
        arima_result = forecaster.train_arima_model(data, param)
        
        # Save model comparison 
        save_model_comparison(param, lstm_result, arima_result)
        
        results[param] = {
            'lstm': lstm_result,
            'arima': arima_result
        }
    
    # Make predictions
    return forecaster.predict(station_id, parameters, prediction_date)

def save_model_comparison(parameter, lstm_result, arima_result):
    """
    Save a side-by-side comparison of LSTM and ARIMA model results to a file.
    
    Parameters:
    -----------
    parameter : str
        The parameter name (e.g., 'temperature')
    lstm_result : dict
        Dictionary containing LSTM model results
    arima_result : dict
        Dictionary containing ARIMA model results
    """
    # Get the metrics and data
    lstm_metrics = lstm_result['metrics'] 
    arima_metrics = arima_result['metrics']
    
    # Get actual and predicted values
    lstm_actual = lstm_result['actual']
    lstm_pred = lstm_result['predictions']
    arima_actual = arima_result['actual']
    arima_pred = arima_result['predictions']
    
    # Get the minimum number of samples for comparison
    num_samples = min(len(lstm_actual), len(arima_actual), 20)
    
    # Create the output file
    comparison_file = os.path.join('outputs', f'{parameter}_model_comparison.txt')
    try:
        # Create output directory if needed
        os.makedirs(os.path.dirname(comparison_file), exist_ok=True)
        
        with open(comparison_file, 'w') as f:
            f.write(f'Water Quality Forecasting: {parameter.upper()} Model Comparison\n')
            f.write('=' * 70 + '\n\n')
            
            # Write performance metrics
            f.write('Performance Metrics:\n')
            f.write('-' * 70 + '\n')
            f.write(f"{'Metric':<10} | {'LSTM':<15} | {'ARIMA':<15}\n")
            f.write('-' * 70 + '\n')
            f.write(f"{'RMSE':<10} | {lstm_metrics['rmse']:<15.4f} | {arima_metrics['rmse']:<15.4f}\n")
            f.write(f"{'MAE':<10} | {lstm_metrics['mae']:<15.4f} | {arima_metrics['mae']:<15.4f}\n")
            f.write(f"{'R²':<10} | {lstm_metrics['r2']:<15.4f} | {arima_metrics['r2'] if arima_metrics['r2'] is not None else 'N/A':<15}\n")
            if lstm_metrics['mape'] is not None and arima_metrics['mape'] is not None:
                f.write(f"{'MAPE (%)':<10} | {lstm_metrics['mape']:<15.2f} | {arima_metrics['mape']:<15.2f}\n")
            f.write('\n')
            
            # Write sample predictions
            f.write('Sample Test Predictions:\n')
            f.write('-' * 70 + '\n')
            f.write(f"{'Index':<6} | {'Actual':<8} | {'LSTM Pred':<10} | {'LSTM Err':<8} | {'ARIMA Pred':<10} | {'ARIMA Err':<8}\n")
            f.write('-' * 70 + '\n')
            
            for i in range(num_samples):
                # LSTM values
                lstm_act = lstm_actual[i][0] if isinstance(lstm_actual[i], np.ndarray) else lstm_actual[i]
                lstm_p = lstm_pred[i][0] if isinstance(lstm_pred[i], np.ndarray) else lstm_pred[i]
                lstm_err = lstm_act - lstm_p
                
                # ARIMA values
                arima_act = arima_actual.iloc[i] if hasattr(arima_actual, 'iloc') else arima_actual[i]
                arima_p = arima_pred.iloc[i] if hasattr(arima_pred, 'iloc') else arima_pred[i]
                arima_err = arima_act - arima_p
                
                f.write(f"{i:<6} | {lstm_act:<8.2f} | {lstm_p:<10.2f} | {lstm_err:<8.2f} | {arima_p:<10.2f} | {arima_err:<8.2f}\n")
            
            # Add conclusion
            f.write('\nConclusion:\n')
            if lstm_metrics['rmse'] < arima_metrics['rmse']:
                f.write(f"LSTM model performs better for {parameter} prediction with lower RMSE ({lstm_metrics['rmse']:.4f} vs {arima_metrics['rmse']:.4f}).\n")
            else:
                f.write(f"ARIMA model performs better for {parameter} prediction with lower RMSE ({arima_metrics['rmse']:.4f} vs {lstm_metrics['rmse']:.4f}).\n")
        
        print(f"Model comparison saved to {comparison_file}")
        
    except Exception as e:
        print(f"Error saving model comparison: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Example usage
    try:
        station_id = "02336300"  # Example station ID
        parameters = ["temperature"]
        prediction_date = "2023-01-01"
        
        predictions = predict_water_quality(station_id, parameters, prediction_date)
        
        print(f"Predictions for station {station_id} on {prediction_date}:")
        for param, value in predictions.items():
            print(f"{param}: {value:.2f}")
    
    except Exception as e:
        print(f"Error: {str(e)}") 