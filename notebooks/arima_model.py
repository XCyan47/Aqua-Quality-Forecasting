#!/usr/bin/env python3
"""
Multivariate ARIMA Models for Water Quality Prediction
This script implements separate ARIMA models for time series forecasting 
of multiple water quality parameters.
"""

# Import required libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical computations
import matplotlib.pyplot as plt  # For creating visualizations
from statsmodels.tsa.arima.model import ARIMA  # For ARIMA modeling
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # For ACF and PACF plots
import sys  # For system-specific parameters and functions
import os  # For interacting with the operating system

# Add the parent directory to the system path
# This allows importing modules from the src directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import custom functions from the src package
from src.data_processing import load_water_quality_data, clean_and_preprocess

def train_arima_model(series, order=(1, 1, 1)):
    """
    Train an ARIMA model on the provided time series
    
    Args:
        series (pd.Series): Time series data
        order (tuple): ARIMA order parameters (p, d, q)
        
    Returns:
        model_fit: Fitted ARIMA model
    """
    # Create ARIMA model with specified order
    # p: AR (autoregression) order - number of lag observations
    # d: Integration order - number of differencing required to make the time series stationary
    # q: MA (moving average) order - size of the moving average window
    model = ARIMA(series, order=order)
    
    # Fit the model to the time series data
    model_fit = model.fit()
    
    return model_fit

def main():
    # Announce the start of ARIMA model training
    print("Starting Multivariate ARIMA model analysis...")
    
    # Get and print the current working directory for debugging purposes
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")
    
    # Create output directories for storing results and visualizations
    output_dir = os.path.join(current_dir, 'outputs')
    os.makedirs(output_dir, exist_ok=True)  # Create if not exists
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)  # Create if not exists
    print(f"Plots will be saved to: {plots_dir}")
    
    # Load and preprocess the water quality data
    print("Loading and preprocessing data...")
    data = load_water_quality_data('data/water_quality_data.csv')
    data = clean_and_preprocess(data)
    # Set datetime as index for time series analysis
    data = data.set_index('dateTime')
    
    # Define the water quality parameters to model
    water_params = ['temperature', 'dissolved_oxygen', 'pH']
    print(f"Modeling parameters: {water_params}")
    print(f"Data shape: {data.shape}")
    
    # Create a dictionary to store model results for all parameters
    model_results = {}
    
    # Train an ARIMA model for each water quality parameter
    for param in water_params:
        # Print separator for better readability in output
        print(f"\n{'='*50}")
        print(f"Modeling parameter: {param}")
        print(f"{'='*50}")
        
        # Extract the time series for the current parameter and remove missing values
        time_series = data[param].dropna()
        print(f"Time series length: {len(time_series)} data points")
        
        # Display basic statistics about the time series
        print("\nTime Series Statistics:")
        print(f"Mean: {time_series.mean():.2f}")
        print(f"Standard Deviation: {time_series.std():.2f}")
        print(f"Min: {time_series.min():.2f}")
        print(f"Max: {time_series.max():.2f}")
        
        # Split the data into training and testing sets (80% train, 20% test)
        train_size = int(len(time_series) * 0.8)
        train_data = time_series[:train_size]
        test_data = time_series[train_size:]
        print(f"Training set size: {len(train_data)}, Test set size: {len(test_data)}")
        
        # Define ARIMA order parameters
        # These could be optimized using auto_arima or grid search
        p, d, q = 1, 1, 1  # Default order for all parameters
        
        # Try to fit the ARIMA model
        print("\nFitting ARIMA model...")
        try:
            # Train the ARIMA model using the training data
            model_fit = train_arima_model(train_data, order=(p, d, q))
            
            # Display model summary statistics for model evaluation
            print("\nARIMA Model Summary Statistics:")
            print(f"AIC: {model_fit.aic:.2f}")  # Akaike Information Criterion - lower is better
            print(f"BIC: {model_fit.bic:.2f}")  # Bayesian Information Criterion - lower is better
            
            # Generate predictions for the test period
            print("\nGenerating predictions for test data...")
            predictions = model_fit.forecast(steps=len(test_data))
            
            # Calculate error metrics on the test data
            test_mse = np.mean((test_data.values - predictions) ** 2)  # Mean Squared Error
            test_rmse = np.sqrt(test_mse)  # Root Mean Squared Error
            test_mae = np.mean(np.abs(test_data.values - predictions))  # Mean Absolute Error
            
            # Display error metrics
            print("\nTest Data Performance Metrics:")
            print(f"Mean Squared Error (MSE): {test_mse:.4f}")
            print(f"Root Mean Squared Error (RMSE): {test_rmse:.4f}")
            print(f"Mean Absolute Error (MAE): {test_mae:.4f}")
            
            # Save test results to a text file
            test_results_file = os.path.join(output_dir, f'{param}_arima_test_results.txt')
            with open(test_results_file, 'w') as f:
                f.write(f"ARIMA Model Test Results for {param}\n")
                f.write(f"Order: ({p}, {d}, {q})\n")
                f.write(f"Test MSE: {test_mse:.4f}\n")
                f.write(f"Test RMSE: {test_rmse:.4f}\n")
                f.write(f"Test MAE: {test_mae:.4f}\n")
            print(f"\nTest results saved to {test_results_file}")
            
            # Generate a forecast for future periods
            forecast_steps = 30  # Forecast for the next 30 days
            print(f"\nGenerating forecast for next {forecast_steps} days...")
            forecast = model_fit.forecast(steps=forecast_steps)
            
            # Generate confidence intervals for the forecast
            forecast_ci = model_fit.get_forecast(steps=forecast_steps).conf_int()
            
            # Display the first few forecast values and their confidence intervals
            print("\nForecast Results (first 5 days):")
            for i in range(min(5, len(forecast))):
                lower_ci = forecast_ci.iloc[i, 0]  # Lower bound of 95% CI
                upper_ci = forecast_ci.iloc[i, 1]  # Upper bound of 95% CI
                print(f"Day {i+1}: {forecast[i]:.2f} (95% CI: {lower_ci:.2f} to {upper_ci:.2f})")
            
            # Extract model residuals for diagnostics
            residuals = model_fit.resid
            
            # Calculate error metrics on the training data
            mse = np.mean(residuals ** 2)  # Mean Squared Error
            rmse = np.sqrt(mse)  # Root Mean Squared Error
            mae = np.mean(np.abs(residuals))  # Mean Absolute Error
            
            # Display training performance metrics
            print("\nTraining Model Performance Metrics:")
            print(f"Mean Squared Error (MSE): {mse:.4f}")
            print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
            print(f"Mean Absolute Error (MAE): {mae:.4f}")
            
            # Create a plot of actual test values vs. predictions
            print("\nCreating prediction plot...")
            plt.figure(figsize=(12, 6))
            plt.plot(test_data.index, test_data.values, 'b-', label='Actual')
            plt.plot(test_data.index, predictions, 'r--', label='Predicted')
            plt.title(f'ARIMA Model: Actual vs Predicted {param.capitalize()}')
            plt.xlabel('Date')
            plt.ylabel(f'{param.capitalize()}')
            plt.legend()
            plt.grid(True)
            
            # Save the prediction plot
            plot_path = os.path.join(plots_dir, f'arima_prediction_{param}.png')
            print(f"Attempting to save plot to: {plot_path}")
            plt.savefig(plot_path)
            print(f"Plot saved successfully to: {plot_path}")
            plt.close()
            
            # Create a plot of the forecast with confidence intervals
            plt.figure(figsize=(12, 6))
            
            # Plot recent historical data for context (last 100 observations)
            hist_data = time_series[-100:]
            plt.plot(hist_data.index, hist_data.values, 'b-', label='Historical Data')
            
            # Generate dates for the forecast period
            last_date = data.index[-1]
            future_dates = pd.date_range(start=last_date, periods=forecast_steps+1)[1:]
            
            # Plot the forecast and its confidence interval
            plt.plot(future_dates, forecast, 'r--', label='Forecast')
            plt.fill_between(future_dates, 
                            forecast_ci.iloc[:, 0],  # Lower bound
                            forecast_ci.iloc[:, 1],  # Upper bound
                            color='pink', alpha=0.3, label='95% Confidence Interval')
            
            plt.title(f'ARIMA Model: {forecast_steps}-Day Forecast for {param.capitalize()}')
            plt.xlabel('Date')
            plt.ylabel(f'{param.capitalize()}')
            plt.legend()
            plt.grid(True)
            
            # Save the forecast plot
            forecast_path = os.path.join(plots_dir, f'arima_forecast_{param}.png')
            plt.savefig(forecast_path)
            print(f"Forecast plot saved to: {forecast_path}")
            plt.close()
            
            # Store model results in the dictionary for later use
            model_results[param] = {
                'model': model_fit,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'train_rmse': rmse,
                'train_mae': mae,
                'aic': model_fit.aic,
                'bic': model_fit.bic,
                'order': (p, d, q)
            }
            
        except Exception as e:
            # Handle any errors that occur during model training
            print(f"Error training ARIMA model for {param}: {e}")
            continue
    
    # Save consolidated results for all parameters to a file
    try:
        results_file = os.path.join(output_dir, 'multivariate_arima_results.txt')
        with open(results_file, 'w') as f:
            f.write("Multivariate ARIMA Model Results\n\n")
            # Write results for each parameter
            for param, results in model_results.items():
                f.write(f"{param.capitalize()}:\n")
                f.write(f"  Order: ({results['order'][0]}, {results['order'][1]}, {results['order'][2]})\n")
                f.write(f"  AIC: {results['aic']:.2f}\n")
                f.write(f"  BIC: {results['bic']:.2f}\n")
                f.write(f"  Training RMSE: {results['train_rmse']:.4f}\n")
                f.write(f"  Training MAE: {results['train_mae']:.4f}\n")
                f.write(f"  Test RMSE: {results['test_rmse']:.4f}\n")
                f.write(f"  Test MAE: {results['test_mae']:.4f}\n\n")
        print(f"\nOverall results saved to {results_file}")
    except Exception as e:
        # Handle any errors during file writing
        print(f"Error saving overall results: {e}")
    
    # Indicate successful completion
    print("\nMultivariate ARIMA model execution completed successfully!")

# Execute main function when script is run directly
if __name__ == "__main__":
    main() 