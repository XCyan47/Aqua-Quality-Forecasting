import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

def check_stationarity(time_series, window=12, plot=True, figsize=(12, 8)):
    """
    Check if a time series is stationary using the Augmented Dickey-Fuller test
    and rolling statistics.
    
    Parameters:
    -----------
    time_series : pandas Series
        The time series to check
    window : int
        Window size for rolling statistics
    plot : bool
        Whether to generate plots
    figsize : tuple
        Figure size for the plots
        
    Returns:
    --------
    tuple
        (is_stationary, p_value) where is_stationary is a boolean and p_value is the ADF test p-value
    """
    # Perform ADF test
    result = adfuller(time_series.dropna())
    adf_stat, p_value = result[0], result[1]
    is_stationary = p_value <= 0.05
    
    if plot:
        # Create the figure
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        
        # Original time series
        axes[0].plot(time_series)
        axes[0].set_title('Original Time Series')
        axes[0].set_xlabel('Date')
        
        # Rolling mean and std
        rolling_mean = time_series.rolling(window=window).mean()
        rolling_std = time_series.rolling(window=window).std()
        
        axes[1].plot(time_series, label='Original')
        axes[1].plot(rolling_mean, label=f'Rolling Mean (window={window})')
        axes[1].plot(rolling_std, label=f'Rolling Std (window={window})')
        axes[1].set_title('Rolling Statistics')
        axes[1].legend()
        
        # Autocorrelation
        plot_acf(time_series.dropna(), ax=axes[2])
        axes[2].set_title('Autocorrelation Function')
        
        plt.tight_layout()
        plt.show()
        
        # Print ADF test results
        print(f'ADF Statistic: {adf_stat:.4f}')
        print(f'p-value: {p_value:.4f}')
        print(f'Is Stationary: {is_stationary}')
        
        # Print critical values
        for key, value in result[4].items():
            print(f'Critical Value ({key}): {value:.4f}')
    
    return is_stationary, p_value

def difference_series(time_series, order=1):
    """
    Apply differencing to make a time series stationary.
    
    Parameters:
    -----------
    time_series : pandas Series
        The time series to difference
    order : int
        The order of differencing
        
    Returns:
    --------
    pandas Series
        The differenced time series
    """
    diff_series = time_series.copy()
    for _ in range(order):
        diff_series = diff_series.diff().dropna()
    return diff_series

def fit_arima_model(time_series, order=(1, 1, 1), seasonal_order=None):
    """
    Fit an ARIMA or SARIMA model to the time series.
    
    Parameters:
    -----------
    time_series : pandas Series
        The time series to model
    order : tuple
        ARIMA order (p, d, q)
    seasonal_order : tuple or None
        Seasonal order (P, D, Q, s) for SARIMA
        
    Returns:
    --------
    statsmodels ARIMA model
        The fitted model
    """
    if seasonal_order:
        # SARIMA model
        model = ARIMA(
            time_series, 
            order=order, 
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
    else:
        # ARIMA model
        model = ARIMA(
            time_series, 
            order=order
        )
    
    return model.fit()

def forecast_arima(model, steps=10, alpha=0.05):
    """
    Generate forecasts from a fitted ARIMA model.
    
    Parameters:
    -----------
    model : statsmodels ARIMA model
        The fitted ARIMA model
    steps : int
        Number of steps to forecast
    alpha : float
        Significance level for prediction intervals
        
    Returns:
    --------
    tuple
        (forecast, confidence_intervals) forecast is a Series and 
        confidence_intervals is a DataFrame with lower and upper bounds
    """
    forecast_result = model.get_forecast(steps=steps, alpha=alpha)
    forecast = forecast_result.predicted_mean
    confidence_intervals = forecast_result.conf_int()
    
    return forecast, confidence_intervals

def evaluate_forecast(actual, predicted):
    """
    Evaluate forecast performance using various metrics.
    
    Parameters:
    -----------
    actual : array-like
        Actual values
    predicted : array-like
        Predicted values
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    
    # Calculate MAPE (Mean Absolute Percentage Error) if no zeros in actual
    if not np.any(actual == 0):
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    else:
        mape = None
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape
    }

def plot_forecast(actual, predicted, confidence_intervals=None, title='Forecast vs Actual', figsize=(12, 6)):
    """
    Plot actual values against forecasted values.
    
    Parameters:
    -----------
    actual : pandas Series
        Actual values with datetime index
    predicted : pandas Series
        Predicted values with datetime index
    confidence_intervals : pandas DataFrame or None
        Confidence intervals for the forecast
    title : str
        Plot title
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib Figure
        The generated plot
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot actual values
    ax.plot(actual, label='Actual', color='blue', marker='o')
    
    # Plot predicted values
    ax.plot(predicted, label='Forecast', color='red', marker='x')
    
    # Plot confidence intervals if provided
    if confidence_intervals is not None:
        ax.fill_between(
            confidence_intervals.index,
            confidence_intervals.iloc[:, 0],
            confidence_intervals.iloc[:, 1],
            color='pink', alpha=0.3,
            label='95% Confidence Interval'
        )
    
    ax.set_title(title)
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    
    return fig

def find_optimal_arima_order(time_series, p_range=range(0, 3), d_range=range(0, 3), q_range=range(0, 3)):
    """
    Find the optimal ARIMA order using AIC.
    
    Parameters:
    -----------
    time_series : pandas Series
        The time series to model
    p_range : range
        Range of p values to try
    d_range : range
        Range of d values to try
    q_range : range
        Range of q values to try
        
    Returns:
    --------
    tuple
        The optimal order (p, d, q)
    """
    best_aic = float('inf')
    best_order = None
    
    for p in p_range:
        for d in d_range:
            for q in q_range:
                try:
                    model = ARIMA(time_series, order=(p, d, q))
                    result = model.fit()
                    aic = result.aic
                    
                    if aic < best_aic:
                        best_aic = aic
                        best_order = (p, d, q)
                        print(f'ARIMA{best_order} - AIC: {best_aic}')
                except:
                    continue
    
    return best_order

def save_model_comparison(actual, lstm_pred, arima_pred, lstm_metrics, arima_metrics, output_file):
    """
    Save a comparison of model predictions and actual values to a text file.
    
    Parameters:
    -----------
    actual : array-like
        Actual values
    lstm_pred : array-like
        LSTM model predictions
    arima_pred : array-like
        ARIMA model predictions
    lstm_metrics : dict
        Dictionary containing LSTM evaluation metrics
    arima_metrics : dict
        Dictionary containing ARIMA evaluation metrics
    output_file : str
        Path to the output file
        
    Returns:
    --------
    None
    """
    try:
        # Print debug info
        print(f"Saving comparison to: {output_file}")
        print(f"Data shapes - actual: {len(actual)}, lstm_pred: {len(lstm_pred)}, arima_pred: {len(arima_pred)}")
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir:  # Only create directory if path contains directory
            print(f"Creating directory if it doesn't exist: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write('Temperature Predictions Comparison (Actual vs Predicted)\n')
            f.write('===========================================\n\n')
            f.write('Day     | Actual | LSTM (RMSE={:.2f}) | ARIMA (RMSE={:.2f})\n'.format(
                lstm_metrics['rmse'], arima_metrics['rmse']))
            f.write('-------------------------------------------\n')
            
            for i in range(len(actual)):
                f.write('Day {:2d} | {:6.2f} | {:6.2f}        | {:6.2f}\n'.format(
                    i+1, actual[i], lstm_pred[i], arima_pred[i]))
            
            f.write('\nLSTM Performance Metrics:\n')
            f.write('  RMSE: {:.2f}\n'.format(lstm_metrics['rmse']))
            f.write('  MAE: {:.2f}\n'.format(lstm_metrics['mae']))
            f.write('  R²: {:.2f}\n'.format(lstm_metrics['r2']))
            
            f.write('\nARIMA Performance Metrics:\n')
            f.write('  RMSE: {:.2f}\n'.format(arima_metrics['rmse']))
            f.write('  MAE: {:.2f}\n'.format(arima_metrics['mae']))
            if arima_metrics['r2'] is not None:
                f.write('  R²: {:.2f}\n'.format(arima_metrics['r2']))
        
        print(f"Model comparison saved to {output_file}")
        
    except Exception as e:
        print(f"Error saving model comparison: {str(e)}")
        import traceback
        traceback.print_exc() 