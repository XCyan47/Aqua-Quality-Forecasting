import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import matplotlib.ticker as ticker

# Set the aesthetics for all plots
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

def plot_time_series(df, column, title=None, figsize=(12, 6), marker=None, color='steelblue'):
    """
    Plot a time series with enhanced styling.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the time series data
    column : str
        Column name to plot
    title : str or None
        Plot title
    figsize : tuple
        Figure size
    marker : str or None
        Marker style
    color : str
        Line color
        
    Returns:
    --------
    matplotlib Figure
        The generated plot
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Ensure datetime index
    plot_df = df.copy()
    if 'dateTime' in plot_df.columns:
        plot_df.set_index('dateTime', inplace=True)
    
    # Plot the time series
    if marker:
        ax.plot(plot_df.index, plot_df[column], marker=marker, color=color, linewidth=2)
    else:
        ax.plot(plot_df.index, plot_df[column], color=color, linewidth=2)
    
    # Set title and labels
    if title is None:
        title = f'Time Series of {column}'
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel(column, fontsize=14)
    
    # Format the date axis
    date_format = DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()  # Rotate date labels
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Tight layout
    plt.tight_layout()
    
    return fig

def plot_multiple_series(df, columns, title=None, figsize=(15, 7), palette='tab10'):
    """
    Plot multiple time series on the same axis.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the time series data
    columns : list
        List of column names to plot
    title : str or None
        Plot title
    figsize : tuple
        Figure size
    palette : str or list
        Color palette for the series
        
    Returns:
    --------
    matplotlib Figure
        The generated plot
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Ensure datetime index
    plot_df = df.copy()
    if 'dateTime' in plot_df.columns:
        plot_df.set_index('dateTime', inplace=True)
    
    # Get color palette
    colors = sns.color_palette(palette, len(columns))
    
    # Plot each series
    for i, column in enumerate(columns):
        ax.plot(plot_df.index, plot_df[column], label=column, color=colors[i], linewidth=2)
    
    # Set title and labels
    if title is None:
        title = 'Time Series Comparison'
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Value', fontsize=14)
    
    # Format the date axis
    date_format = DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()  # Rotate date labels
    
    # Add legend
    ax.legend(loc='best', fontsize=12)
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Tight layout
    plt.tight_layout()
    
    return fig

def plot_station_comparison(df, column, stations, title=None, figsize=(15, 7)):
    """
    Compare the same parameter across different stations.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the data
    column : str
        Column name to compare
    stations : list
        List of station IDs to compare
    title : str or None
        Plot title
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib Figure
        The generated plot
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Filter data for each station and plot
    colors = sns.color_palette('viridis', len(stations))
    
    for i, station in enumerate(stations):
        station_df = df[df['site_no'] == station].copy()
        if 'dateTime' in station_df.columns:
            station_df.set_index('dateTime', inplace=True)
        
        ax.plot(
            station_df.index, 
            station_df[column], 
            label=f'Station {station}', 
            color=colors[i], 
            linewidth=2
        )
    
    # Set title and labels
    if title is None:
        title = f'Comparison of {column} Across Stations'
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel(column, fontsize=14)
    
    # Format the date axis
    date_format = DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(date_format)
    fig.autofmt_xdate()  # Rotate date labels
    
    # Add legend
    ax.legend(loc='best', fontsize=12)
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Tight layout
    plt.tight_layout()
    
    return fig

def plot_correlation_heatmap(df, columns=None, title='Correlation Heatmap', figsize=(12, 10)):
    """
    Create a correlation heatmap for the selected columns.
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the data
    columns : list or None
        List of column names to include. If None, use all numeric columns.
    title : str
        Plot title
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib Figure
        The generated plot
    """
    # Select columns
    if columns is None:
        columns = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Calculate correlation matrix
    corr_matrix = df[columns].corr()
    
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    heatmap = sns.heatmap(
        corr_matrix, 
        mask=mask,
        annot=True, 
        fmt=".2f", 
        cmap='coolwarm',
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    
    # Set title
    ax.set_title(title, fontsize=16, pad=20)
    
    # Rotate y-axis labels
    plt.yticks(rotation=0)
    
    # Tight layout
    plt.tight_layout()
    
    return fig

def plot_seasonal_decomposition(decomposition, title='Time Series Decomposition', figsize=(12, 10)):
    """
    Plot the components of a seasonal decomposition.
    
    Parameters:
    -----------
    decomposition : statsmodels DecomposeResult
        The result of seasonal decomposition
    title : str
        Plot title
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib Figure
        The generated plot
    """
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
    
    # Plot observed data
    decomposition.observed.plot(ax=axes[0], color='steelblue')
    axes[0].set_ylabel('Observed', fontsize=12)
    axes[0].set_title(title, fontsize=16)
    
    # Plot trend
    decomposition.trend.plot(ax=axes[1], color='darkred')
    axes[1].set_ylabel('Trend', fontsize=12)
    
    # Plot seasonal
    decomposition.seasonal.plot(ax=axes[2], color='forestgreen')
    axes[2].set_ylabel('Seasonal', fontsize=12)
    
    # Plot residual
    decomposition.resid.plot(ax=axes[3], color='darkorange')
    axes[3].set_ylabel('Residual', fontsize=12)
    
    # Add grid for better readability
    for ax in axes:
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Format date axis
    fig.autofmt_xdate()
    
    # Tight layout
    plt.tight_layout()
    
    return fig

def plot_prediction_vs_actual(actual, predicted, title='Prediction vs Actual', figsize=(12, 6)):
    """
    Plot predicted values against actual values.
    
    Parameters:
    -----------
    actual : array-like
        Actual values
    predicted : array-like
        Predicted values
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
    
    # Ensure inputs are arrays
    actual = np.array(actual).flatten()
    predicted = np.array(predicted).flatten()
    
    # Plot diagonal line (perfect predictions)
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    
    margin = (max_val - min_val) * 0.1
    ax.plot(
        [min_val - margin, max_val + margin], 
        [min_val - margin, max_val + margin], 
        'k--', label='Perfect Prediction'
    )
    
    # Plot scatter of actual vs predicted
    scatter = ax.scatter(
        actual, 
        predicted, 
        alpha=0.6, 
        c='steelblue', 
        edgecolors='k',
        s=80
    )
    
    # Set title and labels
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Actual Values', fontsize=14)
    ax.set_ylabel('Predicted Values', fontsize=14)
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Equal aspect ratio
    ax.set_aspect('equal')
    
    # Set limits with margin
    ax.set_xlim(min_val - margin, max_val + margin)
    ax.set_ylim(min_val - margin, max_val + margin)
    
    # Add legend
    ax.legend(loc='best')
    
    # Tight layout
    plt.tight_layout()
    
    return fig

def plot_prediction_error(actual, predicted, title='Prediction Error', figsize=(12, 6)):
    """
    Plot prediction errors (residuals).
    
    Parameters:
    -----------
    actual : array-like
        Actual values
    predicted : array-like
        Predicted values
    title : str
        Plot title
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib Figure
        The generated plot
    """
    # Calculate errors
    errors = np.array(actual) - np.array(predicted)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Histogram of errors
    sns.histplot(errors, kde=True, ax=ax1, color='steelblue')
    ax1.set_title('Distribution of Errors', fontsize=14)
    ax1.set_xlabel('Error', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    
    # Add a vertical line at x=0
    ax1.axvline(x=0, color='r', linestyle='--')
    
    # QQ plot of errors
    from scipy import stats
    stats.probplot(errors, plot=ax2)
    ax2.set_title('Q-Q Plot of Errors', fontsize=14)
    
    # Set main title
    fig.suptitle(title, fontsize=16)
    
    # Tight layout
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    
    return fig 