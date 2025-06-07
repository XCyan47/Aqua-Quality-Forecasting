#!/usr/bin/env python3
"""
Exploratory Data Analysis for Water Quality Data
This script performs exploratory analysis on the water quality datasets.
"""

# Import required libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical computations
import matplotlib.pyplot as plt  # For creating visualizations
import seaborn as sns  # For enhanced visualizations
import sys  # For system-specific parameters and functions
import os  # For interacting with the operating system

# Add the parent directory to the system path
# This allows importing modules from src directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import custom functions from the src package
from src.data_processing import load_water_quality_data, clean_and_preprocess

def main():
    # Start the exploratory data analysis process
    print("Starting exploratory data analysis...")
    # Load the water quality dataset from CSV file
    data = load_water_quality_data('data/water_quality_data.csv')
    
    # Create the outputs directory if it doesn't already exist
    # exist_ok=True prevents errors if directory already exists
    os.makedirs('outputs', exist_ok=True)
    
    # Clean and preprocess the data
    print("Loading and preprocessing data...")
    data = clean_and_preprocess(data)
    
    # Set the dateTime column as the DataFrame's index
    # This allows for time-based operations and plotting
    data = data.set_index('dateTime')
    
    # Display basic information about the dataset
    # Output the dimensions of the dataset (rows, columns)
    print(f"\nDataset shape: {data.shape}")
    # Show the time range covered by the dataset
    print(f"Dataset date range: {data.index.min()} to {data.index.max()}")
    # Display the total number of samples
    print(f"Number of data points: {len(data)}")
    
    # Show the data types of each column
    print("\nColumn data types:")
    print(data.dtypes)
    
    # Display statistical summaries of all numeric columns
    # This includes count, mean, std, min, 25%, 50%, 75%, max
    print("\nBasic statistics:")
    stats = data.describe()
    print(stats)
    
    # Check for missing values in the dataset
    # Count the number of NaN values in each column
    missing_values = data.isnull().sum()
    print("\nMissing values per column:")
    print(missing_values)
    
    # Define the most important water quality parameters to analyze
    important_params = ['temperature', 'pH', 'dissolved_oxygen']
    
    # Display detailed statistics for each important parameter
    print("\nParameter statistics:")
    for param in important_params:
        # Check if the parameter exists in the dataset
        if param in data.columns:
            # Get descriptive statistics for the parameter
            param_stats = data[param].describe()
            # Print formatted statistics
            print(f"\n{param.capitalize()} Statistics:")
            print(f"Count: {param_stats['count']}")
            print(f"Mean: {param_stats['mean']:.2f}")
            print(f"Std: {param_stats['std']:.2f}")
            print(f"Min: {param_stats['min']:.2f}")
            print(f"25%: {param_stats['25%']:.2f}")
            print(f"50%: {param_stats['50%']:.2f}")
            print(f"75%: {param_stats['75%']:.2f}")
            print(f"Max: {param_stats['max']:.2f}")
    
    # Calculate and display correlation matrix
    # Shows how strongly parameters are related to each other
    print("\nCorrelation Matrix:")
    corr_matrix = data.corr()
    print(corr_matrix)
    
    # Save all calculated statistics to a file
    try:
        # Define the output file path
        stats_file = os.path.join('outputs', 'exploratory_stats.txt')
        # Open the file in write mode
        with open(stats_file, 'w') as f:
            # Write header information
            f.write("Water Quality Data Exploratory Analysis\n")
            f.write("=======================================\n\n")
            f.write(f"Dataset shape: {data.shape}\n")
            f.write(f"Date range: {data.index.min()} to {data.index.max()}\n")
            f.write(f"Total samples: {len(data)}\n\n")
            
            # Write statistics for each important parameter
            f.write("Parameter Statistics\n")
            f.write("-------------------\n")
            for param in important_params:
                if param in data.columns:
                    param_stats = data[param].describe()
                    f.write(f"\n{param.capitalize()}:\n")
                    f.write(f"  Mean: {param_stats['mean']:.2f}\n")
                    f.write(f"  Std: {param_stats['std']:.2f}\n")
                    f.write(f"  Min: {param_stats['min']:.2f}\n")
                    f.write(f"  Max: {param_stats['max']:.2f}\n")
            
            # Write correlation matrix
            f.write("\nCorrelation Matrix\n")
            f.write("-----------------\n")
            f.write(corr_matrix.to_string())
            
        # Confirm file was saved successfully
        print(f"\nStatistics saved to {stats_file}")
    except Exception as e:
        # Handle any errors that occur during file writing
        print(f"Error saving statistics: {e}")
    
    # Indicate successful completion of the analysis
    print("\nExploratory analysis completed successfully!")

# Execute the main function when script is run directly
if __name__ == "__main__":
    main() 