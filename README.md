# Aqua Quality Forecasting

A machine learning project for predicting water quality parameters in rivers and streams using time series analysis.

## Project Overview

This project uses historical water quality data from monitoring stations to predict future water quality parameters such as:
- Dissolved Oxygen (DO)
- pH
- Temperature

The models implemented utilize both traditional time series forecasting methods (ARIMA) and deep learning approaches (LSTM networks) to provide accurate predictions for environmental monitoring. Our analysis showed that LSTM significantly outperforms ARIMA for these parameters.

## Dataset

The dataset contains water quality measurements from various monitoring stations in Georgia, USA. Each station has recorded measurements over time for parameters that are crucial indicators of water quality and ecosystem health.

## Features

- **Data preprocessing** utilities for cleaning and preparing time series data
- **Exploratory data analysis** of water quality parameters
- **ARIMA model** implementation for traditional statistical forecasting
- **LSTM neural network** for advanced deep learning predictions
- **Comparative analysis** showing LSTM outperforms ARIMA by 76.26% in terms of RMSE
- **Visualization tools** for time series data and predictions
- **Test data prediction** functionality for making forecasts on new data

## Project Structure

```
Aqua-Quality-Forecasting/
│
├── data/                      # Data directory
│   ├── water_quality_data.csv # Consolidated dataset
│   └── raw_station_data/      # Individual station data
│
├── notebooks/                 # Python scripts
│   ├── exploratory_analysis.py  # EDA script
│   ├── lstm_model.py          # LSTM model training
│   ├── arima_model.py         # ARIMA model training
│   ├── model_comparison.py    # Compare LSTM vs ARIMA
│   └── make_predictions.py    # Make predictions with trained model
│
├── models/                    # Saved models
│   └── lstm_model.h5          # Trained LSTM model
│
├── test_data/                 # Test data directory
│   └── test1/                 # Test case subdirectory
│       ├── test1.csv          # Test data file
│       └── test1_prediction.csv # Generated predictions
│
├── outputs/                   # Output files and figures
│   ├── plots/                 # Generated plots
│   ├── predictions/           # Prediction results
│   ├── model_comparison/      # Model comparison results
│   └── exploratory_stats.txt  # Statistical summaries
│
├── src/                       # Source code
│   ├── data_processing.py     # Data cleaning and preparation
│   ├── time_series_utils.py   # Time series utilities
│   ├── visualization.py       # Plotting functions
│   └── main.py                # Main application entry point
│
├── venv/                      # Virtual environment (not in version control)
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Project Execution Steps

To run the water quality prediction models, follow these steps:

1. **Set up the environment**
   ```bash
   # Clone the repository or unzip the project
   # Navigate to the project directory
   cd Aqua-Quality-Forecasting
   
   # Create and activate a virtual environment
   python -m venv venv
   venv\Scripts\activate  # On Windows
   source venv/bin/activate  # On Unix/MacOS
   
   # Install required dependencies
   pip install -r requirements.txt
   ```

2. **Prepare the data**
   ```bash
   # Ensure your data file is in the data directory
   # The project is set up to use water_quality_data.csv by default
   ```

3. **Run the Exploratory Data Analysis**
   ```bash
   python notebooks/exploratory_analysis.py
   ```
   This will generate statistical summaries of your water quality data and save them to `outputs/exploratory_stats.txt`.

4. **Run the ARIMA Model**
   ```bash
   python notebooks/arima_model.py
   ```
   This will train an ARIMA model on the time series data, make predictions, and save results to the outputs directory.

5. **Run the LSTM Model**
   ```bash
   python notebooks/lstm_model.py
   ```
   This will train a deep learning LSTM model, make predictions, and save the model to `models/lstm_model.h5`.

6. **Compare Model Performance**
   ```bash
   python notebooks/model_comparison.py
   ```
   This will compare the performance of LSTM and ARIMA models and generate comprehensive comparison reports.

7. **Make Predictions on New Data**
   ```bash
   # For a specific test dataset in a subdirectory
   python notebooks/make_predictions.py --subdir test1
   
   # For the default test dataset
   python notebooks/make_predictions.py
   ```
   This will make predictions using the trained LSTM model on the test data and save results to the test data directory.

## Model Results

The models generate the following outputs:

- **Exploratory Analysis**: Statistical summaries of the water quality parameters in `outputs/exploratory_stats.txt`.
- **ARIMA Model**: Time series forecasting results with performance metrics.
- **LSTM Model**: Deep learning model for multi-step forecasting with performance metrics. The model is saved to `models/lstm_model.h5`.
- **Model Comparison**: Detailed comparison of LSTM vs. ARIMA performance in `outputs/model_comparison/`.
- **Predictions**: Forecasts for future values based on test data in `test_data/[test_name]/[test_name]_prediction.csv`.

## Model Performance

Based on our comprehensive analysis:

- **LSTM Model**:
  - Temperature: R² = 0.97, RMSE = 0.74
  - Dissolved Oxygen: R² = 0.97, RMSE = 0.15
  - pH: R² = 0.67, RMSE = 0.24

- **ARIMA Model**:
  - Temperature: R² = 0.23, RMSE = 3.69
  - Dissolved Oxygen: R² = 0.21, RMSE = 0.74
  - pH: R² = 0.13, RMSE = 0.39

**Overall Improvement**: LSTM outperforms ARIMA by an average of 76.26% in RMSE across all parameters.

## Test Data Requirements

The prediction functionality requires:
- CSV files with at least 10 consecutive days of data
- Data for temperature, dissolved oxygen, and pH (at minimum)
- Date column named either 'date' or 'dateTime'

Test data should be organized in subdirectories within the `test_data` directory, with each subdirectory named the same as the CSV file it contains.

## Future Improvements

- Incorporate weather data as additional predictors
- Implement ensemble methods combining multiple forecast models
- Add spatial analysis to account for geographic relationships between stations
- Develop a web-based dashboard for interactive visualization

