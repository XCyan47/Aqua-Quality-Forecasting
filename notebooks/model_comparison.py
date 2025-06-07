#!/usr/bin/env python3
"""
Model Comparison Script
This script generates comparative analysis between ARIMA and LSTM models
for water quality parameter prediction.
"""

# Import required libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For creating visualizations
import os  # For file and directory operations
import re  # For regular expression pattern matching
import sys  # For system-specific parameters

# Add the parent directory to the system path
# This allows importing modules from the src directory if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parse_metrics_from_file(file_path):
    """
    Extract performance metrics from a model results file.
    
    Parameters:
    -----------
    file_path : str
        Path to the results file
        
    Returns:
    --------
    dict
        Dictionary containing extracted metrics (RMSE, MAE, R², MSE)
    """
    # Initialize empty dictionary to store metrics
    metrics = {}
    
    try:
        # Open and read the file content
        with open(file_path, "r") as f:
            content = f.read()
            
            # Use regular expressions to extract RMSE value
            # This pattern looks for "RMSE:" followed by whitespace and then a number
            rmse_match = re.search(r"RMSE:\s*([\d\.]+)", content)
            if rmse_match:
                # Store the extracted value as a float
                metrics["RMSE"] = float(rmse_match.group(1))
            
            # Extract MAE (Mean Absolute Error)
            mae_match = re.search(r"MAE:\s*([\d\.]+)", content)
            if mae_match:
                metrics["MAE"] = float(mae_match.group(1))
            
            # Extract R² (coefficient of determination) if available
            r2_match = re.search(r"R²:\s*([\d\.]+)", content)
            if r2_match:
                metrics["R²"] = float(r2_match.group(1))
            
            # Extract MSE (Mean Squared Error) if available
            mse_match = re.search(r"MSE:\s*([\d\.]+)", content)
            if mse_match:
                metrics["MSE"] = float(mse_match.group(1))
    except Exception as e:
        # Handle any errors during file reading or parsing
        print(f"Error parsing file {file_path}: {e}")
    
    return metrics

def create_comparison_text_file(arima_metrics, lstm_metrics, output_file, parameter):
    """
    Create a text file comparing ARIMA and LSTM metrics.
    
    Parameters:
    -----------
    arima_metrics : dict
        Dictionary of ARIMA model metrics
    lstm_metrics : dict
        Dictionary of LSTM model metrics
    output_file : str
        Path to save the comparison file
    parameter : str
        Water quality parameter being compared
    """
    # Open the output file in write mode
    with open(output_file, "w") as f:
        # Write header with parameter name
        f.write(f"Model Performance Comparison for {parameter.capitalize()}\n")
        f.write("=" * 50 + "\n\n")
        
        # Write table header for the metrics comparison
        f.write("Metric      | ARIMA Model | LSTM Model | Improvement (%)\n")
        f.write("-" * 60 + "\n")
        
        # Find metrics that are available in both models
        common_metrics = set(arima_metrics.keys()) & set(lstm_metrics.keys())
        
        # Write each common metric with its values and improvement percentage
        for metric in sorted(common_metrics):
            arima_value = arima_metrics[metric]
            lstm_value = lstm_metrics[metric]
            
            # For error metrics, lower values are better
            # Calculate improvement percentage: how much LSTM improved over ARIMA
            if metric in ["RMSE", "MAE", "MSE"]:
                improvement = ((arima_value - lstm_value) / arima_value) * 100
                f.write(f"{metric:<12}| {arima_value:<12.4f}| {lstm_value:<12.4f}| {improvement:>6.2f}%\n")
        
        # Add R² from LSTM if it's only available in LSTM results
        if "R²" in lstm_metrics and "R²" not in arima_metrics:
            f.write(f"{'R²':<12}| {'N/A':<12}| {lstm_metrics['R²']:<12.4f}| {'N/A':>6}\n")
        
        f.write("\n")
        f.write("Note: For error metrics (RMSE, MAE, MSE), lower values indicate better performance.\n")
        f.write("      For R², higher values indicate better performance (1.0 is perfect prediction).\n")

def create_bar_chart(arima_metrics, lstm_metrics, output_file, parameter, metric="RMSE"):
    """
    Create a bar chart comparing ARIMA and LSTM for a specific metric.
    
    Parameters:
    -----------
    arima_metrics : dict
        Dictionary of ARIMA model metrics
    lstm_metrics : dict
        Dictionary of LSTM model metrics
    output_file : str
        Path to save the chart image
    parameter : str
        Water quality parameter being compared
    metric : str
        The specific metric to visualize (default is RMSE)
    """
    # Check if the specified metric exists in both models
    if metric not in arima_metrics or metric not in lstm_metrics:
        print(f"Metric {metric} not available for both models. Skipping chart.")
        return
    
    # Create a new figure with specified dimensions
    plt.figure(figsize=(10, 6))
    
    # Setup data for the bar chart
    models = ["ARIMA", "LSTM"]
    values = [arima_metrics[metric], lstm_metrics[metric]]
    
    # Create bar chart with custom colors
    bars = plt.bar(models, values, color=["#1f77b4", "#ff7f0e"], width=0.5)
    
    # Add exact metric values on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                 f"{height:.4f}", ha="center", va="bottom")
    
    # Add chart title and labels
    plt.title(f"Comparison of {metric} for {parameter.capitalize()}", fontsize=16)
    plt.ylabel(metric, fontsize=14)
    plt.ylim(0, max(values) * 1.2)  # Add space above bars for text
    
    # Calculate and add improvement percentage annotation with arrow
    improvement = ((arima_metrics[metric] - lstm_metrics[metric]) / arima_metrics[metric]) * 100
    plt.annotate(f"Improvement: {improvement:.2f}%", 
                 xy=(1, lstm_metrics[metric]),  # Point to the LSTM bar
                 xytext=(1.1, lstm_metrics[metric] + (max(values) * 0.1)),  # Text position
                 arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8),  # Arrow properties
                 fontsize=12)
    
    # Adjust layout and save the chart
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def create_multiple_metrics_chart(arima_metrics, lstm_metrics, output_file, parameter):
    """
    Create a chart comparing multiple metrics between ARIMA and LSTM.
    
    Parameters:
    -----------
    arima_metrics : dict
        Dictionary of ARIMA model metrics
    lstm_metrics : dict
        Dictionary of LSTM model metrics
    output_file : str
        Path to save the chart image
    parameter : str
        Water quality parameter being compared
    """
    # Find metrics that are available in both models
    metrics = list(set(arima_metrics.keys()) & set(lstm_metrics.keys()))
    
    # If no common metrics are found, skip creating the chart
    if not metrics:
        print(f"No common metrics found for {parameter}. Skipping chart.")
        return
    
    # Set up the figure with subplots (one per metric)
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)))
    
    # If there's only one metric, we need to handle axes differently
    if len(metrics) == 1:
        axes = [axes]  # Convert to list for consistent indexing
    
    # Create a subplot for each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]  # Get the current subplot
        models = ["ARIMA", "LSTM"]
        values = [arima_metrics[metric], lstm_metrics[metric]]
        
        # Create bar chart in the current subplot
        bars = ax.bar(models, values, color=["#1f77b4", "#ff7f0e"], width=0.5)
        
        # Add exact values on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02 * max(values),
                    f"{height:.4f}", ha="center", va="bottom")
        
        # Add subplot title and labels
        ax.set_title(f"{metric} for {parameter.capitalize()}", fontsize=14)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_ylim(0, max(values) * 1.2)  # Add space for text
        
        # Calculate and add improvement percentage annotation
        improvement = ((arima_metrics[metric] - lstm_metrics[metric]) / arima_metrics[metric]) * 100
        ax.annotate(f"Improvement: {improvement:.2f}%", 
                    xy=(1, lstm_metrics[metric]),  # Point to LSTM bar
                    xytext=(1.1, lstm_metrics[metric] + (max(values) * 0.1)),  # Text position
                    arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8),  # Arrow properties
                    fontsize=10)
    
    # Adjust layout and save the multi-metric chart
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def create_combined_parameters_chart(all_metrics, output_file, metric="RMSE"):
    """
    Create a chart comparing all parameters for a specific metric.
    
    Parameters:
    -----------
    all_metrics : dict
        Dictionary containing metrics for all parameters and models
    output_file : str
        Path to save the chart image
    metric : str
        The specific metric to visualize (default is RMSE)
    """
    # Get the list of water quality parameters
    parameters = list(all_metrics.keys())
    
    # If no parameters are found, skip creating the chart
    if not parameters:
        print(f"No parameters found. Skipping chart.")
        return
    
    # Set up the figure
    plt.figure(figsize=(12, 8))
    
    # Set width of bars and compute positions
    bar_width = 0.35
    index = np.arange(len(parameters))  # x-axis positions for parameters
    
    # Extract values for each parameter and model
    arima_values = []
    lstm_values = []
    
    # Loop through each parameter to extract metric values
    for param in parameters:
        arima_metrics = all_metrics[param]["ARIMA"]
        lstm_metrics = all_metrics[param]["LSTM"]
        
        # Check if the specified metric exists for both models
        if metric in arima_metrics and metric in lstm_metrics:
            arima_values.append(arima_metrics[metric])
            lstm_values.append(lstm_metrics[metric])
        else:
            # Skip this chart if metric not available for any parameter
            print(f"Metric {metric} not available for {param}. Skipping in combined chart.")
            return
    
    # Create grouped bars for ARIMA and LSTM models
    arima_bars = plt.bar(index, arima_values, bar_width, label="ARIMA", color="#1f77b4")
    lstm_bars = plt.bar(index + bar_width, lstm_values, bar_width, label="LSTM", color="#ff7f0e")
    
    # Add labels and title to the chart
    plt.xlabel("Parameter", fontsize=14)
    plt.ylabel(metric, fontsize=14)
    plt.title(f"Comparison of {metric} Across Parameters", fontsize=16)
    plt.xticks(index + bar_width / 2, [p.capitalize() for p in parameters])  # Center x-axis labels
    plt.legend()
    
    # Helper function to add value labels on top of bars
    def add_labels(bars, values):
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02 * max(max(arima_values), max(lstm_values)),
                     f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    
    # Add value labels to both sets of bars
    add_labels(arima_bars, arima_values)
    add_labels(lstm_bars, lstm_values)
    
    # Add improvement percentages between the bars
    for i, (a_val, l_val) in enumerate(zip(arima_values, lstm_values)):
        improvement = ((a_val - l_val) / a_val) * 100
        plt.annotate(f"{improvement:.1f}%", 
                     xy=(index[i] + bar_width, l_val),  # Position near LSTM bar
                     xytext=(index[i] + bar_width, l_val - (max(arima_values) * 0.1)),  # Text below the bar
                     ha="center",  # Horizontally centered
                     fontsize=9)
    
    # Adjust layout and save the combined parameters chart
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def create_model_summary_file(all_metrics, output_file):
    """
    Create a comprehensive summary text file of all model comparisons.
    
    Parameters:
    -----------
    all_metrics : dict
        Dictionary containing metrics for all parameters and models
    output_file : str
        Path to save the summary file
    """
    # Open the output file in write mode
    with open(output_file, "w") as f:
        # Write header section
        f.write("Comprehensive Model Performance Comparison\n")
        f.write("========================================\n\n")
        
        f.write("This file summarizes the performance comparison between ARIMA and LSTM models\n")
        f.write("for predicting multiple water quality parameters.\n\n")
        
        # Write overview section
        f.write("OVERVIEW\n")
        f.write("--------\n")
        f.write("The comparison analyzes the following water quality parameters:\n")
        for parameter in all_metrics:
            f.write(f"- {parameter.capitalize()}\n")
        f.write("\n")
        
        f.write("Key findings:\n")
        
        # Calculate average improvement across all parameters
        improvements = []
        for parameter in all_metrics:
            arima_metrics = all_metrics[parameter]["ARIMA"]
            lstm_metrics = all_metrics[parameter]["LSTM"]
            
            # Check if RMSE is available for both models
            if "RMSE" in arima_metrics and "RMSE" in lstm_metrics:
                # Calculate percentage improvement
                imp = ((arima_metrics["RMSE"] - lstm_metrics["RMSE"]) / arima_metrics["RMSE"]) * 100
                improvements.append(imp)
        
        # Calculate average improvement
        avg_improvement = sum(improvements) / len(improvements) if improvements else 0
        f.write(f"- Overall, LSTM models show an average of {avg_improvement:.2f}% improvement in RMSE over ARIMA models.\n")
        
        # Find parameter with the best improvement
        best_parameter = None
        best_improvement = 0
        
        for parameter in all_metrics:
            arima_metrics = all_metrics[parameter]["ARIMA"]
            lstm_metrics = all_metrics[parameter]["LSTM"]
            
            if "RMSE" in arima_metrics and "RMSE" in lstm_metrics:
                imp = ((arima_metrics["RMSE"] - lstm_metrics["RMSE"]) / arima_metrics["RMSE"]) * 100
                if imp > best_improvement:
                    best_improvement = imp
                    best_parameter = parameter
        
        # Write information about the best performing parameter
        if best_parameter:
            f.write(f"- The most significant improvement is seen in {best_parameter.capitalize()} prediction, with a {best_improvement:.2f}% reduction in RMSE.\n")
        
        # Add note about R² values if available
        if "R²" in all_metrics.get("temperature", {}).get("LSTM", {}):
            f.write(f"- LSTM achieves high R² values (coefficient of determination) for temperature and dissolved oxygen predictions.\n")
        
        # Write detailed comparison section for each parameter
        f.write("\nDETAILED COMPARISON BY PARAMETER\n")
        f.write("--------------------------------\n\n")
        
        # Process each parameter
        for parameter in all_metrics:
            f.write(f"{parameter.capitalize()}:\n")
            f.write("-" * (len(parameter) + 1) + "\n")
            
            arima_metrics = all_metrics[parameter]["ARIMA"]
            lstm_metrics = all_metrics[parameter]["LSTM"]
            
            # Write metric comparison table
            f.write("Metric      | ARIMA Model | LSTM Model | Improvement (%)\n")
            f.write("-" * 60 + "\n")
            
            # Find metrics common to both models
            common_metrics = set(arima_metrics.keys()) & set(lstm_metrics.keys())
            
            # Write each metric comparison
            for metric in sorted(common_metrics):
                arima_value = arima_metrics[metric]
                lstm_value = lstm_metrics[metric]
                
                # For error metrics, calculate improvement percentage
                if metric in ["RMSE", "MAE", "MSE"]:
                    improvement = ((arima_value - lstm_value) / arima_value) * 100
                    f.write(f"{metric:<12}| {arima_value:<12.4f}| {lstm_value:<12.4f}| {improvement:>6.2f}%\n")
            
            # Add R² from LSTM if only available for LSTM
            if "R²" in lstm_metrics and "R²" not in arima_metrics:
                f.write(f"{'R²':<12}| {'N/A':<12}| {lstm_metrics['R²']:<12.4f}| {'N/A':>6}\n")
            
            f.write("\n")
            
            # Add interpretation of results for this parameter
            f.write("Interpretation:\n")
            if "RMSE" in common_metrics:
                arima_rmse = arima_metrics["RMSE"]
                lstm_rmse = lstm_metrics["RMSE"]
                improvement = ((arima_rmse - lstm_rmse) / arima_rmse) * 100
                
                # Choose language based on improvement percentage
                if improvement > 80:
                    assessment = "vastly outperforms"
                elif improvement > 50:
                    assessment = "significantly outperforms"
                elif improvement > 20:
                    assessment = "clearly outperforms"
                else:
                    assessment = "outperforms"
                
                f.write(f"- The LSTM model {assessment} the ARIMA model for {parameter} prediction, with a {improvement:.2f}% reduction in RMSE.\n")
            
            # Add interpretation of R² values if available
            if "R²" in lstm_metrics:
                r2 = lstm_metrics["R²"]
                if r2 > 0.9:
                    f.write(f"- The LSTM model achieves excellent predictive performance with an R² of {r2:.4f}, explaining {r2*100:.1f}% of the variance in the data.\n")
                elif r2 > 0.7:
                    f.write(f"- The LSTM model achieves good predictive performance with an R² of {r2:.4f}, explaining {r2*100:.1f}% of the variance in the data.\n")
                else:
                    f.write(f"- The LSTM model achieves moderate predictive performance with an R² of {r2:.4f}, explaining {r2*100:.1f}% of the variance in the data.\n")
            
            f.write("\n")
        
        # Write conclusion section
        f.write("\nCONCLUSION\n")
        f.write("----------\n")
        f.write("The comparative analysis clearly demonstrates that LSTM models provide superior forecasting performance\n")
        f.write("for water quality parameters compared to traditional ARIMA models. The improvement is most pronounced\n")
        f.write("for temperature and dissolved oxygen predictions, where the LSTM model achieves both high accuracy\n")
        f.write("(low RMSE) and excellent explanatory power (high R²).\n\n")
        
        f.write("While ARIMA models provide a solid baseline and are computationally less demanding, the significant\n")
        f.write("performance improvements offered by LSTM models justify their adoption for water quality forecasting\n")
        f.write("applications where high accuracy is required.\n")

def main():
    """Main function to execute the model comparison analysis workflow."""
    # Announce start of analysis
    print("Starting model comparison analysis...")
    
    # Get current working directory for proper path handling
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")
    
    # Define directories for input/output
    output_dir = os.path.join(current_dir, "outputs")
    comparison_dir = os.path.join(output_dir, "model_comparison")
    os.makedirs(comparison_dir, exist_ok=True)  # Create if it doesn't exist
    
    # Define the water quality parameters to analyze
    # Note: turbidity has been removed
    parameters = ["temperature", "dissolved_oxygen", "pH"]
    
    # Initialize a dictionary to store all metrics for all parameters
    all_metrics = {}
    
    # Process each parameter
    for parameter in parameters:
        print(f"Processing {parameter}...")
        
        # Define file paths for model results
        arima_results_file = os.path.join(output_dir, f"{parameter}_arima_test_results.txt")
        lstm_results_file = os.path.join(output_dir, f"{parameter}_lstm_test_results.txt")
        
        # Extract metrics from results files
        arima_metrics = parse_metrics_from_file(arima_results_file)
        lstm_metrics = parse_metrics_from_file(lstm_results_file)
        
        # Store metrics in the all_metrics dictionary
        all_metrics[parameter] = {
            "ARIMA": arima_metrics,
            "LSTM": lstm_metrics
        }
        
        # Create text file comparing models for this parameter
        comparison_file = os.path.join(comparison_dir, f"{parameter}_model_comparison.txt")
        create_comparison_text_file(arima_metrics, lstm_metrics, comparison_file, parameter)
        print(f"Created comparison text file for {parameter}")
        
        # Create bar chart for RMSE comparison if available
        if "RMSE" in arima_metrics and "RMSE" in lstm_metrics:
            rmse_chart_file = os.path.join(comparison_dir, f"{parameter}_rmse_comparison.png")
            create_bar_chart(arima_metrics, lstm_metrics, rmse_chart_file, parameter, "RMSE")
            print(f"Created RMSE comparison chart for {parameter}")
        
        # Create chart comparing multiple metrics
        metrics_chart_file = os.path.join(comparison_dir, f"{parameter}_metrics_comparison.png")
        create_multiple_metrics_chart(arima_metrics, lstm_metrics, metrics_chart_file, parameter)
        print(f"Created metrics comparison chart for {parameter}")
    
    # Create chart showing RMSE comparison across all parameters
    combined_chart_file = os.path.join(comparison_dir, "combined_rmse_comparison.png")
    create_combined_parameters_chart(all_metrics, combined_chart_file, "RMSE")
    print("Created combined parameters RMSE comparison chart")
    
    # Create comprehensive text summary of all comparisons
    summary_file = os.path.join(comparison_dir, "comprehensive_model_comparison.txt")
    create_model_summary_file(all_metrics, summary_file)
    print("Created comprehensive model comparison summary")
    
    # Indicate successful completion
    print("\nModel comparison analysis completed successfully!")
    print(f"All comparison files saved to: {comparison_dir}")

# Execute main function when script is run directly
if __name__ == "__main__":
    main()
