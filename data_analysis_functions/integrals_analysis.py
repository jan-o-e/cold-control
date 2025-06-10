import pandas as pd
import numpy as np
import re
import os
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from typing import Union 


def extract_voltages(filename: str) -> Union[tuple[float, float], None]:
    """
    Extracts the first two voltage values from a filename string
    formatted as 'swept_{voltage1}V_{voltage2}V_{time}s'.

    Args:
        filename (str): The name of the file, e.g., 'swept_1.4V_6.0V_500s.csv'.

    Returns:
        tuple[float, float] | None: A tuple containing the two voltage values
                                     as floats, or None if the pattern is not found.
    """
    # Define a regular expression pattern to capture the two voltage values.
    # - `swept_`: Matches the literal string "swept_".
    # - `(\d+\.?\d*)`: This is the first capturing group.
    #   - `\d+`: Matches one or more digits (for the whole number part).
    #   - `\.?`: Matches an optional decimal point.
    #   - `\d*`: Matches zero or more digits (for the fractional part after the decimal).
    # - `V_`: Matches the literal string "V_" after the first voltage.
    # - `(\d+\.?\d*)`: This is the second capturing group, identical to the first.
    # - `V_`: Matches the literal string "V_" after the second voltage.
    # - `.*`: Matches any characters that follow (e.g., the time part and file extension).
    pattern = r"swept_(\d+\.?\d*)V_(\d+\.?\d*)V_.*"
    
    # Search for the pattern in the given filename.
    match = re.search(pattern, filename)
    
    if match:
        try:
            # Extract the captured groups and convert them to floats.
            voltage1 = float(match.group(1))
            voltage2 = float(match.group(2))
            return voltage1, voltage2
        except ValueError:
            # This should ideally not happen if the regex is correct,
            # but it's good practice to handle potential conversion errors.
            print(f"Warning: Could not convert extracted values to float for filename: {filename}")
            return None
    else:
        return None
    


def plot_contour_from_csv(csv_filepath: str):
    """
    Reads data from a CSV, extracts voltages from filenames, and generates
    a contour plot of 'average_integral' against the two voltage values.

    Args:
        csv_filepath (str): The path to the input CSV file.
    """
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_filepath)
        print(f"Successfully loaded data from '{csv_filepath}'. Shape: {df.shape}")
        # print("DataFrame head:\n", df.head()) # Uncomment to inspect data

        # Apply the voltage extraction function to the 'folder' column
        # and create new columns for power_voltage and frequency_voltage.
        # Use .apply(pd.Series) to expand the tuple output into two separate columns.
        df[['power_voltage', 'frequency_voltage']] = df['folder'].apply(
            lambda x: pd.Series(extract_voltages(x))
        )
        
        # Drop rows where voltage extraction failed (i.e., 'power_voltage' is NaN)
        df.dropna(subset=['power_voltage', 'frequency_voltage'], inplace=True)
        print(f"Data after extracting voltages and dropping NaNs: {df.shape}")

        # Ensure that integral column is numeric
        df['average_integral'] = pd.to_numeric(df['average_integral'], errors='coerce')
        df.dropna(subset=['average_integral'], inplace=True)
        print(f"Data after ensuring 'average_integral' is numeric: {df.shape}")

        if df.empty:
            print("No valid data remaining after processing for contour plot. Exiting.")
            return

        # Prepare data for contour plot
        # X-axis: power_voltage
        # Y-axis: frequency_voltage
        # Z-axis: average_integral
        x = df['power_voltage'].values
        y = df['frequency_voltage'].values
        z = df['average_integral'].values

        # Create a regular grid for the contour plot
        # Determine the range for X and Y
        xi = np.linspace(x.min(), x.max(), 100) # 100 points along X-axis
        yi = np.linspace(y.min(), y.max(), 100) # 100 points along Y-axis
        
        # Use griddata to interpolate the Z values onto the new grid
        # 'linear' interpolation is a good starting point.
        zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='linear')

        # Create the contour plot
        plt.figure(figsize=(10, 8))
        
        # Filled contour plot
        contourf = plt.contourf(xi, yi, zi, levels=20, cmap='viridis', alpha=0.8)
        
        # Add contour lines for better visualization (optional)
        contour = plt.contour(xi, yi, zi, levels=20, colors='black', linewidths=0.5)
        plt.clabel(contour, inline=True, fontsize=8, fmt='%.2e') # Label contour lines

        # Scatter plot of original data points for reference
        plt.scatter(x, y, c=z, cmap='viridis', edgecolors='k', s=50, label='Original Data Points')

        # Add a color bar
        cbar = plt.colorbar(contourf)
        cbar.set_label('Average Integral Value')

        # Set labels and title
        plt.xlabel('Power Voltage (V)')
        plt.ylabel('Frequency Voltage (V)')
        plt.title('Contour Plot of Average Integral vs. Power and Frequency Voltages')
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout() # Adjust layout to prevent labels from overlapping
        plt.show()

    except FileNotFoundError:
        print(f"Error: The file '{csv_filepath}' was not found. Please check the path.")
    except KeyError as e:
        print(f"Error: Missing expected column in CSV: {e}. Please ensure 'folder' and 'average_integral' columns exist.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
plot_contour_from_csv(r"C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\data\2025-06-10\12-21-56_imaging_sweep1\summary_integrals.csv")
