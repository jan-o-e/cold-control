import pandas as pd

import matplotlib.pyplot as plt

def plot_csv_data(file_path):
    """
    Reads a CSV file and plots the data from 'Channel 1' and 'Channel 2' against 'Time'.

    Parameters:
        file_path (str): Path to the CSV file.
    """
    try:
        # Read the CSV file
        data = pd.read_csv(file_path)

        # Ensure required columns exist
        if not {'Time (s)', 'Channel 1 Voltage (V)', 'Channel 2 Voltage (V)'}.issubset(data.columns):
            raise ValueError("CSV file must contain 'Time', 'Channel 1', and 'Channel 2' columns.")

        # Plot the data
        plt.figure(figsize=(10, 6))
        plt.plot(data['Time (s)'], data['Channel 1 Voltage (V)'], label='Channel 1', color='blue')
        plt.plot(data['Time (s)'], data['Channel 2 Voltage (V)'], label='Channel 2', color='red')

        # Add labels, title, and legend
        plt.xlabel('Time')
        plt.ylabel('Data')
        plt.title('Channel Data Over Time')
        plt.legend()
        plt.grid(True)

        # Show the plot
        plt.show()

    except Exception as e:
        print(f"Error: {e}")

# Example usage
# Replace 'your_file.csv' with the path to your CSV file
# plot_csv_data('your_file.csv')
plot_csv_data(r"C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\data\2025-05-21\A_16-52-15_channels_1_2_data")
plot_csv_data(r"C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\data\2025-05-21\A_16-52-15_channels_1_2_data")
plot_csv_data(r"C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\data\2025-05-21\A_12-06-12_channels_1_2_data")
plot_csv_data(r"C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\data\2025-05-21\A_12-06-16_channels_1_2_data")
plot_csv_data(r"C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\data\2025-05-21\A_12-06-20_channels_1_2_data")
plot_csv_data(r"C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\data\2025-05-21\A_12-06-25_channels_1_2_data")
plot_csv_data(r"C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\data\2025-05-21\A_12-05-59_channels_1_2_data")
plot_csv_data(r"C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\data\2025-05-21\A_12-05-46_channels_1_2_data")
plot_csv_data(r"C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\data\2025-05-21\A_12-05-51_channels_1_2_data")
plot_csv_data(r"C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\data\2025-05-21\A_12-05-55_channels_1_2_data")