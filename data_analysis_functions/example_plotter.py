import pandas as pd
import os
import matplotlib.pyplot as plt

# Load the CSV file
#df = pd.read_csv(r"C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\data\2025-05-29\A_15-46-37_channels_1_3_data")
#df = pd.read_csv(r"C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\data\2025-05-28\A_17-09-59_channels_1_3_data")
#df = pd.read_csv(r"C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\data\2025-05-29\A_15-46-43_channels_1_3_data")



def plot_data(root_path, plot_all=True, plot_range=(1, 1), v_ch=4, window_size=32, show_markers=True):
    """    
    Plots the average voltage data from multiple CSV files in a specified directory.
    """
    if plot_all:
        all_files = os.listdir(root_path)
        filenames = [f for f in all_files if f.endswith(".csv")]
        csv_files = [f"{root_path}\\{name}" for name in filenames]
    else:
        csv_files = []
        for i in range(plot_range[0],plot_range[1] + 1):
            file_path = f"{root_path}\iteration_{i}_data.csv"
            csv_files.append(file_path)

    # Read each CSV into a DataFrame
    dfs = [pd.read_csv(f) for f in csv_files]

    # Ensure all DataFrames have the same shape and time points
    for i, df in enumerate(dfs[1:], 1):
        assert df.shape == dfs[0].shape, f"Shape mismatch in file: {csv_files[i]}"
        #assert all(df['Time (s)'] == dfs[0]['Time (s)']), f"Time mismatch in file: {csv_files[i]}"

    # Stack the voltage data into a 3D array for averaging
    voltage_columns = ['Channel 1 Voltage (V)', f'Channel {v_ch} Voltage (V)', 'Channel 2 Voltage (V)']  
    stacked_data = {
        col: pd.concat([df[col] for df in dfs], axis=1).mean(axis=1)
        for col in voltage_columns
    }

    # Build the final averaged DataFrame
    average_df = pd.DataFrame({
        'Time (s)': dfs[0]['Time (s)'],
        'Channel 1 Voltage (V)': stacked_data['Channel 1 Voltage (V)'],
        f'Channel {v_ch} Voltage (V)': stacked_data[f'Channel {v_ch} Voltage (V)'],
        'Channel 2 Voltage (V)': stacked_data['Channel 2 Voltage (V)']  # <--- añadido canal 2
    })

    df = average_df


    # Create the figure and first axis
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Channel 1 Voltage on the left y-axis
    ax1.plot(df['Time (s)'], df['Channel 1 Voltage (V)'], color='blue', label='Channel 1 Voltage (V)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Channel 1 Voltage (V)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create a second y-axis sharing the same x-axis
    ax2 = ax1.twinx()

    # Apply rolling average to Channel 3 Voltage
    df[f'Channel {v_ch} Rolling Avg'] = df[f'Channel {v_ch} Voltage (V)'].rolling(window=window_size, center=True).mean()

    # Plot the rolling average
    ax2.plot(df['Time (s)'], df[f'Channel {v_ch} Rolling Avg'], color='red', linestyle='--', label=f'Channel {v_ch} Voltage (Rolling Avg)')
    ax2.set_ylabel(f'Channel {v_ch} Voltage (V)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    if show_markers == True:
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 100)) 
        ax3.plot(df['Time (s)'], df['Channel 2 Voltage (V)'], color='green', label='Channel 2')
        ax3.set_ylabel('Imaging Marker', color='green')
        ax3.tick_params(axis='y', labelcolor='green')


        ax1.axvline(x=0.0012, color='black', linestyle=':', linewidth=1.5) 
        ax1.text(0.0012, ax1.get_ylim()[1], 'AWG start', rotation=90, verticalalignment='bottom', color='black')  # <--- etiqueta


    # Add title and grid
    plt.title('New data')
    ax1.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_all = True # Set to True to plot avg of all iterations
    plot_range = (2,2)  # Adjust this range as needed
    v_ch = 4 # Channel number to plot rolling average for
    window_size = 32 # How many samples to calculate the rolling average over
    show_markers = True  # Set to True to show markers for imaging data
    while True:
        response= input("Enter the path to the data directory, 'exit' to quit or 'settings' to change settings: ")
        if response.lower() in ['settings', "s", "set", "options", "config"]:
            print("\n--- Settings Menu ---")
            print("Press Enter to keep the current value.\n")

            new_plot_all = input(f"Plot all iterations (True/False) [{plot_all}]: ").strip().lower()
            if new_plot_all in ['true', 'false']: plot_all = new_plot_all == 'true'

            new_range = input(f"Plot range as two numbers separated by comma (e.g. 1,3) [{plot_range[0]},{plot_range[1]}]: ").strip()
            if new_range:
                try:
                    start, end = map(int, new_range.split(","))
                    plot_range = (start, end)
                except ValueError:
                    print("Invalid input for plot range. Keeping previous values.")

            new_ch = input(f"Channel number for rolling average [{v_ch}]: ").strip()
            if new_ch.isdigit(): v_ch = int(new_ch)

            new_window = input(f"Rolling average window size [{window_size}]: ").strip()
            if new_window.isdigit(): window_size = int(new_window)

            new_markers = input(f"Show markers (True/False) [{show_markers}]: ").strip().lower()
            if new_markers in ['true', 'false']: show_markers = new_markers == 'true'

            print("\nSettings updated.\n")
            continue
        
        if response.lower() in ['exit', 'quit', "x", "e", "q"]:
            print("Exiting the program.")
            break


        path_r = fr"{response}"
        plot_data(path_r, plot_all=plot_all, plot_range=plot_range, v_ch=v_ch,
                  window_size=window_size, show_markers=show_markers)