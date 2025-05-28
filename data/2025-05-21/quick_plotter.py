import pandas as pd

import matplotlib.pyplot as plt


def plot_pd_data(data):
    """
    Reads a CSV file and plots the data from 'Channel 1' and 'Channel 2' against 'Time'.

    Parameters:
        file_path (str): Path to the CSV file.
    """
    try:


        # Ensure required columns exist
        # if not {'Time (s)', 'Channel 1 Voltage (V)', 'Channel 2 Voltage (V)'}.issubset(data.columns):
        #     raise ValueError("CSV file must contain 'Time', 'Channel 1', and 'Channel 2' columns.")

        # # Plot the data
        # plt.figure(figsize=(10, 6))
        # plt.plot(data['Time (s)'], data['Channel 1 Voltage (V)'], label='Channel 1', color='blue')
        # plt.plot(data['Time (s)'], data['Channel 2 Voltage (V)'], label='Channel 2', color='red')

        # # Add labels, title, and legend
        # plt.xlabel('Time')
        # plt.ylabel('Data')
        # plt.title('Channel Data Over Time')
        # plt.legend()
        # plt.grid(True)

        # # Show the plot
        # plt.show()
        
        fig, ax1 = plt.subplots()

        # Plot Channel 1 on the left y-axis
        ax1.plot(data['Time (s)'], data['Channel 1 Voltage (V)'], label='Channel 1', color='blue')
        ax1.set_ylabel('Channel 1 Voltage (V)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Create a second y-axis sharing the same x-axis
        ax2 = ax1.twinx()

        # Plot Channel 2 on the right y-axis
        ax2.plot(data['Time (s)'], data['Channel 3 Voltage (V)'], label='Channel 2', color='red')
        ax2.set_ylabel('Channel 3 Voltage (V)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # Set x-axis limits to zoom in
        #ax1.set_xlim(0, 200e-6)

        # X-axis label
        ax1.set_xlabel('Time (s)')

        # Optional: improve layout
        plt.title('Channel 1 and Channel 2 Voltages Over Time')
        fig.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error: {e}")


experiment_times = ["17-09-30", "17-02-18", "17-02-21", "17-02-25", "17-02-28", "17-02-31", "17-02-35", "17-02-38", "17-02-41", "17-02-44"]


data1 = pd.read_csv(f"C:\\Users\\apc\\Documents\\Python Scripts\\Cold Control Heavy\\data\\2025-05-28\\A_{experiment_times[0]}_channels_1_3_data")
data2 = pd.read_csv(f"C:\\Users\\apc\\Documents\\Python Scripts\\Cold Control Heavy\\data\\2025-05-28\\A_{experiment_times[1]}_channels_1_3_data")
data3 = pd.read_csv(f"C:\\Users\\apc\\Documents\\Python Scripts\\Cold Control Heavy\\data\\2025-05-28\\A_{experiment_times[2]}_channels_1_3_data")
data4 = pd.read_csv(f"C:\\Users\\apc\\Documents\\Python Scripts\\Cold Control Heavy\\data\\2025-05-28\\A_{experiment_times[3]}_channels_1_3_data")
data5 = pd.read_csv(f"C:\\Users\\apc\\Documents\\Python Scripts\\Cold Control Heavy\\data\\2025-05-28\\A_{experiment_times[4]}_channels_1_3_data")
data6 = pd.read_csv(f"C:\\Users\\apc\\Documents\\Python Scripts\\Cold Control Heavy\\data\\2025-05-28\\A_{experiment_times[5]}_channels_1_3_data")
data7 = pd.read_csv(f"C:\\Users\\apc\\Documents\\Python Scripts\\Cold Control Heavy\\data\\2025-05-28\\A_{experiment_times[6]}_channels_1_3_data")
data8 = pd.read_csv(f"C:\\Users\\apc\\Documents\\Python Scripts\\Cold Control Heavy\\data\\2025-05-28\\A_{experiment_times[7]}_channels_1_3_data")
data9 = pd.read_csv(f"C:\\Users\\apc\\Documents\\Python Scripts\\Cold Control Heavy\\data\\2025-05-28\\A_{experiment_times[8]}_channels_1_3_data")
data10 = pd.read_csv(f"C:\\Users\\apc\\Documents\\Python Scripts\\Cold Control Heavy\\data\\2025-05-28\\A_{experiment_times[9]}_channels_1_3_data")


#plot_pd_data(data1)

#"""
tot_data = pd.DataFrame()
#all_data = [data1, data2, data3, data4, data5, data6, data7, data8, data9, data10]
all_data=[data1]
for data in all_data:
    time_col = data['Time (s)'].to_numpy()
    channel1_col = data['Channel 1 Voltage (V)'].to_numpy() 
    channel2_col = data['Channel 3 Voltage (V)'].to_numpy()
    if tot_data.empty:
        tot_data['Time (s)'] = time_col
        tot_data['Channel 1 Voltage (V)'] = channel1_col
        tot_data['Channel 3 Voltage (V)'] = channel2_col
    else:
        tot_data['Channel 1 Voltage (V)'] = tot_data['Channel 1 Voltage (V)'].to_numpy() + channel1_col
        tot_data['Channel 3 Voltage (V)'] = tot_data['Channel 3 Voltage (V)'].to_numpy() + channel2_col

    #plot_pd_data(data)

tot_data['Channel 1 Voltage (V)'] = tot_data['Channel 1 Voltage (V)'] / len(all_data)
tot_data['Channel 3 Voltage (V)'] = tot_data['Channel 3 Voltage (V)'] / len(all_data)

# Plot the data
plot_pd_data(tot_data)
#"""












#plot_csv_data(r"C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\data\2025-05-22\A_16-12-54_channels_1_2_data")
# plot_csv_data(r"C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\data\2025-05-21\A_16-52-15_channels_1_2_data")
# plot_csv_data(r"C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\data\2025-05-21\A_12-06-12_channels_1_2_data")
# plot_csv_data(r"C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\data\2025-05-21\A_12-06-16_channels_1_2_data")
# plot_csv_data(r"C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\data\2025-05-21\A_12-06-20_channels_1_2_data")
# plot_csv_data(r"C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\data\2025-05-21\A_12-06-25_channels_1_2_data")
# plot_csv_data(r"C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\data\2025-05-21\A_12-05-59_channels_1_2_data")
# plot_csv_data(r"C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\data\2025-05-21\A_12-05-46_channels_1_2_data")
# plot_csv_data(r"C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\data\2025-05-21\A_12-05-51_channels_1_2_data")
# plot_csv_data(r"C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\data\2025-05-21\A_12-05-55_channels_1_2_data")