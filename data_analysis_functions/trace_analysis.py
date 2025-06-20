import matplotlib.pyplot as plt
import re, os
import pandas as pd
from numpy import trapz
import numpy as np
import datetime
from scipy.interpolate import interp1d


MARKER_DROP = 7.45 # The level below which the AWG marker will drop
#T_RISE = 1.57e-3 # The time at which the fluorescence is expected to first rise
IMG_WIDTH = 500e-6 # The width of the imaging pulse
TARGET_TIME = 1.46e-3 # The expected time of the AWG marker
TOLERANCE = 50e-6 # How far around the target time to check for the marker

MOT_DROP = 19.7e-3 # The level below which the fluorescence will drop after the MOT is turned off
MOT_DROP_TIME = 600e-6 # The expected time of the MOT drop marker, when the MOT is turned off
T_RISE = MOT_DROP_TIME + 1e-3



def get_folder_paths(directory_path):
    """
    Gets the full filepaths (as raw strings) of only the top-level subfolders
    within a given directory.

    Args:
        directory_path (str): The path to the directory to search.

    Returns:
        list[str]: A list of full filepaths of only the top-level subfolders,
                   or an empty list if the directory doesn't exist or
                   contains no top-level subfolders.
    """
    folder_paths = []
    # Check if the provided path is a valid directory
    if not os.path.isdir(directory_path):
        print(f"Error: '{directory_path}' is not a valid directory.")
        return []

    # Iterate through all entries in the directory
    for entry_name in os.listdir(directory_path):
        full_path = os.path.join(directory_path, entry_name)
        # Check if the entry is a directory (folder)
        if os.path.isdir(full_path):
            # Add the raw string representation of the path
            folder_paths.append(r"{}".format(full_path))
    return folder_paths



def average_and_plot_voltage(dfs, time_col='Time (s)', voltage_col='Voltage (V)', num_points=1000):
    """
    Averages voltage traces from multiple DataFrames using interpolation and plots the result.

    Parameters:
    - dfs: list of pandas.DataFrame, each with time and voltage columns.
    - time_col: name of the time column.
    - voltage_col: name of the voltage column.
    - num_points: number of points in the common time grid.
    """

    # Step 1: Define a common time base (only overlap range)
    min_time = max(df[time_col].min() for df in dfs)
    max_time = min(df[time_col].max() for df in dfs)
    common_time = np.linspace(min_time, max_time, num_points)

    # Step 2: Interpolate each DataFrame to the common time
    interpolated_voltages = []
    for df in dfs:
        interp_func = interp1d(df[time_col], df[voltage_col], kind='linear', bounds_error=False, fill_value="extrapolate")
        interpolated = interp_func(common_time)
        interpolated_voltages.append(interpolated)

    # Step 3: Stack and average
    stacked = np.vstack(interpolated_voltages)
    mean_voltage = np.mean(stacked, axis=0)

    # Step 4: Plot
    plt.figure(figsize=(10, 5))
    for v in interpolated_voltages:
        plt.plot(common_time, v, color='gray', alpha=0.3)

    plt.plot(common_time, mean_voltage, color='red', linewidth=2, label='Average')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.title('Average Voltage Trace')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return pd.DataFrame({'Time (s)': common_time, 'Average Voltage (V)': mean_voltage})


def calculate_integrals_single_trace(data, i=0):
        """
        Function to calculate the fluorescence for a particular trace.
        """
        processed_df = pd.DataFrame()
        # imaging beam on
        ch4 = data['Channel 4 Voltage (V)']
        ch2 = data["Channel 2 Voltage (V)"]
        time = data['Time (s)']

        # Step 1: Find the first index where ch2 drops below 7.45
        below = ch2 < MARKER_DROP
        if not below.any():
            print("Channel 2 never drops below {MARKER_DROP} V")
        else:
            drop_index = below.idxmax()
            drop_time = time[drop_index]

            # Step 2: From that point onward, find the first time it goes back above 7.45
            above = ch2[drop_index+1:] > MARKER_DROP
            if not above.any():
                print(f"Channel 2 drops below {MARKER_DROP} V at {drop_time} s and never goes back above.")
            else:
                rise_index = above.idxmax()
                rise_time = time[rise_index]
                print(f"Channel 2 drops below {MARKER_DROP} V at {drop_time} s")
                print(f"Channel 2 goes back above {MARKER_DROP} V at {rise_time} s")

        # big increase when imaging start
        # mask_rise = (time >= 1.77e-3) & (time <= 2.0e-3)  # imaging starts at 2ms
        # ch4_smooth_rise = ch4_smooth[mask_rise]
        # time_rise = time[mask_rise]
        # deriv_rise = np.gradient(ch4_smooth_rise, time_rise)

        # idx_rise_rel = np.argmax(deriv_rise)
        # idx_rise = time_rise.index[idx_rise_rel]
        # t_rise = time.iloc[idx_rise]

        # big decrease when imaging stops
        t_drop = T_RISE + IMG_WIDTH

        mask_fl = (time >= T_RISE) & (time <= t_drop)
        ch4_segment_fl = data.loc[mask_fl, ['Time (s)', 'Channel 4 Voltage (V)']].copy()

        t_start_ref = t_drop + 50e-6  # after imaging stops
        t_end_ref = data['Time (s)'].iloc[-1] 
        mask_ref = (data['Time (s)'] >= t_start_ref) & (data['Time (s)'] <= t_end_ref)
        ch4_segment_ref = data.loc[mask_ref, ['Time (s)', 'Channel 4 Voltage (V)']].copy()
        average = ch4_segment_ref['Channel 4 Voltage (V)'].mean(axis=0)

        # integration area below curve, taking average as a zero reference
        area = trapz(ch4_segment_fl['Channel 4 Voltage (V)'] - [average]*len(ch4_segment_fl['Channel 4 Voltage (V)']), ch4_segment_fl['Time (s)'])

        # only consider the result if the timing of the awg marker is correct
        if abs(drop_time - TARGET_TIME) <= TOLERANCE:
            print("The drop time is close enough to the expected time")
        
            if area is not None and not np.isnan(area):
                processed_df[f'Time (s) {i}'] = data['Time (s)']
                processed_df[f'Channel 1 Voltage (V) {i}'] = data['Channel 1 Voltage (V)']
                processed_df[f'Channel 4 Voltage (V) {i}'] = data['Channel 4 Voltage (V)']
                processed_df[f"Channel 2 Voltage (V) {i}"] = data["Channel 2 Voltage (V)"]

            print(f"Average background: {average}, Integrated area: {area}\n")

            return (area, average, processed_df)
        else:
            return (area, average, None)



def calculate_integrals_align_mot(data, i=0):
        """
        Function to calculate the fluorescence for a particular trace.
        """


        processed_df = pd.DataFrame()
        # imaging beam on
        ch4 = data['Channel 4 Voltage (V)']
        ch2 = data["Channel 2 Voltage (V)"]
        time = data['Time (s)']

        # Step 1: Find the first index where ch2 drops below 7.45
        below = ch4 < MOT_DROP
        if not below.any():
            print("Channel 2 never drops below {MOT_DROP} V")
        else:
            drop_index = below.idxmax()
            drop_time = time[drop_index]
            print(f"Channel 4 drops below {MOT_DROP} V at {drop_time} s")

        # Adjust the time to align the drop time with the expected drop time
        difference = drop_time - MOT_DROP_TIME


        time = time - difference  # Shift the time to align with the expected drop time

        t_end = T_RISE + IMG_WIDTH

        mask_fl = (time >= T_RISE) & (time <= t_end)
        ch4_segment_fl = data.loc[mask_fl, ['Time (s)', 'Channel 4 Voltage (V)']].copy()

        t_start_ref = t_end + 50e-6  # after imaging stops
        t_end_ref = data['Time (s)'].iloc[-1] 
        mask_ref = (data['Time (s)'] >= t_start_ref) & (data['Time (s)'] <= t_end_ref)
        ch4_segment_ref = data.loc[mask_ref, ['Time (s)', 'Channel 4 Voltage (V)']].copy()
        average = ch4_segment_ref['Channel 4 Voltage (V)'].mean(axis=0)

        # integration area below curve, taking average as a zero reference
        area = trapz(ch4_segment_fl['Channel 4 Voltage (V)'] - [average]*len(ch4_segment_fl['Channel 4 Voltage (V)']), ch4_segment_fl['Time (s)'])

        
        if area is not None and not np.isnan(area):
            processed_df[f'Time (s) {i}'] = data['Time (s)']
            processed_df[f'Channel 1 Voltage (V) {i}'] = data['Channel 1 Voltage (V)']
            processed_df[f'Channel 4 Voltage (V) {i}'] = data['Channel 4 Voltage (V)']
            processed_df[f"Channel 2 Voltage (V) {i}"] = data["Channel 2 Voltage (V)"]

        print(f"Average background: {average}, Integrated area: {area}\n")

        return (area, average, processed_df)



def plot_shot_results(folder_path):
    pattern = re.compile(r'^iteration_\d+_data\.csv$')
    files = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if pattern.match(filename)]
    all_measurements = [] 

    window_size = 64
    for file_path in files:
        data = pd.read_csv(file_path)
        data['Channel 1 Voltage (V)'] = data['Channel 1 Voltage (V)']
        data['Channel 4 Voltage (V)'] = data['Channel 4 Voltage (V)'].rolling(window=window_size, center=True, min_periods=1).mean()
        data["Channel 2 Voltage (V)"] = data['Channel 2 Voltage (V)']

        all_measurements.append(data)

    measurements = pd.DataFrame()
    valid_meas = pd.DataFrame()

    ref_0 = []
    fluor = []
    integrals_fl = []

    for i,data in enumerate(all_measurements):
        time = data["Time (s)"]
        ch1 = data['Channel 1 Voltage (V)']
        ch2 = data["Channel 2 Voltage (V)"]
        ch4 = data["Channel 4 Voltage (V)"]

        measurements[f'Time (s) {i}'] = time
        measurements[f'Channel 1 Voltage (V) {i}'] = ch1
        measurements[f'Channel 2 Voltage (V) {i}'] = ch2
        measurements[f"Channel 4 Voltage (V) {i}"] = ch4



        idx_sorted = (ch1 - 1).abs().sort_values().index
        idx_ch1_1 = idx_sorted[0] # ch1 crosses 1V (trigger value)

        (area, average, processed_df) = calculate_integrals_align_mot(data, i)
        
        if processed_df is not None:
            integrals_fl.append(area)
            ref_0.append(average)
            valid_meas = pd.concat([valid_meas, processed_df], axis = 1)


        fig, ax1 = plt.subplots(figsize=(15, 8))

        ax1.plot(time, ch4, label='CH4 raw', alpha=0.5, color='tab:blue')
        ax1.plot(time, ch4, label='CH4 smooth', linewidth=2, color='tab:cyan')
        ax1.axvline(T_RISE, color='green', linestyle='--', label='t_rise (subida)')
        ax1.axvline((T_RISE+IMG_WIDTH), color='red', linestyle='--', label='t_drop (bajada)')
        ax1.axvline(TARGET_TIME, color = "black", linestyle="--", label="target marker time")
        ax1.axhline(average, color='purple', linestyle='--', label='average ref')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Channel 4 Voltage (V)', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.plot(time, ch2, label='CH2 (marker)', color='tab:orange')
        ax2.set_ylabel('Channel 2 Voltage (V)', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

        # Combinar leyendas de ambos ejes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.title('Detecting rise and fall of imaging beam with CH2 Marker')
        plt.tight_layout()
        plt.show()



    valid_integrals = [val for val in integrals_fl if val is not None and not np.isnan(val)]# and val >= 0]


    average_int = np.mean(valid_integrals)
    std_int = np.std(valid_integrals)
    max_int = np.max(valid_integrals)
    min_int = np.min(valid_integrals)
    print(f'Average integrated area: {average_int}, ')
    print(f"Standard deviation of area: {std_int}")
    print(f"Number of integrals calculated: {len(valid_integrals)} out of {len(integrals_fl)} shots")



    # mean signal
    mean_time = measurements.filter(like='Time (s)').mean(axis=1)
    mean_ch1 = measurements.filter(like='Channel 1 Voltage (V)').mean(axis=1)
    mean_ch4 = measurements.filter(like='Channel 4 Voltage (V)').mean(axis=1)
    mean_ch2 = measurements.filter(like="Channel 2 Voltage (V)").mean(axis=1)

    fig1, ax1 = plt.subplots(figsize=(11, 5))
    ax1.plot(mean_time, mean_ch1, linewidth=1.5, color='tab:blue', label='Mean CH 1')
    ax1.set_xlabel(r'Time (s)')
    ax1.set_ylabel(r'Mean Intensity (a.u) CH 1', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.plot(mean_time, mean_ch4, linewidth=1.5, color='tab:orange', label='Mean CH 3')
    ax2.set_ylabel(r'Mean Intensity (a.u) CH 4', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 100)) 
    ax3.plot(mean_time, mean_ch2, linewidth = 0.5, color='green', label='Channel 2')
    ax3.set_ylabel('Imaging Marker', color='green')
    ax3.tick_params(axis='y', labelcolor='green')

    fig1.suptitle('Mean CH1 and CH4 across all files')
    fig1.tight_layout()

    # plot only good data
    time_val = valid_meas.filter(like='Time (s)').mean(axis=1)
    ch1_val = valid_meas.filter(like='Channel 1 Voltage (V)').mean(axis=1)
    ch2_val = valid_meas.filter(like="Channel 2 Voltage (V)").mean(axis=1)
    ch4_val = valid_meas.filter(like='Channel 4 Voltage (V)').mean(axis=1)

    fig2, ax1 = plt.subplots(figsize=(11, 5))
    ax1.plot(time_val, ch1_val, linewidth=1.5, color='tab:blue', label='Mean CH 1')
    ax1.set_xlabel(r'Time (s)')
    ax1.set_ylabel(r'Mean Intensity (a.u) CH 1', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.plot(time_val, ch4_val, linewidth=1.5, color='tab:orange', label='Mean CH 3')
    ax2.set_ylabel(r'Mean Intensity (a.u) CH 4', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 100)) 
    ax3.plot(time_val, ch2_val, linewidth = 0.5, color='green', label='Channel 2')
    ax3.set_ylabel('Imaging Marker', color='green')
    ax3.tick_params(axis='y', labelcolor='green')

    fig2.suptitle('Only data where the AWG marker is at the right time, and the integral is a number')
    fig2.tight_layout()
    fig2.savefig(os.path.join(folder_path, 'results_plot.png'))


    plt.show()



def calculate_all_integrals(root_directory, shots_to_include=[], window_size=32,
                        folders_to_process=None):
    """ 
    Calculate integrals of fluorescence data from multiple folders. Saves the summary of 
    the results in the same folder as the input data.
    Args:
        root_directory (str): The root directory containing subfolders with data.
        shots_to_include (list): List of specific shots to include in the analysis or 
        leave empty to include all.
        window_size (int): Size of the rolling window for smoothing the data.
        folders_to_process (list): List of folder paths to process. If None,
        it will automatically get all top-level subfolders in the root directory.
    """
    if folders_to_process is None:
        # Get all top-level subfolders in the root directory
        folders_to_process = get_folder_paths(root_directory)
        print(folders_to_process)
        if not folders_to_process:
            print(f"No subfolders found in {root_directory}.")
            return


    today = datetime.datetime.now().strftime("%d-%m")
    output_data = []

    for folder_path in folders_to_process:
        folder_name = os.path.basename(folder_path)
        print(f'Procesando carpeta: {folder_name}')

        pattern = re.compile(r'^iteration_(\d+)_data\.csv$')

        files = []
        for root, _, file_list in os.walk(folder_path):
            if shots_to_include:
                if not any(shot in root for shot in shots_to_include):
                    continue  
            for f in file_list:
                if pattern.match(f):
                    full_path = os.path.join(root, f)
                    files.append(full_path)

        integrals_fl = []
        ref_0 = []

        for file_path in files:
            file_name = os.path.basename(file_path)
            match = pattern.match(file_name)
            iteration_number = match.group(1)

            data = pd.read_csv(file_path)
            data['Channel 1 Voltage (V)'] = data['Channel 1 Voltage (V)'].rolling(window=window_size, center=True, min_periods=1).mean()
            data['Channel 4 Voltage (V)'] = data['Channel 4 Voltage (V)'].rolling(window=window_size, center=True, min_periods=1).mean()

            ch1 = data['Channel 1 Voltage (V)']
            idx_sorted = (ch1 - 1).abs().sort_values().index
            idx_ch1_1 = idx_sorted[0]

            (area, average, processed_df) = calculate_integrals_align_mot(data)

            integrals_fl.append(area)
            ref_0.append(average)

            # # Guardar resultado por iteración
            # integrals_fl_df = pd.DataFrame({'integral': [area], 'ref 0': [average]})
            # output_dir = rf'C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\data\integrals_data_analysis\{today}\{folder_name}'
            # os.makedirs(output_dir, exist_ok=True)
            # integrals_fl_df.to_csv(os.path.join(output_dir, f'integrated_area_iteration_{iteration_number}.csv'), index=False)

        # Filter out NaN and negative values
        valid_integrals = [val for val in integrals_fl if val is not None and not np.isnan(val) and val >= 0]

        if valid_integrals:
            average_int = np.mean(valid_integrals)
            std_int = np.std(valid_integrals)
            max_int = np.max(valid_integrals)
            min_int = np.min(valid_integrals)
        else:
            average_int = std_int = max_int = min_int = np.nan  # or handle differently

        print(f'→ Promedio en {folder_name}: {average_int:.3e}')

        output_data.append({
            'folder': folder_name,
            'average_integral': average_int,
            'std_integral': std_int,
            'max_integral': max_int,
            'min_integral': min_int,
            'n_files': len(files)
        })

    # Guardar resumen final
    summary_df = pd.DataFrame(output_data)
    # summary_output_dir = rf'C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\data\integrals_data_analysis\{today}'
    # os.makedirs(summary_output_dir, exist_ok=True)
    summary_output_path = os.path.join(root_directory, 'summary_integrals.csv')
    summary_df.to_csv(summary_output_path, index=False)

    print(f"\nResumen guardado en: {summary_output_path}")


if __name__ == "__main__":
    root_directory = r"d:\pulse_shaping_data\2025-06-13\16-08-00"
    single_shot_path = r'D:\pulse_shaping_data\2025-06-13\16-08-00\sweep_55_opt_126_80\shot0'

    # folders_to_process = [
    #     r"C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\data\2025-06-09\16-08-07_low_fluoresce\sweeped_pump_175ns_20_stokes_175ns_0_2_126_80",
    #     r"C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\data\2025-06-09\16-08-07_low_fluoresce\sweeped_pump_optimized_stokes_optimized_126_80",
    #     r'C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\data\2025-06-09\16-08-07_low_fluoresce\sweeped_pump_zero_175_stokes_zero_175_126_80'
    # ]
    
    while True:
        response = input("Plot data from a single shot (1/single) or calculate integrals (2/calc):\n")
        if response in ["0", "x", "exit", "q", "quit"]:
            break
        path = input("Enter the path to the data:\n")
        if response.lower() in ["1", "s", "single"]:
            plot_shot_results(path)
        elif response.lower() in ["2", "c", "calc"]:
            calculate_all_integrals(path)
        else:
            print("Invalid response")
        

    
    #calculate_integrals(root_directory)
    #plot_shot_results(single_shot_path)