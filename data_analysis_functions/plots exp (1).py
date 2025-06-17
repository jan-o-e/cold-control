import matplotlib.pyplot as plt
import re, os
import pandas as pd
from numpy import trapz
import numpy as np
import datetime

plotter = False



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

        measurements[f'Time (s) {i}'] = data['Time (s)']
        measurements[f'Channel 1 Voltage (V) {i}'] = data['Channel 1 Voltage (V)']
        measurements[f'Channel 4 Voltage (V) {i}'] = data['Channel 4 Voltage (V)']
        measurements[f"Channel 2 Voltage (V) {i}"] = data["Channel 2 Voltage (V)"]

        ch1 = data['Channel 1 Voltage (V)']

        idx_sorted = (ch1 - 1).abs().sort_values().index
        idx_ch1_1 = idx_sorted[0] # ch1 crosses 1V (trigger value)

        

        # imaging beam on
        ch4 = data['Channel 4 Voltage (V)']
        ch2 = data["Channel 2 Voltage (V)"]
        time = data['Time (s)']

        # Step 1: Find the first index where ch2 drops below 7.45
        below = ch2 < 7.45
        if not below.any():
            print("Channel 2 never drops below 7.45 V")
        else:
            drop_index = below.idxmax()
            drop_time = time[drop_index]

            # Step 2: From that point onward, find the first time it goes back above 7.45
            above = ch2[drop_index+1:] > 7.45
            if not above.any():
                print(f"Channel 2 drops below 7.45 V at {drop_time} s and never goes back above.")
            else:
                rise_index = above.idxmax()
                rise_time = time[rise_index]
                print(f"Channel 2 drops below 7.45 V at {drop_time} s")
                print(f"Channel 2 goes back above 7.45 V at {rise_time} s")

        ch4_smooth = ch4#.rolling(window=144, center=True, min_periods=1).mean()

        # big increase when imaging start
        # mask_rise = (time >= 1.77e-3) & (time <= 2.0e-3)  # imaging starts at 2ms
        # ch4_smooth_rise = ch4_smooth[mask_rise]
        # time_rise = time[mask_rise]
        # deriv_rise = np.gradient(ch4_smooth_rise, time_rise)

        # idx_rise_rel = np.argmax(deriv_rise)
        # idx_rise = time_rise.index[idx_rise_rel]
        # t_rise = time.iloc[idx_rise]
        t_rise = 1.82e-3

        # big decrease when imaging stops
        t_drop = t_rise + 500e-6

        mask_fl = (time >= t_rise) & (time <= t_drop)
        ch4_segment_fl = data.loc[mask_fl, ['Time (s)', 'Channel 4 Voltage (V)']].copy()

        t_start_ref = t_drop + 50e-6  # after imaging stops
        t_end_ref = data['Time (s)'].iloc[-1] 
        mask_ref = (data['Time (s)'] >= t_start_ref) & (data['Time (s)'] <= t_end_ref)
        ch4_segment_ref = data.loc[mask_ref, ['Time (s)', 'Channel 4 Voltage (V)']].copy()
        average = ch4_segment_ref['Channel 4 Voltage (V)'].mean(axis=0)

        # integration area below curve, taking average as a zero reference
        area = trapz(ch4_segment_fl['Channel 4 Voltage (V)'] - [average]*len(ch4_segment_fl['Channel 4 Voltage (V)']), ch4_segment_fl['Time (s)'])

        target_time = 0.001604375
        tolerance = 5e-6

        # only consider the result if the timing of the awg marker is correct
        if abs(drop_time - target_time) <= tolerance:
            print("The drop time is close enough to the expected time")
        
            integrals_fl.append(area)
            ref_0.append(average)

            if area is not None and not np.isnan(area):
                valid_meas[f'Time (s) {i}'] = data['Time (s)']
                valid_meas[f'Channel 1 Voltage (V) {i}'] = data['Channel 1 Voltage (V)']
                valid_meas[f'Channel 4 Voltage (V) {i}'] = data['Channel 4 Voltage (V)']
                valid_meas[f"Channel 2 Voltage (V) {i}"] = data["Channel 2 Voltage (V)"]

        print(f"Average background: {average}, Integrated area: {area}\n")

        # # check if it's working
        # plt.figure(figsize=(15, 8))
        # plt.plot(time, ch3, label='CH4 raw', alpha=0.5)
        # plt.plot(time, ch3_smooth, label='CH4 smooth', linewidth=2)
        # plt.axvline(t_rise, color='green', linestyle='--', label='t_rise (subida)')
        # plt.axvline(t_drop, color='red', linestyle='--', label='t_drop (bajada)')
        # plt.axhline(average, color='purple', linestyle='--', label='average ref')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Channel 4 Voltage (V)')
        # plt.title('Detecting rise and fall of imaging beam')


        # plt.legend()
        # plt.tight_layout()
        # plt.show()


        fig, ax1 = plt.subplots(figsize=(15, 8))

        ax1.plot(time, ch4, label='CH4 raw', alpha=0.5, color='tab:blue')
        ax1.plot(time, ch4_smooth, label='CH4 smooth', linewidth=2, color='tab:cyan')
        ax1.axvline(t_rise, color='green', linestyle='--', label='t_rise (subida)')
        ax1.axvline(t_drop, color='red', linestyle='--', label='t_drop (bajada)')
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
    ch4_val = valid_meas.filter(like='Channel 4 Voltage (V)').mean(axis=1)
    ch2_val = valid_meas.filter(like="Channel 2 Voltage (V)").mean(axis=1)

    fig1, ax1 = plt.subplots(figsize=(11, 5))
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

    fig1.suptitle('Only data where the AWG marker is at the right time, and the integral is a number')
    fig1.tight_layout()


    plt.show()



def calculate_integrals(root_directory, shots_to_include=[], window_size=32,
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

            ch3 = data['Channel 4 Voltage (V)']
            time = data['Time (s)']
            ch3_smooth = ch3.rolling(window=144, center=True, min_periods=1).mean()

            mask_rise = (time >= 1.77e-3) & (time <= 2.0e-3)
            ch3_smooth_rise = ch3_smooth[mask_rise]
            time_rise = time[mask_rise]
            deriv_rise = np.gradient(ch3_smooth_rise, time_rise)

            idx_rise_rel = np.argmax(deriv_rise)
            idx_rise = time_rise.index[idx_rise_rel]
            t_rise = time.iloc[idx_rise]
            t_drop = t_rise + 500e-6

            mask_fl = (time >= t_rise) & (time <= t_drop)
            ch3_segment_fl = data.loc[mask_fl, ['Time (s)', 'Channel 4 Voltage (V)']].copy()

            t_start_ref = t_drop + 50e-6
            t_end_ref = data['Time (s)'].iloc[-1]
            mask_ref = (data['Time (s)'] >= t_start_ref) & (data['Time (s)'] <= t_end_ref)
            ch3_segment_ref = data.loc[mask_ref, ['Time (s)', 'Channel 4 Voltage (V)']].copy()
            average = ch3_segment_ref['Channel 4 Voltage (V)'].mean()

            area = trapz(ch3_segment_fl['Channel 4 Voltage (V)'] - average, ch3_segment_fl['Time (s)'])
            integrals_fl.append(area)
            ref_0.append(average)

            # Guardar resultado por iteración
            integrals_fl_df = pd.DataFrame({'integral': [area], 'ref 0': [average]})
            output_dir = rf'C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\data\integrals_data_analysis\{today}\{folder_name}'
            os.makedirs(output_dir, exist_ok=True)
            integrals_fl_df.to_csv(os.path.join(output_dir, f'integrated_area_iteration_{iteration_number}.csv'), index=False)

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
            calculate_integrals(path)
        else:
            print("Invalid response")
        

    
    #calculate_integrals(root_directory)
    #plot_shot_results(single_shot_path)