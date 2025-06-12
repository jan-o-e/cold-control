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
        # apply rolling mean to every channel
        data['Channel 1 Voltage (V)'] = data['Channel 1 Voltage (V)'].rolling(window=window_size, center=True, min_periods=1).mean()
        data['Channel 4 Voltage (V)'] = data['Channel 4 Voltage (V)'].rolling(window=window_size, center=True, min_periods=1).mean()
        all_measurements.append(data)

    measurements = pd.DataFrame()

    for i, data in enumerate(all_measurements):
        measurements[f'Time (s) {i}'] = data['Time (s)']
        measurements[f'Channel 1 Voltage (V) {i}'] = data['Channel 1 Voltage (V)']
        measurements[f'Channel 4 Voltage (V) {i}'] = data['Channel 4 Voltage (V)']

    ref_0 = []
    fluor = []
    integrals_fl = []

    for data in all_measurements:
        ch1 = data['Channel 1 Voltage (V)']

        idx_sorted = (ch1 - 1).abs().sort_values().index
        idx_ch1_1 = idx_sorted[0] # ch1 crosses 1V (trigger value)

        
        # if ref == 'drop':
        #     ch3 = data['Channel 4 Voltage (V)']
        #     time = data['Time (s)']
        #     ch3_smooth = ch3.rolling(window=15, center=True, min_periods=1).mean()

        #     mask_window = (time >= -1e-4) & (time <= 1e-4)
        #     ch3_smooth_window = ch3_smooth[mask_window]
        #     time_window = time[mask_window]

        #     deriv_window = np.gradient(ch3_smooth_window, time_window)
        #     idx_drop = np.argmin(deriv_window)
        #     idx_rise = None
        #     for idx in range(idx_drop + 1, len(deriv_window)):
        #         if deriv_window[idx] > 0:
        #             idx_rise = idx
        #             break

        #     if idx_rise is not None:
        #         t_start_ref = time_window.iloc[idx_rise]
        #     else:
        #         t_start_ref = time_window.iloc[0]

        #     t_end_ref = data.loc[idx_ch1_1, 'Time (s)']

        # checking if t_start_ref is ok
        # plt.figure(figsize=(8, 3))
        # plt.plot(time, ch3, label='CH3 raw', alpha=0.5)
        # plt.plot(time, ch3_smooth, label='CH3 smooth', linewidth=2)
        # plt.axvline(t_start_ref, color='red', linestyle='--', label='t_start_ref')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Channel 4 Voltage (V)')
        # plt.title('Detection of t_start_ref')
        # plt.legend()
        # plt.tight_layout()
        # plt.show()

        # imaging beam on
        ch3 = data['Channel 4 Voltage (V)']
        time = data['Time (s)']
        ch3_smooth = ch3.rolling(window=144, center=True, min_periods=1).mean()

        # big increase when imaging starts
        mask_rise = (time >= 1.77e-3) & (time <= 2.0e-3)  # imaging starts at 2ms
        ch3_smooth_rise = ch3_smooth[mask_rise]
        time_rise = time[mask_rise]
        deriv_rise = np.gradient(ch3_smooth_rise, time_rise)

        idx_rise_rel = np.argmax(deriv_rise)
        idx_rise = time_rise.index[idx_rise_rel]
        t_rise = time.iloc[idx_rise]

        # big decrease when imaging stops
        t_drop = t_rise + 500e-6

        mask_fl = (time >= t_rise) & (time <= t_drop)
        ch3_segment_fl = data.loc[mask_fl, ['Time (s)', 'Channel 4 Voltage (V)']].copy()

        t_start_ref = t_drop + 50e-6  # after imaging stops
        t_end_ref = data['Time (s)'].iloc[-1] 
        mask_ref = (data['Time (s)'] >= t_start_ref) & (data['Time (s)'] <= t_end_ref)
        ch3_segment_ref = data.loc[mask_ref, ['Time (s)', 'Channel 4 Voltage (V)']].copy()
        average = ch3_segment_ref['Channel 4 Voltage (V)'].mean(axis=0)

        # integration area below curve, taking average as a zero reference
        area = trapz(ch3_segment_fl['Channel 4 Voltage (V)'] - [average]*len(ch3_segment_fl['Channel 4 Voltage (V)']), ch3_segment_fl['Time (s)'])

        print(f"Average background: {average}, Integrated area: {area}")

        # check if it's working
        plt.figure(figsize=(15, 8))
        plt.plot(time, ch3, label='CH4 raw', alpha=0.5)
        plt.plot(time, ch3_smooth, label='CH4 smooth', linewidth=2)
        plt.axvline(t_rise, color='green', linestyle='--', label='t_rise (subida)')
        plt.axvline(t_drop, color='red', linestyle='--', label='t_drop (bajada)')
        plt.axhline(average, color='purple', linestyle='--', label='average ref')
        plt.xlabel('Time (s)')
        plt.ylabel('Channel 4 Voltage (V)')
        plt.title('Detecting rise and fall of imaging beam')
        plt.legend()
        plt.tight_layout()
        plt.show()

        
        integrals_fl.append(area)
        ref_0.append(average)   

    #today = datetime.datetime.now().strftime("%d-%m")
    #output_dir = f'c:\\Users\\apc\\Documents\\marina\\06_jun\\{today}'
    #os.makedirs(output_dir, exist_ok=True)
    #integrals_fl_df = pd.DataFrame({'integral': integrals_fl, 'ref 0': ref_0})
    average_int = np.mean(integrals_fl)
    std_int = np.std(integrals_fl)
    max_int = np.max(integrals_fl)
    min_int = np.min(integrals_fl)
    print(f'Average integrated area: {average_int}, ')
    print(f"Standard deviation of area: {std_int}")
    print(f"Number of integrals calculated: {len(integrals_fl)}")
    #integrals_fl_df.to_csv(os.path.join(output_dir, 'integrated_area_155.csv'), index=False)


    
    # only one file
    # last_file = sorted(files)[-1]  
    # data = pd.read_csv(last_file)

    # fig, ax1 = plt.subplots(figsize=(11, 3))
    # ax1.plot(data['Time (s)'], data['Channel 1 Voltage (V)'], linewidth=1.5, color='tab:blue', label='CH 1')
    # ax1.set_xlabel(r'Time (s)')
    # ax1.set_ylabel(r'Intensity (a.u) CH 1', color='tab:blue')
    # ax1.tick_params(axis='y', labelcolor='tab:blue')

    # ax2 = ax1.twinx()
    # ax2.plot(data['Time (s)'], data['Channel 4 Voltage (V)'], linewidth=1.5, color='tab:orange', label='CH 3')
    # ax2.set_ylabel(r'Intensity (a.u) CH 3', color='tab:orange')
    # ax2.tick_params(axis='y', labelcolor='tab:orange')

    # fig.suptitle(f'CH1 and CH3 - {os.path.basename(last_file)}')
    # fig.tight_layout()
    # plt.show()



    # mean signal
    mean_time = measurements.filter(like='Time (s)').mean(axis=1)
    mean_ch1 = measurements.filter(like='Channel 1 Voltage (V)').mean(axis=1)
    mean_ch3 = measurements.filter(like='Channel 4 Voltage (V)').mean(axis=1)

    fig, ax1 = plt.subplots(figsize=(11, 5))
    ax1.plot(mean_time, mean_ch1, linewidth=1.5, color='tab:blue', label='Mean CH 1')
    ax1.set_xlabel(r'Time (s)')
    ax1.set_ylabel(r'Mean Intensity (a.u) CH 1', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.plot(mean_time, mean_ch3, linewidth=1.5, color='tab:orange', label='Mean CH 3')
    ax2.set_ylabel(r'Mean Intensity (a.u) CH 4', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    fig.suptitle('Mean CH1 and CH4 across all files')
    fig.tight_layout()
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

        average_int = np.mean(integrals_fl)
        std_int = np.std(integrals_fl)
        max_int = np.max(integrals_fl)
        min_int = np.min(integrals_fl)

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
    root_directory = r"D:\pulse_shaping_data\2025-06-12\16-23-23"
    single_shot_path = r'D:\pulse_shaping_data\2025-06-12\16-23-23\sweep_no_pulse_126_80\shot0'

    # folders_to_process = [
    #     r"C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\data\2025-06-09\16-08-07_low_fluoresce\sweeped_pump_175ns_20_stokes_175ns_0_2_126_80",
    #     r"C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\data\2025-06-09\16-08-07_low_fluoresce\sweeped_pump_optimized_stokes_optimized_126_80",
    #     r'C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\data\2025-06-09\16-08-07_low_fluoresce\sweeped_pump_zero_175_stokes_zero_175_126_80'
    # ]
 

    calculate_integrals(root_directory)
    #plot_shot_results(single_shot_path)