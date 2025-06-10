import os
import re
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


class DataFolderManager:
    """
    A class to manage and analyse data saved in a specified base directory. This class
    is particularly to manage data from the MOT Fluorescence experiment.
    """
    def __init__(self, base_path, window_size=32):
        # Base path is the path to the directory containing all the data folders for a single sweep
        self.base_path = base_path
        self.folder_paths = self.get_folder_paths()
        self.summary_path = self.calculate_integrals()
        self.window_size = window_size




    def get_folder_paths(self):
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
        if not os.path.isdir(self.base_path):
            print(f"Error: '{self.base_path}' is not a valid directory.")
            return []

        # Iterate through all entries in the directory
        for entry_name in os.listdir(self.base_path):
            full_path = os.path.join(self.base_path, entry_name)
            # Check if the entry is a directory (folder)
            if os.path.isdir(full_path):
                # Add the raw string representation of the path
                folder_paths.append(r"{}".format(full_path))
        return folder_paths
    
    def calculate_integrals(self, shots_to_include=[], window_size=32,
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
            folders_to_process = self.folder_paths


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
                data['Channel 1 Voltage (V)'] = data['Channel 1 Voltage (V)'].rolling(window=self.window_size, center=True, min_periods=1).mean()
                data['Channel 4 Voltage (V)'] = data['Channel 4 Voltage (V)'].rolling(window=self.window_size, center=True, min_periods=1).mean()

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

                area = np.trapz(ch3_segment_fl['Channel 4 Voltage (V)'] - average, ch3_segment_fl['Time (s)'])
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
        summary_output_path = os.path.join(self.base_path, 'summary_integrals.csv')
        summary_df.to_csv(summary_output_path, index=False)

        print(f"\nResumen guardado en: {summary_output_path}")
        return summary_output_path


    def get_data_from_summary(self):
        """
        Reads the summary dataframe and extracts the relevant data for further analysis.
        Returns:
            pd.DataFrame: A DataFrame containing the relevant data from the summary.
        """

    @staticmethod
    def extract_voltages(filename):
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
                voltage1 = float(match.group(1))
                voltage2 = float(match.group(2))
                return voltage1, voltage2
            except ValueError:
                print(f"Warning: Could not convert extracted values to float for filename: {filename}")
                return None
        else:
            return None
        
    def plot_contour_from_csv(self):
        """
        Reads data from a CSV, extracts voltages from filenames, and generates
        a contour plot of 'average_integral' against the two voltage values.

        Args:
            csv_filepath (str): The path to the input CSV file.
        """
        try:
            # Read the CSV file into a pandas DataFrame
            df = pd.read_csv(self.summary_path)
            print(f"Successfully loaded data from '{self.summary_path}'. Shape: {df.shape}")
            # print("DataFrame head:\n", df.head()) # Uncomment to inspect data

            # Apply the voltage extraction function to the 'folder' column
            # and create new columns for power_voltage and frequency_voltage.
            # Use .apply(pd.Series) to expand the tuple output into two separate columns.
            df[['power_voltage', 'frequency_voltage']] = df['folder'].apply(
                lambda x: pd.Series(self.extract_voltages(x))
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
            print(f"Error: The file '{self.summary_path}' was not found. Please check the path.")
        except KeyError as e:
            print(f"Error: Missing expected column in CSV: {e}. Please ensure 'folder' and 'average_integral' columns exist.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


    def plot_shot_results(self, shot_path):
        """
        Plots the results of a specific shot from the data folder.
        Args:
            shot_path (str): The path to the folder containing the shot data, relative to
            the base path.
        """
        folder_path = os.path.join(self.base_path, shot_path)

        pattern = re.compile(r'^iteration_\d+_data\.csv$')
        files = [os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if pattern.match(filename)]
        all_measurements = [] 

        for file_path in files:
            data = pd.read_csv(file_path)
            # apply rolling mean to every channel
            data['Channel 1 Voltage (V)'] = data['Channel 1 Voltage (V)'].rolling(window=self.window_size, center=True, min_periods=1).mean()
            data['Channel 4 Voltage (V)'] = data['Channel 4 Voltage (V)'].rolling(window=self.window_size, center=True, min_periods=1).mean()
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
            print(average)

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

            # integration area below curve, taking average as a zero reference
            area = np.trapz(ch3_segment_fl['Channel 4 Voltage (V)'] - [average]*len(ch3_segment_fl['Channel 4 Voltage (V)']), ch3_segment_fl['Time (s)'])
            
            integrals_fl.append(area)
            ref_0.append(average)   

        today = datetime.datetime.now().strftime("%d-%m")
        output_dir = f'c:\\Users\\apc\\Documents\\marina\\06_jun\\{today}'
        os.makedirs(output_dir, exist_ok=True)
        integrals_fl_df = pd.DataFrame({'integral': integrals_fl, 'ref 0': ref_0})
        average_int = integrals_fl_df['integral'].mean()
        print(f'Average: {average_int}, ')
        integrals_fl_df.to_csv(os.path.join(output_dir, 'integrated_area_155.csv'), index=False)


        
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

    @staticmethod
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
    # Example usage
    base_path = r"C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\data\2023-10-16"
    manager = DataFolderManager(base_path)
    

    # Plot contour from summary CSV
    manager.plot_contour_from_csv()

    # Plot specific shot results
    manager.plot_shot_results('iteration_1_data')

    # Plot data from a specific folder
    DataFolderManager.plot_data(base_path, plot_all=True, v_ch=4, window_size=32, show_markers=True)