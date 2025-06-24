import pandas as pd
import numpy as np
import os, re
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import warnings

class ExperimentalDataProcessor:
    """
    A class to process experimental CSV data with multiple shots and channels.
    """
    
    def __init__(self, marker_channel=2, fluorescence_channel=4, rolling_window=None):
        self.marker_channel = marker_channel
        self.fluorescence_channel = fluorescence_channel
        self.rolling_window = rolling_window  # Optional smoothing window size
        self.time_col = "Time (s)"
        self.channel_cols = {
            1: "Channel 1 Voltage (V)",
            2: "Channel 2 Voltage (V)", 
            3: "Channel 3 Voltage (V)",
            4: "Channel 4 Voltage (V)"
        }
    
    def load_shot_data(self, shot_folder: str) -> List[pd.DataFrame]:
        """
        Load all CSV files from a shot folder.
        
        Args:
            shot_folder: Path to the shot folder containing CSV files
            
        Returns:
            List of DataFrames, one for each CSV file
        """
        file_pattern = re.compile(r"^iteration_\d+_data\.csv$", re.IGNORECASE)

        # Get all CSV files in the folder
        all_csv_files = list(Path(shot_folder).glob("*.csv"))

        # Filter the files based on the defined pattern
        csv_files = [
            csv_file for csv_file in all_csv_files
            if file_pattern.match(csv_file.name)
        ]

        if len(csv_files) == 0:
            raise ValueError(f"No CSV files found in {shot_folder}")
        
        dataframes = []
        for csv_file in sorted(csv_files):
            try:
                df = pd.read_csv(csv_file)
                # Apply rolling average if specified
                if self.rolling_window:
                    df = self.apply_rolling_average(df, self.rolling_window)
                dataframes.append(df)
            except Exception as e:
                warnings.warn(f"Could not load {csv_file}: {e}")
        
        return dataframes
    
    def apply_rolling_average(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """
        Apply rolling average to voltage channels to reduce noise.
        
        Args:
            df: Input DataFrame
            window: Rolling window size
            
        Returns:
            DataFrame with smoothed voltage data
        """
        df_smooth = df.copy()
        
        # Apply rolling average to all voltage channels
        for channel_num, col_name in self.channel_cols.items():
            if col_name in df_smooth.columns:
                df_smooth[col_name] = df_smooth[col_name].rolling(
                    window=window, center=True, min_periods=1
                ).mean()
        
        return df_smooth
    
    def validate_data(self, df: pd.DataFrame, 
                     marker_time_range: Tuple[float, float] = None,
                     fluor_drop_voltage: float = None,
                     fluor_drop_time_range: Tuple[float, float] = None) -> bool:
        """
        Validate that data meets experimental conditions.
        
        Args:
            df: DataFrame to validate
            marker_time_range: (min_time, max_time) for acceptable marker pulse (voltage drop)
            fluor_drop_voltage: Voltage threshold for fluorescence drop
            fluor_drop_time_range: (min_time, max_time) for acceptable fluorescence drop
            
        Returns:
            True if data is valid, False otherwise
        """
        # Check marker pulse timing (channel 2) - looking for voltage minimum (drop)
        if marker_time_range and self.channel_cols[self.marker_channel] in df.columns:
            marker_data = df[self.channel_cols[self.marker_channel]]
            # Find minimum of marker pulse (voltage drop)
            marker_min_idx = marker_data.idxmin()
            marker_min_time = df.iloc[marker_min_idx][self.time_col]
            
            if not (marker_time_range[0] <= marker_min_time <= marker_time_range[1]):
                return False
        
        # Check fluorescence drop timing (channel 4) - when it drops below threshold
        if (fluor_drop_voltage is not None and fluor_drop_time_range and 
            self.channel_cols[self.fluorescence_channel] in df.columns):
            
            fluor_data = df[self.channel_cols[self.fluorescence_channel]]
            time_data = df[self.time_col]
            
            # Find when fluorescence first drops below threshold
            drop_indices = fluor_data < fluor_drop_voltage
            if not drop_indices.any():
                return False
                
            first_drop_idx = drop_indices.idxmax()  # First True index
            first_drop_time = time_data.iloc[first_drop_idx]
            
            if not (fluor_drop_time_range[0] <= first_drop_time <= fluor_drop_time_range[1]):
                return False
        
        return True
    
    def average_shot_data(self, shot_folder: str, 
                         marker_time_range: Tuple[float, float] = None,
                         fluor_drop_voltage: float = None,
                         fluor_drop_time_range: Tuple[float, float] = None,
                         output_path: str = None) -> pd.DataFrame:
        """
        Average all valid CSV files from a shot folder.
        
        Args:
            shot_folder: Path to shot folder
            marker_time_range: Time range for valid marker pulse (voltage drop)
            fluor_drop_voltage: Voltage threshold for fluorescence drop
            fluor_drop_time_range: Time range for valid fluorescence drop
            output_path: Path to save averaged CSV (optional)
            
        Returns:
            DataFrame with averaged data
        """
        dataframes = self.load_shot_data(shot_folder)
        
        # Filter valid data
        valid_dfs = []
        for df in dataframes:
            if self.validate_data(df, marker_time_range, fluor_drop_voltage, fluor_drop_time_range):
                valid_dfs.append(df)
        
        if len(valid_dfs) == 0:
            raise ValueError("No valid data found in shot folder")
        
        print(f"Using {len(valid_dfs)} out of {len(dataframes)} files from {shot_folder}")
        
        # Average the data
        # Assume all dataframes have the same structure
        averaged_df = valid_dfs[0].copy()
        
        # Average all voltage columns
        for channel_num, col_name in self.channel_cols.items():
            if col_name in averaged_df.columns:
                values = np.array([df[col_name].values for df in valid_dfs])
                averaged_df[col_name] = np.mean(values, axis=0)
        
        # Save if output path provided
        if output_path:
            averaged_df.to_csv(output_path, index=False)
        
        return averaged_df
    
    def average_shot_data_aligned(self, shot_folder: str,
                                 fluor_drop_voltage: float,
                                 marker_time_range: Tuple[float, float] = None,
                                 fluor_drop_time_range: Tuple[float, float] = None,
                                 time_before_drop: float = 1.1e-3,
                                 time_after_drop: float = 4e-3,
                                 num_points: int = 50000,
                                 output_path: str = None) -> pd.DataFrame:
        """
        Average CSV files after aligning them based on fluorescence drop timing.
        Uses interpolation to create aligned time series.
        
        Args:
            shot_folder: Path to shot folder
            fluor_drop_voltage: Voltage threshold for fluorescence drop alignment
            marker_time_range: Time range for valid marker pulse (optional validation)
            fluor_drop_time_range: Time range for valid fluorescence drop (optional validation)
            time_before_drop: Time range before drop to include (seconds)
            time_after_drop: Time range after drop to include (seconds)
            num_points: Number of points in interpolated time series
            output_path: Path to save averaged CSV (optional)
            
        Returns:
            DataFrame with time-aligned and averaged data
        """
        from scipy import interpolate
        
        dataframes = self.load_shot_data(shot_folder)
        
        # Filter valid data and find drop times
        valid_dfs = []
        drop_times = []
        
        for df in dataframes:
            # if not self.validate_data(df, marker_time_range, fluor_drop_voltage, fluor_drop_time_range):
            #     continue
                
            # Find fluorescence drop time
            fluor_data = df[self.channel_cols[self.fluorescence_channel]]
            time_data = df[self.time_col]
            
            drop_indices = fluor_data < fluor_drop_voltage
            if not drop_indices.any():
                continue
                
            first_drop_idx = drop_indices.idxmax()
            drop_time = time_data.iloc[first_drop_idx]
            
            valid_dfs.append(df)
            drop_times.append(drop_time)
        
        if len(valid_dfs) == 0:
            raise ValueError("No valid data found in shot folder")
        
        print(f"Aligning {len(valid_dfs)} out of {len(dataframes)} files from {shot_folder}")
        
        # Create aligned time axis (relative to drop time)
        aligned_time = np.linspace(-time_before_drop, time_after_drop, num_points)
        #print(f"Aligned time axis: {aligned_time[0]} to {aligned_time[-1]} seconds")
        
        # Initialize arrays for interpolated data
        interpolated_data = {}
        interpolated_data[self.time_col] = aligned_time
        
        for channel_num, col_name in self.channel_cols.items():
            if col_name in valid_dfs[0].columns:
                interpolated_data[col_name] = []
        
        # Interpolate each valid dataset to the aligned time axis
        for df, drop_time in zip(valid_dfs, drop_times):
            # Shift time relative to drop
            relative_time = df[self.time_col].values - drop_time
            #print(f"Relative time axis: {relative_time[0]} to {relative_time[-1]} seconds")
            
            # Interpolate each channel
            for channel_num, col_name in self.channel_cols.items():
                if col_name not in df.columns:
                    continue
                    
                channel_data = df[col_name].values
                
                # Create interpolation function
                f_interp = interpolate.interp1d(
                    relative_time, channel_data, 
                    kind='linear', bounds_error=False, fill_value=np.nan
                )
                
                # Interpolate to aligned time axis
                interpolated_channel = f_interp(aligned_time)
                interpolated_data[col_name].append(interpolated_channel)
        
        # Average the interpolated data
        averaged_data = {self.time_col: aligned_time}
        for channel_num, col_name in self.channel_cols.items():
            if col_name in interpolated_data and len(interpolated_data[col_name]) > 0:
                channel_stack = np.array(interpolated_data[col_name])
                # Use nanmean to handle any NaN values from interpolation
                averaged_data[col_name] = np.nanmean(channel_stack, axis=0)
        
        # Create averaged DataFrame
        averaged_df = pd.DataFrame(averaged_data)
        
        # Save if output path provided
        if output_path:
            averaged_df.to_csv(output_path, index=False)
        
        return averaged_df
    
    def calculate_channel4_metrics(self, df: pd.DataFrame,
                                 background_time_range: Tuple[float, float],
                                 integration_time_range: Tuple[float, float]) -> Dict:
        """
        Calculate average and integral of channel 4 with uncertainty.
        
        Args:
            df: DataFrame with averaged data
            background_time_range: (start_time, end_time) for background calculation
            integration_time_range: (start_time, end_time) for integration
            
        Returns:
            Dictionary with metrics and uncertainties
        """
        channel4_col = self.channel_cols[self.fluorescence_channel]
        
        if channel4_col not in df.columns:
            raise ValueError(f"Channel {self.fluorescence_channel} not found in data")
        
        # Get time and channel 4 data
        time_data = df[self.time_col].values
        channel4_data = df[channel4_col].values
        
        # Calculate background average and std
        bg_mask = (time_data >= background_time_range[0]) & (time_data <= background_time_range[1])
        background_data = channel4_data[bg_mask]
        bg_average = np.mean(background_data)
        bg_std = np.std(background_data)
        #print(bg_average, bg_std)
        
        # Calculate integral over integration range
        int_mask = (time_data >= integration_time_range[0]) & (time_data <= integration_time_range[1])
        integration_time = time_data[int_mask]
        integration_data = channel4_data[int_mask]-bg_average  # Subtract background average
        
        # Numerical integration using trapezoidal rule
        integral_value = np.trapz(integration_data, integration_time)
        
        # Uncertainty estimation based on background std
        integration_duration = integration_time_range[1] - integration_time_range[0]
        integral_uncertainty = bg_std * integration_duration #* np.sqrt(len(integration_data)) 
        
        return {
            'background_average': bg_average,
            'background_std': bg_std,
            'integral_value': integral_value,
            'integral_uncertainty': integral_uncertainty,
            'integration_duration': integration_duration,
            'num_integration_points': len(integration_data)
        }
    
    def process_all_experiments(self, root_folder: str,
                              marker_time_range: Tuple[float, float] = None,
                              fluor_drop_voltage: float = None,
                              fluor_drop_time_range: Tuple[float, float] = None,
                              background_time_range: Tuple[float, float] = (0, 1),
                              integration_time_range: Tuple[float, float] = (2, 3),
                              save_averaged_csvs: bool = True,
                              use_alignment: bool = False,
                              alignment_params: Dict = None) -> pd.DataFrame:
        """
        Process all experiments in the root folder.
        
        Args:
            root_folder: Path to root experimental folder
            marker_time_range: Time range for valid marker pulse (voltage drop)
            fluor_drop_voltage: Voltage threshold for fluorescence drop
            fluor_drop_time_range: Time range for valid fluorescence drop
            background_time_range: Time range for background calculation
            integration_time_range: Time range for integration
            save_averaged_csvs: Whether to save averaged CSV files
            use_alignment: Whether to use time-aligned averaging
            alignment_params: Parameters for alignment (time_before_drop, time_after_drop, num_points)
            
        Returns:
            DataFrame with summary results for all experiments
        """
        if alignment_params is None:
            alignment_params = {
                'time_before_drop': 1.1e-3,
                'time_after_drop': 4e-3,
                'num_points': 50000
            }
        root_path = Path(root_folder)
        results = []
        
        # Iterate through parameter folders
        for param_folder in root_path.iterdir():
            if not param_folder.is_dir():
                continue
                
            print(f"Processing parameter folder: {param_folder.name}")
            
            # Iterate through shot folders
            for shot_folder in param_folder.iterdir():
                if not shot_folder.is_dir() or not shot_folder.name.startswith('shot'):
                    continue
                
                try:
                    print(f"  Processing {shot_folder.name}")
                    
                    # Average the shot data
                    output_path = None
                    if save_averaged_csvs:
                        suffix = "_aligned" if use_alignment else "_averaged"
                        output_path = shot_folder / f"{shot_folder.name}{suffix}.csv"
                    
                    if use_alignment:
                        if fluor_drop_voltage is None:
                            raise ValueError("fluor_drop_voltage must be specified for alignment")
                        
                        averaged_df = self.average_shot_data_aligned(
                            str(shot_folder),
                            fluor_drop_voltage,
                            marker_time_range,
                            fluor_drop_time_range,
                            output_path=output_path,
                            **alignment_params
                        )
                    else:
                        averaged_df = self.average_shot_data(
                            str(shot_folder), 
                            marker_time_range, 
                            fluor_drop_voltage,
                            fluor_drop_time_range,
                            output_path
                        )
                    
                    # Calculate channel 4 metrics
                    metrics = self.calculate_channel4_metrics(
                        averaged_df,
                        background_time_range,
                        integration_time_range
                    )
                    
                    # Store results
                    result = {
                        'parameter_folder': param_folder.name,
                        'shot_folder': shot_folder.name,
                        'shot_path': str(shot_folder),
                        **metrics
                    }
                    results.append(result)
                    
                except Exception as e:
                    print(f"    Error processing {shot_folder.name}: {e}")
                    continue
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(results)
        
        # Save summary
        if use_alignment:
            summary_path = root_path / "experiment_summary_aligned.csv"
        else:
            summary_path = root_path / "experiment_summary_averaged.csv"

        summary_df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to: {summary_path}")
        
        return summary_df

# Example usage
if __name__ == "__main__":
    # Initialize processor with optional rolling average (e.g., 5-point window)
    processor = ExperimentalDataProcessor(
        marker_channel=2, 
        fluorescence_channel=4,
        rolling_window=32  # Apply 5-point rolling average to reduce noise
    )

    MARKER_DROP = 7.45 # The level below which the AWG marker will drop
    IMG_WIDTH = 500e-6 # The width of the imaging pulse
    TARGET_TIME = 1.46e-3 # The expected time of the AWG marker
    TOLERANCE = 50e-6 # How far around the target time to check for the marker
    MOT_DROP = 19.7e-3 # The level below which the fluorescence will drop after the MOT is turned off
    MOT_DROP_TIME = 600e-6 # The expected time of the MOT drop, when the MOT is turned off
    T_RISE = MOT_DROP_TIME + 1e-3

    
    # Example parameters (adjust these based on your data)
    root_folder = r"D:\pulse_shaping_data\2025-06-23\17-37-40"
    marker_time_range = (TARGET_TIME-TOLERANCE, TARGET_TIME+TOLERANCE)  # When marker voltage drop should occur
    fluor_drop_voltage = MOT_DROP  # Voltage threshold for fluorescence drop
    fluor_drop_time_range = (MOT_DROP_TIME-TOLERANCE, MOT_DROP_TIME+TOLERANCE)  # When fluorescence drop should occur
    integration_time_range = (T_RISE, T_RISE+IMG_WIDTH)  # Time range for integration
    background_time_range = (T_RISE+IMG_WIDTH+0.1e-3, 1)  # Time range for background calculation
    
    # Process all experiments with standard averaging
    try:
        summary_standard = processor.process_all_experiments(
            root_folder=root_folder,
            marker_time_range=marker_time_range,
            fluor_drop_voltage=fluor_drop_voltage,
            fluor_drop_time_range=fluor_drop_time_range,
            background_time_range=background_time_range,
            integration_time_range=integration_time_range,
            save_averaged_csvs=True,
            use_alignment=False
        )
        
        print("Standard averaging complete!")
        def adjust_time(timings, offset):
            return tuple(np.subtract(timings, (offset, offset)))
        
        # Process with time alignment
        summary_aligned = processor.process_all_experiments(
            root_folder=root_folder,
            marker_time_range=adjust_time(marker_time_range, MOT_DROP_TIME),
            fluor_drop_voltage=fluor_drop_voltage,
            fluor_drop_time_range=adjust_time(fluor_drop_time_range, MOT_DROP_TIME),
            background_time_range=adjust_time(background_time_range, MOT_DROP_TIME),  # Background after drop
            integration_time_range=adjust_time(integration_time_range, MOT_DROP_TIME),   # Integration after drop
            save_averaged_csvs=True,
            use_alignment=True,
            alignment_params={
                'time_before_drop': 1.0e-3,  # 1.1ms before drop
                'time_after_drop': 3.9e-3,   # 4ms after drop
                'num_points': 50000          # High resolution
            }
        )
        
        print("Time-aligned averaging complete!")
        print(f"Processed {len(summary_aligned)} shots total")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Please check your file paths and parameters.")