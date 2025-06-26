""""
Class to manage the conversion between the Rabi frequencies experienced by atoms in the MOT
and the voltage amplitudes of the electronic waveforms sent by the AWG to the AOMs (through an amplifier).

Authors: Jan Ole Ernst, Matt King
Date: 23 June 2025
"""

import os
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from scipy.interpolate import interp1d




class RabiFreqVoltageConverter:
    def __init__(self, csv_path):
        # Load data
        self.df = pd.read_csv(csv_path)
        self.data_dir = os.path.dirname(os.path.abspath(csv_path))

        # Extract amplitude (x) and Rabi frequency (y)
        self.x = self.df['amplitude_cal'].values
        self.y = self.df['rabi_measured_no_ang'].values
        self.waist_size = self.df['waist_size'].values[0]
        self.cg = float(self.df['cg_ang'].values[0])

        # Save bounds
        self.min_voltage = np.min(self.x)
        self.max_voltage = np.max(self.x)
        self.min_rabi = np.min(self.y)
        self.max_rabi = np.max(self.y)

        # Interpolation: voltage -> rabi (safe to use raw x and y)
        self._volt_to_rabi_interp = interp1d(
            self.x, self.y, kind='cubic', fill_value="extrapolate", assume_sorted=False
        )

        # Interpolation: rabi -> voltage — must sort and deduplicate
        df_clean = pd.DataFrame({'rabi_v': self.y, 'amp': self.x})
        df_clean = df_clean.groupby('rabi_v', as_index=False).mean()  # remove duplicates by averaging
        df_clean = df_clean.sort_values(by='rabi_v')  # ensure sorted for interp1d

        self.sorted_y = df_clean['rabi_v'].values
        self.sorted_x = df_clean['amp'].values

        self._rabi_to_volt_interp = interp1d(
            self.sorted_y, self.sorted_x, kind='cubic', fill_value="extrapolate", assume_sorted=True
        )

        # Plot and save
        self._save_plot(self.x, self.y)

    def _save_plot(self, x, y):
        plt.figure(figsize=(8, 5))
        plt.plot(x, y, 'o-', label='Voltage vs Rabi Frequency')
        plt.xlabel('Amplitude Cal (Voltage)')
        plt.ylabel('Rabi Frequency/ d_cg (MHz)')
        plt.title('Voltage to Rabi Frequency Mapping')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plot_path = os.path.join(self.data_dir, f'voltage_vs_rabi_{self.waist_size}mu_waist.pdf')
        plt.savefig(plot_path)
        plt.close()

    def voltage_to_rabi(self, voltage):
        """
        Convert voltage amplitude to Rabi frequency.
        input:
        voltage: Voltage amplitude in V
        returns:
        rabi frequency (not normalised to angular CG)
        """
        if not (self.min_voltage <= voltage <= self.max_voltage):
            raise ValueError(f"Voltage {voltage} out of bounds ({self.min_voltage} - {self.max_voltage})")
        return float(self._volt_to_rabi_interp(voltage)*np.abs(self.cg))

    def rabi_to_voltage(self, rabi):
        """
        Convert Rabi frequency to voltage amplitude.
        input:
        rabi: Rabi frequency in MHz (not normalised to angular CG)
        """
        print(f"Rabi frequency limits are {self.get_rabi_limits(print_info=False)}")
        rabi=rabi/np.abs(self.cg)
        if not (self.min_rabi <= rabi <= self.max_rabi):
            raise ValueError(f"Rabi frequency {rabi} out of bounds ({self.min_rabi} - {self.max_rabi})")
        return float(self._rabi_to_volt_interp(rabi))
    

    def rescale_csv(self, rabi, csv_in, csv_out, normalised = True):
        """
        Function to scale a waveform to have the correct Rabi frequency at the peak of the pulse
        inputs:
         - rabi (float): The desired peak Rabi frequency
         - csv_in (str): The path to the csv to convert rescale
         - csv_out (str): The path to save the csv to
         - normalised (bool): Whether the input waveform is normalised or not. Assumed to be true
        """
        if rabi == 0:
            rescale_factor = 0
        else:
            rescale_factor = self.rabi_to_voltage(rabi)

        # Step 1: Read CSV with a single row
        df = pd.read_csv(csv_in, header=None)

        # Step 2: Normalize to [0, 1]
        values = df.iloc[0].values.astype(float)
        if normalised or rescale_factor == 0:
            norm_values = values
        else:
            norm_values = (values - np.min(values)) / (np.max(values) - np.min(values))

        # Step 3: Rescale
        rescaled = norm_values * rescale_factor

        # Step 4: Save to CSV
        rescaled_df = pd.DataFrame([rescaled])
        rescaled_df.to_csv(csv_out, index=False, header=False)

        print(f"Processed data saved to: {csv_out}")

    def get_rabi_limits(self, print_info=True):
        act_max = np.abs(self.max_rabi*self.cg)/(2*np.pi)
        act_min = np.abs(self.min_rabi*self.cg)/(2*np.pi)

        if print_info:
            print(f"The maximum and minimum values for the transition normalised Rabi frequency are: ")
            print(f"Max: {self.max_rabi}, Min: {self.min_rabi}")

            print("This corresponds to actual Rabi frequencies of:")
            print(f"Max: {act_max}, Min: {act_min}")

        return (act_max, act_min)