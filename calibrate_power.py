'''
Script to calibrate the power. Outputs the required input amplitude to achieve target power. 
Additionally saves the Rabi frequency data corresponding to each amplitude value.

@author: marina llano, Jan Ole Ernst
'''


import os
import time
import datetime
import ast
from instruments.WX218x.WX218x_awg import Channel
import lab_control_functions.calibration_functions as calibrate
import numpy as np
from scipy.constants import c, epsilon_0, hbar
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import csv
from scipy.interpolate import interp1d


# plt.rcParams.update({
#     'text.usetex': True,
#     'text.latex.preamble': r'\usepackage{amsmath}',
#     'font.family': 'serif',
#     'font.size': 12,
#     'axes.labelsize': 14,
#     'axes.titlesize': 15,
#     'legend.fontsize': 12,
#     'xtick.labelsize': 12,
#     'ytick.labelsize': 12,
#     'axes.linewidth': 1.1,
#     'xtick.direction': 'in',
#     'ytick.direction': 'in',
#     'xtick.major.size': 5,
#     'ytick.major.size': 5,
# })

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

        rescale_factor = self.rabi_to_voltage(rabi)

        # Step 1: Read CSV with a single row
        df = pd.read_csv(csv_in, header=None)

        # Step 2: Normalize to [0, 1]
        values = df.iloc[0].values.astype(float)
        if normalised:
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


# general coefficients
#gamma_d1 = 5.746*np.pi
gamma_d2= 6*np.pi
typical_waist_size=100 #mu m
#d_d1 = 2.537 * 10**(-29)
d_d2= 2.853 * 10**(-29) 

# V-STIRAP re-preparation coefficients
cg_d2_stokes = np.sqrt(1/30)
cg_d2_pump = -np.sqrt(5/24)
#rabi_stirap_d1 = 41
rabi_stirap_d2 = 50*2*np.pi

# OPT PUMPING coefficients
cg_d2_p1 = np.sqrt(1/24)
cg_d2_p2 = np.sqrt(1/8)
rabi_p1_d1 = 34 
rabi_p1_d2 = 57.5
rabi_p2_d1 = 24
rabi_p2_d2 = 25.5

today = datetime.datetime.now().strftime("%d-%m")


def rabi_to_laserpower(omega, d, cg, beam_waist):
    """
    Convert Rabi frequency to laser power
    Input agrs:
    omega: Rabi frequency in MHz
    d: dipole moment in C*m
    cg: angular CG dependence
    beam_waist: beam waist in micron"""

    efield=(hbar*(omega*10**6))/(np.abs(d*cg))
    intensity=(efield**2*epsilon_0*c)/(2)
    return (intensity*np.pi*(beam_waist*10**(-6))**2)*10**(3) # in mW

def laserpower_to_rabi(power, d, cg, beam_waist):
    """
    Convert laser power to Rabi frequency
    Input agrs:
    power: power in mW
    d: dipole moment in C*m
    cg: angular CG dependence
    beam_waist: beam waist in micron"""

    intensity=power/(np.pi*(beam_waist*10**(-6))**2*10**3)
    efield=np.sqrt((2*intensity)/(epsilon_0*c))
    omega=(d*cg*efield)/(hbar*10**6)
    return np.abs(omega) #in MHz with angular dependence

pulse = 'pump'  # 'stokes', 'pump', 'P1', 'P2'
channel = 1  # AWG channel
amplitude = 0.2
amplitude_cal = 0.00
diff = 1
results_dict = {}
# Finding the voltage amplitude that corresponds to this power
awg_chan_freqs_map = {1: [116], 2: [80], 3: [62.35], 4: [82.5]}


if __name__ == "__main__":

    if channel==1:
        config_save_path="calibrations\jan\STIRAP_ELYSA"
    elif channel==2:
        config_save_path="calibrations\jan\STIRAP_DL_PRO"
    elif channel==3:
        config_save_path="calibrations\jan\OPT_PUMP_ELYSA"
    elif channel==4:
        config_save_path="calibrations\jan\OPT_PUMP_DL_PRO"
    else:
        raise ValueError("Channel must be 1, 2, 3 or 4")


    while abs(amplitude_cal/amplitude -1) > 0.1 or diff > 2e-5:
    # Pause to change the fiber/ the power distrubution
        response = input('Press Enter to continue')
        if response.lower() in ['exit', 'ex', 'q', 'quit', "x"]:
            print("Exiting...")
            break
        

        cg_d2_map = {'stokes': cg_d2_stokes,'pump': cg_d2_pump, 'P1': cg_d2_p1, 'P2': cg_d2_p1}
        rabi_d2_map = {'stokes': rabi_stirap_d2,'pump': rabi_stirap_d2, 'P1': rabi_p1_d2, 'P2': rabi_p2_d2}

        target_power_d2 = rabi_to_laserpower(rabi_d2_map[pulse], d_d2, cg_d2_map[pulse] , typical_waist_size) # in mW
        target_power_d2 *= 10**(-3) # to W
        print(f'Target Power for desired Rabi Freq: {target_power_d2*1e3} mW')



        awg_channels_dict = {1:Channel.CHANNEL_1, 2:Channel.CHANNEL_2, 3:Channel.CHANNEL_3, 4:Channel.CHANNEL_4}
        amplitude_cal, diff, power, results_dict = calibrate.finding_amplitude_from_power(awg_chan_freqs_map[channel], target_power_d2, awg_channels_dict[channel], n_steps = 75, repeats=3, delay=0.3,\
                                    calibration_lims = (0.1,0.25), save_all=True, results_dict=results_dict)
        
        df = pd.DataFrame({'amplitude_cal': results_dict['level'],'power': results_dict['read_value']})
        df['rabi_measured_no_ang'] = df['power'].apply(lambda p: laserpower_to_rabi(p * 1e3,d_d2,
            cg_d2_map[pulse],typical_waist_size))/np.abs(cg_d2_map[pulse])
        df['target power']=target_power_d2
        df['rabi_des_no_cg'] = np.abs(rabi_d2_map[pulse]/cg_d2_map[pulse])
        df['closest level']=amplitude_cal
        df['waist_size'] = typical_waist_size
        df['cg_ang']=cg_d2_map[pulse]

        #join config_save_path with a new folder with today's date
        today = datetime.datetime.now().strftime("%d-%m")

        config_path_date = os.path.join(config_save_path, today)
        full_folder_path = os.path.join(config_path_date, f"{awg_chan_freqs_map[channel][0]}MHz")
        if not os.path.exists(full_folder_path):
            os.makedirs(full_folder_path)

        

        output_file = os.path.join(full_folder_path, f'rabi_data_{pulse}.csv')
        df.to_csv(output_file, index=False)

        print("Instantiating RabiFreqVoltageConverter...")
        converter = RabiFreqVoltageConverter(output_file)
        


    # #file_path = r'C:\Users\apc\Documents\Python Scripts\017-data-analysis\flatg_0.2_ch4_50us.csv'
    # file_path = r'c:\Users\apc\Documents\marina\06_jun\05-06\x_optimized_21_0.6.csv'
    # #file_path = r'c:\Users\apc\Documents\marina\06_jun\02-06\0.2\pump\x_optimized_27_0.6.csv'
    # opt_input = pd.read_csv(file_path, header=None)
    # opt_input = opt_input.T.to_numpy().flatten()
    # opt_input = opt_input/opt_input.max()*amplitude_cal

    # output_dir = f'c:\\Users\\apc\\Documents\\marina\\06_jun\\{today}\\opt_from_{amplitude}_to_{amplitude_cal}'
    # os.makedirs(output_dir, exist_ok=True)
    # waveform_filename = os.path.join(output_dir, f'{pulse}_optimized.csv')
    # with open(waveform_filename, 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(opt_input)

