'''
Script to calibrate the power. Outputs the required input amplitude to achieve target power. 
Additionally saves the Rabi frequency data corresponding to each amplitude value.

@author: marina llano, Jan Ole Ernst, Matt King
'''


import os
import numpy as np
from scipy.constants import c, epsilon_0, hbar
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from instruments.WX218x.WX218x_awg import Channel
import lab_control_functions.calibration_functions as calibrate
from rabi_voltage_converter import RabiFreqVoltageConverter


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

calib_tuples = [
    (1, "pump", 126),
    #(1, "pump", 116),
    (2, "stokes", 80),
    #(2, "stokes", 70)
]

pulse = 'pump'  # 'stokes', 'pump', 'P1', 'P2'
channel = 1  # AWG channel
amplitude = 0.2
amplitude_cal = 0.00
diff = 1
results_dict = {}
# Finding the voltage amplitude that corresponds to this power
#awg_chan_freqs_map = {1: [126], 2: [80], 3: [62.35], 4: [82.5]}


if __name__ == "__main__":

    for channel, pulse, freq in calib_tuples:

        print(f"\n\nCalibrating {pulse} pulse on channel {channel} at {freq} MHz\n\n")

        if channel==1:
            config_save_path="calibrations\pulse_shaping_expt\STIRAP_ELYSA"
        elif channel==2:
            config_save_path="calibrations\pulse_shaping_expt\STIRAP_DL_PRO"
        elif channel==3:
            config_save_path="calibrations\pulse_shaping_expt\OPT_PUMP_ELYSA"
        elif channel==4:
            config_save_path="calibrations\pulse_shaping_expt\OPT_PUMP_DL_PRO"
        else:
            raise ValueError("Channel must be 1, 2, 3 or 4")
        

        cg_d2_map = {'stokes': cg_d2_stokes,'pump': cg_d2_pump, 'P1': cg_d2_p1, 'P2': cg_d2_p1}
        rabi_d2_map = {'stokes': rabi_stirap_d2,'pump': rabi_stirap_d2, 'P1': rabi_p1_d2, 'P2': rabi_p2_d2}

        target_power_d2 = rabi_to_laserpower(rabi_d2_map[pulse], d_d2, cg_d2_map[pulse] , typical_waist_size) # in mW
        target_power_d2 *= 10**(-3) # to W
        print(f'Target Power for desired Rabi Freq: {target_power_d2*1e3} mW')



        awg_channels_dict = {1:Channel.CHANNEL_1, 2:Channel.CHANNEL_2, 3:Channel.CHANNEL_3, 4:Channel.CHANNEL_4}
        amplitude_cal, diff, power, results_dict = calibrate.finding_amplitude_from_power([freq], target_power_d2, awg_channels_dict[channel], n_steps = 75, repeats=3, delay=0.3,\
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
        #today = datetime.datetime.now().strftime("%d-%m")

        #config_path_date = os.path.join(config_save_path, today)
        full_folder_path = os.path.join(config_save_path, f"{freq}MHz")
        if not os.path.exists(full_folder_path):
            os.makedirs(full_folder_path)

        

        output_file = os.path.join(full_folder_path, f'rabi_data.csv')
        df.to_csv(output_file, index=False)

        print("Instantiating RabiFreqVoltageConverter...")
        converter = RabiFreqVoltageConverter(output_file)
        converter.get_rabi_limits(print_info=True)
        


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

