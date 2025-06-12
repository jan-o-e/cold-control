from calibrate_power import RabiFreqVoltageConverter
import numpy as np
import os

rabi_start=44
rabi_finish=100

rabi_sweep=np.linspace(rabi_start, rabi_finish, 5)

save_dir=r"C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\waveforms\pulse_shaping_exp\rabi_sweep\30mhz_70mhz_10steps"
stokes_pulse_loc=r"C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\waveforms\pulse_shaping_exp\normalised_pulses\optimised\stokes_optimized_avui.csv"
pump_pulse_loc=r"C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\waveforms\pulse_shaping_exp\normalised_pulses\optimised\pump_optimized_avui.csv"

RabiClassStokes=RabiFreqVoltageConverter(r'C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\calibrations\jan\STIRAP_DL_PRO\12-06\rabi_data_stokes.csv')
RabiClassPump=RabiFreqVoltageConverter(r'C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\calibrations\jan\STIRAP_ELYSA\12-06\rabi_data_pump.csv')

for rabi_freq in rabi_sweep:
    rounded_rabi = np.round(rabi_freq, 1)
    full_save_dir = os.path.join(save_dir, f"{rounded_rabi}")
    os.makedirs(full_save_dir, exist_ok=True)
    stokes_save_path = os.path.join(full_save_dir, f'stokes_optimised.csv')
    pump_save_path = os.path.join(full_save_dir, f'pump_optimised.csv')

    RabiClassStokes.rescale_csv(rabi_freq, stokes_pulse_loc, stokes_save_path, )
    RabiClassPump.rescale_csv(rabi_freq, pump_pulse_loc, pump_save_path)
