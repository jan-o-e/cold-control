from calibrate_power import RabiFreqVoltageConverter
import numpy as np
import os

rabi_start=55*2*np.pi
rabi_finish=55*2*np.pi
steps = 1

rabi_sweep=np.linspace(rabi_start, rabi_finish, steps)

#save_dir=rf"C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\waveforms\pulse_shaping_exp\rabi_sweep\{rabi_start/(2*np.pi):.1f}mhz_{rabi_finish/(2*np.pi):.1f}mhz_{steps}steps"
save_dir = r"C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\waveforms\pulse_shaping_exp\rabi_sweep\pulse_sweep\55_non_opt" \
""
stokes_pulse_loc=r"C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\waveforms\pulse_shaping_exp\normalised_pulses\non_optimised\stokes_175ns_0.2.csv"
pump_pulse_loc=r"C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\waveforms\pulse_shaping_exp\normalised_pulses\non_optimised\pump_175ns_20.csv"

RabiClassStokes=RabiFreqVoltageConverter(r'C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\calibrations\jan\STIRAP_DL_PRO\13-06\rabi_data_stokes.csv')
RabiClassPump=RabiFreqVoltageConverter(r'C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\calibrations\jan\STIRAP_ELYSA\13-06\rabi_data_pump.csv')

for rabi_freq in rabi_sweep:
    rounded_rabi = np.round((rabi_freq)/(2*np.pi), 1)
    full_save_dir = os.path.join(save_dir, f"{rounded_rabi}")
    os.makedirs(full_save_dir, exist_ok=True)
    stokes_save_path = os.path.join(full_save_dir, f'stokes_optimised.csv')
    pump_save_path = os.path.join(full_save_dir, f'pump_optimised.csv')

    RabiClassStokes.rescale_csv(rabi_freq, stokes_pulse_loc, stokes_save_path, )
    RabiClassPump.rescale_csv(rabi_freq, pump_pulse_loc, pump_save_path)
