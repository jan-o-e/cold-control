from calibrate_power import RabiFreqVoltageConverter
import numpy as np
import os

rabi_start=50*2*np.pi
rabi_finish=50*2*np.pi
steps = 1

rabi_sweep=np.linspace(rabi_start, rabi_finish, steps)

#save_dir=rf"C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\waveforms\pulse_shaping_exp\rabi_sweep\{rabi_start/(2*np.pi):.1f}mhz_{rabi_finish/(2*np.pi):.1f}mhz_{steps}steps"
save_dir = r"C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\waveforms\pulse_shaping_exp\rabi_sweep\frequency_sweep\55_opt"
stokes_pulse_loc=r"C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\waveforms\pulse_shaping_exp\normalised_pulses\optimised\stokes_optimized_avui.csv"
pump_pulse_loc=r"C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\waveforms\pulse_shaping_exp\normalised_pulses\optimised\pump_optimized_avui.csv"

RabiClassStokes=RabiFreqVoltageConverter(r'C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\calibrations\jan\STIRAP_DL_PRO\13-06\rabi_data_stokes.csv')
RabiClassPump=RabiFreqVoltageConverter(r'C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\calibrations\jan\STIRAP_ELYSA\13-06\rabi_data_pump.csv')
stokes_lims = RabiClassStokes.get_rabi_limits(print_info=False)
pump_lims = RabiClassPump.get_rabi_limits(print_info=False)

print(f"The range of Rabi frequencies for the Stokes beam: {stokes_lims}")
print(f"The range of Rabi frequencies for the pump beam: {pump_lims}")

# for rabi_freq in rabi_sweep:
#     rounded_rabi = np.round((rabi_freq)/(2*np.pi), 1)
#     full_save_dir = os.path.join(save_dir, f"{rounded_rabi}")
#     os.makedirs(full_save_dir, exist_ok=True)
#     stokes_save_path = os.path.join(full_save_dir, f'stokes.csv')
#     pump_save_path = os.path.join(full_save_dir, f'pump.csv')

#     RabiClassStokes.rescale_csv(rabi_freq, stokes_pulse_loc, stokes_save_path, )
#     RabiClassPump.rescale_csv(rabi_freq, pump_pulse_loc, pump_save_path)
