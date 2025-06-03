import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
import csv
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
import os

# def gaussian_rise_fall(t, t_rise, t_flat, amplitude):
#     sigma = t_rise / (2 * np.sqrt(2 * np.log(10))) 
#     t_fall_start = t_rise + t_flat
#     rise = amplitude * np.exp(-(t - t_rise)**2 / (2 * sigma**2))
#     flat = amplitude * (t >= t_rise) * (t < t_fall_start)
#     fall = amplitude * np.exp(-(t - t_fall_start)**2 / (2 * sigma**2))
#     signal = rise * (t < t_rise) + flat + fall * (t >= t_fall_start)
    
#     return signal


# rise_time = 41.02e-9  # (41.02e-9=16.12 ns entre 10 i 90%, 60.61e-9=23.82 ns entre 10 i 90%)
# flat_time = 20e-9     
# amplitude = 0.2       
# t_total = 100e-9      
# dt = 1e-9           
# t = np.arange(0, t_total, dt)

# signal = gaussian_rise_fall(t, rise_time, flat_time, amplitude)

# time_0_02 = t[np.where(signal >= 0.02)[0][0]]
# time_0_18 = t[np.where(signal >= 0.18)[0][0]]
# time_difference = time_0_18 - time_0_02
# print(f"Rise time: {time_difference}")

# with open('flat_20_gaussian_ch4.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(signal) 

# plt.plot(t, signal)
# plt.show()

# file_path =r'c:\Users\apc\Documents\marina\pump_150ns_20.csv'
# x = pd.read_csv(file_path, header=None)
# x = x.T.to_numpy().flatten()
# t_original = np.linspace(0, 150, 150)

# t_new = np.linspace(0, 75, 75)
# new = np.interp(t_new, t_original * (75 / 150), x)


# continuous sine
# f = 75e6
# fs = 1e9
# N = round(600e-9 * f)

# while True:
#     T = N / f
#     t_new = np.arange(0, T, 1/fs)
#     new = np.sin(2 * np.pi * f * t_new)
    
#     if np.isclose(new[-1], 0, atol=1e-12):
#         break    
#     N += 0.001


# cnt amplitude
# fs = 1e9
# T = 500e-9
# samples = int(fs * T)

# amplitude = np.full(samples, 0.1)

# with open('flat_ch4.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(amplitude)

# plt.plot(amplitude)
# plt.grid()
# plt.show()



########################################################################
# Edit previous .csv files (amplitudes)

# file_path = r'c:\Users\apc\Documents\marina\pump_175ns_30.csv'
# x = pd.read_csv(file_path, header=None).T.to_numpy().flatten()

# amplitude_cal =  0.2
# x= (x/ x.max()) * amplitude_cal

# plt.plot(x)
# plt.show()

# with open(f'pump_175ns_20.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(x)


########################################################################
# # Create zeros files

# x= [0]*1001000
# with open('zero_1.001ms.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(x)


########################################################################
# # Create new top hats of diff lengths

length = 50000
amplitude = 0.2
x = [amplitude] * length
start = [1e-05,1.3499999999999998e-05,1.82e-05,2.435e-05,3.24e-05,4.295e-05,5.6591499999999995e-05,7.4231e-05,9.68825e-05,0.000125813,0.0001625655,0.0002090035,0.000267363,0.0003403075,0.00043098649999999996,0.000543098,0.0006809495,0.000849521,0.0010545215,0.001302444,0.001600608,0.001957191,0.0023812435,0.002882681,0.003472255,0.0041614915,0.0049625925,0.0058883085,0.00695176,0.008166223,0.00954487,0.0111004625,0.012845009,0.014789381999999998,0.0169429075,0.019312928,0.0219043615,0.0247192535,0.027756352,0.031010709999999997,0.034473336,0.0381309125,0.0419655935,0.0459548995,0.0500717255,0.054284467,0.0585572775,0.0628504605,0.0671209865,0.071323145,0.0754093015,0.079330764,0.083038718,0.0864852255,0.0896242405,0.0924126315,0.0948111625,0.0967854085,0.09830658,0.0993522185,0.099906751]
end = [0.099961873,0.0995167575,0.0985780745,0.0971598265,0.0952829975,0.0929750405,0.090269206,0.0872037485,0.083821025,0.080166526,0.076287862,0.072233741,0.0680529655,0.0637934785,0.0595014835,0.05522065949999999,0.050991486,0.0468506945,0.0428308485,0.038960059,0.0352618275,0.031755014,0.028453918499999998,0.0253684625,0.0225044545,0.0198639275,0.0174455245,0.0152449225,0.013255273499999998,0.0114676505,0.009871485,0.0084549845,0.007205521499999999,0.0061099875,0.005155105,0.004327698,0.00361492,0.0030044345,0.0024845575,0.002044361,0.001673742,0.0013634575,0.0011051395,0.0008912815,0.000715212,0.000571054,0.00045367150000000006,0.000358615,0.0002820575,0.000220734,0.000171879,0.0001331675,0.00010265900000000001,7.8744e-05,6.0098e-05,4.565e-05,3.45e-05,2.5949999999999997e-05,1.94e-05]

final = start + x + end

with open(f'flatg_{amplitude}_ch4_50us.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(final) 

#########################################################
# # Convert feather files to csv

# folder = 'data/2025-06-02/freq 107 i 107'
# for filename in os.listdir(folder):
#     path_in = os.path.join(folder, filename)
#     path_out = os.path.join(folder, filename + '.csv')  # añade .csv al nombre original

#     try:
#         df = pd.read_feather(path_in)
#         df.to_csv(path_out, index=False)
#         print(f'Convertido: {filename}')
#     except Exception as e:
#         print(f'No se pudo convertir {filename}: {e}')

