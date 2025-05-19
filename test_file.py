# import pyvisa as visa
import re
import numpy as np
import os
from typing import List, Dict

from Config import ConfigReader, DaqReader
from DAQ import DAQ_controller, DAQ_channel


config_reader = ConfigReader(os.getcwd() + '/configs/rootConfig.ini')
daq_config_fname = config_reader.get_daq_config_fname()# gets the name of the config file for the DAQ cards
daq_reader = DaqReader(daq_config_fname)

channels:List[DAQ_channel] = []
for _,v in daq_reader.config['DAQ channels'].items():
    #This line uses the map() function to apply a series of type conversions to the configuration data.
    channelArgs = map(lambda x,y:x(y), [int, str, lambda x: (float(x[0]), float(x[1])), float, eval, str],
                            [v['chNum'],v['chName'],v['chLimits'],v['default value'],v['UIvisible'],v['calibrationFname']])
    channels.append(DAQ_channel(*channelArgs))

#print(channels)

def main_loop():
    print("\n\n**Starting program to convert between voltages and calibration values**\n")
    print("enter the number of the channel to find a value for:")
    ch_num = input("")
    
    try:
        ch_num = int(ch_num)
    except ValueError:
        print("Invalid channel number")
        if ch_num in ["x", "e", "q", "quit", "exit"]:
            return
        main_loop()
        return
    
    channel_found = False
    for channel in channels:
        #print(channel.chNum)
        if channel.chNum == ch_num:
            calib_to_V = channel.calibrationToVFunc
            calib_from_V = channel.calibrationFromVFunc
            channel_found = True
            print(f"This channel is {channel.chName}")
    
    if not channel_found:
        print("channel not found")
        main_loop()
        return
    
    print("Do you want to find a voltage (v) or a calibration value (c)?")
    conv_type = input("").lower()
    if conv_type in ["v", "volts", "voltage"]:
        conv_type = 0
    elif conv_type in ["c", "calib", "calibration"]:
        conv_type = 1
    else:
        print("invalid conversion type")
        main_loop()
        return

    print("What value do you want to convert?")
    value = input("")
    try:
        value = float(value)
    except ValueError:
        print("Invalid value to convert")
        main_loop()
        return
    
    print("result:")
    if conv_type:
        print(calib_from_V(value))
    else:
        print(calib_to_V(value))
    
    main_loop()
    return


main_loop()


# cool_upper_freq =r"C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\calibrations\jan\cool_upper_freq.txt" 
# cool_lower_freq =r"C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\calibrations\jan\cool_lower_freq.txt"
# cool_centre_freq = r"C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\calibrations\jan\cool_centre_freq.txt"
# filename = cool_lower_freq
# r"C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\calibrations\jan\cool_lower_freq.txt"
# value = 102


# def calibrate_from_txt(calibrationFname, reReadIn = r'([\+|\-]?[\d|\.]+)[ \t]*([\+|\-]?[\d|\.]+)'):


#     #print("WARNING: calibrate_from_txt() METHOD IS DEPRECATED. USE CALIBRATE WITH CSV FILES INSTEAD.")

#     vData, calData = [], []
#     with open(calibrationFname) as f:
#         calibrationUnits = re.split(r'[ \t]*', f.readline())[-1].strip()
#         for line in f.readlines():
#             match = re.match(reReadIn, line.strip())
#             if match:
#                 vData.append(float(match.group(1)))
#                 calData.append(float(match.group(2)))
                
#     if calData[0] <= calData[-1]:
#         calibrationToVFunc = lambda x: np.interp(x, calData, vData)
#     else:
#         calibrationToVFunc = lambda x: np.interp(x, [x for x in reversed(calData)], [x for x in reversed(vData)])
        
#     if vData[0] <= vData[-1]:    
#         calibrationFromVFunc = lambda x: np.interp(x, vData, calData)
#     else:
#         calibrationFromVFunc = lambda x: np.interp(x, [x for x in reversed(vData)], [x for x in reversed(calData)])

#     return calibrationToVFunc, calibrationFromVFunc

# calib_to_V, calib_from_V = calibrate_from_txt(filename)
