'''
Helper functions for the calibration functions.

@author: Matt King
'''

import os
import serial
import time
import numpy as np
from Config import ConfigReader, DaqReader
import matplotlib.pyplot as plt
import re
from instruments.TF930 import TF930
from instruments.ThorlabsPM100 import ThorlabsPM100
import pyvisa as visa
import pandas as pd
'''
Load required classes for awg driven AOM calibration
'''
from instruments.WX218x.WX218x_awg import WX218x_awg, Channel
from instruments.WX218x.WX218x_DLL import WX218x_OperationMode, WX218x_Waveform, WX218x_OutputMode
from ExperiementalRunner import Waveform
    
class CalibrationException(Exception):
    
    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        super(CalibrationException, self).__init__(message)

  

class testWaveform(Waveform):
    '''
    Subclass of waveform to enable us to easily generate rf signals of different amplitudes.
    Level is in (0,1) - it corresponds to the (-1,1) range that waveforms loaded into the
    awg are defined in.  Note that a negative level only flips the phase of the rf signal,
    hence why we only need to calibrate for positive levels.

    Inputs:
    samp_rate (float): sample rate of the awg
    level (float): see above
    mod_freq (float): inherited from Waveform class
    '''
    def __init__(self, samp_rate, level=1, mod_freq=75.25*10**6):
        self.mod_frequency = mod_freq
        self.phases = []
        self.t_step = 2*np.pi/samp_rate
        self.calib(level, self.mod_frequency)
        
    def calib(self, level, mod_freq):
        self.level = level
        self.data = [self.level]*800
        self.mod_frequency = mod_freq



def getCalibName(aom_name, freq):
        return '{0}_amp_at_{1}MHz'.format(aom_name, freq)


def default_v_step():
    '''
    Returns the smallest resolvable voltage step a 4096 bit digital output corresponding to -10 to 10V can make.
    i.e. the voltage resolution of the DAQ-2502 cards.
    '''
    f = lambda x: np.interp(x, (0,4095), (-10,10))
    return f(1) - f(0)


def get_power_meter(return_inst = False, debug_mode = False):
    '''
    Finds a Thor Labs PM100A power_meter if one is connected and returns a ThorlabsPM100 instance
    for it.  If no power meter is found the function raises an exception
    Inputs:
     - return_inst (bool): If true then the visa resource will be returned along with the power_meter. *This allows the instrument to be closed*
     - debug_mode (bool): If True then all available resources will be listed
    '''

    rm = visa.ResourceManager()
    all_res = rm.list_resources()
    power_meter = None
    # the VISA addresses of the 3 thorlabs powermeters we have are in the list below:
    pm_addresses = ["USB0::0x1313::0x8079::P1002563::0::INSTR","USB0::0x1313::0x8079::P1000416::0::INSTR",\
                     "USB0::0x1313::0x8079::P1002564::0::INSTR", "USB0::0x1313::0x8079::P1002347::0::INSTR"]
    # THIS WILL NEED TO BE CHANGED IF A DIFFERENT POWERMETER IS USED!

    if debug_mode:
        print(all_res)
        pm_addresses = all_res


    for resource in pm_addresses:
        try:
            inst = rm.open_resource(resource)
            #inst = rm.get_instrument(resource)
            print(inst.query("*IDN?").split(','))
            if inst.query("*IDN?").split(',')[1] == 'PM100A':
                power_meter = ThorlabsPM100(inst)
                break # --> Thorlabs,PM100A,P1002563,2.3.0
        except visa.errors.VisaIOError as e:
            print(f"powermeter with address {resource} is not available.")
            if debug_mode:
                print(f"error message: {e}")

    
    if power_meter == None:
        print('Calibration failed - power meter could not be found')
        raise CalibrationException('Calibration failed - power meter could not be found')
    if return_inst:
        return inst, power_meter
    else:
        return power_meter


def configure_power_meter(power_meter:ThorlabsPM100, nMeasurmentCounts = 1):
    '''Configures the power meter, see https://pypi.python.org/pypi/ThorlabsPM100 for full details for a full list of commands.
    Most options are hard coded as I don't see any need to change them right now.
    
        power_meter = A ThorlabsPM100 instance to configure.
        nMeasurementCounts - How many measurements to take and average over when reading a value from the power meter
                             (note 1 measurement is about 3ms).
    
    '''
    power_meter.sense.correction.wavelength = 780
    power_meter.sense.power.dc.range.auto = 'ON'
    power_meter.sense.power.dc.unit = 'W'
    power_meter.sense.average.count = nMeasurmentCounts
    




def create_file_txt(fname, levelData, parsedData, units):
    """
    Saves a .txt file containing the voltage levels and the calibration data. DEPRECATED.
    """
    fname = '{0}.txt'.format(fname)
    
    f = open(fname, 'a')
    print('created: ', fname)
    lineArgs =  [('V', units)]
    lineArgs += zip(levelData, parsedData)
    for args in lineArgs:
        f.write('{0}\t{1}\n'.format(*args))
    f.close()
    print('written: ', fname)


def create_file(fname, levelData, parsedData, calib_units, level_units = "Voltage (V)"):
    """
    Saves data to a CSV file using pandas.

    Args:
        fname (str): The base filename for the output file.
        levelData (list): A list of voltage levels.
        parsedData (list): A list of corresponding calibration data.
        calib_units (str): The units of the calibration levels (e.g., "W", "uW").
        level_units (str): The title to give the column containing the independent variable data from the calibration run
    Returns:
        None
    """
    if level_units == "Voltage (V)":
        df = pd.DataFrame({"Voltage (V)": levelData, 
                       f"Calibration Data ({calib_units})": parsedData})
    else:
        df = pd.DataFrame({f"{level_units}": levelData, 
                       f"Calibration Data ({calib_units})": parsedData})
        
    # Get the directory path from the filename
    directory = os.path.dirname(fname) 

    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)  # Create directory if it doesn't exist

    df.to_csv(f"{fname}.csv", index=False) 
    print(f"created: {fname}.csv") 

    


def save_plot(fname, vData, calData, units, title, level_units = "Voltage (V)"):
    """
    Saves a plot of voltage against calibration data.
    """
    fig = plt.figure()
    
    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)
    ax.set_title(title)
    
    ax.set_xlabel(f'{level_units}')
    ax.set_ylabel(f"Calibration data ({units})")
    #ax.set_ylabel("test")
    
    ax.plot(vData, calData)

    # Get the directory path from the filename
    directory = os.path.dirname(fname) 

    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)  # Create directory if it doesn't exist
    
    plt.savefig(fname)
    print ('saved img: ', fname)
