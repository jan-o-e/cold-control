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


def get_power_meter():
    '''Finds a Thor Labs PM100A power_meter if one is connected and returns a ThorlabsPM100 instance
    for it.  If no power meter is found the function returns None'''
    rm = visa.ResourceManager()
    power_meter = None
    for resource in rm.list_resources():
        try:
            inst = rm.get_instrument(resource)
            print (inst.query("*IDN?").split(','))
            if inst.query("*IDN?").split(',')[1] == 'PM100A':
                power_meter = ThorlabsPM100(inst)
                break # --> Thorlabs,PM100A,P1002563,2.3.0
        except:
            pass

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
    




def create_file(fname, levelData, parsedData, units):
    """
    Saves a .txt file containing the voltage levels and the calibration data
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
    


def save_plot(fname, vData, calData, units, title):
    """
    Saves a plot of voltage against calibration data.
    """
    fig = plt.figure()
    
    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)
    ax.set_title(title)
    
    ax.set_xlabel('V')
    ax.set_ylabel(units)
    
    ax.plot(vData, calData)
    
    plt.savefig(fname)
    print ('saved img: ', fname)
