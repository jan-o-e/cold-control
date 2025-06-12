'''
File containing functions to generate calibration data for driving AOMs with DAQ cards and the AWG.

Refactored 09/12/2024

@author: Matt King, Marina
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
from ExperimentalConfigs import Waveform

# helper functions in a separate file
from lab_control_functions.calibration_helper_functions import *






def daq_driven_aom_response(daq_controller:DaqReader, frequency_channel:int,\
                    amp_channel:int, frequency_voltage_pairs:dict, v_range, v_step = default_v_step(),\
                    delay=0.1, repeats = 3, save_folder="unfiled_data"):
    '''
    Creates a calibration file between the voltage given to an AOM and absolute power output.

    Inputs:
        daq_controller (DaqReader) - a DAQ_controller object to un the DAQ cards.
        frequency_channel (int) - DAQ channel number corresponding to AOM frequency
        amp_channel (int) - The channel number which is corresponds to the AOM amplitude
        frequency_voltage_pairs (dict) - dictionary with keys corresponding to AOM frequencies and values corresponding
                                         to the voltages required to drive the AOM at those frequencies
        v_range (tuple) - A voltage range to calibrate between of the form (V_min, V_max).
        v_step (float) - The voltage step over between taking calibration measurements. The default is calculated to be
                          equivilent to increasing the digital output on a 4096-bit channel with a -10 to 10V output range.
        delay (float) - How long to wait between writing a new voltage and querying the frequency counter.
        repeats (int)  - How many measurements to take and average over when reading a value from the power meter
                          (note 1 measurement is about 3ms).
    '''

    file_path = os.path.join(os.getcwd(), 'calibrations', save_folder)

    # Find and configure a power meter connected to the computer
    inst, power_meter = get_power_meter(return_inst=True)
    power_meter:ThorlabsPM100 = power_meter #declare the type for easier editing
    configure_power_meter(power_meter, nMeasurmentCounts=repeats)

    # loop through the different frequencies (and their associated voltages)
    for freq, v in frequency_voltage_pairs.items():
        calib_name = f"amp_at_{freq}MHz"
        daq_controller.updateChannelValue(frequency_channel, v)
        time.sleep(3)

        # apply a range of voltages to the daq card and measure the output amplitudes
        voltage_data = np.arange(v_range[0], v_range[1]+v_step, v_step)
        amp_data = np.empty(len(voltage_data))
        print ('Running through voltages...might take a while...')
        for i in range(len(voltage_data)):
            print(voltage_data[i])
            daq_controller.updateChannelValue(amp_channel, voltage_data[i])
            time.sleep(delay)
            amp_data[i] = float(power_meter.read)
        print ('...finished!')

        units = str(power_meter.sense.power.dc.unit.split('\n')[0])
        #print(type(units), repr(units))
        # Just a hack to convert W to uW as it's nicer.    
        if units == 'W':
            amp_data = amp_data * 10**6
            units = 'uW'


        # save the data and the plot
        create_file(os.path.join(file_path, calib_name), voltage_data, amp_data, units)
        save_plot(os.path.join(file_path, f"{calib_name}_plot.png"), voltage_data, amp_data, units, f"freq = {freq}MHz")

    inst.close()
        




    
def awg_driven_aom_response(freqs, name, awg_channel, n_steps = 20, repeats=3, delay=0.2,\
                            calibration_lims = (0,1), save_folder = "unfiled_data"):
    
    """Creates a calibration file detailing the dependence of the power through the aom depending on the
    voltage level of the awg waveform.
    
    Inputs:
        freqs (list) - frequencies at which the awg should drive the aom
        name (str) - name of the laser producing the beam
        awg_channel (Channel) - awg channel that drives the aom
        n_steps (int) - 
        repeats (int) - How many measurements to take and average over when reading a value from the power meter
                          (note 1 measurement is about 3ms).
        delay (float) - How long to wait between writing a new voltage and querying the frequency counter
        calibration_lims (tuple) - limits of the calibration
        save_folder (str) - name of the folder to save the calibration results in (under /calibrations)
    """


    # get complete path to folder to save data in
    save_location = os.path.join(os.getcwd(), 'calibrations',save_folder)


    # Open and configure the AWG
    sample_rate = 1.25*10**9
    print ('Creating AWG instance')
    awg = WX218x_awg()
    print ('Connecting...')
    awg.open(reset=False)
    
    awg.configure_operation_mode(awg_channel, WX218x_OperationMode.CONTINUOUS)
    awg.configure_output_mode(WX218x_OutputMode.ARBITRARY)
    awg.configure_sample_rate(sample_rate)
    awg.configure_arb_gain(awg_channel, 2)

    inst, power_meter = get_power_meter(return_inst=True)
    power_meter:ThorlabsPM100 = power_meter #declare the type for easier editing
    configure_power_meter(power_meter, nMeasurmentCounts=repeats)
    
    for freq in freqs:
        # Run through the voltages and record the TF930 output
        levelData = np.linspace(*calibration_lims, n_steps)
        calData = np.empty(n_steps)
        print ('Running through awg levels...might take a while...')

        for i, level in enumerate(levelData):
            print ('Level:', level)

            wf = testWaveform(sample_rate, level=level, mod_freq=freq*10**6)
            awg.create_custom_adv(wf.get(sample_rate), wf.get(sample_rate))
             
            awg.enable_channel(awg_channel)
            time.sleep(delay)
            calData[i] = (power_meter.read)
            
            print(calData[i])
            
            awg.disable_channel(awg_channel)
            
        print ('...finished taking data')
        
        save_plot_location = os.path.join(save_location, 'plots')
        if not os.path.isdir(save_plot_location):
            os.makedirs(save_plot_location)

        #save microWatts
        calData = calData*10**6 #[x*10**6 for x in calData]
        save_plot(os.path.join(save_plot_location,f"{freq}MHz_abs_power.png"), levelData, calData, "uW", f"{freq}MHz: Power vs level")
        create_file(os.path.join(save_location,f"{name}_{awg_channel}_{freq}MHz_abs"), levelData, calData, 'uW', level_units='level')

        
        
        end_on_max = False
        while not end_on_max:
#         indexMin, indexMax = calData.index(min(calData)), calData.index(max(calData))
            indexMin = min(range(len(calData)), key=lambda i: abs(calData[i]- (min(calData) + calibration_lims[0]*max(calData)) ))
            indexMax = min(range(len(calData)), key=lambda i: abs(calData[i]- (max(calData)*calibration_lims[1])) )
         
            levelData, calData = levelData[indexMin:indexMax+1], calData[indexMin:indexMax+1]
            
            if np.argmax(calData) != len(calData)-1:
                calibration_lims = (calibration_lims[0], calibration_lims[1]-0.1)
            else:
                end_on_max = True
        
        print(f'Calibration limits set to {calibration_lims} to avoid a maximum in the middle of the calibration range.')
     
        def normalise(values):
            mi, ma = np.min(values),np.max(values)
            return (values - mi)/(ma - mi)
         
        calData = 100 * normalise(calData)

        save_plot(os.path.join(save_location,f"{name}_{awg_channel}_{freq}MHz_rel_power_plot.png"), levelData, calData, "%", f"{freq}MHz: Rel Power vs level")
        create_file(os.path.join(save_location,f"{name}_{awg_channel}_{freq}MHz_rel"), levelData, calData, '%', level_units='level')
    
    print ('Resetting awg...',)
    awg.reset()
    print ('calibration finished.')
    awg.close()

    inst.close()
 







def calibrate_frequency(daq_controller, chNum_to_calibrate, calibration_V_range = (0,10),
                        calibration_V_step = default_v_step(),
                        writeToQueryDelay=0.1, queryToReadDelay=0.3):
    '''
    NOT YET FIXED

    Creates a calibration file between the voltage given to an AOM and the frequency output.
        daq_controller - a DAQ_controller object to un the DAQ cards.
        chNum_to_calibrate - The channel number which is attached to the AOM input
        calibration_V_range - A voltage range to calibrate between of the form (V_min, V_max).
        calibration_V_step - The voltage step over between taking calibration measurements. The default is calculated to be
                             equivalent to increasing the digital output on a 4096-bit channel with a -10 to 10V output range.
        writeToQueryDelay - How long to wait between writing a new voltage and querying the frequency counter
        queryToReadDelay - How long to wait between querying the frequency counter and reading the output
                           NOTE: the shortest measurement time on the TF930 is 0.3s
    '''

    try:
        counter = TF930.TF930(port='COM5')
    except serial.serialutil.SerialException as err:
        print ('Calibration failed - frequency counter could not be found')
        raise err

    # Run through the voltages and record the TF930 output
    vData, calData = [], []
    print ('Running through voltages...might take a while...')
    for v in np.arange(calibration_V_range[0], calibration_V_range[1]+calibration_V_step, calibration_V_step):
        daq_controller.updateChannelValue(chNum_to_calibrate, v)
        time.sleep(writeToQueryDelay)
        vData.append(v)
        calData.append(counter.query('N?', delay=queryToReadDelay))
    print ('...finished!')
    # Parse the output, once for units and once for values
    r = r'([\d|\.|e|\+]+)([a-zA-Z]*)\r\n'

    units = ''
    while units == '':
        for i in range(0, len(calData)):
            match = re.match(r, calData[i])
            if match:
                units = match.group(2)
                break

    parsedData = []
    nBadPoints = 0
    for i in range(0, len(calData)):
        match = re.match(r, calData[i])
        if match:
            parsedData.append(match.group(1))
        else:
            # If there was unexpected output (e.g. when the delays before reading are wrong)
            # then remove the corresponding data point from vData
            nBadPoints += 1
            vData.pop(i - nBadPoints)

    print ('Removed {0} bad data points'.format(nBadPoints))

    # Just a hack to convert Hz to MHz as it's nicer.
    if units == 'Hz':
        parsedData = map(lambda x: float(x)/10**6, parsedData)
        units = 'MHz'

    return vData, parsedData, units





def frequency_timeseries_mx(t_max,
                        writeToQueryDelay=0.1, queryToReadDelay=0.3):
    '''
    NOT YET FIXED

    Creates a calibration file between the voltage given to an AOM and the frequency output.

    Inputs:
        t_max - Maximal time (in s) for which to measure a frequency,
        writeToQueryDelay - How long to wait between writing a new voltage and querying the frequency counter
        queryToReadDelay - How long to wait between querying the frequency counter and reading the output
                           NOTE: the shortest measurement time on the TF930 is 0.3s
    '''

    try:
        counter = TF930.TF930(port='COM5')
    except serial.serialutil.SerialException as err:
        print ('Calibration failed - frequency counter could not be found')
        raise err

    # record the TF930 output
    t_data, calData = [], []
    print ('Running through the measurements...')
    for t_step in np.arange(0,t_max, writeToQueryDelay+queryToReadDelay):
        print(t_step)
        time.sleep(writeToQueryDelay)
        t_data.append(t_step)
        calData.append(counter.query('N?', delay=queryToReadDelay))
    print ('...finished!')
    # Parse the output, once for units and once for values
    r = r'([\d|\.|e|\+]+)([a-zA-Z]*)\r\n'

    units = ''
    while units == '':
        for i in range(0, len(calData)):
            match = re.match(r, calData[i])
            if match:
                units = match.group(2)
                break

    parsedData=calData
    #parsedData = []
    #nBadPoints = 0
    #for i in range(0, len(calData)):
    #    match = re.match(r, calData[i])
    #    if match:
    #        parsedData.append(match.group(1))
    #    else:
            # If there was unexpected output (e.g. when the delays before reading are wrong)
            # then remove the corresponding data point from vData
    #        nBadPoints += 1
    #        vData.pop(i - nBadPoints)

    #print ('Removed {0} bad data points'.format(nBadPoints))

    # Just a hack to convert Hz to MHz as it's nicer.
    if units == 'Hz':
        parsedData = map(lambda x: float(x) / 10 ** 6, parsedData)
        units = 'MHz'

    return t_data, parsedData, units




def percentage_power(daq_controller, chNum_to_calibrate, calibration_V_range = (0,7), calibration_perc_lims = (0,0.9),
                    calibration_V_step = default_v_step(), writeToQueryDelay=0.1,
                    nMeasurmentCounts = 3):
    '''
    NOT YET FIXED

    Creates a calibration file between the voltage given to an AOM and percentage of the maximum power output.
    Note that for this to work you will want to check that the maximum power output from the AOM is given at a
    voltage within the calibration_V_range!
    Inputs:
    daq_controller      - a DAQ_controller object to un the DAQ cards.
    chNum_to_calibrate  - The channel number which is attached to the AOM input
    calibration_V_range - A voltage range to calibrate between of the form (V_min, V_max).
    calibration_perc_lims - The percentage power range to allow calibration between.
                            i.e. (0,90) will only give the user access to 0 to 90% of the power.
                            This can be more stable if the calibation if very sensitive at the extreme ranges.
    calibration_V_step  - The voltage step over between taking calibration measurements. The default is calculated to be
                          equivilent to increasing the digital output on a 4096-bit channel with a -10 to 10V output range.
    writeToQueryDelay   - How long to wait between writing a new voltage and querying the frequency counter.
    nMeasurementCounts  - How many measurements to take and average over when reading a value from the power meter
                          (note 1 measurement is about 3ms).
    '''
    # Find and configure a power meter connected to the computer
    power_meter = get_power_meter()
    configure_power_meter(power_meter, nMeasurmentCounts=nMeasurmentCounts)
    
    # Run through the voltages and record the TF930 output
    vData, calData = [], []
    print ('Running through voltages...might take a while...')
    for v in np.arange(calibration_V_range[0], calibration_V_range[1]+calibration_V_step, calibration_V_step):
        print (v)
        daq_controller.updateChannelValue(chNum_to_calibrate, v)
        time.sleep(writeToQueryDelay)
        vData.append(v)
        calData.append(power_meter.read)
    print( '...finished!')

    absMinIndex, absMaxIndex = calData.index(min(calData)),  calData.index(max(calData))
    
    calData=calData[:absMaxIndex+1]
    
    indexMin = min(range(len(calData)), key=lambda i: abs(calData[i]- (min(calData) + calibration_perc_lims[0]*max(calData)) ))
    indexMax = min(range(len(calData)), key=lambda i: abs(calData[i]- (max(calData)*calibration_perc_lims[1])) )
    
    vData, calData = vData[indexMin:indexMax+1], calData[indexMin:indexMax+1]

    def normalise(values):
        mi, ma = min(values),max(values)
        ran = ma - mi
        return [(l-mi)/ran for l in values]
    
    calData = [100*x for x in normalise(calData)]

    units = '%'
    
    power_meter.close()
        
    return vData, calData, units
    



def test_stirap_aom_freq_response(level=0.5,
                                  freqs=range(60,90,1),
                                  nMeasurmentCounts=3,
                                  writeToQueryDelay=0.2):
    """
    NOT YET FIXED
    """


    # Open and configure the AWG
    sample_rate = 1.25*10**9

    print ('Creating AWG instance')
    awg = WX218x_awg()
    print ('Connecting...')
    awg.open(reset=False)
    
    awg.configure_operation_mode(Channel.CHANNEL_1, WX218x_OperationMode.CONTINUOUS)
    awg.configure_output_mode(WX218x_OutputMode.ARBITRARY)
    awg.configure_sample_rate(sample_rate)
    awg.configure_arb_gain(Channel.CHANNEL_1, 2)
    awg.configure_arb_gain(Channel.CHANNEL_2, 2)
    
    power_meter = get_power_meter()
    configure_power_meter(power_meter, nMeasurmentCounts=nMeasurmentCounts)
    
    calData = []
    
    for freq in freqs:
        # Run through the voltages and record the TF930 output
    
        print ('freq:', freq)
        
        wf = testWaveform(sample_rate, level=level, mod_freq=freq*10**6)
        awg.create_custom_adv(wf.get(sample_rate), wf.get(sample_rate))
        
        awg.enable_channel(Channel.CHANNEL_1)
        time.sleep(writeToQueryDelay)
        calData.append(power_meter.read)
        
        print (calData[-1])
        
        awg.disable_channel(Channel.CHANNEL_1)

    awg.reset()
    awg.close()

    power_meter.close()
   
    fig = plt.figure()
    
    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)
    
    ax.set_xlabel('freq')
    ax.set_ylabel('W')
    
    ax.plot(freqs, calData)
    


def finding_amplitude_from_power(freqs, target_power, awg_channel, n_steps=20, repeats=3, delay=0.2,
                                 calibration_lims=(0, 1), save_all=False, results_dict=None):
    """ Provides with the value of the voltage amplitude we have to send to the AWG if we want an output 
    of the aom with a specific power value.
    
    Inputs:
        freqs (list) - frequencies at which the awg should drive the aom
        target_power (float) - target power value to achieve
        awg_channel (Channel) - awg channel that drives the aom
        n_steps (int) - number of steps to divide the voltage range into
        repeats (int) - How many measurements to take and average over when reading a value from the power meter
                          (note 1 measurement is about 3ms).
        delay (float) - How long to wait between writing a new voltage and querying the frequency counter
        calibration_lims (tuple) - limits of the calibration
    """

    sample_rate = 1.25 * 10**9
    print('Creating AWG instance')
    awg = WX218x_awg()
    print('Connecting...')
    awg.open(reset=False)
    
    awg.configure_operation_mode(awg_channel, WX218x_OperationMode.CONTINUOUS)
    awg.configure_output_mode(WX218x_OutputMode.ARBITRARY)
    awg.configure_sample_rate(sample_rate)
    awg.configure_arb_gain(awg_channel, 2)

    inst, power_meter = get_power_meter(return_inst=True)
    power_meter: ThorlabsPM100 = power_meter  # declare the type for easier editing
    configure_power_meter(power_meter, nMeasurmentCounts=repeats)
    
    closest_level = None
    closest_diff = float('inf')
    last_read_Value = None

    if save_all and results_dict is not None:
        results_dict['level'] = []
        results_dict['read_value'] = []
        results_dict['rabi']=[]

    for freq in freqs:
        levelData = np.linspace(*calibration_lims, n_steps)
        calData = np.empty(n_steps)
        print('Running through awg levels...might take a while...')

        for i, level in enumerate(levelData):
            wf = testWaveform(sample_rate, level=level, mod_freq=freq * 10**6)
            awg.create_custom_adv(wf.get(sample_rate), wf.get(sample_rate))
             
            awg.enable_channel(awg_channel)
            time.sleep(delay)
            read_value = float(power_meter.read)
            print(f'{level}V, {read_value*1e3}mW')

            calData[i] = read_value
            
            awg.disable_channel(awg_channel)

            diff = abs(read_value - target_power)
            if diff < closest_diff:
                closest_diff = diff
                closest_level = level
                last_read_value = read_value

            if save_all and results_dict is not None:
                results_dict['level'].append(level)
                results_dict['read_value'].append(read_value)

            if diff < 1e-8:  # Adjust the tolerance as needed
                print(f'Target power {target_power} achieved with level {level}: {read_value}. Diff: {diff}')
                awg.reset()
                awg.close()
                inst.close()
                return level, diff, read_value, results_dict
        
        print('...finished taking data')

    print('Resetting awg...')
    awg.reset()
    print('Calibration finished.')
    awg.close()

    inst.close()
    print(f'Closest level found: {closest_level} with difference: {closest_diff}')
    return closest_level, closest_diff, read_value, results_dict  # Return the closest level found if the target power is not achieved