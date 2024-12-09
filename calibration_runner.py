'''
Script runner to generate different types of calibration files.

Refactored 09/12/2024

@author: Matt King
'''

import os
import time
from Config import ConfigReader, DaqReader
from instruments.WX218x.WX218x_awg import Channel

import awg_functions.calibration_runner as calibrate


# Select the type of calibration script to run.
CALIB_TYPE = "absolute_power" # options are "absolute_power", "stirap_aom_response", "another one"


    

if __name__ == "__main__" and CALIB_TYPE == "absolute_power":
    """
    Varies the frequency of the AOM signal, and calibrates the output power against voltage at
    each of the specified frequencies.
    """

    AOM_NAME = 'cool_upper'
    FREQ_CHAN = 0
    AMP_CHAN = 4
    FREQ_V = [4.141,5.089,6.017,6.383]
    AOM_FREQS = [90,95,100,102]

    # AOM_NAME = 'vStirap_ref'
    # AMP_CHAN = 9
    # FREQ_CHAN = 8
    # FREQ_V = [6.104]
    # FREQ_V = [6.412,6.051,5.113,4.16,3.2087]
    # AOM_FREQS = [76,78,80,82,84]
        
    # AOM_NAME = 'cool_lower'
    # AMP_CHAN = 5
    # FREQ_CHAN = 1
    # FREQ_V = [6.412,6.051,5.113,4.16]
    # AOM_FREQS = [102,100,95,90]

    # AOM_NAME = 'cool_centre'
    # FREQ_CHAN = 2
    # AMP_CHAN = 6
    # FREQ_V = [4.141,5.089,6.017,6.383]
    # AOM_FREQS = [90,95,100,102]

    config_reader = ConfigReader(os.getcwd() + '/configs/rootConfig')
    daq_config_fname = config_reader.get_daq_config_fname()
    daq_controller = DaqReader(daq_config_fname).load_DAQ_controller()
    daq_controller.continuousOutput=True

    calibrate.daq_driven_aom_response(daq_controller, AOM_FREQS, FREQ_V, FREQ_CHAN, AMP_CHAN,\
                              (0.2, 1.75),v_step = calibrate.default_v_step()*5, \
                              delay = 0.5, save_folder = "jan/{AOM_NAME}")







elif __name__ == "__main__" and CALIB_TYPE == "stirap_aom_response":
    """
    AWG driven AOM calib
    """

    AWG_CHAN_1_FREQS = [105, 107, 109]
    AWG_CHAN_2_FREQS = [76, 78.5, 80]

    calibrate.awg_driven_aom_response(AWG_CHAN_1_FREQS, 'stirap_elysa', Channel.CHANNEL_1, repeats=5,\
                                       delay=0.3, save_folder = "jan/awg_driven")

    calibrate.awg_driven_aom_response(AWG_CHAN_2_FREQS, 'stirap_dl_pro', Channel.CHANNEL_2, repeats=5,\
                                       delay=0.3, save_folder = "jan/awg_driven")

    #need to write new calibration routine for opical pumping where I am only producing a square pulse


elif __name__ == "__main__" and CALIB_TYPE == "another one":
    #     Freq calib DAQ
    """
        aom_name = 'Optical Pump 2'
        freq_ch = 14
        config_reader = ConfigReader(os.getcwd() + '/configs/rootConfig')
        daq_config_fname = config_reader.get_daq_config_fname()
        daq_controller = DaqReader(daq_config_fname).load_DAQ_controller()
        daq_controller.continuousOutput=True
        daq_controller.updateChannelValue(15, 1) # for manual control of amplitude input (in V)

        calibName = "{0}_freq".format(aom_name)
        vData, calData, units = calibrate_frequency(daq_controller,freq_ch, (0,10), calibration_V_step = get_default_calibration_Vstep())

        create_calibration_file(os.getcwd() + '/calibrations/jan/{0}'.format(calibName), vData, calData, units)
        save_calibration_plot(os.getcwd() + '/calibrations/jan/{0}_plot.png'.format(calibName), vData, calData, units, '{0}_plot'.format(calibName))
        daq_controller.releaseAll()
    """

else:
    raise ValueError("Invalid choice for CALIB_TYPE. Calibration could not be run.")