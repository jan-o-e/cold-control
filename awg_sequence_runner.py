from time import sleep
import os
import time

from Config import ConfigReader, DaqReader
from ExperiementalRunner import PhotonProductionConfiguration, AwgConfiguration, TdcConfiguration, Waveform
from configobj import ConfigObj
from awg_functions.process_awg_config import run_awg


path_to_config = r'C:\Users\apc\Documents\Python Scripts\Cold Control Heavy\configs\photon production\newPhotonProductionConfigJan'



if __name__ == '__main__':

    GLOB_TRUE_BOOL_STRINGS = ['true', 't', 'yes', 'y']

    def toBool(string):
        return string.lower() in GLOB_TRUE_BOOL_STRINGS
    
    # Converts the specified file to a "config object"
    config = ConfigObj(path_to_config)
    # Reads the awg properties from the config object, and creates a new awg configuration with those settings        
    awg_config = AwgConfiguration(sample_rate = float(config['AWG']['sample rate']),
                                    burst_count = int(config['AWG']['burst count']),
                                    waveform_output_channels = list(config['AWG']['waveform output channels']),
                                    waveform_output_channel_lags = map(float, config['AWG']['waveform output channel lags']),
                                    marked_channels = list(config['AWG']['marked channels']),
                                    marker_width = eval(config['AWG']['marker width']),
                                    waveform_aom_calibrations_locations = list(config['AWG']['waveform aom calibrations locations']))
    # Same as above but for the tdc
    tdc_config = TdcConfiguration(counter_channels = map(eval, config['TDC']['counter channels']),
                                    marker_channel = int(config['TDC']['marker channel']),
                                    timestamp_buffer_size = int(config['TDC']['timestamp buffer size']))
    
    # Reads the waveforms from the config object, and creates a list of Waveforms with those properties
    waveforms = []
    for x,v in config['waveforms'].items():
        waveforms.append(Waveform(fname = v['filename'],
                                    mod_frequency= float(v['modulation frequency']),
                                    phases=map(float, v['phases'])))
        

    # Sets the general settings for the whole process as a photon production configuration
    photon_production_config = PhotonProductionConfiguration(save_location = config['save location'],
                                                                mot_reload  = eval(config['mot reload']),
                                                                iterations = int(config['iterations']),
                                                                waveform_sequence = list(eval(config['waveform sequence'])),
                                                                waveforms = waveforms,
                                                                waveform_stitch_delays = list(eval(config['waveform stitch delays'])),
                                                                interleave_waveforms = toBool(config['interleave waveforms']),
                                                                awg_configuration = awg_config,
                                                                tdc_configuration = tdc_config)
    

    # Calls the configure_awg function with the values extracted from the config object
    # This function used to be called "configure_awg"
    awg_test=run_awg(awg_config, photon_production_config)


    # Opens a new config file as a "config reader" object.
    config_reader = ConfigReader(os.getcwd() + '/configs/rootConfig')
    for i in range(1,10000):
        daq_config_fname = config_reader.get_daq_config_fname()# gets the name of the config file for the DAQ cards
        daq_controller = DaqReader(daq_config_fname).load_DAQ_controller()# reads the config file to create a "daq reader" object

        # The below lines all control the DAQ cards manually aside from the config file
        daq_controller.continuousOutput=True
        daq_controller.updateChannelValue(22, 2.6) # for manual control of amplitude input (in V)
        daq_controller.updateChannelValue(14, 2.485)
        daq_controller.updateChannelValue(8, 0.0048)
        daq_controller.releaseAll()
        time.sleep(0.2)

'''waveform sequence = '[0,1,0,1],[2,5,2], [3], [4]'
waveform stitch delays = '[-1,[3]],[-1,[3,0]],[1,[0,1,0,1]], [1,[0,1,0,1]]'
sample rate = 1228750000.0 '''

'''waveform sequence = '[0,1,0,1],[2,4,2], [3]'
waveform stitch delays = '[-1,[3]],[-1,[3,0]],[1,[0,1,0,1]]'''