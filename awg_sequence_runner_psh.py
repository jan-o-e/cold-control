from time import sleep
import os
import time

from Config import ConfigReader, DaqReader
from ExperimentalRunner import AWGSequenceConfiguration, AwgConfiguration, Waveform
from configobj import ConfigObj
from lab_control_functions.awg_control_functions_psh import run_awg
from lab_control_functions.awg_control_functions_single_psh import run_awg_single
from instruments.WX218x.WX218x_awg import WX218x_awg, Channel
import pyvisa 
import re, ast


def toBool(string):
    GLOB_TRUE_BOOL_STRINGS = ['true', 't', 'yes', 'y']
    return string.lower() in GLOB_TRUE_BOOL_STRINGS


path_to_config = 'waveforms/pulse shaping exp/newPhotonProductionConfigA'
path_to_config_single = 'waveforms/pulse shaping exp/newPhotonProductionConfigCH4A'

# path_to_config = 'waveforms/pulse shaping exp/newPhotonProductionConfigB'
# path_to_config_single = 'waveforms/pulse shaping exp/newPhotonProductionConfigCH4B'


if __name__ == '__main__':
    # Converts the specified file to a "config object"
    config = ConfigObj(path_to_config)     # permite acceder a los valores de configuración como un diccionario.
    config_single = ConfigObj(path_to_config_single)  

    # Reads the awg properties from the config object, and creates a new awg configuration with those settings        
    awg_config = AwgConfiguration(sample_rate = float(config['AWG']['sample rate']),
                                    burst_count = int(config['AWG']['burst count']),
                                    waveform_output_channels = list(config['AWG']['waveform output channels']),
                                    waveform_output_channel_lags = map(float, config['AWG']['waveform output channel lags']),  # Retrasos asociados a los canales de salida.
                                    marked_channels = list(config['AWG']['marked channels']),
                                    marker_width = eval(config['AWG']['marker width']),
                                    waveform_aom_calibrations_locations = list(config['AWG']['waveform aom calibrations locations']))
    
    # Reads the waveforms from the config object, and creates a list of Waveforms with those properties
    waveforms = []
    for x,v in config['waveforms'].items():
        if v['phases']: 
            phases_str = ' '.join(v['phases'])
            phases_str = re.sub(r'\(([^)]+) ([^)]+)\)', r'(\1, \2)', phases_str)
            phases_str = phases_str.replace(') (', '), (')
            phases = ast.literal_eval(phases_str)
        else:
            phases = [] 
        waveforms.append(Waveform(fname = v['filename'],
                                    mod_frequency= float(v['modulation frequency']),
                                    phases = phases)) # map(float, v['phases']))) 
         

    # Sets the general settings for the whole process as a photon production configuration
    photon_production_config = AWGSequenceConfiguration(save_location = config['save location'],
                                                                mot_reload  = eval(config['mot reload']),
                                                                iterations = int(config['iterations']),
                                                                waveform_sequence = list(eval(config['waveform sequence'])),
                                                                waveforms = waveforms,
                                                                waveform_stitch_delays = list(eval(config['waveform stitch delays'])), #  Retrasos entre formas de onda.
                                                                interleave_waveforms = toBool(config['interleave waveforms']),  # Indica si las formas de onda deben intercalarse.
                                                                awg_configuration = awg_config)
    

    # Calls the configure_awg function with the values extracted from the config object
    # This function used to be called "configure_awg"
     

    awg_config_single = AwgConfiguration(sample_rate = float(config_single['AWG']['sample rate']),
                                         burst_count = int(config_single['AWG']['burst count']),
                                         waveform_output_channels = list(config_single['AWG']['waveform output channels']),
                                         waveform_output_channel_lags = map(float, config_single['AWG']['waveform output channel lags']),
                                         marked_channels = list(config_single['AWG']['marked channels']),
                                         marker_width = eval(config_single['AWG']['marker width']),
                                         waveform_aom_calibrations_locations = list(config_single['AWG']['waveform aom calibrations locations']))

    waveforms_single = []
    for x,v in config_single['waveforms'].items():
        waveforms_single.append(Waveform(fname = v['filename'],
                                         mod_frequency= float(v['modulation frequency']),
                                         phases=map(float, v['phases'])))

    photon_production_config_single = AWGSequenceConfiguration(save_location = config_single['save location'],
                                                                    mot_reload  = eval(config_single['mot reload']),
                                                                    iterations = int(config_single['iterations']),
                                                                    waveform_sequence = list(eval(config_single['waveform sequence'])),
                                                                    waveforms = waveforms_single,
                                                                    waveform_stitch_delays = list(eval(config_single['waveform stitch delays'])),
                                                                    interleave_waveforms = toBool(config_single['interleave waveforms']),
                                                                    awg_configuration = awg_config_single,
                                                                    )

    rm = pyvisa.ResourceManager()
    awg = rm.open_resource("USB0::0x168C::0x1284::0000215582::0::INSTR")   
    awg.write(":SYSTem:REBoot") 
    awg.close()

    awg_test=run_awg_single(awg_config_single, photon_production_config_single)
    awg_test=run_awg(awg_config, photon_production_config) 


    
    # Opens a new config file as a "config reader" object.
    config_reader = ConfigReader(os.getcwd() + '/configs/rootConfig.ini')
    for i in range(1,1000):
        daq_config_fname = config_reader.get_daq_config_fname()# gets the name of the config file for the DAQ cards
        daq_controller = DaqReader(daq_config_fname).load_DAQ_controller()# reads the config file to create a "daq reader" object

        # The below lines all control the DAQ cards manually aside from the config file
        daq_controller.continuousOutput=True
        daq_controller.updateChannelValue(22, 2.6) # for manual control of amplitude input (in V)
        daq_controller.updateChannelValue(14, 2.485)
        daq_controller.updateChannelValue(8, 0.0048)
        daq_controller.releaseAll()
        time.sleep(1)

'''waveform sequence = '[0,1,0,1],[2,5,2], [3], [4]'
waveform stitch delays = '[-1,[3]],[-1,[3,0]],[1,[0,1,0,1]], [1,[0,1,0,1]]'
sample rate = 1228750000.0 '''

'''waveform sequence = '[0,1,0,1],[2,4,2], [3]'
waveform stitch delays = '[-1,[3]],[-1,[3,0]],[1,[0,1,0,1]]'''