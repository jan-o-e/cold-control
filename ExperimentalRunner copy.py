'''
This module is intended to hold the top profiles for running assorted experiments from 
configuration through to data acquisition.

Created on 13 Aug 2016, Refactored on 22 May 2025

@authors: Tom Barrett, Matt King

This module contains configuration classes that are used to store the parameters for a 
particular experiment, usually loaded from a configuration file. It also contains 
experiment classes that are used to run the experiments getting the details from the
configuration classes. The (experimental) configuration classes should inherit from the
generic configuration class, GenericConfiguration. The experiment classes should inherit
from the GenericExperiment class.

'''
from time import sleep
import copy
import os
import time
import numpy as np
import threading
from PIL import Image
import csv
import pandas as pd
import glob
import re
import collections
import _tkinter
from typing import List, Tuple, Dict, Any
import oscilloscope_manager as osc
import pyvisa as visa
from datetime import datetime


from DAQ import DAQ_controller, DaqPlayException, DAQ_channel
from instruments.WX218x.WX218x_awg import WX218x_awg, Channel
from instruments.WX218x.WX218x_DLL import WX218x_MarkerSource, WX218x_OutputMode, WX218x_OperationMode, \
    WX218x_SequenceAdvanceMode, WX218x_TraceMode, WX218x_TriggerImpedance, WX218x_TriggerMode,\
    WX218x_TriggerSlope, WX218x_Waveform 
from instruments.quTAU.TDC_quTAU import TDC_quTAU
from instruments.quTAU.TDC_BaseDLL import TDC_SimType, TDC_DevType, TDC_SignalCond
from instruments.pyicic.IC_ImagingControl import IC_ImagingControl
from instruments.pyicic.IC_Exception import IC_Exception
from instruments.pyicic.IC_Camera import IC_Camera
from instruments.TF930 import TF930
from Sequence import IntervalStyle, Sequence
from sympy.physics.units import frequency
from serial.serialutil import SerialException



def make_property(attr_name):
    return property(
        fget=lambda self: getattr(self, attr_name),
        fset=lambda self, value: setattr(self, attr_name, value),
        fdel=lambda self: delattr(self, attr_name),
    )

"""
class ExperimentSessionConfig (object):
    
    def __init__(self,
                 save_location,
                 summary_fname,
                 automated_experiment_configurations,
                 daq_channel_update_steps,
                 daq_channel_update_delay):
        
        self.save_location = save_location
        self.summary_fname = summary_fname
        self.automated_experiment_configurations = automated_experiment_configurations
        self.daq_channel_update_steps = daq_channel_update_steps
        self.daq_channel_update_delay = daq_channel_update_delay

    def get_summary_fname(self):
        return self.__summary_fname


    def set_summary_fname(self, value):
        self.__summary_fname = value


    def del_summary_fname(self):
        del self.__summary_fname


    def get_daq_channel_update_steps(self):
        return self.__daq_channel_update_steps


    def get_daq_channel_update_delay(self):
        return self.__daq_channel_update_delay


    def set_daq_channel_update_steps(self, value):
        self.__daq_channel_update_steps = value


    def set_daq_channel_update_delay(self, value):
        self.__daq_channel_update_delay = value


    def del_daq_channel_update_steps(self):
        del self.__daq_channel_update_steps


    def del_daq_channel_update_delay(self):
        del self.__daq_channel_update_delay


    def get_save_location(self):
        return self.__save_location


    def get_automated_experiment_configurations(self):
        return self.__automated_experiment_configurations


    def set_save_location(self, value):
        self.__save_location = value


    def set_automated_experiment_configurations(self, value):
        self.__automated_experiment_configurations = value


    def del_save_location(self):
        del self.__save_location


    def del_automated_experiment_configurations(self):
        del self.__automated_experiment_configurations

    save_location = property(get_save_location, set_save_location, del_save_location, "save_location's docstring")
    automated_experiment_configurations = property(get_automated_experiment_configurations, set_automated_experiment_configurations, del_automated_experiment_configurations, "automated_experiment_configurations's docstring")
    daq_channel_update_steps = property(get_daq_channel_update_steps, set_daq_channel_update_steps, del_daq_channel_update_steps, "daq_channel_update_steps's docstring")
    daq_channel_update_delay = property(get_daq_channel_update_delay, set_daq_channel_update_delay, del_daq_channel_update_delay, "daq_channel_update_delay's docstring")
    summary_fname = property(get_summary_fname, set_summary_fname, del_summary_fname, "summary_fname's docstring")
#"""

class ExperimentSessionConfig:
    """
    ExperimentSessionConfig manages high-level configuration for an automated experimental session.
    Previously called ExperimentalAutomationConfiguration.

    This includes:
    - The location where experiment data and summaries should be saved
    - A list of individual experiment configurations to run
    - Parameters controlling how frequently DAQ channels are updated

    Intended to coordinate and control the behavior of a full session involving multiple experiments.
    """
    def __init__(self,
                 save_location,
                 summary_fname,
                 automated_experiment_configurations,
                 daq_channel_update_steps,
                 daq_channel_update_delay):
        
        self._save_location = save_location
        self._summary_fname = summary_fname
        self._automated_experiment_configurations = automated_experiment_configurations
        self._daq_channel_update_steps = daq_channel_update_steps
        self._daq_channel_update_delay = daq_channel_update_delay


    summary_fname = make_property('_summary_fname')
    save_location = make_property('_save_location')
    automated_experiment_configurations = make_property('_automated_experiment_configurations')
    daq_channel_update_steps = make_property('_daq_channel_update_steps')
    daq_channel_update_delay = make_property('_daq_channel_update_delay')


class GenericConfiguration:
    """
    GenericConfiguration is a placeholder for any configuration that doesn't fit into the other categories.
    This class is not intended to be used directly but serves as a base for other configuration classes.
    """
    def __init__(self,
                 save_location,
                 mot_reload,
                 iterations,):
        
        self._save_location = save_location
        self._mot_reload = mot_reload# in milliseconds
        self._iterations = iterations

    save_location = make_property('_save_location')
    mot_reload = make_property('_mot_reload')
    iterations = make_property('_iterations')

    def set_mot_reload(self, value):
        """Sets the value of the MOT reload time in milliseconds."""
        self._mot_reload = value
    
    def set_iterations(self, value):
        self._iterations = value

class MotFluoresceConfiguration(GenericConfiguration):
    """
    Configuration for a MOT fluorescence experiment. More details can be found in the MotFluoresceExperiment class.
    
    The data used to configure the experiment should be loaded from a configuration file with (currently) the 
    "ExperimentConfigReader" class and the get_mot_fluoresce_config method. This class must be passed to the 
    MotFluoresceExperiment class to run the experiment.

    inputs:
     - save_location: The location to save the data collected in the experiment
     - mot_reload: The time in milliseconds to wait for the MOT to reload
     - iterations: The number of times to repeat the experiment
     - use_cam: Boolean to determine whether to use the camera for imaging
     - use_scope: Boolean to determine whether to use the scope for data acquisition
     - cam_dict: Dictionary containing camera configuration parameters (if use_cam is True)
     - scope_dict: Dictionary containing scope configuration parameters (if use_scope is True)
    """
    def __init__(self, save_location, mot_reload, iterations, use_cam, use_scope,
                 cam_dict:Dict = None, scope_dict:Dict = None):
        super().__init__(save_location, mot_reload, iterations)

        # self.use_scope = use_scope
        self.use_cam = use_cam

        if use_cam == True:
            self.cam_exposure = cam_dict["cam_exposure"]
            self.cam_gain = cam_dict["cam_gain"]
            self.camera_trigger_channel = cam_dict["camera_trigger_channel"]
            self.camera_trigger_level = cam_dict["camera_trigger_level"]
            self.camera_pulse_width = cam_dict["camera_pulse_width"]
            self.save_images = cam_dict["save_images"]
        else:
            print("No camera will be used. Functionality may not be fully implemented")

        # if self.use_scope:
        #     self.scope_trigger_channel = scope_dict["trigger_channel"]
        #     self.scope_trigger_level = scope_dict["trigger_level"]
        #     self.scope_sample_rate = scope_dict["sample_rate"]
        #     self.scope_time_range = scope_dict["time_range"]
        #     self.scope_centered_0 = scope_dict["centered_0"]
        #     self.scope_data_channels = scope_dict["data_channels"]


class PhotonProductionConfiguration(GenericConfiguration):
    """
    PhotonProductionConfiguration stores all configuration parameters
    required for a photon production experiment.

    This includes:
    - Save location and MOT reload time
    - Number of iterations
    - A waveform sequence and its associated waveforms
    - Interleaving and stitching behavior for waveforms
    - Configuration objects for the AWG and TDC systems
    """

    def __init__(self,
                 save_location,
                 mot_reload,
                 iterations,
                 waveform_sequence,
                 waveforms,
                 interleave_waveforms,
                 waveform_stitch_delays,
                 awg_configuration,
                 tdc_configuration):
        
        super().__init__(save_location, mot_reload, iterations)

        self._waveform_sequence = waveform_sequence
        self.waveforms: List[Waveform] = waveforms
        self.interleave_waveforms: bool = interleave_waveforms
        self.waveform_stitch_delays = waveform_stitch_delays
        
        self._awg_configuration: AwgConfiguration = awg_configuration
        self._tdc_configuration: TdcConfiguration = tdc_configuration


    # --- waveform_sequence ---
    @property
    def waveform_sequence(self):
        return self._waveform_sequence
    @waveform_sequence.setter
    def waveform_sequence(self, value):
        print('Setting waveform sequence to', value, [type(x) for x in value])
        self._waveform_sequence = value
    @waveform_sequence.deleter
    def waveform_sequence(self):
        del self._waveform_sequence

    
    awg_configuration = make_property('_awg_configuration')
    tdc_configuration = make_property('_tdc_configuration')


class AbsorbtionImagingConfiguration(GenericConfiguration):
    '''
    This object stores and presents for editing the settings for absorbtion imaging experiments.
        
        scan_abs_img_freq - TODO
        abs_img_freq_ch - TODO
        abs_img_freqs - TODO
        camera_trig_ch, imag_power_ch - The DAQ channels that trigger the camera and control the imaging light power.
        camera_pulse_width, imag_pulse_width - How long to make the trigger pulse and absorbtion imaging flash in microseconds.
        t_imgs - The times at which to take images (in microseconds where 0 is the beginning of the sequence).
        mot_reload_time - The MOT reload time in ms
        bkg_off_channels - A list of channels (specified by channel number) to turn off during background pictures.
        n_backgrounds - The number of background images to take for each absorbtion image.
        cam_gain - The gain setting for the camera when taking the picture.
        cam_exposure - How long the camera exposure should be.  Passes as an integer x which corresponds to an exposure time of 1/x seconds.
        save_location - The folder to save images to as 'save_location/{date}/{time}/'
        save_raw_images - Boolean determining whether the raw images (i.e. processed absorbtion images and all background contributing to 
                          the background average) are saved.
        save_processed_images - Boolean determining whether the processed images (i.e. absorbtion images after background subtraction and
                                average backgrounds) are automatically saved.
        review_processed_images - Boolean determining whether the Absorbtion_imaging_review_UI is launched after the images are processed
                                  to allow the user to review the images, add notes and decide whether to save or not. Note that since the
                                  user is given the chance to review the processed images, the option to automatically save them is disabled
                                  when review_processed_images=True.
    '''
        
    def __init__(self,
                 scan_abs_img_freq, abs_img_freq_ch, abs_img_freqs,
                 camera_trig_ch, imag_power_ch, 
                 camera_trig_levs, imag_power_levs, 
                 camera_pulse_width, imag_pulse_width,
                 t_imgs, 
                 mot_reload, 
                 n_backgrounds, bkg_off_channels, 
                 cam_gain, cam_exposure, 
                 cam_gain_lims, cam_exposure_lims,
                 save_location,
                 save_raw_images, save_processed_images, review_processed_images, iterations=1):
        super().__init__(save_location, mot_reload, iterations)
        
        self.scan_abs_img_freq = scan_abs_img_freq
        self.abs_img_freq_ch = abs_img_freq_ch
        self.abs_img_freqs = abs_img_freqs
        self.camera_trig_ch = camera_trig_ch
        self.imag_power_ch = imag_power_ch
        self.camera_trig_levs = camera_trig_levs
        self.imag_power_levs = imag_power_levs
        self.camera_pulse_width = camera_pulse_width
        self.imag_pulse_width = imag_pulse_width
        self.t_imgs = t_imgs
        self.mot_reload_time = mot_reload
        self.n_backgrounds = n_backgrounds
        self.bkg_off_channels = bkg_off_channels
        self.cam_gain = cam_gain
        self.cam_exposure = cam_exposure
        self.cam_gain_lims = cam_gain_lims
        self.cam_exposure_lims = cam_exposure_lims
        self.save_location = save_location
        self.save_raw_images = save_raw_images
        self.save_processed_images = save_processed_images
        self.review_processed_images = review_processed_images

"""
class SingleExperimentConfig(object):
    
    def __init__(self,
                 daq_channel_static_values,
                 sequence_fname,
                 sequence,
                 iterations,
                 mot_reload,
                 modulation_frequencies
                 ):
        self.daq_channel_static_values = daq_channel_static_values
        self.sequence_fname = sequence_fname
        self.sequence = sequence
        self.iterations = iterations
        self.mot_reload = mot_reload
        self.modulation_frequencies = modulation_frequencies

    def get_daq_channel_static_values(self):
        return self.__daq_channel_static_values


    def get_sequence_fname(self):
        return self.__sequence_fname


    def get_iterations(self):
        return self.__iterations


    def get_mot_reload_time(self):
        return self.__mot_reload_time


    def set_daq_channel_static_values(self, value):
        self.__daq_channel_static_values = value


    def set_sequence_fname(self, value):
        self.__sequence_fname = value


    def set_iterations(self, value):
        self.__iterations = value


    def set_mot_reload_time(self, value):
        self.__mot_reload_time = value


    def del_daq_channel_static_values(self):
        del self.__daq_channel_static_values


    def del_sequence_fname(self):
        del self.__sequence_fname


    def del_iterations(self):
        del self.__iterations


    def del_mot_reload_time(self):
        del self.__mot_reload_time

    daq_channel_static_values = property(get_daq_channel_static_values, set_daq_channel_static_values, del_daq_channel_static_values, "daq_channel_static_values's docstring")
    sequence_fname = property(get_sequence_fname, set_sequence_fname, del_sequence_fname, "sequence_fname's docstring")
    iterations = property(get_iterations, set_iterations, del_iterations, "iterations's docstring")
    mot_reload_time = property(get_mot_reload_time, set_mot_reload_time, del_mot_reload_time, "mot_reload_time's docstring")
#"""

class SingleExperimentConfig(GenericConfiguration):
    """
    SingleExperimentConfig defines the configuration for a single automated experiment run.
    Previously called AutomatedExperimentConfiguration.

    This includes:
    - Static DAQ channel values
    - The filename and contents of the experiment sequence
    - The number of times to repeat the experiment
    - The MOT (Magneto-Optical Trap) reload time
    - Frequencies used for modulation during the experiment

    Intended to be used as part of a larger experiment session, or independently for individual runs.
    """
    def __init__(self,
                 daq_channel_static_values,
                 sequence_fname,
                 sequence,
                 iterations,
                 mot_reload,
                 modulation_frequencies,
                 save_location=None):
        
        self._daq_channel_static_values = daq_channel_static_values
        self._sequence_fname = sequence_fname
        self._sequence = sequence
        self._modulation_frequencies = modulation_frequencies
        super().__init__(save_location, mot_reload, iterations)

    daq_channel_static_values = make_property('_daq_channel_static_values')
    sequence_fname = make_property('_sequence_fname')
    iterations = make_property('_iterations')
    mot_reload_time = make_property('_mot_reload_time')
    sequence = make_property('_sequence')
    modulation_frequencies = make_property('_modulation_frequencies')

class GenericExperiment:
    '''
    A generic base class for all experiments.  This is not intended to be used directly, but is what other experiments should inherit from.
    '''
    def __init__(self, daq_controller:DAQ_controller, sequence:Sequence, configuration:GenericConfiguration):
        '''
        Constructor
        '''        
        self.daq_controller = daq_controller
        self.sequence = sequence
        self.config = configuration
        

    def configure(self):
        raise NotImplementedError()
    
    def daq_cards_on(self):
        """
        Function to turn on the DAQ card output.  This is required to run a sequence.
        """
        self.isDaqContinuousOutput = self.daq_controller.continuousOutput
        if not self.isDaqContinuousOutput:
            print("DAQ output must be on to run a sequence - turning it on.")
            self.daq_controller.toggleContinuousOutput()
    
    def run(self):
        raise NotImplementedError()
    
    def daq_cards_off(self):
        if self.isDaqContinuousOutput:
            print("Returning to free running DAQ values.")
            self.daq_controller.writeChannelValues()
        else:
            print("Reverting DAQ output to off.")
            self.daq_controller.toggleContinuousOutput()
            self.daq_controller.writeChannelValues()

    def close(self):
        raise NotImplementedError()

    def run_in_thread(self, start_thread=True):
        '''
        Run the experiment with the experimental loop in a separate thread.
        '''
        def run_and_close():
            self.run()
            self.close()
        
        thread = threading.Thread(name='Cold Control Experiment Thread',
                                  target=run_and_close)
        
        if start_thread:
            thread.start()
        return thread
    

class AbsorbtionImagingExperiment(GenericExperiment):
        
    shutter_lag = 4.8 #The camera response time to the trigger.  Hard coded as it is a physical camera property.
     
    def __init__(self, daq_controller:DAQ_controller, sequence:Sequence, absorbtion_imaging_configuration:AbsorbtionImagingConfiguration, ic_imaging_control:IC_ImagingControl):
        '''
        Runs an absorbtion imaging experiment. Takes a number of parameters, namely:
            daq_controller - The DAQ Controller object for running the daq channels
            sequence - The sequence to run whilst taking the images.  Note that the camera trigger and
                       imaging power channels are overwritten by this experiment and so need not be configured
                       in the original sequence.
            absorbtion_imaging_configuration - An instance of AbsorbtionImagingConfiguration.
        '''
        
        super().__init__(daq_controller, sequence, absorbtion_imaging_configuration)
        # the configuration object is called self.config
        c:AbsorbtionImagingConfiguration = self.config
        self.results_ready = False # A flag to determine if the experiment has finished running and the results exist yet
        
        # Use the externally provided IC_ImagingControl instances if one is provided
        if ic_imaging_control != None:
            self.ic_ic = ic_imaging_control
            if not self.ic_ic.initialised:
                self.ic_ic.init_library()
            self.external_ic_ic_provided = True
        # Otherwise create and initialise one of our own, setting a flag so it is closed again later.
        else:
            self.ic_ic = IC_ImagingControl()
            self.ic_ic.init_library()
            self.external_ic_ic_provided = False
        
        if self.sequence.t_step > 100:
            print('WARNING: Sequence step size is > 100us.  This means the fastest possible imaging flash is longer than recommended.\n',
                   'Typically the order of 10us flashes are appropriate. Continuing with the sequence anyway...')
        
        self.save_location = os.path.join(c.save_location, time.strftime("%y-%m-%d/%H-%M-%S"))
        if not os.path.exists(self.save_location):
            os.makedirs(self.save_location)
        
        # If the trigger levs have not been set, default them to 0 and the maximum permitted value on the camera/imaging channels.
        if c.camera_trig_levs == None:
            c.camera_trig_levs = (0, next(ch.chLimits[1] if not ch.isCalibrated else ch.calibrationFromVFunc(ch.chLimits[1])
                                           for ch in daq_controller.getChannels() if ch.chNum == c.camera_trig_ch))
        if c.imag_power_levs == None:
            c.imag_power_levs = (0, next(ch.chLimits[1] if not ch.isCalibrated else ch.calibrationFromVFunc(ch.chLimits[1])
                                           for ch in daq_controller.getChannels() if ch.chNum == c.imag_power_ch))
            
            
        # If the camera trigger pulse is shorter than a single time step in the sequence, make it longer so it actually happens! 
        if c.camera_pulse_width < sequence.t_step:
            c.camera_pulse_width = sequence.t_step
        # If the imaging pulse width has not been set or is too short, default it to one point in the sequence (i.e. as short
        # as possible),
        if c.imag_pulse_width == None or c.imag_pulse_width < sequence.t_step:
            c.imag_pulse_width = sequence.t_step
         
        # Check the absorbtion imaging flash will be on for background images and other sanity checks
        if c.imag_power_ch in c.bkg_off_channels:
            print('WARNING: You specified no to to have the absorption imaging flash on while taking backgrounds. \n' +\
                   'This will lead to poor background correction so it will be left on for background regardless.')
            c.bkg_off_channels.remove(c.imag_power_ch)
        if len(c.bkg_off_channels)==0:
            print('WARNING: No channels are turned off when taking background images - this means the images will be taken \n' +\
                   'using an identical sequence to the absorption images.  Please consider turning the MOT repump off to remove \n' +\
                   'atoms from the background images.' )

    
    def run(self, analyse=True, bkg_test=False):
        '''
        Run the absorbtion imaging.  This first generates a series of sequences to run, then
        opens and configures the camera, takes the images, saves them and closes the camera.
        '''
        print('Running absorbtion imaging experiment')
        
        self.__configureExperiment()
        
        try:
            self.__configureCamera()
            img_arrs, bkg_arrs = self.__takeImages(save_raw_images=self.config.save_raw_images)
            if bkg_test:
                self.corr_img_arrs, self.ave_bkg_arrs, self.raw_images\
                      = None, [np.mean(bkgs, axis=0, dtype=float) for bkgs in bkg_arrs], None#[sum([b.astype(float)/len(bkgs) for b in bkgs]) for bkgs in bkg_arrs]
            elif analyse:
                self.corr_img_arrs, self.ave_bkg_arrs, self.raw_images\
                      = self.__analyseImages(img_arrs, bkg_arrs, save_processed_images=self.config.save_processed_images)
            else:
                self.corr_img_arrs, self.ave_bkg_arrs = None, None, None
            self.results_ready = True
                
        # It is important to properly close the camera before exiting, otherwise the computer can be crashed.
        finally:
            self.close()
            
        return self.getResults()
    

        
    def __configureExperiment(self):
        '''
        Perform and configuration that has to occur before the experiment can be safely run.
        '''
        # Make a list of sequences (in time order) to run in order to take the imaging pictures
        self.sequences: List[Sequence] = []
        self.bkg_sequences: List[Sequence] = []

        c:AbsorbtionImagingConfiguration = self.config

        t_lag = AbsorbtionImagingExperiment.shutter_lag
#         t_offset = AbsorbtionImagingExperiment.flash_offset
        t_offset = ((1./c.cam_exposure)*10**6)/2

        for t in c.t_imgs:
            if t==0: t += self.sequence.t_step
            sequenceCopy = copy.deepcopy(self.sequence)
            # Note the camera triggers on the down slope of the square wave trigger sent to it.
            
            print('cam trig: {0}-{1}'.format(np.clip(t-c.camera_pulse_width,0,self.sequence.getLength()), t))
            
            sequenceCopy.updateChannel(c.camera_trig_ch, [(0,c.camera_trig_levs[0]),
                                                        (np.clip(t-c.camera_pulse_width,0,self.sequence.getLength()), c.camera_trig_levs[1]), 
                                                        (t,c.camera_trig_levs[0])],
                                       [IntervalStyle.FLAT]*3)
            
            print('flash: {0}-{1}'.format(np.clip(t+t_lag+t_offset, 0, sequenceCopy.getLength()-c.imag_pulse_width), np.clip(t+t_lag+t_offset+c.imag_pulse_width, 0, sequenceCopy.getLength()-sequenceCopy.t_step )))
            sequenceCopy.updateChannel(c.imag_power_ch, [(0,c.imag_power_levs[0]),
                                                        (np.clip(t+t_lag+t_offset, 0, sequenceCopy.getLength()-c.imag_pulse_width) ,c.imag_power_levs[1]), 
                                                        (np.clip(t+t_lag+t_offset+c.imag_pulse_width, 0, sequenceCopy.getLength()-sequenceCopy.t_step),c.imag_power_levs[0])],
                                       [IntervalStyle.FLAT]*3)
#             t=1000
#             sequenceCopy.updateChannel(c.imag_power_ch, [(0,c.imag_power_levs[0]),
#                                                         ( np.clip( t+t_lag+t_offset, 0, sequenceCopy.getLength()-c.imag_pulse_width) ,c.imag_power_levs[1]), 
#                                                         ( np.clip( t+t_lag+t_offset+c.imag_pulse_width, 0, sequenceCopy.getLength()-sequenceCopy.t_step ),c.imag_power_levs[0])],
#                                        [IntervalStyle.FLAT]*3)
            
            self.sequences.append(sequenceCopy)
            
            sequenceBackground = copy.deepcopy(sequenceCopy)
            # Set any channels that should be off for background images to zero.
            for ch in c.bkg_off_channels:
                sequenceBackground.updateChannel(ch, [(0,0)],[IntervalStyle.FLAT])
            self.bkg_sequences.append(sequenceBackground)
        
        # Make a list of labels for the sequences - this is what the images will be saved as.
        self.sequence_labels = list(map(int,c.t_imgs))
            
        if c.scan_abs_img_freq:
            new_seqs = []
            new_labs = []
            try:
                calib_units,_,fromVFunc = self.daq_controller.getChannelCalibrationDict()[c.abs_img_freq_ch]
            except KeyError:
                calib_units,_,fromVFunc = 'V', None, lambda x: x
            
            for freq in c.abs_img_freqs:
                seqs_copy = [copy.deepcopy(seq) for seq in self.sequences]
                for seq in seqs_copy:
                    seq.updateChannel(c.abs_img_freq_ch, [(0, freq)], [IntervalStyle.FLAT])
                new_seqs += seqs_copy
                new_labs += ["{0}_{1}{2}".format(lab, fromVFunc(freq), calib_units) for lab in self.sequence_labels]
                    
            self.sequences = new_seqs
            self.bkg_sequences = self.bkg_sequences*len(c.abs_img_freqs)
            self.sequence_labels = new_labs
            
        super().daq_cards_on()

     
    def __configureCamera(self):
        # open first available camera device
        cam_names = self.ic_ic.get_unique_device_names()
        self.cam:IC_Camera = None
        cam:IC_Camera = None
        self.cam = cam = self.ic_ic.get_device(cam_names[0])
#         self.cam_frame_timeout = int(self.sequences[0].getLength()*10**-3 + (1./self.config.cam_exposure)*10**3)
        self.cam_frame_timeout = 5000
        print('Timeout set to {0}ms'.format(self.cam_frame_timeout))
        print('Opened connection to camera {0}', cam_names[0])
        
        if not cam.is_open():
            cam.open()
            
        # change camera settings
        cam.gain.auto = False
        cam.exposure.auto = False
        cam.gain.value = self.config.cam_gain
        cam.exposure.value = self.config.cam_exposure
        formats = cam.list_video_formats()
        cam.set_video_format(formats[0])        # use first available video format
        cam.enable_continuous_mode(True)        # image in continuous mode
        cam.start_live(show_display=False)       # start imaging
                    
        # print cam.is_triggerable()
        cam.enable_trigger(True)              # camera will wait for trigger
        
        if not cam.callback_registered:
            cam.register_frame_ready_callback() # needed to wait for frame ready callback
        
        # Clear out the memory of any rogue image still in there
        try:
            cam.wait_til_frame_ready(self.cam_frame_timeout)
            cam.get_image_data()
        except IC_Exception as err:
            print("Caught IC_Exception with error: {0}".format(err.message))
        cam.reset_frame_ready()
        
    def __takeImages(self, save_raw_images):
            img_arrs = []
            bkg_arrs = []
            
                 
            for seq, bkg_seq, label in zip(self.sequences, self.bkg_sequences, self.sequence_labels):
                # Write the persistance values and wait for the MOT to reload
                self.daq_controller.load(seq.getArray())
                self.daq_controller.writeChannelValues()
                print('Loading MOT for {0}ms...'.format(self.config.mot_reload))
                sleep(self.config.mot_reload*10**-3) # convert from ms to s 
                
                if save_raw_images: 
                    # Make dirs for saving pictures
                    img_dir = os.path.join(self.save_location, 'raw')
    #                 os.makedirs(img_dir)
                    bkg_dir = os.path.join(img_dir,'img{0}-backgrounds'.format(label))
                    os.makedirs(bkg_dir)   
                
                # Load and play imaging sequence       
#                 self.daq_controller.load(seq.getArray())
                self.daq_controller.play(float(seq.t_step), clearCards=True)
                
                # Grab image and save as bmp
                self.cam.wait_til_frame_ready(self.cam_frame_timeout)    
                data = self.cam.get_image_data()
                img = Image.frombuffer('RGB', (data[1], data[2]), data[0], 'raw', 'RGB',0,1).convert('L').transpose(Image.FLIP_TOP_BOTTOM)
                img_arrs.append(np.array(img))
                if save_raw_images:
                    img.save("{0}/img{1}raw.bmp".format(img_dir, label), "bmp")
                
#                 sleep(1)
                
                self.cam.reset_frame_ready()
                    
                # Take the background images
                bkgs = []
                self.daq_controller.load(bkg_seq.getArray())
                for i in range(self.config.n_backgrounds):
                    # Load and play background sequence
                    sleep(0.5)   
                    self.daq_controller.play(float(bkg_seq.t_step), clearCards=False)
                    self.cam.wait_til_frame_ready(self.cam_frame_timeout) 
                    
                    data = self.cam.get_image_data()
                    img = Image.frombuffer('RGB', (data[1], data[2]), data[0], 'raw', 'RGB',0,1).convert('L').transpose(Image.FLIP_TOP_BOTTOM)
                    bkgs.append(np.array(img))
                    if save_raw_images:
                        img.save("{0}/{1}.bmp".format(bkg_dir, i), "bmp")
                    
                    self.cam.reset_frame_ready()
                    
                bkg_arrs.append(bkgs)
                    
                self.daq_controller.clearCards()
                
            return img_arrs, bkg_arrs
        
    def __analyseImages(self, img_arrs: List[np.ndarray], bkg_arrs: List[np.ndarray], save_processed_images:bool, process_type=1):
        '''
        Takes images and backgrounds as lists of np.arrays containing there pixel values, averages the backgrounds for each time step
        and corrects the raw image for this background.
            imgs_arrs = [img1, img2, ...]
            bkg_arrs = [[bkg1.1,bkg1.2,...], [bkg2.1,bkg2.2,...], ...]
        Returns a list of corrected images (as arrays), and of the averaged background images (also as arrays).
        
        If the save_processed_images flag is True then the background corrected images and the average backgrounds for each time step are saved.
        
        So a bit of jiggery-pokery here to analyse the image:
            1. Average the background together, converting them to arrays of floats before doing so.
            2. Convert the raw images to floats and subract them from the backgrounds.  This means light in the background absorbed by the MOT will now appear as
            high pixel values (i.e. bright in the final image).
            3. Clip the corrected array between 0 and 255 as these are valid pixel values.  If the is not done we get excess noise from points where the corrected
            image was negative or greater than 255.  This also means that extra light present in the raw image when compared to the background (e.g. MOT beams
            clipping on mirrors) are removed from the corrected image.  Note that pixels between 0,255 are only necessary for PIL - matplotlip.pyplot can handle
            wider ranges.
            4. Rescale the pixel values from to span 0-255 for maximum visibility.  Finally convert the images back to unit8 arrays as PIL expects. 
        '''
        bkg_aves_float = [np.mean(bkgs, axis=0, dtype=float) for bkgs in bkg_arrs]

        if process_type == 1:
            print("processing images by subtracting the image from the background and clipping to be between 0 and 255")# This was Tom's original method
            unscaled_corr_imgs = [np.clip(np.round(bkg - img), 0, 255) for img, bkg in zip(img_arrs, bkg_aves_float)]
        elif process_type == 2:
            print("processing images by subtracting the background from the image and clipping to be between 0 and 255")
            unscaled_corr_imgs = [np.clip(np.round(img - bkg), 0, 255) for img, bkg in zip(img_arrs, bkg_aves_float)]
        elif process_type == 3:
            print("images aren't being processed.")
            unscaled_corr_imgs = img_arrs
        else:
            raise ValueError(f"process_type cannot take the value {process_type}")
        
        bkg_aves = [bkg.astype(np.uint8) for bkg in bkg_aves_float]
        corr_images = [(arr * 255.0/arr.max()).astype(np.uint8) for arr in unscaled_corr_imgs]
#         corr_images = [arr.astype(np.uint8) for arr in unscaled_corr_imgs]
        
        if save_processed_images:
            print('Saving processed images...')
            bkg_dir = os.path.join(self.save_location,'backgrounds')
            os.makedirs(bkg_dir)
            for img, bkg_img, label in zip(corr_images, bkg_aves, self.sequence_labels):
                Image.fromarray(img).save("{0}/{1}.bmp".format(self.save_location, label), "bmp")
                Image.fromarray(bkg_img).save("{0}/{1}.bmp".format(bkg_dir, label), "bmp")
            print('Processed images saved')

        raw_images = img_arrs
        
        return corr_images, bkg_aves, raw_images
    
    def saveProcessedImages(self, notes=None):
        if not self.results_ready:
            raise Exception('The abosrbtion imaging experiment has not been run yet. There are no results to save.')
        bkg_dir = os.path.join(self.save_location,'backgrounds')
        os.makedirs(bkg_dir)

        if self.corr_img_arrs == None and self.ave_bkg_arrs == None:
            print("No images to save.")
        elif self.corr_img_arrs == None:
            # this code runs when only a background experiment has been run
            for bkg_img, label in zip(self.ave_bkg_arrs, self.sequence_labels):
                # Normalize the floating-point image to the range 0-255
                if bkg_img.dtype == float:
                    bkg_img = (255 * (bkg_img - np.min(bkg_img)) / (np.ptp(bkg_img) + 1e-10)).astype(np.uint8)
                Image.fromarray(bkg_img).save(f"{bkg_dir}/{label}.bmp", "bmp")
        else:
            for img, bkg_img, label in zip(self.corr_img_arrs, self.ave_bkg_arrs, self.sequence_labels):
                # Normalize the floating-point image to the range 0-255
                if bkg_img.dtype == float:
                    bkg_img = (255 * (bkg_img - np.min(bkg_img)) / (np.ptp(bkg_img) + 1e-10)).astype(np.uint8)
                if img.dtype == float:
                    img = (255 * (img - np.min(img)) / (np.ptp(img) + 1e-10)).astype(np.uint8)
                Image.fromarray(img).save("{0}/{1}.bmp".format(self.save_location, label), "bmp")
                Image.fromarray(bkg_img).save("{0}/{1}.bmp".format(bkg_dir, label), "bmp")
            
        if notes:
            fname = os.path.join(self.save_location,'notes.txt')
            f = open(fname, 'w')
            print('write: ', fname)
            f.write(notes)
            f.close()
            
    def getResults(self):
        if not self.results_ready:
            raise Exception('The abosrbtion imaging experiment has not been run yet.')
        print('Returning absorbtion imaging results.', len(self.ave_bkg_arrs))
        return self.corr_img_arrs, self.ave_bkg_arrs, self.raw_images, self.sequence_labels
    
    def close(self):
        '''
        Perform any tidying up.
        '''
        print('closing camera...')
        self.cam.enable_trigger(False)
        self.cam.stop_live()
        self.cam.close()
        
        if not self.external_ic_ic_provided:
            self.ic_ic.close_library()
        print('...closed')
    
        super().daq_cards_off()  
    


class PhotonProductionExperiment(GenericExperiment):
    
    def __init__(self, daq_controller:DAQ_controller, sequence:Sequence, photon_production_configuration:PhotonProductionConfiguration):
        super().__init__(daq_controller, sequence, photon_production_configuration)
        # the configuration object is a PhotonProductionConfiguration object and called self.config
        self.photon_production_config:PhotonProductionConfiguration = self.config
        self.tdc_config = self.photon_production_config.tdc_configuration
        self.awg_config = self.photon_production_config.awg_configuration

        c = self.config
        
        self.iterations = c.iterations
        self.mot_reload_time = c.mot_reload# in ms
        print('MOT reload time (ms)', self.mot_reload_time)
        self.is_live = False # Experiment is not running yet
        self.forced_stop = False # Flag for if the experiment is forcibly stopped early.
        self.data_queue = None # Queue to push data into
        
    def configure(self):
        super().daq_cards_on()
        self.daq_controller.load(self.sequence.getArray())
        # Configure the awg and record the length of the waveform loaded onto it (in seconds)
        self.awg, self.waveform_length = self.__configure_awg()
        self.tdc = self.__configure_tdc()
#         try:
#             self.counter = TF930.TF930(port='COM5')
#         except SerialException:
#             print 'Cannot find counter. Ignoring and carrying on.'
#             self.counter = None
        self.counter = None
        # Get tdc timebase in ps - do i once now, rather than every time we poll
        # the tdc for data (just for performance reasons).
        self.tdc_timebase = self.tdc.get_timebase()*10**12
        self.data_saver = PhotonProductionDataSaver(self.tdc_timebase,
                                                    self.tdc_config.marker_channel,
                                                    self.photon_production_config.save_location,
                                                    data_queue = self.data_queue,
                                                    create_log=False)
        
    def configure_data_queue(self, data_queue):
        self.data_queue = data_queue
        try:
            self.data_saver.data_queue = self.data_queue
        except AttributeError:
            pass
    
    def run(self):
        self.tdc.enable_tdc_input(True)
        self.tdc.freeze_buffers(True)
        self.tdc.clear_buffer()
        time.sleep(1)
        
        self.is_live = True
        i = 1
        tdc_read_thread = None
        self.daq_controller.load(self.sequence.getArray())
        
        while i <= self.iterations and self.is_live:
            print('iter: {0}'.format(i))
            
            sleep(self.mot_reload_time*10**-3) # convert from ms to s
            
            if tdc_read_thread: tdc_read_thread.join(timeout=5000)
#             self.daq_controller.load(self.sequence.getArray()) # TODO: can we load only once at start?
            print('unfreeze')
            self.tdc.freeze_buffers(False)
#             sleep(1)
            print('play')
            self.daq_controller.play(float(self.sequence.t_step), clearCards=False)
            print('freeze')
            tdc_read_thread = threading.Thread(name='PhotonProductionExperiment_read TDC buffer and start save thread',
                                  target=self.__save_throw_data,
                                  args=(i,))
            tdc_read_thread.start()
#             self.__save_throw_data(throw_number=i)
# 
#             if (i%100 == 0 or i==1)  and self.counter != None:
#                 self.data_saver.log_in_thread(['Repump offset VCO is at ', self.counter.query_frequency],
#                                               throw_number=i)
            
            self.daq_controller.writeChannelValues()
            
            i+=1
            
        self.daq_controller.clearCards()
        self.is_live = False
        if tdc_read_thread: tdc_read_thread.join(timeout=5000)
        self.tdc.enable_tdc_input(False)



    def close(self):
        print("Closing connection to AWG..."),
        self.awg.disable_channel(Channel.CHANNEL_1)
        self.awg.disable_channel(Channel.CHANNEL_2)
        self.awg.disable_channel(Channel.CHANNEL_3)
        self.awg.disable_channel(Channel.CHANNEL_4)
        self.awg.close()
        print("...closed")

        print("Closing connection to TDC...",
        self.tdc.close())
        print("...closed")

        if self.counter!=None:
            print('Closing connection to TF930')
            self.counter.close()

        super().daq_cards_off()
    
        print('Consolidating experimental data...',
        self.data_saver.combine_saves())
        print('done.')
     
    def __save_throw_data(self, throw_number):
        t=time.time()
        sleep( 200*10**-3 )
        self.tdc.freeze_buffers(True)   
        print('reading tdc')
        timestamps, channels, valid =  self.tdc.get_timestamps(True)
        print('throw {0}: counts on tdc={1}, tdc read time={2}ms'.format(throw_number,valid,(time.time()-t)*10**3))
        
        self.data_saver.save_in_thread(timestamps, channels, valid, throw_number)
#         save_thread.join(timeout=5000)
#         fname = r'C:/Users/apc/Desktop/test/'
#         f = open(os.path.join(fname, '{0}.txt'.format(throw_number)), 'w')
#         print 'writing to:', os.path.join(fname, '/{0}.txt'.format(throw_number))
#         for line in zip(timestamps[:valid], channels[:valid]):
#             f.write('{0},{1}\n'.format(*line))
#         f.close()
        
        
        

    def __configure_awg(self):
        print('Connecting to AWG...')
        awg = WX218x_awg()
        awg.open(reset=False)
        print('...connected')
        awg.clear_arbitrary_sequence()
        awg.clear_arbitrary_waveform()
        awg.configure_sample_rate(self.awg_config.sample_rate)
        
        awg_chs = self.awg_config.waveform_output_channels
        
        awg.configure_output_mode(WX218x_OutputMode.ARBITRARY)
        awg.configure_couple_enabled(True)
                
        for ch in [awg_chs[x] for x in range(len(awg_chs)) if x%2==0]:
            print('Configuring trigger options for', ch)
            awg.configure_burst_count(ch, self.awg_config.burst_count)
            awg.configure_operation_mode(ch, WX218x_OperationMode.TRIGGER)
            time.sleep(1)
            awg.configure_trigger_source(ch, WX218x_TriggerMode.EXTERNAL)
            awg.configure_trigger_level(ch, 2)
            awg.configure_trigger_slope(ch, WX218x_TriggerSlope.POSITIVE)
        
            
        channel_absolute_offsets = [np.rint(x*10**-6 * self.awg_config.sample_rate) for x in self.awg_config.waveform_output_channel_lags]
        channel_relative_offsets = list(map(lambda x, m=max(channel_absolute_offsets): int(m-x), channel_absolute_offsets))
        print("Channel relative lags (in awg steps are)", channel_relative_offsets)
        print("Channel absolute offsets (in awg steps are)", channel_absolute_offsets)
        
        marker_levs, marker_waveform_levs = (0,1.2), (0,1)
        marker_wid  = int(self.awg_config.marker_width*10**-6 * self.awg_config.sample_rate)
        
        def get_waveform_calib_fnc(calib_fname, max_eff=0.9):
            calib_data = np.genfromtxt(calib_fname,skip_header=1)
            calib_data[:,1] /= 100. # convert % as saved to decimal efficiencies
            calib_data = calib_data[(calib_data[:,1]<=max_eff)] # remove all elements with greater than the maximum efficiency
            calib_data[:,1] /= max(calib_data[:,1]) # rescale effiencies
            
            return lambda x: np.interp(np.abs(x),
                                        calib_data[:,1],
                                        calib_data[:,0])
        
        #this takes the array of waveform information and channel based waveform sequencing and returns the waveforms in the order they are to be played
        seq_waveforms = [[self.photon_production_config.waveforms[i] for i in ch_waveforms]
                        for ch_waveforms in self.photon_production_config.waveform_sequence]
        
        print('seq_waveforms={}', seq_waveforms)
        
        queud_markers = []
        
        seq_waveform_data, seq_marker_data = [[] for _ in range(len(awg_chs))], []
        
        # Note we deep copy the config stitch delays so that updating them below doesn't change the configuration settings.
        seq_waveforms_stitch_delays = copy.deepcopy(self.photon_production_config.waveform_stitch_delays)
                
        if self.photon_production_config.interleave_waveforms:
#             '''Add a delay to the front of the i-th channel to wait for the 0,1...,i-1 channels
#             first waveforms to finish'''
#             for i in range(1, len(seq_waveform_data)):
#                 seq_waveform_data[i] += [0]*(len(seq_waveform_data[i-1]) + seq_waveforms[i-1][0].get_n_samples())
#                          
#             '''For other channels/waveforms, add the length of the interleaved waveform from
#             the opposite channel to the stich delay'''
#             for i in range(0, len(seq_waveforms)):
#                 for j in range(0, len(seq_waveforms[i])):
#                     for k in [l for l in range(len(seq_waveforms)) if l!=i]:
#                         if j!=len(seq_waveforms[i])-1 or k>i:
# #                             print i,j,k
#                             seq_waveforms_stitch_delays[i][j] += seq_waveforms[k][j].get_n_samples()
#      
            #TODO this needs updating
            interleave_channels = [(0,1),(0,2)]
             
            def get_j_segments_max_length(seq_waveforms, channels, j, seq_stitch_delays=None):
                waveform_lengths = []
                if seq_stitch_delays==None:
                    seq_stitch_delays = np.zeros(np.array(seq_waveforms).shape).tolist()
                for ch in channels:
                    try:
                        waveform_lengths.append(seq_waveforms[ch][j].get_n_samples() + seq_stitch_delays[ch][j])
                    except IndexError:
                        pass
                print(waveform_lengths)
                return int(max(waveform_lengths)) if waveform_lengths != [] else 0    
              
            for i in range(len(seq_waveforms)):
                woven_channels = sorted([k for pair in [x for x in interleave_channels if i in x] for k in pair if k!=i])
                bck_woven_channels = [k for k in woven_channels if k<i]
                fwd_woven_channels = [k for k in woven_channels if k>i]
                for j in range(0, len(seq_waveforms[i])):
                    if j==0:
                        max_bck_waveforms = get_j_segments_max_length(seq_waveforms, bck_woven_channels, j)
                        print('Pre-padding channel{0}(seg{1}) with {2}'.format(i+1, j, max_bck_waveforms))
                        seq_waveform_data[i] += [0]*max_bck_waveforms
 
                    max_bck_waveforms = get_j_segments_max_length(seq_waveforms, bck_woven_channels, j+1)
                    max_fwd_waveforms = get_j_segments_max_length(seq_waveforms, fwd_woven_channels, j, seq_waveforms_stitch_delays) 
                     
                    print('Post-padding channel{0}(seg{1}) with max({2},{3})'.format(i+1,j,max_bck_waveforms,max_fwd_waveforms))
                    seq_waveforms_stitch_delays[i][j] += max(max_bck_waveforms,max_fwd_waveforms)#
                    
                    '''CURRENT ISSUE IS IF THE DELAY IN A BACK WAVEFORM TAKES THE PREVIOUS J-SEGMENT [AST THE START OF THE CURRENT
                    TARGET WAVEFORM/SEGMENT.'''

        j = 0
        for channel, waveform_data, waveforms, delays, channel_abs_offset in zip(awg_chs, seq_waveform_data, seq_waveforms, seq_waveforms_stitch_delays, channel_absolute_offsets):
           
            waveform_aom_calibs = {}
            aom_calibration_loc = self.awg_config.waveform_aom_calibrations_locations[j]
            print('For {0} using aom calibrations in {1}'.format(channel, os.path.join(aom_calibration_loc, '*MHz.txt')))
            for filename in glob.glob(os.path.join(aom_calibration_loc, '*MHz.txt')):
                try:
                    waveform_aom_calibs[float(re.match(r'\d+\.*\d*', os.path.split(filename)[1]).group(0))] = get_waveform_calib_fnc(filename)
                except AttributeError:
                    pass
           
            marker_data = []
           
            print('Writing onto channel:', channel)
           
            for waveform, delay in zip(waveforms, delays):
           
                if not waveform_aom_calibs:
                    calib_fun = lambda x: x
                else:
                    calib_fun = waveform_aom_calibs[min(waveform_aom_calibs,
                                                        key=lambda calib_freq: np.abs(calib_freq - waveform.get_mod_frequency()*10**-6))]
                    print('\tFor waveform with freq {0}MHz, using calib for {1}MHz'.format(waveform.get_mod_frequency()*10**-6, 
                                                                                         min(waveform_aom_calibs, key=lambda calib_freq: np.abs(calib_freq - waveform.get_mod_frequency()*10**-6))))
                
                seg_length = waveform.get_n_samples() + delay
                marker_pos = []
#                 for i in range(len(queud_markers)):
#                     print i, queud_markers
#                     if queud_markers[i] < seg_length:
#                         marker_pos.append(queud_markers.pop(i))
                i=0
                while i < len(queud_markers):
                    print(i, queud_markers)
                    if queud_markers[i] < seg_length:
                        marker_pos.append(queud_markers.pop(i))
                        i-=1
                    i+=1
                if channel_abs_offset <= seg_length:
                    marker_pos.append(channel_abs_offset)
                else:
                    queud_markers.append(channel_abs_offset)
                
                print('\tWriting markers at', marker_pos)
                print('\tWriting waveform {0} with stitch delay {1}'.format(os.path.split(waveform.fname)[1], delay))
                waveform_data += waveform.get(sample_rate=self.awg_config.sample_rate, calibration_function=calib_fun) + [0]*delay
                marker_data   += waveform.get_marker_data(marker_positions=marker_pos, marker_levels=marker_waveform_levs, marker_width=marker_wid, n_pad_right=delay)
    #             marker_data   += waveform.get(sample_rate=self.awg_config.sample_rate) + [0]*delay
                queud_markers = [x-seg_length for x in queud_markers]
            
            '''
            Wrap any makers still queued into the first waveforms markers (presuming we are looping through this sequence multiple times).
            '''
            if queud_markers != []:
                marker_index = 0
                for waveform, delay in zip(waveforms, delays):
                    seg_length = waveform.get_n_samples() + delay
    #             queud_markers = [x-(waveforms[-1].get_n_samples()+self.photon_production_config.waveform_stitch_delays[-1]) for x in queud_markers]
                    if len([x for x in queud_markers if x>=0])>0:
                        print('\tStill in queue:', [x for x in queud_markers if x>=0])
                        markers_in_waveform = [x for x in queud_markers if marker_index <= x <= marker_index+seg_length]
                        print('\tCan wrap from queue:',markers_in_waveform)
                        wrapped_marker_data = waveform.get_marker_data(marker_positions=markers_in_waveform,
                                                                       marker_levels=marker_waveform_levs,
                                                                       marker_width=marker_wid,
                                                                       n_pad_right=delay)
                        marker_data[marker_index:marker_index+len(wrapped_marker_data)] = \
                                                    [marker_waveform_levs[1] if (a==marker_waveform_levs[1] or b==marker_waveform_levs[1]) else marker_waveform_levs[0] if (a==marker_waveform_levs[0] and b==marker_waveform_levs[0])  else a+b
                                                     for a,b in zip(marker_data[:len(wrapped_marker_data)], wrapped_marker_data)]
                        marker_index+=len(wrapped_marker_data)
                        queud_markers = [x-seg_length for x in queud_markers]
            
            seq_waveform_data[j] = waveform_data
            j += 1
            
            print('\t', j, len(seq_waveform_data), [len(x) for x in seq_waveform_data])
            
            '''
            Combine the marker data for each marked channel.
            '''
            print(self.awg_config.marked_channels)
            if channel in self.awg_config.marked_channels:
                print('\tAdding marker data for', channel)
                if seq_marker_data == []:
                    seq_marker_data += marker_data
                else:
                    j1, j2 = map(len,[seq_marker_data, marker_data])
                    
                    if j1<j2: 
                        seq_marker_data = [sum(x) for x in zip(seq_marker_data[:j1], marker_data[:j1])]+ marker_data[j1:]
                    if j2<j1: 
                        seq_marker_data = [sum(x) for x in zip(seq_marker_data[:j2], marker_data[:j2])]+ seq_marker_data[j2:]
                    if j1==j2: 
                        seq_marker_data = [sum(x) for x in zip(seq_marker_data, marker_data)]
            
        # Convert the marker offset (used to account for lags in writing the AWG waveform and the STIRAP pulse being sent through the cavity)
        # from us to AWG units. Ensure the total amount added to the waveform is 16*n samples.
#         marker_offset_2 = (np.floor(marker_offset_1/16) + 1) * 16 - marker_offset_1 if marker_offset_1%16 != 0 else 0
#         
#         print marker_offset_1
#         print marker_offset_2
#         
#         waveform_data = [0]*marker_offset_2 + waveform_data + [0]*marker_offset_1
#         marker_data = [marker_waveform_levs[0]]*marker_offset_1 + marker_data + [marker_waveform_levs[0]]*marker_offset_2
#         
#         # This is a big fix. If the first element of the sequence is 1 (i.e. max high level)
#         # then the channel remains high at the end of the sequence. Don't know why...
#         if marker_data[0]==1: marker_data[0]=0
#         if marker_data[-1]==1: marker_data[-1]=0
    
        '''
        Ensure we write the same number of points to each channel.
        '''
        N = max([len(x) for x in seq_waveform_data])
        for x in seq_waveform_data:
            if len(x) < N:
                x += [0]*(N-len(x))

        '''Previously we wrote marker data onto channel2 - we now try to use the marker channels.  However, the
        above work to produce a waveform-ready set of marker data is kept as it is quick and allows us to easily
        revert back to our previous methods (simply uncomment the following line).'''
#         wave_handle, marker_handle = awg.create_custom_adv(waveform_data, marker_data)
                
        l_mark, l_seq = len(seq_marker_data), len(seq_waveform_data[0])
                
        if   l_mark < l_seq : seq_marker_data += [0]*(l_seq - l_mark)
        elif l_seq  < l_mark: seq_marker_data  = seq_marker_data[:l_seq]
                
        awg.configure_arb_wave_trace_mode(WX218x_TraceMode.SINGLE)
        
        '''Configure each channel for its output data.'''
        for channel, rel_offset, data in zip(awg_chs, channel_relative_offsets, seq_waveform_data):
            # Roll channel data to account for relative offsets (e.g. AOM lags)
            print('Rolling {0} forward by {1} points'.format(channel, rel_offset))
            data = np.roll(np.array(data), rel_offset).tolist()
            
            print('Writing {0} points to {1}'.format(len(data),channel))
            awg.set_active_channel(channel)
            awg.create_arbitrary_waveform_custom(data)
        
              
        for channel in awg_chs:
            awg.enable_channel(channel)
            awg.configure_arb_gain(channel, 2)

        '''Quick hack to write marker data to channel 2'''    
#         awg.set_active_channel(Channel.CHANNEL_2)
#         awg.create_arbitrary_waveform_custom(marker_data)
#         awg.enable_channel(Channel.CHANNEL_2)
#         awg.configure_arb_gain(Channel.CHANNEL_2, 2)
            
        marker_starts = [x[0] for x in enumerate(zip([0]+seq_marker_data[:-1],seq_marker_data)) if x[1][0]==0 and x[1][1]>0]
        
        if len(marker_starts) > 2:
            print('ERROR: There are more markers required than can be set currently using the marker channels!')
            marker_starts = marker_starts[:2]

        print('Writing markers to marker channels at {0}'.format(marker_starts))
        marker_channel_index = 1
        for marker_pos in marker_starts:
            awg.configure_marker(awg_chs[0], 
                                 index = marker_channel_index, 
                                 position = marker_pos - marker_wid/4,
                                 levels = marker_levs,
                                 width = marker_wid/2)
            marker_channel_index += 1

        return awg, len(seq_waveform_data[0])/self.awg_config.sample_rate
    
    def __configure_tdc(self):
        tdc = TDC_quTAU()
        print('Connecting to quTAU tdc..')
        tdc.open()
        print('...opened')
        # Maps converted to lists on the line below. New syntax from python 3
        # https://stackoverflow.com/questions/1303347/getting-a-map-to-return-a-list-in-python-3-x
        print('Enabling channels: ', list(self.tdc_config.counter_channels) + [self.tdc_config.marker_channel])
        tdc.set_enabled_channels(list(self.tdc_config.counter_channels) + [self.tdc_config.marker_channel])
        tdc.set_timestamp_buffer_size(self.tdc_config.timestamp_buffer_size)
        # Four our need: the exposure time determines the rate at which data is put into the buffer.
        # Future proofing: if we use the in-built quTAU functions (e.g. histograms) an exposure time
        # of zero may cause issue.
        tdc.set_exposure_time(0)
    
        # Set the tdc to high impedance
        if tdc.get_dev_type() == TDC_DevType.DEVTYPE_1A:
            # Turn 50 Ohm termination off
            print('Device 1A')
            tdc.switch_termination(False)
        elif tdc.get_dev_type() in [TDC_DevType.DEVTYPE_1B, TDC_DevType.DEVTYPE_1C]:
            print('Device 1B/1C')
            print(self.tdc_config.marker_channel)
            tdc.configure_signal_conditioning(self.tdc_config.marker_channel,
                                              TDC_SignalCond.SCOND_MISC,
                                              edge = 1, # rising edge,
                                              term = 0, # Turn 50 Ohm termination off
                                              threshold = 0.5)
            for ch in self.tdc_config.counter_channels:
                tdc.configure_signal_conditioning(ch,
                                              TDC_SignalCond.SCOND_MISC,
                                              edge = 1, # rising edge,
                                              term = 0, # Turn 50 Ohm termination off
                                              threshold = 0.05)
        
#         tdc.enable_tdc_input(True)
        print('tdc configured')
        return tdc
    
    def set_iterations(self, iterations):
        '''
        Sets the iterations.
        '''
        self.iterations = iterations
        
    def set_mot_reload_time(self, reload_time):
        '''
        Sets the MOT reload time. Takes the reload_time in milliseconds.
        '''
        print(f'Setting reload_time to {reload_time}ms')
        self.mot_reload_time = reload_time
        

class MotFluoresceExperiment(GenericExperiment):
    """
    Experimental runner for the mot fluoresce experiment.

    The MOT fluorescence experiment can be run with or without a camera and/or an 
    oscilloscope, but at least one of these is needed to collect data. The experimental 
    steps are as follows:
    1. The MOT is loaded for a set time.
    2. A sequence of DAQ cards is played. This will consist of turning off the MOT and 
     sending a trigger to the oscilloscope or camera, amongst other things.
    3. The oscilloscope is used to read data captured by a photodiode allowing the
     fluorescence of the atoms to be measured, or the image of the MOT is read from the
     camera

    Inputs:
    - daq_controller: The DAQ controller object that controls the DAQ cards.
    - sequence: The sequence object that contains the DAQ card sequence to be played.
    - mot_fluoresce_configuration: The configuration object that contains the MOT
        fluorescence experiment configuration.
    - ic_imaging_control: Optional. The IC Imaging Control object that controls the camera.

    Methods:
    - configure(): Configures the DAQ cards and loads and configures the oscilloscope.
    - run(): Runs the experiment by playing the sequence and collecting data from the
        oscilloscope.
    - close(): Closes the DAQ cards and the oscilloscope.
    """

    def __init__(self, daq_controller:DAQ_controller, sequence:Sequence, 
                mot_fluoresce_configuration:MotFluoresceConfiguration,
                ic_imaging_control:IC_ImagingControl = None):
        
        super().__init__(daq_controller, sequence, mot_fluoresce_configuration)
        # the configuration object is a MotFluoresceConfiguration object and called self.config
        self.mot_fluoresce_config:MotFluoresceConfiguration = self.config
        self.save_location = self.mot_fluoresce_config.save_location
        self.iterations = self.mot_fluoresce_config.iterations
        self.mot_reload = self.mot_fluoresce_config.mot_reload # in ms
        print('MOT reload time (ms)', self.mot_reload)
        self.with_cam = self.mot_fluoresce_config.use_cam
        # self.with_scope = self.mot_fluoresce_config.use_scope

        if self.with_cam:
            if ic_imaging_control is None:
                # if no imaging control object is provided, create a new one
                self.ic_ic = IC_ImagingControl()
                self.external_ic_ic_provided = False
                self.ic_ic.init_library()
            else:
                self.ic_ic = ic_imaging_control
                if not self.ic_ic.initialised:
                    self.ic_ic.init_library()
                self.external_ic_ic_provided = True

            self.cam_gain = self.mot_fluoresce_config.cam_gain
            self.cam_exposure = self.mot_fluoresce_config.cam_exposure
            self.camera_trigger_channel = self.mot_fluoresce_config.camera_trigger_channel
            self.camera_trigger_level = self.mot_fluoresce_config.camera_trigger_level
            self.camera_pulse_width = self.mot_fluoresce_config.camera_pulse_width
            self.save_images = self.mot_fluoresce_config.save_images
        else:
            print("Not using camera for MOT fluorescence experiment.")

        # if self.with_scope:
        #     self.samp_rate = self.mot_fluoresce_config.scope_sample_rate
        #     self.time_range = self.mot_fluoresce_config.scope_time_range
        #     self.centred_0 = self.mot_fluoresce_config.scope_centered_0
        #     self.trig_ch = self.mot_fluoresce_config.scope_trigger_channel
        #     self.trig_lvl = self.mot_fluoresce_config.scope_trigger_level
        #     self.data_chs = self.mot_fluoresce_config.scope_data_channels



    def __configureCamera(self):
        """
        Private method to configure the camera for the MOT fluorescence experiment.
        """
        # open first available camera device
        cam_names = self.ic_ic.get_unique_device_names()
        self.cam:IC_Camera = None
        cam:IC_Camera = None
        self.cam = cam = self.ic_ic.get_device(cam_names[0])
#         self.cam_frame_timeout = int(self.sequences[0].getLength()*10**-3 + (1./self.config.cam_exposure)*10**3)
        #is this in milliseconds? and does it match the MOT reload time? I think so
        self.cam_frame_timeout = 5000
        print('Timeout set to {0}ms'.format(self.cam_frame_timeout))
        print('Opened connection to camera {0}', cam_names[0])
        
        if not cam.is_open():
            cam.open()
            
        # change camera settings
        cam.gain.auto = False
        cam.exposure.auto = False
        cam.gain.value = self.cam_gain
        cam.exposure.value = self.cam_exposure
        formats = cam.list_video_formats()
        cam.set_video_format(formats[0])        # use first available video format
        cam.enable_continuous_mode(True)        # image in continuous mode
        cam.start_live(show_display=False)       # start imaging
                    
        # print cam.is_triggerable()
        cam.enable_trigger(True)              # camera will wait for trigger
        
        if not cam.callback_registered:
            cam.register_frame_ready_callback() # needed to wait for frame ready callback
        
        # Clear out the memory of any rogue image still in there
        try:
            cam.wait_til_frame_ready(self.cam_frame_timeout)
            cam.get_image_data()
        except IC_Exception as err:
            print("Caught IC_Exception with error: {0}".format(err.message))
        cam.reset_frame_ready()


    def configure(self):
        """
        Configures the experiment. This method should called before the experiment is run.
        """
        super().daq_cards_on()
        self.daq_controller.load(self.sequence.getArray())

        # if self.with_scope:
        #     print("connecting to scope")
        #     self.scope = osc.OscilloscopeManager()
        #     self.scope.configure_scope(samp_rate=self.samp_rate, timebase_range=self.time_range,
        #                                centered_0=self.centred_0)
        #     self.scope.configure_trigger(self.trig_ch, self.trig_lvl)


    def measure_signal(self, osc_manager, channels, samp_rate=1e9, timebase_range=4e-6, window=0, centered_0=False, save_file=False):
        """
        Performs measurements using the oscilloscope and returns the acquired signals for each channel as DataFrames.
        """
        data, filename = osc_manager.acquire_with_trigger_multichannel(
            channels, samp_rate=samp_rate, timebase_range=timebase_range,
            save_file=save_file, centered_0=centered_0, window=window
        )

        ch_marker = channels[0]
        ch_pd = channels[1]

        marker = pd.DataFrame()
        pddata = pd.DataFrame()
        marker['Time (s)'] = data['Time (s)']
        pddata['Time (s)'] = data['Time (s)']
        marker['Voltage (V)'] = data[f'Channel {ch_marker} Voltage (V)']
        pddata['Voltage (V)'] = data[f'Channel {ch_pd} Voltage (V)']

        return marker, pddata, filename


    def __run_with_scope(self):
        """
        Private method to run the experiment with a scope.
        """
        self.daq_controller.load(self.sequence.getArray())
        self.daq_controller.writeChannelValues()
        i = 1
        all_readouts = []

        while i <= self.config.iterations:
            print(f"Iteration {i}")
            print(f"loading mot for {self.config.mot_reload}ms")
            sleep(self.config.mot_reload*10**-3) # convert from ms to s

            #self.scope.set_to_digitize()
            print("playing sequence")
            self.daq_controller.play(float(self.sequence.t_step), clearCards=False)
        
            print("writing channel values")
            self.daq_controller.writeChannelValues()
            
            print("collecting data")
            osc_manager=osc.oscilloscope_manager()
            data_marker, data_pd, filename = self.measure_signal(osc_manager, [1,3], samp_rate=5e6, timebase_range=5e-3, centered_0=False, save_file=False)

            print("processing data")
            # window: time range we want to save after detecting imaging trigger
            processor = MotFluoresceDataProcessor(data_marker, data_pd, window=500e-6, save_path=f'all_readouts_{i}.feather')
            processor.process_and_save()
            processor.extract_readouts()
            all_readouts.append(processor.df_all)

            print(f"data saved to {filename}")
            i += 1

        print("end of the loop")
        print("processing all data")
        mean_df = processor.compute_mean_across_iterations(all_readouts)
        mean_df.to_csv('mean_readouts_across_iterations.csv', index=False)


    def __run_with_cam(self):
        try:# needs to be in a try except. If the camera isn't closed the computer will crash
            self.__configureCamera()
            img_arrs = []
            i = 1
            while i <= self.config.iterations:
                print(f"Iteration {i}")
                print(f"loading mot for {self.config.mot_reload}ms")
                sleep(self.config.mot_reload*10**-3) # convert from ms to s

                print("playing sequence")
                self.daq_controller.play(float(self.sequence.t_step), clearCards=False)

                # Grab image and save as bmp
                self.cam.wait_til_frame_ready(self.cam_frame_timeout)    
                data = self.cam.get_image_data()
                img = Image.frombuffer('RGB', (data[1], data[2]), data[0], 'raw', 'RGB',\
                                       0,1).convert('L').transpose(Image.FLIP_TOP_BOTTOM)
                img_arrs.append(np.array(img))

                # Save the images
                current_date = datetime.now().strftime("%Y-%m-%d")
                current_time = datetime.now().strftime("%H-%M-%S")
                directory = os.path.join(self.save_location, current_date)
                os.makedirs(directory, exist_ok=True) 

                if self.save_images:
                    # Creates full file name including time and parent folders
                    full_name = os.path.join(directory, current_time)
                    img.save(f"{full_name}.bmp", "bmp")

                self.cam.reset_frame_ready()
                print("writing channel values")
                self.daq_controller.writeChannelValues()

                i += 1
        finally:
            self.close()


    def run(self):
        """
        Runs the experiment. This is called by the run_in_thread method.
        """
        if not self.with_cam and self.with_scope:
            self.__run_with_scope()
        elif self.with_cam and not self.with_scope:
            self.__run_with_cam()
        else:
            raise ValueError("Either a camera or a scope must be used, but not both.")


    def close(self):
        '''
        Perform any tidying up.
        '''
        if self.with_cam:
            print('closing camera...')
            self.cam.enable_trigger(False)
            self.cam.stop_live()
            self.cam.close()
        
            if not self.external_ic_ic_provided:
                self.ic_ic.close_library()
            print('...closed')

        self.daq_controller.clearCards()
        self.scope.quit()
        super().daq_cards_off()


class MotFluoresceDataProcessor(object):
    """
    Processes oscilloscope data: detects triggers in the marker channel,
    extracts PD signal after each trigger, and saves all readouts in binary format.
    """
    def __init__(self, data_marker, data_pd, window=500e-6, save_path='all_readouts.feather'):
        """
        Initialize with marker and PD data, window size, and save path.
        """
        self.data_marker = data_marker
        self.data_pd = data_pd
        self.window = window
        self.save_path = save_path
        self.df_all = None  # Will hold all readouts

    def detect_triggers(self):
        """
        Detect all falling edges (triggers) in the marker channel.
        """
        signal = self.data_marker['Voltage (V)'].values
        trigger_val = (signal.max() + signal.min()) / 2
        trigger_idxs = np.where((signal[:-1] > trigger_val) & (signal[1:] < trigger_val))[0] + 1
        trigger_times = self.data_marker['Time (s)'].iloc[trigger_idxs].values
        return trigger_times

    def extract_readouts(self):
        """
        Extract PD signal after each trigger and store all readouts in a DataFrame.
        """
        trigger_times = self.detect_triggers()
        time_array = self.data_pd['Time (s)'].values
        pd_array = self.data_pd['Voltage (V)'].values

        readout = []
        for t0 in trigger_times:
            mask = (time_array >= t0) & (time_array <= t0 + self.window)
            pd_time = time_array[mask] - t0  # relative time to trigger
            pd_signal = pd_array[mask]
            df_temp = pd.DataFrame({'Time (s)': pd_time, 'PD Signal': pd_signal, 'Trigger Time': t0})
            readout.append(df_temp)

        self.df_all = pd.concat(readout, ignore_index=True)

    def save_readouts(self):
        """
        Save the DataFrame with all readouts in feather format.
        """
        if self.df_all is not None:
            # self.df_all.reset_index(drop=True).to_feather(self.save_path)
            self.df_all.to_csv(self.save_path.replace('.feather', '.csv'), index=False)
            print(f"Readouts saved to {self.save_path}")
        else:
            print("No data to save. Run extract_readouts() first.")

    def process_and_save(self):
        """
        Run the full processing and save the result.
        """
        self.extract_readouts()
        self.save_readouts()

    def compute_mean_across_iterations(self, all_readouts):
        """
        Given a list of df_all DataFrames (one per iteration), compute the mean PD Signal
        for each readout index (Trigger Time) across all iterations.
        Returns a DataFrame with columns: 'Time (s)', 'Mean PD Signal', 'Readout Index'
        """
        import pandas as pd

        # Group each iteration's df_all by 'Trigger Time'
        grouped_per_iter = [df.groupby('Trigger Time') for df in all_readouts]

        # Get the sorted list of unique trigger times (readout indices) from the first iteration
        readout_ids = sorted(all_readouts[0]['Trigger Time'].unique())

        mean_readouts = []

        for rid in readout_ids:
            # For each readout index, collect the DataFrame for this readout from every iteration
            readout_list = [g.get_group(rid).set_index('Time (s)')['PD Signal'] for g in grouped_per_iter]
            # Concatenate along columns (axis=1)
            readout_matrix = pd.concat(readout_list, axis=1)
            # Compute the mean row-wise (across iterations)
            mean_signal = readout_matrix.mean(axis=1)
            mean_df = pd.DataFrame({'Time (s)': mean_signal.index, 'Mean PD Signal': mean_signal.values, 'Readout Index': rid})
            mean_readouts.append(mean_df)

        # Concatenate all mean readouts into a single DataFrame
        mean_readouts_df = pd.concat(mean_readouts, ignore_index=True)
        return mean_readouts_df


class PhotonProductionDataSaver(object):
    '''
    This object takes raw data from the TDC, parses it into our desired format and saves
    it to file.  It also enables all of the parsing/saving to be done in a separate thread
    so as to not hold up the experiment.
    '''
    def __init__(self, tdc_timebase, tdc_marker_channel, save_location, data_queue=None, create_log=False):
        '''
        Initialise the object with the information it will need for saving.
        '''
        self.tdc_timebase = tdc_timebase
        self.tdc_marker_channel = tdc_marker_channel
        
        self.experiment_time = time.strftime("%H-%M-%S")
        self.experiment_date = time.strftime("%y-%m-%d")
        
        self.save_location = os.path.join(save_location, self.experiment_date, self.experiment_time)
        self.save_location_raw = os.path.join(self.save_location, 'raw')
        
        if not os.path.exists(self.save_location):
            os.makedirs(self.save_location)
        if not os.path.exists(self.save_location_raw):
            os.makedirs(self.save_location_raw)
            
        if create_log:
            self.log_file = os.path.join(self.save_location, 'log.txt')
            print('Log file at {0}'.format(self.log_file))
        else:
            self.log_file = None
                
        self.data_queue = data_queue
            
        self.threads = []
        
    def save_in_thread(self,timestamps, channels, valid, throw_number):
        '''
        Save the data in a new thread.
        '''
        thread = threading.Thread(name='Throw {0} save'.format(throw_number),
                                  target=self.__save,
                                  args=(timestamps, channels, valid, throw_number))
        thread.start()
        self.threads.append(thread)
        return thread
    
    def log_in_thread(self, log_input, throw_number):
        thread = threading.Thread(name='Throw {0} save'.format(throw_number),
                                  target=self.__log,
                                  args=(log_input, throw_number))
        thread.start()
        self.threads.append(thread)
        return thread
        
    def combine_saves(self):
        '''
        Combine all the individual files from each MOT throw into a single file.
        '''
        t = 0
        # Check only the main thread is still running i.e. nothing is stills saving.
        while True in [thread.is_alive() for thread in self.threads]:
            time.sleep(1)
            t+=1
            if t>60:
                print("Timed-out waiting for save-threads to finish. Abandoning combine_saves().")
                return
            
        combined_file = open(os.path.join(self.save_location, self.experiment_time + '.txt'), 'w')
        for fname in os.listdir(self.save_location_raw):
            if fname.endswith(".txt"):
                data_file = open(os.path.join(self.save_location_raw, fname))
                combined_file.write(data_file.read())
                data_file.close()
                combined_file.write('nan,nan,nan,nan\n') # Batman!
        combined_file.close()
        
    def __save(self, timestamps, channels, valid, throw_number):
        '''
        Save the data returned from the TDC.
        '''
        print('__save iter', throw_number)
        
        t = time.time()
#         print 'Num markers:', channels.tolist().count(self.tdc_marker_channel)
        try:
#             marker_index = channels.tolist().index(self.tdc_marker_channel)
            first_marker_index = next(i for i,elm in enumerate(channels.tolist()) if elm==self.tdc_marker_channel)
            last_marker_index  = next(len(channels)-1-i for i,elm in enumerate(reversed(channels.tolist())) if elm==self.tdc_marker_channel)
        except ValueError as err:
            print("__save(throw={0}) Nothing measured on marker channel - so nothing to save.".format(throw_number))
            print(err)
            return
        print('__save: found first marker index ({0} sec)'.format(time.time()-t))
        x_0 = timestamps[first_marker_index]
#         t_mot_0 = x_0*self.tdc_timebase
#         timestamps = [(x-x_0)*self.tdc_timebase for x in timestamps[marker_index+1:] if x >= 0]
#         channels = [x for x in channels[marker_index+1:] if x >= 0]
        
        # This step essentially does two things:
        #     1. Converts all the timestamps to picoseconds by *tdc_timebase.
        #     2. Makes t=0 be the time of the initial marker pulse, i.e. writes
        #        all timestamps in terms of the so-called mot-time.
        t = time.time()
        timestamps = [(x-x_0)*self.tdc_timebase for x in timestamps[first_marker_index:last_marker_index+1]]
        channels = channels[first_marker_index:last_marker_index+1]
#         print 'create temp data dump'
#         t = open(os.path.join(self.save_location, 'temp.txt'), 'w')
#         for line in zip(timestamps, channels):
#             t.write('{0},{1}\n'.format(*line))
#         t.close()
#         
        print('__save: selected valid timestamps and channels')
        t = time.time()
        pulse_number = 0
        data = []
        data_buffer = []
#         sti_lens = []

        try:
            for t, ch in zip(timestamps, channels):
                if ch == self.tdc_marker_channel:
#                     print 'MARKER', t, ch
                    t_stirap_0 = t
                    pulse_number += 1
                    data += data_buffer
    #                 sti_lens.append(len(data_buffer))
                    data_buffer = []
                else:
#                     print 'DATA', t, ch
                    data_buffer.append((ch, t-t_stirap_0, t, pulse_number))
        except _tkinter.TclError as err:
            print(t, ch)
            raise err
        
        if pulse_number > 25000:
            print('__save: Too many pulses recorded ({0}) - returning.'.format(pulse_number))
            return
        
#         print len(data), sti_lens
        print('__save: creating file')
        f = open(os.path.join(self.save_location_raw, '{0}.txt'.format(throw_number)), 'w')
        print('__save: writing file')
        for line in data:
            f.write('{0},{1},{2},{3}\n'.format(*line))
        print('__save: closing file')
        f.close()
        
        # If a push data function is configured, throw it now
        if self.data_queue:
            print('Queuing  data')
            self.data_queue.put((throw_number, data))
        
        print('iter {0}: counts {1}, pulses recorded {2}'.format(throw_number, len(data), pulse_number))

    def __log(self, log_input, throw_number):
        if self.log_file != None:
            print('__log: writing to log')
            
            if callable(log_input):
                log_input = log_input()
            if type(log_input) in [list, tuple]:
                def flatten(l):
                    for el in l:
                        if isinstance(el, collections.Iterable) and not isinstance(el, str):# replaced basestring with str as an update from python 2 python 3. See
                            # https://stackoverflow.com/questions/60743762/basestring-equivalent-in-python3-str-and-string-types-from-future-and-six-not
                            for sub in flatten(el):
                                yield sub
                        else:
                            yield el
                log_input = ' '.join([str(x) for x in flatten([x() if callable(x) else x for x in log_input])])

            f = open(self.log_file, 'w')
            f.write('Throw {0}: {1}\n'.format(throw_number, log_input))
            f.close()
            print('__log: closed log file')
        else:
            print('__log: Can not write. No log file exists.')


class MOTFluorescenceDataSaver(object):
    '''
    This object takes raw data from the PD Measuring MOT fluorescnece, parses it into our desired format and saves
    it to file.  It also enables all of the parsing/saving to be done in a separate thread
    so as to not hold up the experiment.
    '''
    def __init__(self, scope_timebase, scope_marker_channel, save_location, data_queue=None, create_log=False):
        '''
        Initialise the object with the information it will need for saving.
        '''
        self.scope_timebase = scope_timebase
        self.scope_marker_channel = scope_marker_channel
        
        self.experiment_time = time.strftime("%H-%M-%S")
        self.experiment_date = time.strftime("%y-%m-%d")
        
        self.save_location = os.path.join(save_location, self.experiment_date, self.experiment_time)
        self.save_location_raw = os.path.join(self.save_location, 'raw')
        
        if not os.path.exists(self.save_location):
            os.makedirs(self.save_location)
        if not os.path.exists(self.save_location_raw):
            os.makedirs(self.save_location_raw)
            
        if create_log:
            self.log_file = os.path.join(self.save_location, 'log.txt')
            print('Log file at {0}'.format(self.log_file))
        else:
            self.log_file = None
                
        self.data_queue = data_queue
            
        self.threads = []
        
    def save_in_thread(self,timestamps, channels, valid, throw_number):
        '''
        Save the data in a new thread.
        '''
        thread = threading.Thread(name='Throw {0} save'.format(throw_number),
                                  target=self.__save,
                                  args=(timestamps, channels, valid, throw_number))
        thread.start()
        self.threads.append(thread)
        return thread
    
    def log_in_thread(self, log_input, throw_number):
        thread = threading.Thread(name='Throw {0} save'.format(throw_number),
                                  target=self.__log,
                                  args=(log_input, throw_number))
        thread.start()
        self.threads.append(thread)
        return thread
        
    def combine_saves(self):
        '''
        Combine all the individual files from each MOT throw into a single file.
        '''
        t = 0
        # Check only the main thread is still running i.e. nothing is stills saving.
        while True in [thread.is_alive() for thread in self.threads]:
            time.sleep(1)
            t+=1
            if t>60:
                print("Timed-out waiting for save-threads to finish. Abandoning combine_saves().")
                return
            
        combined_file = open(os.path.join(self.save_location, self.experiment_time + '.txt'), 'w')
        for fname in os.listdir(self.save_location_raw):
            if fname.endswith(".txt"):
                data_file = open(os.path.join(self.save_location_raw, fname))
                combined_file.write(data_file.read())
                data_file.close()
                combined_file.write('nan,nan,nan,nan\n') # Batman!
        combined_file.close()
        
    def __save(self, timestamps, channels, valid, throw_number):
        '''
        Save the data returned from the TDC.
        '''
        print('__save iter', throw_number)
        
        t = time.time()
#         print 'Num markers:', channels.tolist().count(self.tdc_marker_channel)
        try:
#             marker_index = channels.tolist().index(self.tdc_marker_channel)
            first_marker_index = next(i for i,elm in enumerate(channels.tolist()) if elm==self.tdc_marker_channel)
            last_marker_index  = next(len(channels)-1-i for i,elm in enumerate(reversed(channels.tolist())) if elm==self.tdc_marker_channel)
        except ValueError as err:
            print("__save(throw={0}) Nothing measured on marker channel - so nothing to save.".format(throw_number))
            print(err)
            return
        print('__save: found first marker index ({0} sec)'.format(time.time()-t))
        x_0 = timestamps[first_marker_index]
#         t_mot_0 = x_0*self.tdc_timebase
#         timestamps = [(x-x_0)*self.tdc_timebase for x in timestamps[marker_index+1:] if x >= 0]
#         channels = [x for x in channels[marker_index+1:] if x >= 0]
        
        # This step essentially does two things:
        #     1. Converts all the timestamps to picoseconds by *tdc_timebase.
        #     2. Makes t=0 be the time of the initial marker pulse, i.e. writes
        #        all timestamps in terms of the so-called mot-time.
        t = time.time()
        timestamps = [(x-x_0)*self.tdc_timebase for x in timestamps[first_marker_index:last_marker_index+1]]
        channels = channels[first_marker_index:last_marker_index+1]
#         print 'create temp data dump'
#         t = open(os.path.join(self.save_location, 'temp.txt'), 'w')
#         for line in zip(timestamps, channels):
#             t.write('{0},{1}\n'.format(*line))
#         t.close()
#         
        print('__save: selected valid timestamps and channels')
        t = time.time()
        pulse_number = 0
        data = []
        data_buffer = []
#         sti_lens = []

        try:
            for t, ch in zip(timestamps, channels):
                if ch == self.tdc_marker_channel:
#                     print 'MARKER', t, ch
                    t_stirap_0 = t
                    pulse_number += 1
                    data += data_buffer
    #                 sti_lens.append(len(data_buffer))
                    data_buffer = []
                else:
#                     print 'DATA', t, ch
                    data_buffer.append((ch, t-t_stirap_0, t, pulse_number))
        except _tkinter.TclError as err:
            print(t, ch)
            raise err
        
        if pulse_number > 25000:
            print('__save: Too many pulses recorded ({0}) - returning.'.format(pulse_number))
            return
        
#         print len(data), sti_lens
        print('__save: creating file')
        f = open(os.path.join(self.save_location_raw, '{0}.txt'.format(throw_number)), 'w')
        print('__save: writing file')
        for line in data:
            f.write('{0},{1},{2},{3}\n'.format(*line))
        print('__save: closing file')
        f.close()
        
        # If a push data function is configured, throw it now
        if self.data_queue:
            print('Queuing  data')
            self.data_queue.put((throw_number, data))
        
        print('iter {0}: counts {1}, pulses recorded {2}'.format(throw_number, len(data), pulse_number))

    def __log(self, log_input, throw_number):
        if self.log_file != None:
            print('__log: writing to log')
            
            if callable(log_input):
                log_input = log_input()
            if type(log_input) in [list, tuple]:
                def flatten(l):
                    for el in l:
                        if isinstance(el, collections.Iterable) and not isinstance(el, str):# replaced basestring with str as an update from python 2 python 3. See
                            # https://stackoverflow.com/questions/60743762/basestring-equivalent-in-python3-str-and-string-types-from-future-and-six-not
                            for sub in flatten(el):
                                yield sub
                        else:
                            yield el
                log_input = ' '.join([str(x) for x in flatten([x() if callable(x) else x for x in log_input])])

            f = open(self.log_file, 'w')
            f.write('Throw {0}: {1}\n'.format(throw_number, log_input))
            f.close()
            print('__log: closed log file')
        else:
            print('__log: Can not write. No log file exists.')


"""
class PhotonProductionConfiguration(object):
    
    def __init__(self,
                 save_location,
                 mot_reload,
                 iterations,
                 waveform_sequence,
                 waveforms,
                 interleave_waveforms,
                 waveform_stitch_delays,
                 awg_configuration,
                 tdc_configuration):
        self.save_location = save_location
        self.mot_reload = mot_reload
        self.iterations = iterations

        self.waveform_sequence = waveform_sequence
        self.waveforms:List[Waveform] = waveforms
        self.interleave_waveforms:bool = interleave_waveforms
        self.waveform_stitch_delays = waveform_stitch_delays
        
        self.awg_configuration: AwgConfiguration = awg_configuration
        self.tdc_configuration: TdcConfiguration = tdc_configuration

    def get_waveform_sequence(self):
        return self.__waveform_sequence

    def set_waveform_sequence(self, value):
        print('Setting waveform sequence to', value, [type(x) for x in value])
        self.__waveform_sequence = value

    def del_waveform_sequence(self):
        del self.__waveform_sequence

    def get_save_location(self):
        return self.__save_location

    def get_mot_reload(self):
        return self.__mot_reload

    def get_iterations(self):
        return self.__iterations

    def get_awg_configuration(self):
        return self.__awg_configuration

    def get_tdc_configuration(self):
        return self.__tdc_configuration

    def set_save_location(self, value):
        self.__save_location = value

    def set_mot_reload(self, value):
        self.__mot_reload = value

    def set_iterations(self, value):
        self.__iterations = value

    def set_awg_configuration(self, value):
        self.__awg_configuration = value

    def set_tdc_configuration(self, value):
        self.__tdc_configuration = value

    def del_save_location(self):
        del self.__save_location

    def del_mot_reload(self):
        del self.__mot_reload

    def del_iterations(self):
        del self.__iterations

    def del_awg_configuration(self):
        del self.__awg_configuration

    def del_tdc_configuration(self):
        del self.__tdc_configuration

    save_location = property(get_save_location, set_save_location, del_save_location, "save_location's docstring")
    mot_reload = property(get_mot_reload, set_mot_reload, del_mot_reload, "mot_reload's docstring")
    iterations = property(get_iterations, set_iterations, del_iterations, "iterations's docstring")
    awg_configuration = property(get_awg_configuration, set_awg_configuration, del_awg_configuration, "awg_configuration's docstring")
    tdc_configuration = property(get_tdc_configuration, set_tdc_configuration, del_tdc_configuration, "tdc_configuration's docstring")
    waveform_sequence = property(get_waveform_sequence, set_waveform_sequence, del_waveform_sequence, "waveform_sequence's docstring")
#"""



class ExperimentalAutomationRunner(object):
     
    def __init__(self, daq_controller:DAQ_controller, experimental_automation_configuration:ExperimentSessionConfig , photon_production_configuration:PhotonProductionConfiguration):
         
        self.daq_controller = daq_controller
        self.experimental_automation_configuration:ExperimentSessionConfig  = None
        c:ExperimentSessionConfig  = None
        self.experimental_automation_configuration = c = experimental_automation_configuration
        self.photon_production_configuration = photon_production_configuration
         
        self.experiements_to_run = len(c.automated_experiment_configurations)
        self.experiements_iter = 0
         
        self.experiment_time = time.strftime("%H-%M-%S")
        self.experiment_date = time.strftime("%y-%m-%d")
         
        summary_location = os.path.join(c.save_location, self.experiment_date) 
        if not os.path.exists(summary_location):
            os.makedirs(summary_location) 
         
        self.summary_fname = os.path.join(summary_location, '{0}.txt'.format(c.summary_fname))
         
        self.original_daq_channel_values = daq_controller.getChannelValues()
        
        if not self.daq_controller.continuousOutput:
            print("DAQ output must be on to run an experiement - turning it on.")
            self.daq_controller.toggleContinuousOutput()
         
    def get_next_experiment(self):
         
        print('Configuring experiment {0} of {1}'.format(self.experiements_iter+1, self.experiements_to_run))
         
        config:SingleExperimentConfig = self.experimental_automation_configuration.automated_experiment_configurations[self.experiements_iter]
        self.experiements_to_run = len(self.experimental_automation_configuration.automated_experiment_configurations)
        self.experiements_iter += 1
         
        # Update MOT reload, iterations and, if necessary, the modulation frequencies.
        self.photon_production_configuration.iterations = config.iterations
        self.photon_production_configuration.mot_reload = config.mot_reload
        
        if config.modulation_frequencies != []:
            j=0
            waveform:Waveform = None
            for waveform in [self.photon_production_configuration.waveforms[i] for i in self.photon_production_configuration.waveform_sequence[0]]:
                try:
                    waveform.mod_frequency = config.modulation_frequencies[j]
                    print('Set freq', config.modulation_frequencies[j])
                    j+=1
                except IndexError:
                    #If the modulation frequency 'j' is not specified, stop iterating.
                    break
            
        # Reset the daq channels to their original values - this way channel value changes for
        # one experimental run don't persist for others.  Don't bother resetting any channel
        # that is going to be set from the original value for the next experiment though.
        self._reset_daq_channel_static_values([x[0] for x in config.daq_channel_static_values])
         
        # update static DAQ values
        for channel_number, value in config.daq_channel_static_values:
            self._update_daq_channel_static_values(channel_number, value)
         
        # Return photon experiment
        return (PhotonProductionExperiment(daq_controller=self.daq_controller,
                                          sequence=config.sequence,
                                          photon_production_configuration=self.photon_production_configuration),
                config.sequence_fname,
                config.modulation_frequencies)
     
    def write_to_summary_file(self, text):        
#         if not os.path.exists(self.summary_fname):
#             os.path.join(self.summary_location, 'log.txt')
        f = open(self.summary_fname, 'a+')
        f.write(text)
        f.close()
    
    def get_total_iterations(self):
        return sum([x.iterations for x in self.experimental_automation_configuration.automated_experiment_configurations])
         
    def has_next_experiment(self):
        return self.experiements_iter < self.experiements_to_run
         
    def _update_daq_channel_static_values(self, channel_number, new_val):
         
        try:
            channel:DAQ_channel = next(ch for ch in self.daq_controller.getChannels() if ch.chNum==channel_number)
        except StopIteration:
            print('Channel {0} not found, ignoring this channel.'.format(channel_number))
            return
         
        start_val = self.daq_controller.channelValues[channel_number]
        if channel.isCalibrated:
            start_val = channel.calibrationFromVFunc(start_val)
         
        print('Updating channel {0} from {1} to {2}'.format(channel_number, start_val, float(new_val)),)

        for val in np.linspace(start_val, new_val, self.experimental_automation_configuration.daq_channel_update_steps):
            self.daq_controller.updateChannelValue(channel.chNum,
                                                   val if not channel.isCalibrated else channel.calibrationToVFunc(val)) 
            time.sleep(self.experimental_automation_configuration.daq_channel_update_delay)
            print('.'),
        print('channel {0} update.'.format(channel_number))
     
    def _reset_daq_channel_static_values(self, channels_to_ignore=[]):
        
#         bool_mask = np.array([True if x not in channels_to_ignore else False for x in range(len(self.original_daq_channel_values))])
#         for orig_val, curr_val in zip(self.original_daq_channel_values[bool_mask], self.daq_controller.getChannelValues[bool_mask]):
#         
        channel_values_to_reset = []
        current_daq_channel_values = self.daq_controller.getChannelValues()

        for chNum in range(len(self.original_daq_channel_values)):
            
            if chNum not in channels_to_ignore:
                # If the current daq value is not the original reset it back.
                if self.original_daq_channel_values[chNum] != current_daq_channel_values[chNum]:
                    
                    try:
                        channel:DAQ_channel = next(ch for ch in self.daq_controller.getChannels() if ch.chNum==chNum)
                    except StopIteration:
                        print('Channel {0} not found, ignoring this channel.'.format(chNum))
                        return
                                
                    orig_val = self.original_daq_channel_values[chNum]
                    channel_values_to_reset.append((chNum, orig_val if not channel.isCalibrated else channel.calibrationFromVFunc(orig_val)))
        
        print('Resetting DAQ channels {0}'.format([x[0] for x in channel_values_to_reset]))
        for args in channel_values_to_reset:
            self._update_daq_channel_static_values(*args)
    
    def close(self):
        self._reset_daq_channel_static_values()
             


       
"""
class Waveform(object):
    
    def __init__(self, fname, mod_frequency, phases):
        self.fname = fname
        self.mod_frequency = mod_frequency
        self.phases = sorted(phases, key=lambda x:x[1]) # Sort phases into the order that they need to be applied.
        self.data = self.__load_data()
        
    def __load_data(self):
        with open(self.fname, 'rt') as csvfile:
            print('Loading waveform:', self.fname)
            reader = csv.reader(csvfile, delimiter=',')
            data = []
            for row in reader:
                if len(row) > 1:
                    data += list(map(float, row))
                else:
                    data += float(row[0])
        
        return data
        
    def get(self, sample_rate, calibration_function = lambda level: level, constant_voltage=False, double_pass=True):
        # Copy data to new object. We do not want to mutate original data.
        mod_data = [calibration_function(x) for x in self.data]
        if constant_voltage:
            return mod_data
        t_step = 2 * np.pi / sample_rate
        phi = 0.0
        if double_pass:
            # Divided phases by two for double passed AOM.
            phases = [(x[0] / 2, x[1]) for x in self.phases]  # Using list comprehension for clarity
        next_phi, next_i_flip = (None, None) if not phases else phases.pop(0)
        for i in range(len(mod_data)):
            # i here is the index of the phase list
            if i == next_i_flip:
                phi = next_phi
                next_phi, next_i_flip = (None, None) if not phases else phases.pop(0)
            mod_data[i] = mod_data[i] * np.sin(i * t_step * self.mod_frequency + phi)
        return mod_data


    def get_marker_data(self, marker_positions=[], marker_levels=(0,1), marker_width=50, n_pad_right=0, n_pad_left=0):
        '''
        Returns a marker waveform.
        '''
        
        # Use a np array for ease of setting array slices to contant values.
        data = np.array( [marker_levels[0]] * (n_pad_left + len(self.data) + n_pad_right))
        for pos in marker_positions:
            pos = int(pos)  # Ensure pos is an integer
            data[pos:pos+int(marker_width)] = marker_levels[1]
        # This is a big fix. If the first element of the sequence is 1 (i.e. max high level)
        # then the channel remains high at the end of the sequence. Don't know why...
        if data[0]==1:
            data[0]=0
        # Convert np array to list for consitancy with the get() method.
        return data.tolist()

    def get_profile(self):
        return self.data

    def get_n_samples(self):
        return len(self.data)

    def get_t_length(self, sample_rate):
        return len(self.data)*sample_rate

    def get_fname(self):
        return self.__fname

    def get_mod_frequency(self):
        return self.__mod_frequency

    def get_phases(self):
        return self.__phases

    def set_fname(self, value):
        self.__fname = value
        self.data = self.__load_data()

    def set_mod_frequency(self, value):
        self.__mod_frequency = value

    def set_phases(self, value):
        self.__phases = sorted(value, key=lambda x:x[1])

    fname = property(get_fname, set_fname, None, None)
    mod_frequency = property(get_mod_frequency, set_mod_frequency, None, None)
    phases = property(get_phases, set_phases, None, None)
#"""


class Waveform:
    def __init__(self, fname: str, mod_frequency: float, phases: List[Tuple[float, int]]):
        self.__fname = fname
        self.__mod_frequency = mod_frequency
        self.__phases = sorted(phases, key=lambda x: x[1])  # Sort by index
        self.data = self.__load_data()

    def __load_data(self) -> List[float]:
        """Loads waveform data from a CSV file."""
        with open(self.__fname, 'rt') as csvfile:
            print('Loading waveform:', self.__fname)
            reader = csv.reader(csvfile, delimiter=',')
            data = []
            for row in reader:
                if len(row) > 1:
                    data += list(map(float, row))
                else:
                    data += float(row[0])
        return data

    def get(self, sample_rate: float, calibration_function=lambda level: level,
            constant_voltage=False, double_pass=True) -> List[float]:
        """
        Returns the modulated waveform data.

        - Applies the calibration function.
        - If constant_voltage is False, applies sinusoidal modulation.
        """
        mod_data = [calibration_function(x) for x in self.data]
        if constant_voltage:
            return mod_data

        t_step = 2 * np.pi / sample_rate
        phi = 0.0
        if double_pass:
            # Divided phases by two for double passed AOM.
            phases = [(x[0] / 2 if double_pass else x[0], x[1]) for x in self.__phases]
        next_phi, next_i_flip = (None, None) if not phases else phases.pop(0)

        for i in range(len(mod_data)):
            if i == next_i_flip:
                phi = next_phi
                next_phi, next_i_flip = (None, None) if not phases else phases.pop(0)
            mod_data[i] *= np.sin(i * t_step * self.__mod_frequency + phi)

        return mod_data

    def get_marker_data(self, marker_positions=[], marker_levels=(0, 1),
                        marker_width=50, n_pad_right=0, n_pad_left=0) -> List[int]:
        """
        Returns a marker waveform.

        Pads with zeros on both sides, and marks selected positions with high levels.
        """
        data = np.array([marker_levels[0]] * (n_pad_left + len(self.data) + n_pad_right))
        for pos in marker_positions:
            pos = int(pos)
            data[pos:pos + int(marker_width)] = marker_levels[1]

        # Fix for high-start issue
        if data[0] == 1:
            data[0] = 0

        return data.tolist()

    def get_profile(self) -> List[float]:
        """Returns the raw waveform data."""
        return self.data

    def get_n_samples(self) -> int:
        """Returns the number of samples in the waveform."""
        return len(self.data)

    def get_t_length(self, sample_rate: float) -> float:
        """Returns the duration of the waveform at a given sample rate."""
        return len(self.data) * sample_rate
    
    def set_mod_frequency(self, value: float):
        """Sets the modulation frequency."""
        self.__mod_frequency = value

    # --- Properties ---

    @property
    def fname(self) -> str:
        return self.__fname
    @fname.setter
    def fname(self, value: str):
        self.__fname = value
        self.data = self.__load_data()
    
    @property
    def mod_frequency(self) -> float:
        return self.__mod_frequency
    @mod_frequency.setter
    def mod_frequency(self, value: float):
        self.__mod_frequency = value

    @property
    def phases(self) -> List[Tuple[float, int]]:
        return self.__phases
    @phases.setter
    def phases(self, value: List[Tuple[float, int]]):
        self.__phases = sorted(value, key=lambda x: x[1])

"""
class AwgConfiguration(object):
    def __init__(self, 
                 sample_rate,
                 burst_count,
                 waveform_output_channels,
                 waveform_output_channel_lags,
                 marked_channels,
                 marker_width,
                 waveform_aom_calibrations_locations):
        self.sample_rate = sample_rate
        self.burst_count = burst_count
        self.waveform_output_channels = waveform_output_channels
        self.waveform_output_channel_lags = waveform_output_channel_lags
        self.marked_channels = marked_channels
        self.marker_width = marker_width
        self.waveform_aom_calibrations_locations = waveform_aom_calibrations_locations

    def get_sample_rate(self):
        return self.__sample_rate

    def get_burst_count(self):
        return self.__burst_count

    def get_waveform_output_channels(self):
        return self.__waveform_output_channels

    def set_sample_rate(self, value):
        self.__sample_rate = value

    def set_burst_count(self, value):
        self.__burst_count = value

    def set_waveform_output_channels(self, value):
        self.__waveform_output_channels = value

    def del_sample_rate(self):
        del self.__sample_rate

    def del_burst_count(self):
        del self.__burst_count

    def del_waveform_output_channels(self):
        del self.__waveform_output_channels

    sample_rate = property(get_sample_rate, set_sample_rate, del_sample_rate, "sample_rate's docstring")
    burst_count = property(get_burst_count, set_burst_count, del_burst_count, "burst_count's docstring")
    waveform_output_channels = property(get_waveform_output_channels, set_waveform_output_channels, del_waveform_output_channels, "waveform_output_channels' docstring")
#"""


class AwgConfiguration:
    """
    Configuration for an Arbitrary Waveform Generator (AWG), including sample rate,
    output channels, timing lags, marker widths, and calibration locations.
    """
    def __init__(self,
                 sample_rate: float,
                 burst_count: int,
                 waveform_output_channels: List[int],
                 waveform_output_channel_lags: List[float],
                 marked_channels: List[int],
                 marker_width: int,
                 waveform_aom_calibrations_locations: Dict[int, Any]):
        self._sample_rate = sample_rate
        self._burst_count = burst_count
        self._waveform_output_channels = waveform_output_channels

        self.waveform_output_channel_lags = waveform_output_channel_lags
        self.marked_channels = marked_channels
        self.marker_width = marker_width
        self.waveform_aom_calibrations_locations = waveform_aom_calibrations_locations


    sample_rate = make_property('_sample_rate')
    burst_count = make_property('_burst_count')
    waveform_output_channels = make_property('_waveform_output_channels')

    def set_burst_count(self, value: int):
        self._burst_count = value

    def set_sample_rate(self, value: float):
        self._sample_rate = value


"""
class TdcConfiguration(object):
    def __init__(self,
                 counter_channels,
                 marker_channel,
                 timestamp_buffer_size):
        self.counter_channels = counter_channels
        self.marker_channel = marker_channel
        self.timestamp_buffer_size = timestamp_buffer_size

    def get_counter_channels(self):
        return self.__counter_channels

    def get_marker_channel(self):
        return self.__marker_channel

    def get_timestamp_buffer_size(self):
        return self.__timestamp_buffer_size

    def set_counter_channels(self, value):
        self.__counter_channels = value

    def set_marker_channel(self, value):
        self.__marker_channel = value

    def set_timestamp_buffer_size(self, value):
        self.__timestamp_buffer_size = value

    def del_counter_channels(self):
        del self.__counter_channels

    def del_marker_channel(self):
        del self.__marker_channel

    def del_timestamp_buffer_size(self):
        del self.__timestamp_buffer_size

    counter_channels = property(get_counter_channels, set_counter_channels, del_counter_channels, "counter_channels's docstring")
    marker_channel = property(get_marker_channel, set_marker_channel, del_marker_channel, "marker_channel's docstring")
    timestamp_buffer_size = property(get_timestamp_buffer_size, set_timestamp_buffer_size, del_timestamp_buffer_size, "timestamp_buffer_size's docstring")
#"""


class TdcConfiguration:
    """
    Configuration for a Time-to-Digital Converter (TDC), including the channels used for
    counting events, the marker channel for synchronization, and the timestamp buffer size.
    """
    def __init__(self,
                 counter_channels: List[int],
                 marker_channel: int,
                 timestamp_buffer_size: int):
        self._counter_channels = counter_channels
        self._marker_channel = marker_channel
        self._timestamp_buffer_size = timestamp_buffer_size

    counter_channels = make_property('_counter_channels')
    marker_channel = make_property('_marker_channel')
    timestamp_buffer_size = make_property('_timestamp_buffer_size')

    def set_counter_channels(self, value: List[int]):
        self._counter_channels = value
    
    def set_marker_channel(self, value: int):
        self._marker_channel = value

