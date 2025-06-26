'''
This module is intended to hold the top profiles for running assorted experiments from 
configuration through to data acquisition.

Created on 13 Aug 2016, Refactored on 22 May 2025

@authors: Tom Barrett, Matt King, Jan Ole Ernst and Marina Llano Pineiro

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
import glob
import re
import collections
import pandas as pd
import _tkinter
from typing import List, Tuple, Dict, Any
import oscilloscope_manager as osc
import pyvisa
from configobj import ConfigObj
from numpy import trapz
from datetime import datetime
import matplotlib.pyplot as plt
import re, ast

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
from lab_control_functions.awg_control_functions_psh import run_awg
from lab_control_functions.awg_control_functions_single_psh import run_awg_single
from Sequence import IntervalStyle, Sequence
from serial.serialutil import SerialException
from ExperimentalConfigs import GenericConfiguration, AbsorbtionImagingConfiguration,\
    PhotonProductionConfiguration, MotFluoresceConfiguration, AWGSequenceConfiguration,\
    ExperimentSessionConfig, SingleExperimentConfig, Waveform, AwgConfiguration,\
    MotFluoresceConfigurationSweep



def toBool(string):
    GLOB_TRUE_BOOL_STRINGS = ['true', 't', 'yes', 'y']
    return string.lower() in GLOB_TRUE_BOOL_STRINGS

def make_property(attr_name):
    return property(
        fget=lambda self: getattr(self, attr_name),
        fset=lambda self, value: setattr(self, attr_name, value),
        fdel=lambda self: delattr(self, attr_name),
    )


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
        # def run_and_close():
        #     self.run()
        #     self.close()
        # the experiment should close itself in run() method, so we don't need to close it here.
        
        thread = threading.Thread(name='Cold Control Experiment Thread',
                                  target=self.run)
        
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
                ic_imaging_control:IC_ImagingControl = None, sweep=True):
        
        super().__init__(daq_controller, sequence, mot_fluoresce_configuration)
        # the configuration object is a MotFluoresceConfiguration object and called self.config
        self.mot_fluoresce_config:MotFluoresceConfiguration = self.config
        self.save_location = self.mot_fluoresce_config.save_location
        self.iterations = self.mot_fluoresce_config.iterations
        self.mot_reload = self.mot_fluoresce_config.mot_reload # in ms
        print('MOT reload time (ms)', self.mot_reload)
        
        self.with_cam = self.mot_fluoresce_config.use_cam
        self.with_scope = self.mot_fluoresce_config.use_scope
        self.with_awg = self.mot_fluoresce_config.use_awg

        self.sweep = sweep
        #if self.save_with_time and self.iterations != 1:
        #    raise ValueError("Cannot save with time if iterations is not 1. Set save_with_time to False or iterations to 1.")
        self.current_date = datetime.now().strftime("%Y-%m-%d")
        self.current_time = datetime.now().strftime("%H-%M-%S")

        if self.with_awg:
            self.awg_config = self.mot_fluoresce_config.awg_config
            self.awg_sequence_config = self.mot_fluoresce_config.awg_sequence_config
            self.awg_config_single = self.mot_fluoresce_config.awg_config_single
            self.awg_sequence_config_single = self.mot_fluoresce_config.awg_sequence_config_single


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

        if self.with_scope:
            self.samp_rate = self.mot_fluoresce_config.scope_sample_rate
            self.time_range = self.mot_fluoresce_config.scope_time_range
            self.centred_0 = self.mot_fluoresce_config.scope_centered_0
            self.trig_ch = self.mot_fluoresce_config.scope_trigger_channel
            self.trig_lvl = self.mot_fluoresce_config.scope_trigger_level
            self.data_chs = self.mot_fluoresce_config.scope_data_channels


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
        Configures the experiment.
        """
        super().daq_cards_on()
        self.daq_controller.load(self.sequence.getArray())

        if self.with_scope:
            print("connecting to scope")
            self.scope = osc.OscilloscopeManager()
            #self.scope.reset_scope()
            self.scope.configure_scope(self.data_chs, samp_rate=self.samp_rate,
                                       timebase_range=self.time_range, centered_0=self.centred_0)
            
            self.scope.configure_trigger(self.trig_ch, self.trig_lvl)
            #self.scope.set_to_run()

        if self.with_awg:
            print("Configuring AWG")
            self._configure_awg()
            print("AWG configured")
    
    def _configure_awg(self):
        """
        Configures the AWG for the experiment, loads data for all channels"""
        rm = pyvisa.ResourceManager()
        awg = rm.open_resource("USB0::0x168C::0x1284::0000215582::0::INSTR")   
        awg.write(":SYSTem:REBoot") 
        awg.close()
        if self.awg_config_single is not None:
            print("Configuring single AWG")
            run_awg_single(self.awg_config_single, self.awg_sequence_config_single)

        run_awg(self.awg_config, self.awg_sequence_config) 


    def __run_with_scope(self):
        """
        Private method to run the experiment with a scope.
        """
        self.daq_controller.load(self.sequence.getArray())
        self.daq_controller.writeChannelValues()
        print("DAQ controller loaded and channel values written.")

        save_dir = self.save_location
        if save_dir.endswith(".csv"):
            save_dir = os.path.dirname(save_dir)  # Get parent folder if it's a full file path
        
        directory = self.save_location

        if not self.sweep:
            # Get current date and time
            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H-%M-%S")
            # Create full path with date and time subdirectories
            full_directory = os.path.join(directory, date_str, time_str)
        else:
            full_directory = directory

        os.makedirs(full_directory, exist_ok=True)

        print(f"Data will be saved to {full_directory}")

        i = 1
        fails = 0

        #self.scope.set_to_run()
        while i <= self.config.iterations:
            print(f"Iteration {i}")
            print(f"loading mot for {self.config.mot_reload}ms")
            #self.scope.set_to_digitize(self.data_chs)
            sleep(self.config.mot_reload*10**-3) # convert from ms to s

            self.scope.arm_scope()

            print("playing sequence")
            self.daq_controller.play(float(self.sequence.t_step), clearCards=False)
        
            print("writing channel values")
            self.daq_controller.writeChannelValues()

            success = self.scope.wait_for_acquisition()

            if success:
                print("collecting data")
                data = self.scope.read_slow_return_data(self.data_chs)
                filename=f"iteration_{i}_data.csv"
                full_name = os.path.join(full_directory, filename)
                data.to_csv(full_name, index=False)# Saves the data
                print(f"Data saved to {full_name}")
                i += 1
            else:
                fails += 1
                if fails >= 10:
                    self.scope.set_to_stop()
                    break
                print("Failed to trigger, running again")


    def __run_with_cam(self):
        # needs to be in a try except. If the camera isn't closed the computer will crash
        self.__configureCamera()

        if self.with_awg:
            print("Configuring AWG")
            self._configure_awg()
            print("AWG configured")
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


    def run(self):
        """
        Runs the experiment. This is called by the run_in_thread method.
        """
        try:
            self.configure()
            if not self.with_cam and self.with_scope:
                self.__run_with_scope()
            elif self.with_cam and not self.with_scope:
                self.__run_with_cam()
        except Exception as e:
            print(f"An error occurred during the experiment: {e}")
        finally:
            self.close()


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


class MotFluoresceSweepExperiment():
    def __init__(self, sweep_config:MotFluoresceConfigurationSweep,
                 daq_controller:DAQ_controller):
        self.sweep_config = sweep_config
        self.daq_controller = daq_controller


    def run(self):
        """
        This method should create and run a MotFluoresceExperiment for each sweep point
        """

        print(f"This will run a series of {len(self.sweep_config)} experiments.")

        for i, (config, sequence) in enumerate(self.sweep_config):
            print(f"Running experiment with configuration: {i}")
            # Create a new MotFluoresceExperiment with the current configuration
            experiment = MotFluoresceExperiment(self.daq_controller, sequence, config,
                                                sweep=True)
            experiment.run()
            print(f"Experiment {i} completed and closed.")




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

class MotFluoresceDataProcessor(object):
    """
    Processes oscilloscope data: detects triggers in the marker channel,
    extracts PD signal after each trigger, and saves all readouts in binary format.
    """
    def __init__(self, collected_data, window=500e-6, save_dir='outputs'):
        """
        Initialize with marker and PD data, window size, and save path.
        """
        self.data = collected_data
        self.window = window
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.trigger_times = None
        self.integrals_fl = []
        self.ref_0 = []

    def detect_triggers(self):
        """
        Detect all falling edges (triggers) in the marker channel.
        """
        signal = self.data['Channel 2 Voltage (V)'].values
        trigger_val = (signal.max() + signal.min()) / 2
        trigger_idxs = np.where((signal[:-1] > trigger_val) & (signal[1:] < trigger_val))[0] + 1
        self.trigger_times = self.data['Time (s)'].iloc[trigger_idxs].values
    
    def process_readouts(self):
        """
        Processes the PD signals after each trigger and calculates the fluorescence integral.
        """
        if self.trigger_times is None:
            self.detect_triggers()

        time_array = self.data['Time (s)'].values
        ch2_array = self.data['Channel 2 Voltage (V)'].values
        ch4_array = self.data['Channel 4 Voltage (V)'].values

        for t0 in self.trigger_times:
            # Before Imaging Trigger
            mask_pre = (time_array >= t0 - self.window) & (time_array < t0)
            time_pre = time_array[mask_pre] - t0
            ch2_pre = ch2_array[mask_pre]
            ch4_pre = ch4_array[mask_pre]

            self.plot_pre_trigger(time_pre, ch2_pre, ch4_pre, t0)

            # After Imaging Trigger
            mask_post = (time_array >= t0) & (time_array <= t0 + self.window)
            time_post = time_array[mask_post] - t0
            ch4_post = ch4_array[mask_post]

            ch4_smooth = pd.Series(ch4_post).rolling(window=15, center=True, min_periods=1).mean().values

            # Detection imaging start
            mask_rise = (time >= 2.1e-3) & (time <= 2.3e-3)  # imaging starts at 2.1ms
            ch3_smooth_rise = ch4_smooth[mask_rise]
            time_rise = time[mask_rise]
            deriv_rise = np.gradient(ch3_smooth_rise, time_rise)

            idx_rise_rel = np.argmax(deriv_rise)
            idx_rise = time_rise.index[idx_rise_rel]
            t_rise = time.iloc[idx_rise]

            # Detection imaging end
            t_drop = t_rise + 450e-6

            # Fluorescence
            mask_fl = (time_post >= t_rise) & (time_post <= t_drop)
            time_fl = time_post[mask_fl]
            ch4_fl = ch4_post[mask_fl]

            # Reference (background noise)
            t_start_ref = t_drop + 50e-6
            t_end_ref = time_post[-1]
            mask_ref = (time_post >= t_start_ref) & (time_post <= t_end_ref)
            ch4_ref = ch4_post[mask_ref]
            average = np.mean(ch4_ref)

            # Integrated area
            area = trapz(ch4_fl - average, time_fl)

            self.integrals_fl.append(area)
            self.ref_0.append(average)

        today = datetime.datetime.now().strftime("%d-%m")
        output_path = os.path.join(self.save_dir, f'integrated_area_{today}.csv')
        integrals_fl_df = pd.DataFrame({'integral': self.integrals_fl, 'ref 0': self.ref_0})
        integrals_fl_df.to_csv(output_path, index=False)
    
    def plot_pre_trigger(self, time, ch1, ch4, t0):
        """
        Generates and saves a plot of the average signals of channels 1 and 4 before the trigger, 
        as a reference of what is happening before the imaging.
        """
        fig, ax1 = plt.subplots(figsize=(11, 5))
        ax1.plot(time, ch1, linewidth=1.5, color='tab:blue', label='CH 1')
        ax1.set_xlabel(r'Time (s)')
        ax1.set_ylabel(r'Intensity (a.u.) CH 1', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.plot(time, ch4, linewidth=1.5, color='tab:orange', label='CH 4')
        ax2.set_ylabel(r'Intensity (a.u.) CH 4', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

        fig.suptitle(r'Average of CH1 and CH4 before Imaging')
        fig.tight_layout()
        plot_filename = os.path.join(self.save_dir, 'pre_imaging_average.png')
        plt.savefig(plot_filename)
        plt.close()

    def plot_post_trigger(self, time, ch1, ch4, t0, window_size=64):
        """
        Generates and saves a plot of CH1 and rolling-averaged CH4 after the trigger,
        to visualize what happens during/after the imaging.
        """
        mask_post = time >= t0
        time_post = time[mask_post]
        ch1_post = ch1[mask_post]
        ch4_post = ch4[mask_post]

        ch4_smoothed = pd.Series(ch4_post).rolling(window=window_size, center=True, min_periods=1).mean()

        fig, ax1 = plt.subplots(figsize=(11, 5))
        ax1.plot(time_post, ch1_post, linewidth=1.5, color='tab:blue', label='CH 1')
        ax1.set_xlabel(r'Time (s)')
        ax1.set_ylabel(r'Intensity (a.u.) CH 1', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.plot(time_post, ch4_smoothed, linewidth=1.5, linestyle='--', color='tab:orange', label='CH 4 (smoothed)')
        ax2.set_ylabel(r'Intensity (a.u.) CH 4 (smoothed)', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

        fig.suptitle(r'Signals after Trigger: CH1 and Smoothed CH4')
        fig.tight_layout()
        plot_filename = os.path.join(self.save_dir, 'post_imaging_plot.png')
        plt.savefig(plot_filename)
        plt.close()


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
             
