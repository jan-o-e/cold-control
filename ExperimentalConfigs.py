"""
This file contains the configuration objects that are loaded by the experiment objects in
the ExperimentalRunner.py file. The configuration objects should be loaded by reading from
a configuration file (see Config.py) and then they can be passed to the experiment object
which will run the experiment with the specified configuration.

@author: Matt King, Jan Ole Ernst
created: 2025-05-30

"""
from __future__ import annotations
import numpy as np
import threading
import csv
import os
import re
from typing import List, Tuple, Dict, Any
from copy import deepcopy
from datetime import datetime

from Sequence import Sequence

def toBool(string):
    GLOB_TRUE_BOOL_STRINGS = ['true', 't', 'yes', 'y']
    return string.lower() in GLOB_TRUE_BOOL_STRINGS

def make_property(attr_name):
    return property(
        fget=lambda self: getattr(self, attr_name),
        fset=lambda self, value: setattr(self, attr_name, value),
        fdel=lambda self: delattr(self, attr_name),
    )

def sanitize_filename(name: str) -> str:
    # Remove extension
    name = os.path.splitext(name)[0]
    # Remove all non-alphanumeric or underscore characters
    return re.sub(r'[^A-Za-z0-9_]+', '_', name)

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

    def __init__(self,
                 save_location,
                 mot_reload,
                 iterations,
                 use_cam,
                 use_scope,
                 use_awg,
                 awg_dict: Dict = None,
                 cam_dict: Dict = None,
                 scope_dict: Dict = None):
        super().__init__(save_location, mot_reload, iterations)

        self.use_scope = use_scope
        self.use_cam = use_cam
        self.use_awg = use_awg

        if use_cam == True:
            self.cam_exposure = cam_dict["cam_exposure"]
            self.cam_gain = cam_dict["cam_gain"]
            self.camera_trigger_channel = cam_dict["camera_trig_ch"]
            self.camera_trigger_level = cam_dict["camera_trig_levs"]
            self.camera_pulse_width = cam_dict["camera_pulse_width"]
            self.save_images = cam_dict["save_images"]
        else:
            print("No camera will be used.")

        if self.use_scope:
            self.scope_trigger_channel = scope_dict["trigger_channel"]
            self.scope_trigger_level = scope_dict["trigger_level"]
            self.scope_sample_rate = scope_dict["sample_rate"]
            self.scope_time_range = scope_dict["time_range"]
            self.scope_centered_0 = scope_dict["centered_0"]
            self.scope_data_channels = scope_dict["data_channels"]

        if self.use_awg:
            self.awg_config_path_single = awg_dict["config_path_single"]
            self.awg_config_path = awg_dict["config_path_full"]
            self.awg_config = awg_dict["awg_config"]
            self.awg_sequence_config = awg_dict["sequence_config"]
            self.awg_config_single = awg_dict["awg_config_single"]
            self.awg_sequence_config_single = awg_dict[
                "sequence_config_single"]
        else:
            print("No AWG will be used.")


class MotFluoresceConfigurationSweep:

    def __init__(self, base_config: 'MotFluoresceConfiguration', base_sequence: Sequence,
                 sweep_type: str, num_shots:int, sweep_params: Dict[str:Any]):

        self.base_config = base_config
        self.base_sequence = base_sequence
        self.sweep_type = sweep_type
        self.sweep_params = sweep_params
        #print(self.sweep_params)
        self.num_shots = num_shots
        now = datetime.now()
        self.current_date = now.strftime("%Y-%m-%d")
        self.current_time = now.strftime("%H-%M-%S")
        print(f"[DEBUG] date: {self.current_date}")
        print(f"[DEBUG] time: {self.current_time}")
        
        self.configs:List[MotFluoresceConfiguration] = []
        self.sequences:List[Sequence] = []
        print("Creating all MOT fluorescence configurations for the sweep...")

        if sweep_type == "awg_sequence":
            pulses_by_index: List[Tuple[str, str]] = self.sweep_params["pulses_by_index"]
            mod_freqs_ch1: List[float] = self.sweep_params["freq_list_1"]
            mod_freqs_ch2: List[float] = self.sweep_params["freq_list_2"]
            frequency_waveform_indices: List[int] = self.sweep_params["frequency_waveform_indices"]
            pulse_waveform_config_indices: List[int] = self.sweep_params["waveform_config_indices"]
            self.__configure_awg_sweep(pulses_by_index,pulse_waveform_config_indices, mod_freqs_ch1, mod_freqs_ch2, frequency_waveform_indices)
        
        elif sweep_type == "mot_imaging":
            # all these parameters need to be extracted from the config file
            _beam_powers: List[float] = self.sweep_params["beam_powers"]
            _beam_frequencies: List[float] = self.sweep_params["beam_frequencies"]
            _pulse_lengths: List[float] = self.sweep_params["pulse_lengths"]
            self.__configure_imaging_sweep(_beam_powers, _beam_frequencies, _pulse_lengths)

        else:
            raise ValueError("Sweep type not supported")
        
        assert len(self.configs) == len(self.sequences), \
        "configs and sequences must have the same length"


    def __iter__(self):
        return iter(zip(self.configs, self.sequences))

    def __len__(self):
        return len(self.configs)
    
    def __configure_awg_sweep(self, pulses_by_index,pulse_waveform_indices, mod_freqs_ch1, mod_freqs_ch2, frequency_indices):
        print(mod_freqs_ch1)
        print(mod_freqs_ch2)
        print("Warning! If the steps are too small then the frequencies will be overwritten when rounded.")
        for i in range(self.num_shots):
            directories = {os.path.dirname(csv) for csv in pulses_by_index.values()}
            assert len(directories) == 1, "All CSVs must be in the same directory"
            last_folder = os.path.basename(next(iter(directories)))

            # if we are sending no pulses we don't need to sweep the frequency
            if last_folder == r"no_pulse":
                ch1_freqs = [1]
                ch2_freqs = [1]
            else:
                ch1_freqs = mod_freqs_ch1
                ch2_freqs = mod_freqs_ch2
            for freq1 in ch1_freqs:
                for freq2 in ch2_freqs:
                    # Clone and modify base configuration
                    new_config = deepcopy(self.base_config)
                    new_sequence = deepcopy(self.base_sequence)

                    # Modify waveform and frequency settings
                    modified_sequence_config = self.modify_awg_sequence_config(
                        base_config=new_config.awg_sequence_config,
                        waveform_csvs = {idx: pulses_by_index[idx] for idx in pulse_waveform_indices},
                        mod_freqs={
                            frequency_indices[0]: freq1,
                            frequency_indices[1]: freq2
                        })

                    # Update the new config with modified sequence
                    new_config.awg_sequence_config = modified_sequence_config

                    new_config.save_location = os.path.join(
                        self.base_config.save_location,
                        self.current_date,
                        self.current_time,
                        f"sweep_{last_folder}_{freq1/1e6:.1f}_{freq2/1e6:.1f}",
                        f"shot{i}"
                    )

                    if not os.path.exists(self.base_config.save_location):
                        raise FileNotFoundError(f"Base save location does not exist: {self.base_config.save_location}")
                    # Ensure the directory exists
                    save_dir = os.path.dirname(new_config.save_location)
                    os.makedirs(save_dir, exist_ok=True)
                    
                    self.configs.append(new_config)
                    self.sequences.append(new_sequence)

    def __configure_imaging_sweep(self, beam_powers, beam_frequencies, pulse_lengths):
        for i in range(self.num_shots):
            for power in beam_powers:
                for freq in beam_frequencies:
                    for length in pulse_lengths:
                        # Clone and modify the base sequence and config
                        new_config = deepcopy(self.base_config)
                        new_sequence = deepcopy(self.base_sequence)

                        # Modify save location to easily manage data
                        new_config.save_location = os.path.join(
                            self.base_config.save_location,
                            self.current_date,
                            self.current_time,
                            f"swept_{power:.2f}V_{freq:.2f}V_{length}us",
                            f"shot{i}"
                        )

                        # Modifies the sequence
                        freq_ch = 2 # These values shouldn't be hardcoded
                        power_ch = 6
                        new_sequence.updateChannel(freq_ch, [(0, freq),], [0,])
                        tv_pairs = list(new_sequence.get_tV_pairs(power_ch))
                        print(f"The old tv pairs for the imaging channel are: {tv_pairs}")
                        #HACK to change the correct power value and pulse length
                        img_start_tv = tv_pairs[2]# This is a tuple representing a time voltage pair
                        img_end_tv = tv_pairs[3]
                        new_start_tv = (img_start_tv[0], power)
                        new_end_tv = (img_start_tv[0]+length, img_end_tv[1])
                        tv_pairs[2] = new_start_tv
                        tv_pairs[3] = new_end_tv
                        print(f"The new tv pairs for the imaging channel are: {tv_pairs}")
                        new_vint_styles = new_sequence.get_V_intervalStyles(power_ch)
                        new_sequence.updateChannel(power_ch, tv_pairs, new_vint_styles)

                        # Ensure directory exists
                        if not os.path.exists(self.base_config.save_location):
                            raise FileNotFoundError(f"Base save location does not exist: {self.base_config.save_location}")
                        save_dir = os.path.dirname(new_config.save_location)
                        os.makedirs(save_dir, exist_ok=True)

                        # Append sequence and config files to the list
                        self.configs.append(new_config)
                        self.sequences.append(new_sequence)




    
    @staticmethod
    def modify_awg_sequence_config(*, base_config: AWGSequenceConfiguration,
                                waveform_csvs: Dict[int, str],
                                mod_freqs: Dict[int, float]) -> AWGSequenceConfiguration:
        new_config = deepcopy(base_config)

        for idx, wf in enumerate(new_config.waveforms):
            if idx in waveform_csvs:
                wf.fname = waveform_csvs[idx]
            if idx in mod_freqs:
                wf.mod_frequency = mod_freqs[idx]

        return new_config
    

    #this can be used as follows:
    # base_config = MotFluoresceConfiguration(...)

    # waveform_csvs_ch1 = ['waveform1.csv', 'waveform2.csv']
    # waveform_csvs_ch2 = ['waveform3.csv', 'waveform4.csv']

    # mod_freqs_ch1 = [1e6, 2e6]
    # mod_freqs_ch2 = [3e6, 4e6]
    # sweep = MotFluoresceConfigurationSweep(base_config, waveform_csvs_ch1, waveform_csvs_ch2, mod_freqs_ch1, mod_freqs_ch2)
    # for config in sweep:


class AWGSequenceConfiguration():
    """
    AWGSequenceConfiguration stores all configuration parameters
    required for a photon production experiment.

    This includes:
    - Save location and MOT reload time
    - Number of iterations
    - A waveform sequence and its associated waveforms
    - Interleaving and stitching behavior for waveforms
    - Configuration objects for the AWG and TDC systems
    """

    def __init__(self,
                 waveform_sequence,
                 waveforms,
                 interleave_waveforms,
                 waveform_stitch_delays,
                 awg_configuration,
                 ):


        self._waveform_sequence = waveform_sequence
        self.waveforms: List[Waveform] = waveforms
        self.interleave_waveforms: bool = interleave_waveforms
        self.waveform_stitch_delays = waveform_stitch_delays

        self._awg_configuration: AwgConfiguration = awg_configuration

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




class TdcConfiguration:
    """
    Configuration for a Time-to-Digital Converter (TDC), including the channels used for
    counting events, the marker channel for synchronization, and the timestamp buffer size.
    """

    def __init__(self, counter_channels: List[int], marker_channel: int,
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
