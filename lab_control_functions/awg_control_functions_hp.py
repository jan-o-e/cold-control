from time import sleep, perf_counter
import os
import numpy as np
import glob
import re
from ExperimentalConfigs import AWGSequenceConfiguration
from ExperimentalConfigs import AwgConfiguration, Waveform
from instruments.WX218x.WX218x_awg import WX218x_awg, Channel
from instruments.WX218x.WX218x_DLL import (
    WX218x_OutputMode, WX218x_OperationMode, WX218x_TriggerMode, WX218x_TriggerSlope, WX218x_TraceMode
)
import matplotlib.pyplot as plt
import ctypes

# Constants for marker configuration

#marker_levs, marker_waveform_levs = (0,1.2), (0,1)
MARKER_LOW = 0.0
MARKER_HIGH = 1.2
MARKER_WF_LOW = 0.0
MARKER_WF_HIGH = 1
MARKER_WIDTH_FACTOR = 10**-6
ABSOLUTE_OFFSET_FACTOR = 10**-6
# Increasing DEFAULT_MARKER_OFFSET makes the marker pulses happen later.
DEFAULT_MARKER_OFFSET = 4150  # TO DO: MAKE A LIST TO VARY MARKER DELAYS INDEPENDENTLY   si surt 0 en marker channel es aixoooo


MARKER_WF_LEVS = (MARKER_WF_LOW, MARKER_WF_HIGH)
MARKER_LEVS = (MARKER_LOW, MARKER_HIGH)

def connect_awg():
    """Connect to the AWG and clear previous configurations."""
    print("Connecting to AWG...")
    awg = WX218x_awg()
    print(f"Attempting to open AWG: {awg}")

    awg.open(reset=False)
    awg.clear_arbitrary_sequence()
    awg.clear_arbitrary_waveform()
    print("...connected")
    return awg


def configure_awg_general(awg:WX218x_awg, sample_rate, burst_count):
    """Configure general AWG settings like sample rate and output mode."""
    awg.configure_sample_rate(sample_rate)
    awg.configure_output_mode(WX218x_OutputMode.ARBITRARY)
    awg.configure_couple_enabled(True)
    #awg.configure_burst_count(Channel.CHANNEL_1, burst_count)  # Example for one channel


def configure_trigger(awg:WX218x_awg, awg_chs, burst_count):
    """Configure trigger settings for specific channels."""
    for ch in [awg_chs[x] for x in range(len(awg_chs)) if x%2==0]:
        print(f"Configuring trigger options for {ch}")
        awg.configure_burst_count(ch, burst_count)
        awg.configure_operation_mode(ch, WX218x_OperationMode.TRIGGER)
        sleep(1)
        awg.configure_trigger_source(ch, WX218x_TriggerMode.EXTERNAL)
        awg.configure_trigger_level(ch, 2)
        awg.configure_trigger_slope(ch, WX218x_TriggerSlope.POSITIVE)


def calculate_offsets(channel_lags, sample_rate):
    """Calculate absolute and relative channel offsets."""
    absolute_offsets = [np.rint(x * ABSOLUTE_OFFSET_FACTOR * sample_rate) for x in channel_lags]
    max_offset = max(absolute_offsets)
    #relative_offsets = [max_offset - x for x in absolute_offsets]
    relative_offsets = list(map(lambda x, m=max_offset: int(m-x), absolute_offsets))

    print("\n🔍 DEBUG: Offsets calculados")
    for i, offset in enumerate(absolute_offsets):
        print(f"  Canal {i+1}: abs_offset = {offset}, rel_offset = {relative_offsets[i]}")

    print("Channel relative lags (in awg steps are)", relative_offsets)
    print("Channel absolute offsets (in awg steps are)", absolute_offsets)
    return absolute_offsets, relative_offsets



def plot_marker_data(marker_data, debug=False):
    """Plot marker data for visualization."""
    if debug:
        plt.plot(marker_data)
        plt.title("Marker Data")
        plt.show(block=False)
        plt.pause(1)
        plt.close()


def get_waveform_calib_fnc(calib_fname, max_eff=0.9):
    """
    Generates a calibration function from a file containing waveform calibration data.
    Inputs:
        - calib_fname (str): name of the file from which to read calibration data. Has two columns.
        - max_eff (float): maximum efficiency value. Values above this are removed from the data.
    Returns:
        - interp_fct (function): a function that takes in a list of points, x, and returns the
        interpolated values of the calibration data at each of the points.
    """
    calib_data = np.genfromtxt(calib_fname,skip_header=1)
    calib_data[:,1] /= 100. # convert % as saved to decimal efficiencies
    calib_data = calib_data[(calib_data[:,1]<=max_eff)] # remove all elements with greater than the maximum efficiency
    calib_data[:,1] /= max(calib_data[:,1]) # rescale effiencies

    interp_fct = lambda x: np.interp(np.abs(x), calib_data[:,1], calib_data[:,0])

    return interp_fct


def get_multiwaveform_marker_data(inp_data, marker_positions=[], marker_levels=(0,1), marker_width=50, n_pad_right=0, n_pad_left=0):
    '''
    Returns a marker waveform.
    Inputs:
        - inp_data (int): the length of the waveform data
        - marker_positions (list): positions within the waveform to start each marker segment
        - marker_levels (tuple): levels to be used in the waveform in the form (low, high)
        - marker_width (int): width of each marker pulse
        - n_pad_right, n_pad_left (int): number of padding elements to add at the end and beginning (respectively) of the waveform
    '''
    data = np.array( [marker_levels[0]] * (n_pad_left + inp_data + n_pad_right))# Use a np array for ease of setting array slices to contant values.
    for pos in marker_positions:
        data[int(pos):int(pos+marker_width)] = marker_levels[1]
    
    if data[0]==1:# This is a big fix. If the first element of the sequence is 1 (i.e. max high level)
        data[0]=0# then the channel remains high at the end of the sequence. Don't know why...

    # Convert np array to list for consistency with the get() method.
    return data.tolist()
#

def create_waveform_lists(waveforms, waveform_sequence, awg_chs):
    # Creates lists of waveforms to be sent to the awg
    queud_markers = []
    seq_waveforms = [[waveforms[i] for i in ch_waveforms]
                        for ch_waveforms in waveform_sequence]
    #waveform data is the data to be written to the awg
    seq_waveform_data, seq_marker_data = [[] for _ in range(len(awg_chs))], []
    #waveform delays are the delays (in awg units) to be applied to the waveforms to synchronise them across channels,
    #these can be positive or negative accounting for delay after or before waveform respectively
    seq_waveforms_stitched_delays =  [[] for _ in range(len(awg_chs))]

    return seq_waveforms, seq_waveform_data, seq_waveforms_stitched_delays, seq_marker_data, queud_markers


def stitch_waveforms(awg_chs, stitch_delays, waveforms, seq_waveforms_stitched_delays):
    #perform stitching of different waveforms with delays to synchronise across channels  
    print("\n🔍 DEBUG: Stitch delays antes de aplicar")
    for i, delay in enumerate(stitch_delays):
        print(f"  Canal {awg_chs[i]}: stitch_delay = {delay}")


    print('Interleaving waveforms')
    for i in range(len(awg_chs)):
        calculated_delay=0
        if stitch_delays[i][0] == -1:
            if not stitch_delays[i][1]:
                calculated_delay = 0
            else:
                for el in stitch_delays[i][1]:
                    calculated_delay -= waveforms[el].get_n_samples()
        elif stitch_delays[i][0] == 1:
            if not stitch_delays[i][1]:
                calculated_delay = 0
            else:
                for el in stitch_delays[i][1]:
                    calculated_delay += waveforms[el].get_n_samples()
        else:
            raise ValueError('Invalid stitch delay needs to be before (-1) or after (+1) the waveform')
        
        seq_waveforms_stitched_delays[i] = calculated_delay
    print('Stitch delays are', seq_waveforms_stitched_delays)
    return seq_waveforms_stitched_delays


def align_data_length(seq_waveform_data, seq_marker_data):
    '''
    Ensure we write the same number of points to each channel.
    '''
    N = max([len(x) for x in seq_waveform_data])
    for x in seq_waveform_data:
        if len(x) < N:
            x += [0]*(N-len(x))
    for x in seq_waveform_data:
        if N % 16 != 0:
            print('{} points added to waveform data to ensure a multiple of 16 points are written to the AWG.'.format(16-(N % 16)))
            x += [0]*((16-(N % 16)))

    #Ensure number of samples are multiples of 16
    print("Number of samples in each channel modulo 16:{}".format(len(x) % 16))

    l_mark, l_seq = len(seq_marker_data), len(seq_waveform_data[0])
    #this checks whether the marker and sequence data have the same length and pads the marker data with zeros if not        
    if   l_mark < l_seq : seq_marker_data += [0]*(l_seq - l_mark)
    elif l_seq  < l_mark: seq_marker_data  = seq_marker_data[:l_seq]
    return seq_waveform_data, seq_marker_data



def write_markers(marker_data, awg:WX218x_awg, awg_chs, marker_width):
    #finds start of marker pulse if it has been padded
    marker_starts = [x[0] for x in enumerate(zip([0]+marker_data[:-1],marker_data)) if x[1][0]==0 and x[1][1]>0]
    print('Marker_starts:', marker_starts)
    
    if len(marker_starts) > 2:
        print('ERROR: There are more markers required than can be set currently using the marker channels!')
        marker_starts = marker_starts[:2]

    # print('Writing markers to marker channels at {0}'.format(marker_starts))

    awg.configure_marker(awg_chs[0], 
                                index = 1, 
                                position = marker_starts[0] - marker_width/4,
                                levels = MARKER_LEVS,
                                width = marker_width/2)
    #awg.configure_marker(awg_chs[1], # changed to channel 2 to investigate pulses
    #                            index = 2, 
    #                            position = marker_starts[1] - marker_width/4,
    #                            levels = MARKER_LEVS,
    #                            width = marker_width/2)

    awg.clear_arbitrary_sequence()
    awg.clear_arbitrary_waveform()



def write_channels(awg_chs, _rel_offsets, _wf_data, _awg:WX218x_awg):
    '''Configure each channel for its output data.'''
    for channel, rel_offset, data in zip(awg_chs, _rel_offsets, _wf_data):
        # Roll channel data to account for relative offsets (e.g. AOM lags)
        # print(data)
        print('Rolling {0} forward by {1} points'.format(channel, rel_offset))
        #plt.plot(data)
        #plt.title('Channel {0} data'.format(channel))
        # plt.show(block=False)
        # plt.pause(1)
        # plt.close()

        data = np.roll(np.array(data), rel_offset).tolist()        
        # print('Writing {0} points to {1}'.format(len(data),channel))
        _awg.set_active_channel(channel)
        if channel == Channel.CHANNEL_1 or channel==Channel.CHANNEL_2 or channel==Channel.CHANNEL_3:
            _awg.create_arbitrary_waveform_custom(data)

            
    for channel in awg_chs:
        _awg.enable_channel(channel)
        _awg.configure_arb_gain(channel, 2)


# Refactored subfunctions for run_awg
def _initialize_awg_connection():
    """Initialize AWG connection and clear previous configurations."""
    return connect_awg()

def _setup_awg_configuration(awg, awg_config):
    """Configure general AWG settings and trigger configuration."""
    configure_awg_general(awg, awg_config.sample_rate, awg_config.burst_count)
    configure_trigger(awg, awg_config.waveform_output_channels, awg_config.burst_count)

def _compute_channel_offsets(awg_config):
    """Calculate absolute and relative channel offsets based on channel lags."""
    return calculate_offsets(awg_config.waveform_output_channel_lags, awg_config.sample_rate)

def _initialize_waveform_processing(photon_config, awg_config):
    """Initialize waveform processing parameters and data structures."""
    marker_wid = int(awg_config.marker_width * 10**-6 * awg_config.sample_rate)
    
    wf_list, wf_data, wf_stitched_delays, seq_marker_data, queud_markers = create_waveform_lists(
        photon_config.waveforms,
        photon_config.waveform_sequence, 
        awg_config.waveform_output_channels
    )
    
    return marker_wid, wf_list, wf_data, wf_stitched_delays, seq_marker_data, queud_markers

def _process_waveform_stitching(photon_config, awg_config, wf_stitched_delays):
    """Process waveform stitching if enabled."""
    if photon_config.interleave_waveforms:
        return stitch_waveforms(
            awg_config.waveform_output_channels,
            photon_config.waveform_stitch_delays, 
            photon_config.waveforms, 
            wf_stitched_delays
        )
    else:
        return [0] * len(awg_config.waveform_output_channels)

def _load_aom_calibrations(aom_calibration_loc, channel):
    """Load AOM calibration files for a specific channel."""
    waveform_aom_calibs = {}
    print('For {0} using aom calibrations in {1}'.format(channel, os.path.join(aom_calibration_loc, '*MHz.txt')))
    
    for filename in glob.glob(os.path.join(aom_calibration_loc, '*MHz.txt')):
        try:
            freq_match = re.match(r'\d+\.*\d*', os.path.split(filename)[1])
            if freq_match:
                freq = float(freq_match.group(0))
                waveform_aom_calibs[freq] = get_waveform_calib_fnc(filename)
        except AttributeError:
            print("Warning, waveform_aom_calibs is undefined.")
    
    return waveform_aom_calibs

def _get_calibration_function(waveform_aom_calibs, waveform):
    """Get the appropriate calibration function for a waveform."""
    if not waveform_aom_calibs:
        return lambda x: x
    else:
        closest_freq = min(waveform_aom_calibs,
                          key=lambda calib_freq: np.abs(calib_freq - waveform.get_mod_frequency()*10**-6))
        return waveform_aom_calibs[closest_freq]

def _process_single_waveform_sequence(waveform, waveform_data, marker_data, delay, awg_config, 
                                    calib_fun, constant_V, marker_pos, marker_wid):
    """Process a single waveform in a sequence."""
    if delay < 0:
        waveform_data += [0]*abs(delay) + waveform.get(sample_rate=awg_config.sample_rate, 
                                                       calibration_function=calib_fun, 
                                                       constant_voltage=constant_V)
        marker_data += waveform.get_marker_data(marker_positions=marker_pos, 
                                               marker_levels=MARKER_WF_LEVS, 
                                               marker_width=marker_wid, 
                                               n_pad_left=abs(delay))
    elif delay > 0:
        waveform_data += waveform.get(sample_rate=awg_config.sample_rate, 
                                     calibration_function=calib_fun, 
                                     constant_voltage=constant_V) + [0]*delay
        marker_data += waveform.get_marker_data(marker_positions=marker_pos, 
                                               marker_levels=MARKER_WF_LEVS, 
                                               marker_width=marker_wid, 
                                               n_pad_right=delay)
    else:
        waveform_data += waveform.get(sample_rate=awg_config.sample_rate, 
                                     calibration_function=calib_fun, 
                                     constant_voltage=constant_V)
        marker_data += waveform.get_marker_data(marker_positions=marker_pos, 
                                               marker_levels=MARKER_WF_LEVS, 
                                               marker_width=marker_wid)
    
    return waveform_data, marker_data

def _process_multi_waveform_sequence(waveforms, waveform, ind, waveform_data, marker_data, 
                                   delay, awg_config, calib_fun, constant_V, marker_pos, 
                                   marker_wid, channel_abs_offset, marker_waveform_delays):
    """Process multiple waveforms in a sequence."""
    MARKER_WF_INDICES = [0]
    
    if len(waveforms) > 1:
        marker_pos = []
        for i in MARKER_WF_INDICES:
            if i == 0:
                marker_pos.append(channel_abs_offset + DEFAULT_MARKER_OFFSET)
            else:
                marker_pos.append(channel_abs_offset + DEFAULT_MARKER_OFFSET + sum(marker_waveform_delays[:i]))
        
        marker_length = sum(marker_waveform_delays)
        
        if ind == 0:
            if delay < 0:
                for i, pos in enumerate(marker_pos):
                    marker_pos[i] = abs(delay) + pos
                
                waveform_data += [0]*abs(delay) + waveform.get(sample_rate=awg_config.sample_rate, 
                                                               calibration_function=calib_fun, 
                                                               constant_voltage=constant_V)
                marker_data = get_multiwaveform_marker_data(marker_length, 
                                                           marker_positions=marker_pos, 
                                                           marker_levels=MARKER_WF_LEVS, 
                                                           marker_width=marker_wid, 
                                                           n_pad_left=abs(delay))
            else:
                waveform_data += waveform.get(sample_rate=awg_config.sample_rate, 
                                             calibration_function=calib_fun, 
                                             constant_voltage=constant_V)
                marker_data = get_multiwaveform_marker_data(marker_length, 
                                                           marker_positions=marker_pos, 
                                                           marker_levels=MARKER_WF_LEVS, 
                                                           marker_width=marker_wid, 
                                                           n_pad_right=abs(delay))
        elif ind == len(waveforms) - 1:
            if delay > 0:
                waveform_data += waveform.get(sample_rate=awg_config.sample_rate, 
                                             calibration_function=calib_fun, 
                                             constant_voltage=constant_V) + [0]*delay
            else:
                waveform_data += waveform.get(sample_rate=awg_config.sample_rate, 
                                             calibration_function=calib_fun, 
                                             constant_voltage=constant_V)
        else:
            waveform_data += waveform.get(sample_rate=awg_config.sample_rate, 
                                         calibration_function=calib_fun, 
                                         constant_voltage=constant_V)
    
    return waveform_data, marker_data

def _apply_channel_offset(waveform_data, marker_data, channel_abs_offset):
    """Apply channel offset to waveform and marker data."""
    if channel_abs_offset < 0:
        waveform_data = waveform_data + [0]*abs(int(channel_abs_offset))
        marker_data = marker_data + [0]*abs(int(channel_abs_offset))
    else:
        waveform_data = [0]*abs(int(channel_abs_offset)) + waveform_data
        marker_data = marker_data + [0]*abs(int(channel_abs_offset))
    
    return waveform_data, marker_data

def _process_queued_markers(queud_markers, waveforms, delay, marker_wid, marker_data):
    """Process any queued markers by wrapping them into waveforms."""
    if queud_markers != []:
        marker_index = 0
        for ind, waveform in enumerate(waveforms):
            seg_length = waveform.get_n_samples() + abs(delay)
            
            if len([x for x in queud_markers if x >= 0]) > 0:
                markers_in_waveform = [x for x in queud_markers if marker_index <= x <= marker_index + seg_length]
                
                if ind == 0:
                    if delay < 0:
                        wrapped_marker_data = waveform.get_marker_data(marker_positions=markers_in_waveform,
                                                                     marker_levels=MARKER_WF_LEVS,
                                                                     marker_width=marker_wid,
                                                                     n_pad_right=delay)
                    else:
                        wrapped_marker_data = waveform.get_marker_data(marker_positions=markers_in_waveform,
                                                                     marker_levels=MARKER_WF_LEVS,
                                                                     marker_width=marker_wid)
                
                if ind == len(waveforms) - 1:
                    if delay < 0:
                        wrapped_marker_data = waveform.get_marker_data(marker_positions=markers_in_waveform,
                                                                     marker_levels=MARKER_WF_LEVS,
                                                                     marker_width=marker_wid,
                                                                     n_pad_left=abs(delay))
                    else:
                        wrapped_marker_data = waveform.get_marker_data(marker_positions=markers_in_waveform,
                                                                     marker_levels=MARKER_WF_LEVS,
                                                                     marker_width=marker_wid)
                
                marker_data[marker_index:marker_index+len(wrapped_marker_data)] = \
                    [MARKER_WF_LEVS[1] if (a==MARKER_WF_LEVS[1] or b==MARKER_WF_LEVS[1]) else 
                     MARKER_WF_LEVS[0] if (a==MARKER_WF_LEVS[0] and b==MARKER_WF_LEVS[0]) else a+b
                     for a,b in zip(marker_data[:len(wrapped_marker_data)], wrapped_marker_data)]
                
                marker_index += len(wrapped_marker_data)
                queud_markers = [x-seg_length for x in queud_markers]
    
    return marker_data, queud_markers

def _process_channel_waveforms(channel, waveforms, waveform_data, delay, channel_abs_offset, 
                             awg_config, waveform_aom_calibs, marker_wid, queud_markers):
    """Process all waveforms for a specific channel."""
    constant_V = False
    marker_data = []
    marker_waveform_delays = [w.get_n_samples() for w in waveforms]
    
    print('Writing onto channel:', channel)
    
    for ind, waveform in enumerate(waveforms):
        calib_fun = _get_calibration_function(waveform_aom_calibs, waveform)
        
        seg_length = waveform.get_n_samples() + abs(delay) + abs(channel_abs_offset)
        marker_pos = []
        
        # Process queued markers
        i = 0
        while i < len(queud_markers):
            if queud_markers[i] < seg_length:
                marker_pos.append(queud_markers.pop(i))
                i -= 1
            i += 1
        
        if channel_abs_offset <= seg_length:
            marker_pos.append(channel_abs_offset + DEFAULT_MARKER_OFFSET)
        else:
            queud_markers.append(channel_abs_offset + DEFAULT_MARKER_OFFSET)
        
        # Process waveform sequence
        if len(waveforms) == 1:
            waveform_data, marker_data = _process_single_waveform_sequence(
                waveform, waveform_data, marker_data, delay, awg_config, 
                calib_fun, constant_V, marker_pos, marker_wid
            )
        else:
            waveform_data, marker_data = _process_multi_waveform_sequence(
                waveforms, waveform, ind, waveform_data, marker_data, 
                delay, awg_config, calib_fun, constant_V, marker_pos, 
                marker_wid, channel_abs_offset, marker_waveform_delays
            )
        
        # Apply channel offset
        waveform_data, marker_data = _apply_channel_offset(waveform_data, marker_data, channel_abs_offset)
        
        queud_markers = [x - seg_length for x in queud_markers]
    
    # Process any remaining queued markers
    marker_data, queud_markers = _process_queued_markers(queud_markers, waveforms, delay, marker_wid, marker_data)
    
    return waveform_data, marker_data, queud_markers

def _combine_marker_data(seq_marker_data, marker_data, channel, awg_config):
    """Combine marker data for marked channels."""
    if channel in awg_config.marked_channels:
        if seq_marker_data == []:
            print('seq_marker_data is empty')
            seq_marker_data += marker_data
        else:
            j1, j2 = map(len, [seq_marker_data, marker_data])
            print('combining marker data in seq_marker_data')
            if j1 < j2:
                seq_marker_data = [sum(x) for x in zip(seq_marker_data[:j1], marker_data[:j1])] + marker_data[j1:]
            if j2 < j1:
                seq_marker_data = [sum(x) for x in zip(seq_marker_data[:j2], marker_data[:j2])] + seq_marker_data[j2:]
            if j1 == j2:
                seq_marker_data = [sum(x) for x in zip(seq_marker_data, marker_data)]
    
    return seq_marker_data

def _process_all_channels(awg_config, wf_list, wf_data, wf_stitched_delays, abs_offsets, 
                         marker_wid, seq_marker_data):
    """Process waveforms for all AWG channels."""
    queud_markers = []
    
    for j, (channel, waveform_data, waveforms, delay, channel_abs_offset) in enumerate(
        zip(awg_config.waveform_output_channels, wf_data, wf_list, wf_stitched_delays, abs_offsets)
    ):
        # Load AOM calibrations
        aom_calibration_loc = awg_config.waveform_aom_calibrations_locations[j]
        waveform_aom_calibs = _load_aom_calibrations(aom_calibration_loc, channel)
        
        # Process channel waveforms
        waveform_data, marker_data, queud_markers = _process_channel_waveforms(
            channel, waveforms, waveform_data, delay, channel_abs_offset,
            awg_config, waveform_aom_calibs, marker_wid, queud_markers
        )
        
        wf_data[j] = waveform_data
        
        # Combine marker data
        seq_marker_data = _combine_marker_data(seq_marker_data, marker_data, channel, awg_config)
        
        print('\t', j+1, len(wf_data), [len(x) for x in wf_data])
    
    return wf_data, seq_marker_data

def _finalize_awg_configuration(awg, wf_data, seq_marker_data, awg_config, rel_offsets, marker_wid):
    """Finalize AWG configuration by writing data and configuring channels."""
    # Ensure data length alignment
    wf_data, marker_data = align_data_length(wf_data, seq_marker_data)
    
    # Plot marker data
    plot_marker_data(marker_data)
    
    # Add markers to channels
    write_markers(seq_marker_data, awg, awg_config.waveform_output_channels, marker_wid)
    
    # Configure trace mode
    awg.configure_arb_wave_trace_mode(WX218x_TraceMode.SINGLE)
    
    # Configure channels and write data
    write_channels(awg_config.waveform_output_channels, rel_offsets, wf_data, awg)
    
    return len(wf_data[0]) / awg_config.sample_rate

def run_awg(awg_config: AwgConfiguration, photon_config: AWGSequenceConfiguration):
    """
    Main function to configure the AWG for the experiment.
    
    Args:
        awg_config (AwgConfiguration): AWG hardware configuration
        photon_config (AWGSequenceConfiguration): Photon sequence configuration
        
    Returns:
        tuple: (awg_instance, sequence_duration_seconds)
    """
    
    # 1. Initialize AWG connection
    t0 = perf_counter()
    awg = _initialize_awg_connection()
    print(f"AWG_CONNECTION_INIT: {perf_counter() - t0:.4f} SECONDS")
    
    # 2. Setup AWG configuration
    t1 = perf_counter()
    _setup_awg_configuration(awg, awg_config)
    print(f"AWG_CONFIGURATION_SETUP: {perf_counter() - t1:.4f} SECONDS")
    
    # 3. Compute channel offsets
    t2 = perf_counter()
    abs_offsets, rel_offsets = _compute_channel_offsets(awg_config)
    print(f"CHANNEL_OFFSETS_COMPUTATION: {perf_counter() - t2:.4f} SECONDS")
    
    # 4. Initialize waveform processing
    t3 = perf_counter()
    marker_wid, wf_list, wf_data, wf_stitched_delays, seq_marker_data, queud_markers = \
        _initialize_waveform_processing(photon_config, awg_config)
    print(f"WAVEFORM_PROCESSING_INIT: {perf_counter() - t3:.4f} SECONDS")
    
    # 5. Process waveform stitching
    t4 = perf_counter()
    wf_stitched_delays = _process_waveform_stitching(photon_config, awg_config, wf_stitched_delays)
    print(f"WAVEFORM_STITCHING_PROCESSING: {perf_counter() - t4:.4f} SECONDS")
    
    # 6. Process all channels
    t5 = perf_counter()
    wf_data, seq_marker_data = _process_all_channels(
        awg_config, wf_list, wf_data, wf_stitched_delays, abs_offsets, marker_wid, seq_marker_data
    )
    print(f"ALL_CHANNELS_PROCESSING: {perf_counter() - t5:.4f} SECONDS")
    
    # 7. Finalize AWG configuration
    t6 = perf_counter()
    sequence_duration = _finalize_awg_configuration(awg, wf_data, seq_marker_data, awg_config, rel_offsets, marker_wid)
    print(f"AWG_FINALIZATION: {perf_counter() - t6:.4f} SECONDS")
    
    # Total execution time
    total_time = perf_counter() - t0
    print(f"TOTAL_RUN_AWG_EXECUTION: {total_time:.4f} SECONDS")
    
    print("AWG configuration complete.")
    return awg, sequence_duration
