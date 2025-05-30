from time import sleep
import copy
import os
import time
import numpy as np
import glob
import re 
import matplotlib.pyplot as plt
import csv

from Config import ConfigReader, DaqReader
from ExperimentalConfigs import AbsorbtionImagingConfiguration,\
    PhotonProductionConfiguration, AwgConfiguration, TdcConfiguration, Waveform,\
        ExperimentSessionConfig , SingleExperimentConfig
from instruments.WX218x.WX218x_awg import WX218x_awg, Channel
from instruments.WX218x.WX218x_DLL import WX218x_MarkerSource, WX218x_OutputMode,\
    WX218x_OperationMode, WX218x_SequenceAdvanceMode, WX218x_TraceMode,\
    WX218x_TriggerImpedance, WX218x_TriggerMode, WX218x_TriggerSlope, WX218x_Waveform 
from configobj import ConfigObj

#essentially this loads all the data onto all the channels as well as the markers and awaits an external trigger to play the sequence
def configure_awg(awg_config: AwgConfiguration, photon_production_config: PhotonProductionConfiguration):
    '''Configure the AWG for the current experiment.
    Input args:
    awg_config (AwgConfiguration): Specifies the settings of the awg including: (all in AWG section of config file)
                                    - sample rate
                                    - burst count
                                    - waveform output channels
                                    - waveform output channel lags
                                    - marker channels
                                    - marker width
                                    - waveform aom calibrations locations
    photon_production_config (PhotonProductionConfiguration): photon production configuration instance
    '''

    # Connects to the AWG and clears previous settings
    print('Connecting to AWG...')
    awg = WX218x_awg()
    awg.open(reset=False)
    print('...connected')
    awg.clear_arbitrary_sequence()
    awg.clear_arbitrary_waveform()

    # Configures the awg according to the awg_config and some default settings
    awg.configure_sample_rate(awg_config.sample_rate)
    awg.configure_output_mode(WX218x_OutputMode.ARBITRARY)
    awg.configure_couple_enabled(True)
            
    
    # Configures trigger options for every other channel????
    awg_chs = awg_config.waveform_output_channels
    for ch in [awg_chs[x] for x in range(len(awg_chs)) if x%2==0]:
        print('Configuring trigger options for', ch)
        # Configure the burst count, operation mode, trigger source, trigger level and trigger slope
        awg.configure_burst_count(ch, awg_config.burst_count)
        awg.configure_operation_mode(ch, WX218x_OperationMode.TRIGGER)
        time.sleep(1)# why is this here?
        awg.configure_trigger_source(ch, WX218x_TriggerMode.EXTERNAL)
        awg.configure_trigger_level(ch, 2)
        awg.configure_trigger_slope(ch, WX218x_TriggerSlope.POSITIVE)
    
    
    # Calculates absolute and relative offsets to deal with channel lags
    channel_absolute_offsets = [np.rint(x*10**-6 * awg_config.sample_rate) for x in awg_config.waveform_output_channel_lags]
    channel_relative_offsets = list(map(lambda x, m=max(channel_absolute_offsets): int(m-x), channel_absolute_offsets))
    print("Channel relative lags (in awg steps are)", channel_relative_offsets)
    print("Channel absolute offsets (in awg steps are)", channel_absolute_offsets)
    



    # Defines the shape of the marker pulses
    marker_levs, marker_waveform_levs = (0,1.2), (0,1)
    marker_wid  = int(awg_config.marker_width*10**-6 * awg_config.sample_rate)
    

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
    


    # Creates lists of waveforms to be sent to the awg
    queud_markers = []
    seq_waveforms = [[photon_production_config.waveforms[i] for i in ch_waveforms]
                        for ch_waveforms in photon_production_config.waveform_sequence]
    #waveform data is the data to be written to the awg
    seq_waveform_data, seq_marker_data = [[] for _ in range(len(awg_chs))], []
    #waveform delays are the delays (in awg units) to be applied to the waveforms to synchronise them across channels,
    #these can be positive or negative accounting for delay after or before waveform respectively
    seq_waveforms_stitched_delays =  [[] for _ in range(len(awg_chs))]


    #perform stitching of different waveforms with delays to synchronise across channels        
    if photon_production_config.interleave_waveforms:
        print('Interleaving waveforms')
        for i in range(len(awg_chs)):
            calculated_delay=0
            if photon_production_config.waveform_stitch_delays[i][0] == -1:
                if not photon_production_config.waveform_stitch_delays[i][1]:
                    calculated_delay = 0
                else:
                    for el in photon_production_config.waveform_stitch_delays[i][1]:
                        calculated_delay -=photon_production_config.waveforms[el].get_n_samples()
            elif photon_production_config.waveform_stitch_delays[i][0] == 1:
                if not photon_production_config.waveform_stitch_delays[i][1]:
                    calculated_delay = 0
                else:
                    for el in photon_production_config.waveform_stitch_delays[i][1]:
                        calculated_delay +=photon_production_config.waveforms[el].get_n_samples()
            else:
                raise ValueError('Invalid stitch delay needs to be before (-1) or after (+1) the waveform')
            
            seq_waveforms_stitched_delays[i] = calculated_delay
        print('Stitch delays are', seq_waveforms_stitched_delays)
        

    j = 0
    for channel, waveform_data, waveforms, delay, channel_abs_offset in \
        zip(awg_chs, seq_waveform_data, seq_waveforms, seq_waveforms_stitched_delays, channel_absolute_offsets):
        """
        Loops through each channel, processing the waveform and marker data to be sent to the awg for each channel.
        1. Sets channel output and marker attributes
        2. Loads calibration files
        3. Marker data and delays
        4. Processes each waveform
        5. Handles waveform sequencing
        6. Handles channel offset
        """
        

        # 1 Setting channel output and marker attributes
        if channel==Channel.CHANNEL_3:#IMPORTANT THIS SET CH3 to output a constant voltage
            constant_V=True
        else:
            constant_V=False

        marker_waveforms_indices=[0,2]#IMPORTANT THIS SETS the indices of the waveforms on the marked channel that are actually marked for photon dectection (i.e. the VST pulses)
        # Increasing marker_waveform_offset makes the first (channel 1) marker pulse happen later.
        marker_waveform_offset=500#IMPORTANT THIS SETS the relative marker offset with respect to the delays of the VST pulse

        # 2 Loading calibration files
        waveform_aom_calibs = {}
        aom_calibration_loc = awg_config.waveform_aom_calibrations_locations[j]
        print('For {0} using aom calibrations in {1}'.format(channel, os.path.join(aom_calibration_loc, '*MHz.txt')))
        for filename in glob.glob(os.path.join(aom_calibration_loc, '*MHz.txt')):
            try:
                waveform_aom_calibs[float(re.match(r'\d+\.*\d*', os.path.split(filename)[1]).group(0))] = get_waveform_calib_fnc(filename)
            except AttributeError:
                print("Warning, waveform_aom_calibs is undefined.")
        
        # 3 Marker data and delays
        marker_data = []
        marker_waveform_delays=[]
        for w in waveforms:
            marker_waveform_delays.append(w.get_n_samples())

        print('Writing onto channel:', channel)

        # 4 Processing each waveform
        for ind, waveform in enumerate(waveforms):
            waveform: Waveform
            if not waveform_aom_calibs:
                calib_fun = lambda x: x
            else:
                calib_fun = waveform_aom_calibs[min(waveform_aom_calibs,
                                                    key=lambda calib_freq: np.abs(calib_freq - waveform.get_mod_frequency()*10**-6))]
                print('\tFor waveform with freq {0}MHz, using calib for {1}MHz'.format(waveform.get_mod_frequency()*10**-6, 
                                                                                        min(waveform_aom_calibs, key=lambda calib_freq: np.abs(calib_freq - waveform.get_mod_frequency()*10**-6))))
            
            seg_length = waveform.get_n_samples() + abs(delay) + abs(channel_abs_offset)
            marker_pos = []
            #  for i in range(len(queud_markers)):
            #   print i, queud_markers
            #   if queud_markers[i] < seg_length:
            #   marker_pos.append(queud_markers.pop(i))
            i=0
            while i < len(queud_markers):
                print(i, queud_markers)
                if queud_markers[i] < seg_length:
                    marker_pos.append(queud_markers.pop(i))
                    i-=1
                i+=1
            if channel_abs_offset <= seg_length:
                print('Writing marker pos at', channel_abs_offset+marker_waveform_offset)
                marker_pos.append(channel_abs_offset+marker_waveform_offset)
            else:
                print('Queing marker at', channel_abs_offset+marker_waveform_offset)
                queud_markers.append(channel_abs_offset+marker_waveform_offset)

            
            # 5 Waveform sequence handling
            print('\tWriting markers at', marker_pos)
            print('\tWriting waveform {0} with stitch delay {1}'.format(os.path.split(waveform.fname)[1], delay))
            if len(waveforms)==1:
                print('\tOnly one waveform in sequence')
                #write delay to the single waveform in a sequence
                if delay < 0:
                    waveform_data += [0]*abs(delay) + waveform.get(sample_rate=awg_config.sample_rate, calibration_function=calib_fun, constant_voltage=constant_V )
                    marker_data   += waveform.get_marker_data(marker_positions=marker_pos, marker_levels=marker_waveform_levs, marker_width=marker_wid, n_pad_left=abs(delay))
                elif delay > 0:
                    waveform_data += waveform.get(sample_rate=awg_config.sample_rate, calibration_function=calib_fun, constant_voltage=constant_V) + [0]*delay
                    marker_data   += waveform.get_marker_data(marker_positions=marker_pos, marker_levels=marker_waveform_levs, marker_width=marker_wid, n_pad_right=delay)
                else:
                    #no delay
                    waveform_data += waveform.get(sample_rate=awg_config.sample_rate, calibration_function=calib_fun, constant_voltage=constant_V)
                    marker_data   += waveform.get_marker_data(marker_positions=marker_pos, marker_levels=marker_waveform_levs, marker_width=marker_wid)
            
            else:# if there are multiple waveforms is the sequence
                print('\tMore than one waveform in sequence')
                print('Configuring markers for indexed waveforms:', marker_waveforms_indices)
                marker_pos=[]

                for i in marker_waveforms_indices:
                    if i==0:
                        marker_pos.append(channel_abs_offset+marker_waveform_offset)
                    else:
                        marker_pos.append(channel_abs_offset+marker_waveform_offset+sum(marker_waveform_delays[:i]))
                marker_length=sum(marker_waveform_delays)
                seg_length = marker_length + abs(delay) + abs(channel_abs_offset)

                if ind==0:
                    #write delay to the first waveform in a sequence if a sequence contains more than one waveform
                    if delay < 0:

                        #add delay to marker position
                        for i,pos in enumerate(marker_pos):
                            marker_pos[i]=abs(delay)+pos

                        waveform_data += [0]*abs(delay) + waveform.get(sample_rate=awg_config.sample_rate, calibration_function=calib_fun, constant_voltage=constant_V)
                        #write marker data directly with a single call with multiple defined marker positions inside multi waveform sequence
                        marker_data= get_multiwaveform_marker_data(marker_length,marker_positions=marker_pos, marker_levels=marker_waveform_levs, marker_width=marker_wid, n_pad_left=abs(delay))

                    else:
                        waveform_data += waveform.get(sample_rate=awg_config.sample_rate, calibration_function=calib_fun, constant_voltage=constant_V)
                        #write marker data directly with a single call with multiple defined marker positions inside multi waveform sequence
                        marker_data = get_multiwaveform_marker_data(marker_length,marker_positions=marker_pos, marker_levels=marker_waveform_levs, marker_width=marker_wid, n_pad_right=abs(delay))
                elif ind==len(waveforms)-1:
                    #write delay to the last waveform in a sequence if a sequence contains more than one waveform
                    if delay >0:
                        waveform_data += waveform.get(sample_rate=awg_config.sample_rate, calibration_function=calib_fun, constant_voltage=constant_V) + [0]*delay
                    else:
                        waveform_data += waveform.get(sample_rate=awg_config.sample_rate, calibration_function=calib_fun, constant_voltage=constant_V)
                else:
                    waveform_data += waveform.get(sample_rate=awg_config.sample_rate, calibration_function=calib_fun, constant_voltage=constant_V)

            
            # 6 Channel offset handling
            if channel_abs_offset<0:
                waveform_data=waveform_data+[0]*abs(int(channel_abs_offset))
                marker_data=marker_data+[0]*abs(int(channel_abs_offset))
            else:
                waveform_data=[0]*abs(int(channel_abs_offset))+waveform_data
                marker_data=marker_data+[0]*abs(int(channel_abs_offset))
            
            print('Waveform data length: {}'.format(len(waveform_data)))
            print('Marker data length: {}'.format(len(marker_data)))


            queud_markers = [x-seg_length for x in queud_markers]
        
        '''
        Wrap any makers still queued into the first waveforms markers (presuming we are looping through this sequence multiple times).
        '''
        if queud_markers != []:
            print('Wrapping queued markers into the first waveform marker...')
            marker_index = 0
            for ind, waveform in enumerate(waveforms):
                seg_length = waveform.get_n_samples() + abs(delay)
#             queud_markers = [x-(waveforms[-1].get_n_samples()+self.photon_production_config.waveform_stitch_delays[-1]) for x in queud_markers]
                if len([x for x in queud_markers if x>=0])>0:
                    print('\tStill in queue:', [x for x in queud_markers if x>=0])
                    markers_in_waveform = [x for x in queud_markers if marker_index <= x <= marker_index+seg_length]
                    print('\tCan wrap from queue:',markers_in_waveform)
                    if ind==0:
                        if delay < 0:
                            wrapped_marker_data = waveform.get_marker_data(marker_positions=markers_in_waveform,
                                                                            marker_levels=marker_waveform_levs,
                                                                            marker_width=marker_wid,
                                                                            n_pad_right=delay)
                        else:
                            wrapped_marker_data = waveform.get_marker_data(marker_positions=markers_in_waveform,
                                                                        marker_levels=marker_waveform_levs,
                                                                        marker_width=marker_wid)
                    if ind==len(waveforms)-1:
                        if delay < 0:
                            wrapped_marker_data = waveform.get_marker_data(marker_positions=markers_in_waveform,
                                                                            marker_levels=marker_waveform_levs,
                                                                            marker_width=marker_wid,
                                                                            n_pad_left=abs(delay))
                        else:
                            wrapped_marker_data = waveform.get_marker_data(marker_positions=markers_in_waveform,
                                                                            marker_levels=marker_waveform_levs,
                                                                            marker_width=marker_wid)
                            
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
        print('Marked Channels:', awg_config.marked_channels)
        if channel in awg_config.marked_channels:
            if seq_marker_data == []:
                print('seq_marker_data is empty')
                seq_marker_data += marker_data
            else:
                j1, j2 = map(len,[seq_marker_data, marker_data])
                print('combining marker data in seq_marker_data')
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
    for x in seq_waveform_data:
        if N % 16 != 0:
            print('{} points added to waveform data to ensure a multiple of 16 points are written to the AWG.'.format(16-(N % 16)))
            x += [0]*((16-(N % 16)))

    '''Ensure number of samples are multiples of 16'''
    print("Number of samples in each channel modulo 16:{}".format(len(x) % 16))


    plt.plot(seq_marker_data)
    plt.title('Sequence Marker Data')
    plt.show()        
    l_mark, l_seq = len(seq_marker_data), len(seq_waveform_data[0])


    #this checks whether the marker and sequence data have the same length and pads the marker data with zeros if not        
    if   l_mark < l_seq : seq_marker_data += [0]*(l_seq - l_mark)
    elif l_seq  < l_mark: seq_marker_data  = seq_marker_data[:l_seq]

    #finds start of marker pulse if it has been padded
    marker_starts = [x[0] for x in enumerate(zip([0]+seq_marker_data[:-1],seq_marker_data)) if x[1][0]==0 and x[1][1]>0]
    print('Marker_starts:', marker_starts)
    
    if len(marker_starts) > 2:
        print('ERROR: There are more markers required than can be set currently using the marker channels!')
        marker_starts = marker_starts[:2]

    print('Writing markers to marker channels at {0}'.format(marker_starts))
    marker_channel_index = 1
    #for i, marker_pos in enumerate(marker_starts):
        #this writes different markers to different channels
    #    print(awg_chs[i])
    #    awg.configure_marker(awg_chs[i], 
    #                            index = marker_channel_index, 
    #                            position = marker_pos - marker_wid/4,
    #                            levels = marker_levs,
    #                            width = marker_wid/2)

        #when i uncomment this line it writes one marker which seems to be the first
        #when i leave it there seems to only be the last marker
    #    marker_channel_index += 1
    awg.configure_marker(awg_chs[0], 
                                index = 1, 
                                position = marker_starts[0] - marker_wid/4,
                                levels = marker_levs,
                                width = marker_wid/2)
    awg.configure_marker(awg_chs[2], 
                                index = 2, 
                                position = marker_starts[1] - marker_wid/4,
                                levels = marker_levs,
                                width = marker_wid/2)

    awg.clear_arbitrary_sequence()
    awg.clear_arbitrary_waveform()
    awg.configure_arb_wave_trace_mode(WX218x_TraceMode.SINGLE)
    
    '''Configure each channel for its output data.'''
    for channel, rel_offset, data in zip(awg_chs, channel_relative_offsets, seq_waveform_data):
        # Roll channel data to account for relative offsets (e.g. AOM lags)
        print(data)
        print('Rolling {0} forward by {1} points'.format(channel, rel_offset))
        plt.plot(data)
        plt.title('Channel {0} data'.format(channel))
        plt.show()
        #with open('{0}.csv'.format(channel), 'w') as file:
        #    writer = csv.writer(file)
        #    writer.writerow(data)

        data = np.roll(np.array(data), rel_offset).tolist()        
        print('Writing {0} points to {1}'.format(len(data),channel))
        awg.set_active_channel(channel)
        if channel == Channel.CHANNEL_1 or channel==Channel.CHANNEL_2 or channel==Channel.CHANNEL_3:
            awg.create_arbitrary_waveform_custom(data)

    #awg.set_active_channel(Channel.CHANNEL_4)
    #awg.create_arbitrary_waveform_custom(seq_marker_data)
    
            
    for channel in awg_chs:
        awg.enable_channel(channel)
        awg.configure_arb_gain(channel, 2)

    return awg, len(seq_waveform_data[0])/awg_config.sample_rate
















