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

from ExperiementalRunner import AbsorbtionImagingConfiguration, PhotonProductionConfiguration, AwgConfiguration, TdcConfiguration, Waveform, ExperimentalAutomationConfiguration, AutomatedExperimentConfiguration
from instruments.WX218x.WX218x_awg import WX218x_awg, Channel
from instruments.WX218x.WX218x_DLL import WX218x_MarkerSource, WX218x_OutputMode, WX218x_OperationMode, WX218x_SequenceAdvanceMode, WX218x_TraceMode, WX218x_TriggerImpedance, WX218x_TriggerMode, WX218x_TriggerSlope, WX218x_Waveform 


#essentially this loads all the data onto all the channels as well as the markers and awaits an external trigger to play the sequence
def __configure_awg(self: AwgConfiguration):
    '''Configure the AWG for the current experiment.
    Input args: self: awg instance'''

    print 'Connecting to AWG...'
    awg = WX218x_awg()
    awg.open(reset=False)
    print '...connected'
    awg.clear_arbitrary_sequence()
    awg.clear_arbitrary_waveform()
    awg.configure_sample_rate(self.awg_config.sample_rate)
    
    awg_chs = self.awg_config.waveform_output_channels
    
    awg.configure_output_mode(WX218x_OutputMode.ARBITRARY)
    awg.configure_couple_enabled(True)
            
    for ch in [awg_chs[x] for x in range(len(awg_chs)) if x%2==0]:
        print 'Configuring trigger options for', ch
        # Configure the burst count, operation mode, trigger source, trigger level and trigger slope
        awg.configure_burst_count(ch, self.awg_config.burst_count)
        awg.configure_operation_mode(ch, WX218x_OperationMode.TRIGGER)
        time.sleep(1)
        awg.configure_trigger_source(ch, WX218x_TriggerMode.EXTERNAL)
        awg.configure_trigger_level(ch, 2)
        awg.configure_trigger_slope(ch, WX218x_TriggerSlope.POSITIVE)
    
        
    channel_absolute_offsets = [np.rint(x*10**-6 * self.awg_config.sample_rate) for x in self.awg_config.waveform_output_channel_lags]
    #maximum offset is defined as the maximum of the absolute offsets
    channel_relative_offsets = list(map(lambda x, m=max(channel_absolute_offsets): int(m-x), channel_absolute_offsets))
    print "Channel relative lags (in awg steps are)", channel_relative_offsets
    print "Channel absolute offsets (in awg steps are)", channel_absolute_offsets
    
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
    
    #get the waveforms and stitch delays
    seq_waveforms = [[self.photon_production_config.waveforms[i] for i in ch_waveforms]
                    for ch_waveforms in self.photon_production_config.waveform_sequence]
    
    print 'Waveform sequence:', seq_waveforms
    
    queud_markers = []
    
    seq_waveform_data, seq_marker_data = [[] for _ in xrange(len(awg_chs))], []
    
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
        #why is this manually defined??
        interleave_channels = [(0,1),(2,3)]
            
        def get_j_segments_max_length(seq_waveforms, channels, j, seq_stitch_delays=None):
            waveform_lengths = []
            if seq_stitch_delays==None:
                seq_stitch_delays = np.zeros(np.array(seq_waveforms).shape).tolist()
            for ch in channels:
                try:
                    waveform_lengths.append(seq_waveforms[ch][j].get_n_samples() + seq_stitch_delays[ch][j])
                except IndexError:
                    pass

            return int(max(waveform_lengths)) if waveform_lengths != [] else 0    
            
        for i in range(len(seq_waveforms)):
            #
            woven_channels = sorted([k for pair in [x for x in interleave_channels if i in x] for k in pair if k!=i])
            bck_woven_channels = [k for k in woven_channels if k<i]
            fwd_woven_channels = [k for k in woven_channels if k>i]
            for j in range(0, len(seq_waveforms[i])):
                if j==0:
                    max_bck_waveforms = get_j_segments_max_length(seq_waveforms, bck_woven_channels, j)
                    print 'Pre-padding channel{0}(seg{1}) with {2}'.format(i+1, j, max_bck_waveforms)
                    seq_waveform_data[i] += [0]*max_bck_waveforms
^
                max_bck_waveforms = get_j_segments_max_length(seq_waveforms, bck_woven_channels, j+1)
                max_fwd_waveforms = get_j_segments_max_length(seq_waveforms, fwd_woven_channels, j, seq_waveforms_stitch_delays) 
                    
                print 'Post-padding channel{0}(seg{1}) with max({2},{3})'.format(i+1,j,max_bck_waveforms,max_fwd_waveforms)
                seq_waveforms_stitch_delays[i][j] += max(max_bck_waveforms,max_fwd_waveforms)#
                
                '''CURRENT ISSUE IS IF THE DELAY IN A BACK WAVEFORM TAKES THE PREVIOUS J-SEGMENT [AST THE START OF THE CURRENT
                TARGET WAVEFORM/SEGMENT.'''

    j = 0
    for channel, waveform_data, waveforms, delays, channel_abs_offset in zip(awg_chs, seq_waveform_data, seq_waveforms, seq_waveforms_stitch_delays, channel_absolute_offsets):
        
        waveform_aom_calibs = {}
        aom_calibration_loc = self.awg_config.waveform_aom_calibrations_locations[j]
        print 'For {0} using aom calibrations in {1}'.format(channel, os.path.join(aom_calibration_loc, '*MHz.txt'))
        for filename in glob.glob(os.path.join(aom_calibration_loc, '*MHz.txt')):
            try:
                waveform_aom_calibs[float(re.match(r'\d+\.*\d*', os.path.split(filename)[1]).group(0))] = get_waveform_calib_fnc(filename)
            except AttributeError:
                pass
        
        marker_data = []
        
        print 'Writing onto channel:', channel
        
        for waveform, delay in zip(waveforms, delays):
        
            if not waveform_aom_calibs:
                calib_fun = lambda x: x
            else:
                calib_fun = waveform_aom_calibs[min(waveform_aom_calibs,
                                                    key=lambda calib_freq: np.abs(calib_freq - waveform.get_mod_frequency()*10**-6))]
                print '\tFor waveform with freq {0}MHz, using calib for {1}MHz'.format(waveform.get_mod_frequency()*10**-6, 
                                                                                        min(waveform_aom_calibs, key=lambda calib_freq: np.abs(calib_freq - waveform.get_mod_frequency()*10**-6)))
            
            seg_length = waveform.get_n_samples() + delay
            marker_pos = []
#                 for i in range(len(queud_markers)):
#                     print i, queud_markers
#                     if queud_markers[i] < seg_length:
#                         marker_pos.append(queud_markers.pop(i))
            i=0
            while i < len(queud_markers):
                print i, queud_markers
                if queud_markers[i] < seg_length:
                    marker_pos.append(queud_markers.pop(i))
                    i-=1
                i+=1
            if channel_abs_offset <= seg_length:
                marker_pos.append(channel_abs_offset)
            else:
                queud_markers.append(channel_abs_offset)
            
            print '\tWriting markers at', marker_pos
            print '\tWriting waveform {0} with stitch delay {1}'.format(os.path.split(waveform.fname)[1], delay)
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
                    print '\tStill in queue:', [x for x in queud_markers if x>=0]
                    markers_in_waveform = [x for x in queud_markers if marker_index <= x <= marker_index+seg_length]
                    print '\tCan wrap from queue:',markers_in_waveform
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
        
        print '\t', j, len(seq_waveform_data), [len(x) for x in seq_waveform_data]
        
        '''
        Combine the marker data for each marked channel.
        '''
        print self.awg_config.marked_channels
        if channel in self.awg_config.marked_channels:
            print '\tAdding marker data for', channel
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
        print 'Rolling {0} forward by {1} points'.format(channel, rel_offset)
        data = np.roll(np.array(data), rel_offset).tolist()
        
        print 'Writing {0} points to {1}'.format(len(data),channel)
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
        print 'ERROR: There are more markers required than can be set currently using the marker channels!'
        marker_starts = marker_starts[:2]

    print 'Writing markers to marker channels at {0}'.format(marker_starts)
    marker_channel_index = 1
    for marker_pos in marker_starts:
        awg.configure_marker(awg_chs[0], 
                                index = marker_channel_index, 
                                position = marker_pos - marker_wid/4,
                                levels = marker_levs,
                                width = marker_wid/2)
        marker_channel_index += 1

    return awg, len(seq_waveform_data[0])/self.awg_config.sample_rate