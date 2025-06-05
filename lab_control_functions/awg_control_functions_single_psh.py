
from time import sleep
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


MARKER_LOW = 0.0
MARKER_HIGH = 1.2
MARKER_WF_LOW = 0.0
MARKER_WF_HIGH = 1
MARKER_WIDTH_FACTOR = 10**-6
ABSOLUTE_OFFSET_FACTOR = 10**-6
DEFAULT_MARKER_OFFSET = 50  # Ajustado a un solo canal

MARKER_WF_LEVS = (MARKER_WF_LOW, MARKER_WF_HIGH)
MARKER_LEVS = (MARKER_LOW, MARKER_HIGH)

def connect_awg():
    """Conectar al AWG y limpiar configuraciones previas."""
    print("Conectando al AWG...")
    awg = WX218x_awg()
    awg.open(reset=False)
    awg.clear_arbitrary_sequence()
    awg.clear_arbitrary_waveform()
    print("...conectado")
    return awg

def configure_awg_general(awg: WX218x_awg, sample_rate, burst_count):
    """Configurar parámetros generales del AWG."""
    awg.configure_sample_rate(sample_rate)
    awg.configure_output_mode(WX218x_OutputMode.ARBITRARY)
    awg.configure_couple_enabled(True)

def configure_trigger(awg: WX218x_awg, awg_ch, burst_count):
    """Configurar el trigger para el canal seleccionado."""
    print(f"Configurando trigger para {awg_ch}")
    awg.configure_burst_count(awg_ch, burst_count)
    awg.configure_operation_mode(awg_ch, WX218x_OperationMode.TRIGGER)
    sleep(1)
    awg.configure_trigger_source(awg_ch, WX218x_TriggerMode.EXTERNAL)
    awg.configure_trigger_level(awg_ch, 2)
    awg.configure_trigger_slope(awg_ch, WX218x_TriggerSlope.POSITIVE)

def calculate_offsets(channel_lag, sample_rate):
    """Calcular los offsets para un solo canal."""
    absolute_offset = int(np.rint(channel_lag * ABSOLUTE_OFFSET_FACTOR * sample_rate))
    print("Offset absoluto en pasos AWG:", absolute_offset)
    return absolute_offset

def write_markers(marker_data, awg: WX218x_awg, awg_ch, marker_width):
    """Escribir los marcadores para un solo canal."""
    marker_start = next((i for i, (prev, curr) in enumerate(zip([0] + marker_data[:-1], marker_data)) if prev == 0 and curr > 0), None)
    print('Inicio del marcador:', marker_start)
    
    if marker_start is not None:
        awg.configure_marker(awg_ch, index=1, position=marker_start - marker_width // 4, levels=MARKER_LEVS, width=marker_width // 2)
    
    awg.clear_arbitrary_sequence()
    awg.clear_arbitrary_waveform()

def write_channel(awg_ch, rel_offset, wf_data, awg: WX218x_awg):
    """Configurar y escribir datos en el canal seleccionado."""
    # print(f'Aplicando un retraso de {rel_offset} pasos en {awg_ch}')
    # plt.plot(wf_data)
    # plt.title(f'Datos del Canal {awg_ch}')
    # plt.show(block=False)
    # plt.pause(1)
    # plt.close()

    wf_data = np.roll(np.array(wf_data), rel_offset).tolist()
    awg.set_active_channel(awg_ch)
    awg.create_arbitrary_waveform_custom(wf_data)
    awg.enable_channel(awg_ch)
    awg.configure_arb_gain(awg_ch, 2)

def load_waveform(awg: WX218x_awg, awg_ch, waveform_data):
    """Carga la forma de onda en el canal del AWG."""
    print(f"Cargando forma de onda en {awg_ch}...")

    # plt.figure(figsize=(8, 4))
    # plt.plot(waveform_data)
    # plt.title(f'Waveform Data - Canal {awg_ch}')
    # plt.xlabel('Muestras')
    # plt.ylabel('Amplitud')
    # plt.grid(True)
    # plt.show()
    # plt.pause(1)
    # plt.close()

    awg.set_active_channel(awg_ch)
    awg.create_arbitrary_waveform_custom(waveform_data)
    awg.enable_channel(awg_ch)
    awg.configure_arb_gain(awg_ch, 2)
    print("Forma de onda cargada.")

def get_multiwaveform_marker_data(marker_length, marker_positions, marker_levels, marker_width):
    """Genera los datos del marcador cuando hay múltiples formas de onda en una secuencia."""
    marker_data = [marker_levels[0]] * marker_length
    for pos in marker_positions:
        for i in range(pos, min(pos + marker_width, marker_length)):
            marker_data[i] = marker_levels[1]
    return marker_data

def get_waveform_calib_fnc(filename):
    """Carga la función de calibración desde un archivo de texto."""
    def calibration_function(waveform):
        return waveform  # Aquí se aplicaría la calibración real si se tienen datos
    
    return calibration_function

def stitch_waveforms(channel_list, waveform_stitch_delays, waveforms, sample_rate):
    """Une múltiples formas de onda considerando los retardos entre ellas."""
    stitched_waveforms = []
    for i, waveform in enumerate(waveforms):
        # Verificar que waveform_stitch_delays es una lista y tiene elementos
        if isinstance(waveform_stitch_delays, list) and i < len(waveform_stitch_delays):
            delay = waveform_stitch_delays[i]
            if isinstance(delay, list) and len(delay) > 0:
                delay = delay[0]
            delay = int(delay)
        else:
            delay = 0  
        if delay < 0:
            raise ValueError(f"Delay en posición {i} no puede ser negativo: {delay}")
        waveform_data = waveform.get(sample_rate=sample_rate)
        stitched_waveforms.append(([0] * delay) + waveform_data)
    return stitched_waveforms

def create_waveform_lists(waveforms, waveform_sequence, channels, sample_rate):
    """Crea las listas de formas de onda y sus datos asociados."""
    wf_list = [[] for _ in channels]
    wf_data = [[] for _ in channels]
    wf_stitched_delay = [0 for _ in channels]
    seq_marker_data = []
    
    for waveform in waveforms:
        for i, channel in enumerate(channels):
            wf_list[i].append(waveform)
            wf_data[i] += waveform.get(sample_rate=sample_rate)

    
    return wf_list, wf_data, wf_stitched_delay, seq_marker_data, []

def load_marker_data(awg: WX218x_awg, awg_ch, marker_data, marker_width):
    """Carga los datos del marcador en el canal del AWG."""
    print(f"Cargando datos de marcador en {awg_ch}...")

    marker_starts = [i for i, (prev, curr) in enumerate(zip([0] + marker_data[:-1], marker_data)) if prev == 0 and curr > 0]
    
    if not marker_starts:
        print("⚠️ Advertencia: No se encontraron marcadores en los datos. Asegúrate de que marker_data contenga pulsos.")
        return
    
    marker_start = marker_starts[0] 
    awg.configure_marker(awg_ch, 
                         index=1, 
                         position=marker_start - marker_width // 4,
                         levels=MARKER_LEVS,
                         width=marker_width // 2)

    awg.clear_arbitrary_sequence()
    awg.clear_arbitrary_waveform()

    print(f"✔ Datos de marcador cargados en {awg_ch}. Posición: {marker_start}")





def run_awg_single(awg_config: AwgConfiguration, photon_config: AWGSequenceConfiguration):
    """
    Configures the AWG for a single-channel experiment.
    """
    awg = connect_awg()
    
    # General AWG settings
    configure_awg_general(awg, awg_config.sample_rate, awg_config.burst_count)
    configure_trigger(awg, awg_config.waveform_output_channels[0], awg_config.burst_count)

    # Calculate offsets
    abs_offset = calculate_offsets(list(awg_config.waveform_output_channel_lags)[0], awg_config.sample_rate)

    # Process waveforms and markers
    marker_wid = int(awg_config.marker_width * 10**-6 * awg_config.sample_rate)
    wf_list, wf_data, wf_stitched_delay, seq_marker_data, queued_markers = create_waveform_lists(
        photon_config.waveforms,
        photon_config.waveform_sequence,
        [awg_config.waveform_output_channels[0]],
        awg_config.sample_rate  # <-- Pasamos sample_rate
    )
    
    if photon_config.interleave_waveforms:
        wf_stitched_delay = stitch_waveforms(
            [awg_config.waveform_output_channels[0]],
            [photon_config.waveform_stitch_delays],
            photon_config.waveforms,
            awg_config.sample_rate
        )[0]
    else:
        wf_stitched_delay = 0
    
    channel = awg_config.waveform_output_channels[0]
    waveforms = wf_list[0]
    waveform_data = wf_data[0]
    delay = wf_stitched_delay
    waveform : Waveform

    constant_V=False # IMPORTANT 
    
    # Load calibration files
    waveform_aom_calibs = {}
    aom_calibration_loc = awg_config.waveform_aom_calibrations_locations[0]
    for filename in glob.glob(os.path.join(aom_calibration_loc, '*MHz.txt')):
        try:
            waveform_aom_calibs[float(re.match(r'\d+\.*\d*', os.path.split(filename)[1]).group(0))] = get_waveform_calib_fnc(filename)
        except AttributeError:
            print("Warning, waveform_aom_calibs is undefined.")
    
    # Process each waveform
    marker_data = []
    waveform_data = [[]]
    

    for waveform in waveforms:
        if not waveform_aom_calibs:
            calib_fun = lambda x: x
        else:
            calib_fun = waveform_aom_calibs[min(waveform_aom_calibs, key=lambda calib_freq: abs(calib_freq - waveform.get_mod_frequency() * 10**-6))]
        
        segment_length = waveform.get_n_samples() + abs(delay[0]) + abs(abs_offset)
        marker_pos = [abs_offset + DEFAULT_MARKER_OFFSET]
        
        if len(waveforms) == 1:
            waveform_data[0].extend(waveform.get(sample_rate=awg_config.sample_rate, calibration_function=calib_fun))
            marker_data += waveform.get_marker_data(marker_positions=marker_pos, marker_levels=MARKER_WF_LEVS, marker_width=marker_wid)
        else:
            marker_length = sum(waveform.get_n_samples() for w in waveforms)
            marker_data = get_multiwaveform_marker_data(marker_length, marker_positions=marker_pos, marker_levels=MARKER_WF_LEVS, marker_width=marker_wid)
            waveform_data[0].extend(waveform.get(sample_rate=awg_config.sample_rate, calibration_function=calib_fun))
    
    # Apply channel offset
    if abs_offset < 0:
        waveform_data[0] = [0] * abs(int(abs_offset)) + waveform_data[0]
        marker_data += [0] * abs(int(abs_offset))
    else:
        waveform_data[0] = [0] * abs(int(abs_offset)) + waveform_data[0]
        marker_data = [0] * abs(int(abs_offset)) + marker_data
    
    wf_data[0] = waveform_data[0]
    seq_marker_data = marker_data if not seq_marker_data else [sum(x) for x in zip(seq_marker_data, marker_data)]
    
    # Load waveforms and markers into AWG
    print(f"Waveform length: {len(waveform_data)}")

    extra_points = (16 - (len(waveform_data[0]) % 16)) % 16
    waveform_data[0] += [0] * extra_points 

    load_waveform(awg, channel, waveform_data[0])
    if channel in awg_config.marked_channels:
        load_marker_data(awg, channel, seq_marker_data, marker_wid)
    
    print(f'Configuration complete for channel {channel}')
