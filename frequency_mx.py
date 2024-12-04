import serial
import time
import numpy as np
from Config import ConfigReader, DaqReader
import matplotlib.pyplot as plt
import re
from instruments.TF930 import TF930
import visa
# from sympy.physics.quantum.circuitplot import matplotlib



def frequency_timeseries_mx(t_max,
                        writeToQueryDelay=0.1, queryToReadDelay=0.3):
    '''Creates a plot for the time fluctuation the frequency output.
        t_max - Maximal time (in s) for which to measure a frequency,
        writeToQueryDelay - How long to wait between writing a new voltage and querying the frequency counter
        queryToReadDelay - How long to wait between querying the frequency counter and reading the output
                           NOTE: the shortest measurement time on the TF930 is 0.3s'''

    try:
        counter = TF930.TF930(port='COM5')
    except serial.serialutil.SerialException as err:
        print ('Calibration failed - frequency counter could not be found')
        raise err

    # record the TF930 output
    t_data, calData = [], []
    print ('Running through the measurements...')
    for t_step in np.arange(0,t_max, writeToQueryDelay+queryToReadDelay):
        #print t_step
        time.sleep(writeToQueryDelay)
        t_data.append(t_step)
        calData.append(counter.query('N?', delay=queryToReadDelay))
    print ('...finished!')
    # Parse the output, once for units and once for values
    r = r'([\d|\.|e|\+]+)([a-zA-Z]*)\r\n'

    units = ''
    while units == '':
        for i in range(0, len(calData)):
            match = re.match(r, calData[i])
            if match:
                units = match.group(2)
                break

    parsedData = []
    nBadPoints = 0
    for i in range(0, len(calData)):
        match = re.match(r, calData[i])
        if match:
            parsedData.append(match.group(1))
        else:
            # If there was unexpected output (e.g. when the delays before reading are wrong)
            # then remove the corresponding data point from vData
            nBadPoints += 1
            t_data.pop(i - nBadPoints)

    print ('Removed {0} bad data points'.format(nBadPoints))

    # Just a hack to convert Hz to MHz as it's nicer.
    if units == 'Hz':
        parsedData = map(lambda x: float(x) / 10 ** 6, parsedData)
        units = 'MHz'

    return t_data, parsedData, units


def save_calibration_plot(fname, xData, calData, units, title):
    fig = plt.figure()

    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=0.85)
    ax.set_title(title)

    ax.set_xlabel('t')
    ax.set_ylabel(units)

    ax.plot(xData, calData)

    plt.savefig(fname)
    print ('saved img: ', fname)

if __name__ == "__main__":
    freq_meas=frequency_timeseries_mx(30*60)
    save_calibration_plot("vescent_box_frequ_fluctuations2",freq_meas[0],freq_meas[1], freq_meas[2], "Vescent box Frequency Fluctuation2")