"""
Created on 22/05/2025.
@authors: Marina Llanero Pinero, Matt King


@description: This script contains the OscilloscopeManager class, which is used to manage
the connection to and data acquisition from an oscilloscope.
"""



import numpy as np
import pyvisa as visa
import pandas as pd
import os
from datetime import datetime
import time
import matplotlib.pyplot as plt


class OscilloscopeManager:

    def __init__(self, scope_id='USB0::0x2A8D::0x900E::MY53450121::0::INSTR'):
        self.scope_id = scope_id
        self.read_speed = None

        try:
            # Inicializar VISA y conectar al osciloscopio
            self.rm = visa.ResourceManager()
            self.scope = self.rm.open_resource(scope_id)
            self.scope.timeout = 60000  # 30 segundos de timeout

            # Verificar la comunicación con el osciloscopio
            print("Conectado al osciloscopio:", self.scope.query("*IDN?"))

            # Resetear el osciloscopio (opcional, puedes comentar esta línea)
            # self.scope.write("*RST")
            #self.scope.write("*CLS")  # Limpiar errores

        except visa.Error as e:
            print(f"Error al conectar con el osciloscopio: {e}")
            raise

    def quit(self):
        """
        Function to end the connection to the scope at the end of the program.
        """

        self.scope.close()
        self.rm.close()


    @staticmethod
    def save_data(dataframe, filename, window):
            """
            Static method to save a dataframe to a file. 
            Inputs:
             - dataframe (pd.Dataframe): The dataframe to be stored as a csv
             - filename (str): desired name of the file
             
            Returns:
             - full_name (str): full name of the file including file path
            """

            # Get current date and time
            current_date = datetime.now().strftime("%Y-%m-%d")
            current_time = datetime.now().strftime("%H-%M-%S")

            # Ensure the new directory exists
            directory = os.path.join("data", current_date)
            os.makedirs(directory, exist_ok=True) 

            # Creates full file name including time and parent folders
            full_name = f"{window}_{current_time}_{filename}"
            full_name = os.path.join(directory, full_name)

            # Saves the dataframe
            dataframe.to_csv(full_name, index=False)
            print(f"Data saved to {full_name}")
            return full_name


    @staticmethod   # Se comporta como una función independiente, pero está encapsulada
    #dentro de la clase por razones de organización.
    # no tiene acceso a los atributos de la clase (self o cls)
    def csv_analysis(filename):
        """
        Static method to plot data from a csv
        Inputs:
         - filename (str): path to the file from which to extract the data
        """

        # Load data from CSV
        data = pd.read_csv(filename)

        title = filename.split("\\")[-1]

        # Plot the data
        plt.figure(figsize=(10, 6))
        plt.plot(data['Time (s)'], data['Voltage (V)'], linestyle="None", marker=".", color="black")
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.title(title)
        plt.grid(True)
        plt.show()


    @staticmethod
    def process_scope_data(filename):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(filename)
        
        # Display basic information about the data
        print("Data Overview:")
        print(df.head())
        print("\nSummary Statistics:")
        print(df.describe())
        
        # Plot each channel's voltage data over time
        plt.figure(figsize=(10, 6))
        for column in df.columns:
            if "Voltage (V)" in column:
                plt.plot(df['Time (s)'], df[column], label=column)
        
        # Customize the plot
        plt.title("Oscilloscope Data")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        
        # Show the plot
        plt.show()


    def configure_scope(self, samp_rate=1e10, timebase_range=1e-8, centered_0=False,\
                        high_speed = False):
        """
        Function to configure the general scope settings.
        Inputs:
         - samp_rate (float): Rate at which samples are collected
         - timebase_range (float): How long to collect samples for
         - centered_0 (bool): if True, the time axis will be centered on 0. 
            If false, the time axis will start at 0.
         - high_speed (bool): TODO if True, the scope will be set to read data at high speed.
        """

        self.read_speed = high_speed
        
        print("configuring the scope settings")
        self.scope.write('ACQUIRE:MODE HRESOLUTION')
        self.scope.write(f'ACQUIRE:SRATE:ANALOG {samp_rate}')
        self.scope.write(f'TIMEBASE:RANGE {timebase_range}')

        if centered_0:
            self.scope.write(f"TIMEBASE:POSITION 0")
        else:
            self.scope.write(f'TIMEBASE:POSITION {timebase_range / 2}')

        self.scope.write('WAVEFORM:FORMAT WORD')
        self.scope.write('WAVEFORM:STREAMING OFF')
        print("scope settings configured")


    def configure_trigger(self, trigger_channel, trigger_level, trigger_slope="+"):
        """
        Function to configure the trigger settings of the oscilloscope.
        Inputs:
         - trigger_channel (int): Channel on which to set the trigger
         - trigger_level (float): Voltage level at which to trigger
         - trigger_slope (str): Slope of the trigger, either '+' or '-'
        """
        # Set the trigger level and slope
        self.scope.write(":TRIGGER:SWEEP TRIGGERED")
        self.scope.write(":TRIGGER:MODE EDGE")
        self.scope.write(f":TRIGGER:EDGE:SOURCE CHANNEL{trigger_channel}")
        self.scope.write(f":TRIGGER:EDGE:LEVEL {trigger_level}")
        
        if trigger_slope == "+":
            self.scope.write(":TRIGGER:EDGE:SLOPE POSITIVE")
        elif trigger_slope == "-":
            self.scope.write(":TRIGGER:EDGE:SLOPE NEGATIVE")
        else:
            raise ValueError(f"Invalid value for trigger_slope: {trigger_slope}")



    def set_to_single(self):
        """
        Function to set the scope to single mode.
        TODO: Does this actually work?
        """
        # Set the scope to single mode
        self.scope.write(':SINGLE')
        self.scope.write(':RUN')
        print("Oscilloscope set to single mode.")


    def set_to_run(self):
        """
        Function to set the scope to run mode. 
        """
        # Set the scope to run mode
        self.scope.write(':RUN')
        print("Oscilloscope set to run mode.")
    

    def set_to_digitize(self, channels=[1, 2]):
        """
        Function to set the scope to digitize mode. This is the primary way to collect
        data from the scope. Use this before sending a trigger pulse to the scope.
        """
        # HACK - This is allows all the channels to be digitized at once.
        write_text = "DIGITIZE:"
        for channel in channels:
            write_text += f",CHANNEL{channel}"
        
        self.scope.write(write_text) 

        print(f"Oscilloscope set to digitize mode for channels {channels}.")




    def acquire_slow_return_data(self, channels):   
        """
        Function to sample data from multiple channels when a trigger has been manually 
        set on the oscilloscope.
        
        Inputs:
         - channels (list of int): List of channels to collect data from
         - save_file (bool): Option to save the collected data in a csv file
         - window (int): Name for saving the data

        Returns:
         - collected_data (pd.DataFrame): Datafram with time and voltage values for each channel
        """
        collected_data = None

        if self.read_speed is None:
            raise ValueError("Scope read speed not set. Please configure the scope first.")
        elif self.read_speed is True:
            print("Warning: Scope is set to high speed. Consider using acquire_fast_return_data() instead.")

        for channel in channels:
            self.scope.write('WAVEFORM:BYTEORDER LSBFIRST')
            self.scope.write(f'WAVEFORM:SOURCE CHANNEL{channel}')

            print(f"Collecting data from channel {channel}...")
            y_incr = float(self.scope.query('WAVEFORM:YINCREMENT?'))
            y_orig = float(self.scope.query('WAVEFORM:YORIGIN?'))
            y_data = self.scope.query_binary_values('WAVEFORM:DATA?', datatype='h', container=np.array, is_big_endian=False)
            y_data = y_data * y_incr + y_orig

            if len(y_data) == 0:
                raise ValueError(f"No data collected from channel {channel}.")


            if collected_data is None:
                print("collecting time data")
                x_incr = float(self.scope.query('WAVEFORM:XINCREMENT?'))
                x_orig = float(self.scope.query('WAVEFORM:XORIGIN?'))
                num_points = int(self.scope.query('WAVEFORM:POINTS?'))
                time_data = np.linspace(x_orig, x_orig + x_incr * (num_points - 1), num_points)
                collected_data = pd.DataFrame({'Time (s)': time_data})
            
            collected_data[f'Channel {channel} Voltage (V)'] = y_data

        
        return collected_data
        

    def acquire_slow_save_data(self, channels, window=00):   
        """
        Function to sample data from multiple channels when a trigger has been manually 
        set on the oscilloscope. This is a slower method of acquiring data, and is used
        when the read speed is slow. It saves the data to a file rather than returning it.
        
        Inputs:
         - channels (list of int): List of channels to collect data from
         - window (int): Name for saving the data

        Returns:
         - filename (str): File path of the saved data
        """

        if self.read_speed is None:
            raise ValueError("Scope read speed not set. Please configure the scope first.")
        elif self.read_speed is True:
            print("Warning: Scope is set to high speed. Consider using acquire_slow_return_data() instead.")

        collected_data = None

        for channel in channels:
            self.scope.write('WAVEFORM:BYTEORDER LSBFIRST')
            self.scope.write(f'WAVEFORM:SOURCE CHANNEL{channel}')

            print(f"Collecting data from channel {channel}...")
            y_incr = float(self.scope.query('WAVEFORM:YINCREMENT?'))
            y_orig = float(self.scope.query('WAVEFORM:YORIGIN?'))
            y_data = self.scope.query_binary_values('WAVEFORM:DATA?', datatype='h', container=np.array, is_big_endian=False)
            y_data = y_data * y_incr + y_orig

            if len(y_data) == 0:
                raise ValueError(f"No data collected from channel {channel}.")


            if collected_data is None:
                print("collecting time data")
                x_incr = float(self.scope.query('WAVEFORM:XINCREMENT?'))
                x_orig = float(self.scope.query('WAVEFORM:XORIGIN?'))
                num_points = int(self.scope.query('WAVEFORM:POINTS?'))
                time_data = np.linspace(x_orig, x_orig + x_incr * (num_points - 1), num_points)
                collected_data = pd.DataFrame({'Time (s)': time_data})
            
            collected_data[f'Channel {channel} Voltage (V)'] = y_data

        
        channels_str = "_".join(map(str, channels))
        filename = self.save_data(collected_data, f"channels_{channels_str}_data", window)
        return filename