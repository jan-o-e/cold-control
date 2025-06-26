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

    def __init__(self, scope_id="USB0::0x0957::0x17A0::MY54280441::0::INSTR"):#'USB0::0x2A8D::0x900E::MY53450121::0::INSTR'):
        self.scope_id = scope_id
        self.read_speed = None

        try:
            self.rm = visa.ResourceManager()
            self.scope = self.rm.open_resource(scope_id)
            self.scope.timeout = 30000

            print("Conectado al osciloscopio:", self.scope.query("*IDN?"))
            self.scope.write("*RST")
            self.scope.write("*CLS")

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


    @staticmethod
    def csv_analysis(filename):
        """
        Static method to plot data from a csv
        Inputs:
         - filename (str): path to the file from which to extract the data
        """
        data = pd.read_csv(filename)
        title = filename.split("\\")[-1]
        plt.figure(figsize=(10, 6))
        plt.plot(data['Time (s)'], data['Voltage (V)'], linestyle="None", marker=".", color="black")
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.title(title)
        plt.grid(True)
        plt.show()

    @staticmethod
    def process_scope_data(filename):
        df = pd.read_csv(filename)
        print("Data Overview:")
        print(df.head())
        print("\nSummary Statistics:")
        print(df.describe())
        
        plt.figure(figsize=(10, 6))
        for column in df.columns:
            if "Voltage (V)" in column:
                plt.plot(df['Time (s)'], df[column], label=column)
        
        plt.title("Oscilloscope Data")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        
        plt.show()


    def configure_scope(self, data_chs, samp_rate=1e10, timebase_range=1e-8, centered_0=False,\
                        high_speed = False, high_impedance=True):
        """
        Function to configure the general scope settings.
        Inputs:
         - samp_rate (float): Rate at which samples are collected
         - timebase_range (float): How long to collect samples for
         - centered_0 (bool): if True, the time axis will be centered on 0. 
            If false, the time axis will start at 0.
         - high_speed (bool): TODO if True, the scope will be set to read data at high speed.
         - high_impedance (bool): if True, the scope will be set to high impedance mode.
        """

        self.read_speed = high_speed
        self.scope.write('*CLS')
        print("configuring the scope settings")
        self.scope.write('ACQUIRE:TYPE HRESOLUTION')
        #self.scope.write(f'ACQUIRE:SRATE:ANALOG {samp_rate}') # 9000 series scope
        if not centered_0:
            self.scope.write(':TIMEBASE:REFERENCE LEFT')
        else:
            self.scope.write(f":TIMEBASE:POSITION 0")
        self.scope.write(f':TIMEBASE:RANGE {timebase_range}')
        
        if high_impedance:
            for channel in data_chs:
                #self.scope.write(f":CHANnel{channel}:INPut DC") # 9000 series scope
                self.scope.write(f":CHANNEL{channel}:IMPEDANCE ONEMEG")
        
        self.scope.write('WAVEFORM:FORMAT WORD')
        #self.scope.write('WAVEFORM:STREAMING OFF') # 9000 series scope

        query_result = self.scope.query('*OPC?')  # Wait for all operations to complete
        if query_result.strip() != '1':
            raise RuntimeError(f"Oscilloscope configuration failed: {query_result.strip()}")
        
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
        self.scope.write(":TRIGGER:SWEEP NORMAL")#TRIGGERED") for 9000 series scope
        self.scope.write(":TRIGGER:MODE EDGE")
        self.scope.write(f":TRIGGER:EDGE:COUPLING DC")
        self.scope.write(f":TRIGGER:EDGE:SOURCE CHANNEL{trigger_channel}")
        self.scope.write(f":TRIGGER:EDGE:LEVEL {trigger_level}")
        
        if trigger_slope == "+":
            self.scope.write(":TRIGGER:EDGE:SLOPE POSITIVE")
        elif trigger_slope == "-":
            self.scope.write(":TRIGGER:EDGE:SLOPE NEGATIVE")
        else:
            raise ValueError(f"Invalid value for trigger_slope: {trigger_slope}")

        query_result = self.scope.query('*OPC?')  # Wait for all operations to complete
        if query_result.strip() != '1':
            raise RuntimeError(f"Trigger configuration failed: {query_result.strip()}")
        print(f"Trigger configured on channel {trigger_channel} with level {trigger_level} V and slope '{trigger_slope}'.")

    

    def set_to_digitize(self, channels=[1, 2]):
        """
        Function to set the scope to digitize mode. This is the primary way to collect
        data from the scope. Use this before sending a trigger pulse to the scope.
        """
        # # # HACK - This allows multiple channels to be digitized at once.
        # write_text = "DIGITIZE:"
        # for channel in channels:
        #     write_text += f" CHANNEL{channel},"

        # write_text = write_text[:-1]  # Remove the last comma

        #print(f"Oscilloscope set to digitize mode for channels {channels}.")
        query_result = self.scope.query(':DIGitize;*OPC?')
        if query_result.strip() == '1':
            print(f"Oscilloscope digitized channels {channels}.")
        
        return query_result.strip() == '1'

    def reset_scope(self):
        """
        Function to reset the oscilloscope. This will clear all settings and data.
        """
        self.scope.clear()
        self.scope.write('*RST')

    def clear_scope(self):
        """
        Function to clear the oscilloscope. This will clear all settings and data.
        """
        self.scope.clear()
        print("Oscilloscope cleared.")

    def check_errors(self):
        """
        Function to return any errors that may have occurred during the operation of the oscilloscope.
        """
        error = self.scope.query('SYSTem:ERRor?')
        if error != '0,"No error"':
            return f"Oscilloscope Error: {error}"
        else:
            return "No errors detected in the oscilloscope."


    def read_slow_return_data(self, channels):   
        """
        Function to sample data from multiple channels when a trigger has been manually 
        set on the oscilloscope. This is a slower method of acquiring data, and is used
        when the read speed is slow. It returns the data as a DataFrame rather than saving
        it to a file.
        
        Inputs:
         - channels (list of int): List of channels to collect data from
         - window (int): Name for saving the data

        Returns:
         - filename (str): File path of the saved data
        """

        if self.read_speed is None:
            raise ValueError("Scope read speed not set. Please configure the scope first.")
        elif self.read_speed is True:
            print("Warning: Scope is set to high speed.")

        collected_data = None

        self.scope.write('WAVEFORM:BYTEORDER LSBFIRST')


        for channel in channels:
            self.scope.write(f'WAVEFORM:SOURCE CHANNEL{channel}')

            print(f"Collecting data from channel {channel}...")
            errors = self.scope.query('SYSTem:ERRor?')  # Check for errors
            print(f"Errors: {errors.strip()}")  # Print any errors that might have occurred
            preamble = self.scope.query('WAVEFORM:PREAMBLE?')  # Get preamble information
            opc = self.scope.query('*OPC?')  # Wait for operation complete
            if opc.strip() != '1':
                raise RuntimeError(f"Operation did not complete successfully. OPC returned: {opc.strip()}")
            print(f"Preamble info: {preamble}")
            y_incr = float(self.scope.query('WAVEFORM:YINCREMENT?'))
            y_orig = float(self.scope.query('WAVEFORM:YORIGIN?'))
            y_data = self.scope.query_binary_values('WAVEFORM:DATA?', datatype='H', container=np.array, is_big_endian=False)
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
            y_data = self.scope.query_binary_values('WAVEFORM:DATA?', datatype='H', container=np.array, is_big_endian=False)
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
    




    def arm_scope(self, max_acq_wait_sec=10, poll_interval_sec=0.1):
        """
        Function to arm the oscilloscope ready to collect data when it receives a trigger.
        """
        self.scope.write(':SINGLE')

        print("Waiting for oscilloscope to arm (polling :AER?)...\n") 
        StartTime = time.perf_counter()
        armed_status = 0
        acq_started = False

        while armed_status != 1 and (time.perf_counter() - StartTime) <= max_acq_wait_sec:
            try:
                # Query :AER?. It returns 1 when armed and is cleared upon reading [3, 33].
                # We need to capture the 1 when it first appears.
                query_result = self.scope.query(":AER?")
                armed_status = int(query_result)
                if armed_status == 1:
                    acq_started = True # Armed state indicates acquisition is waiting for trigger
                    break # Exit loop once armed
            except Exception as e:
                print(f"Error during arming poll: {e}")
                acq_started = False
                break
            time.sleep(poll_interval_sec)

        if not acq_started:
            print("Oscilloscope did not arm within the maximum wait time.")
            self.scope.clear() 
            self.scope.close()
            raise  RuntimeError("Oscilloscope failed to arm within the specified time.")

        print("Oscilloscope is armed and ready for trigger!")
        return True
    

    def wait_for_acquisition(self, max_acq_wait_sec=1, poll_interval_sec=0.01):
        # --- Wait for Acquisition to Complete ---
        # Now that the trigger pulse is sent, the oscilloscope should trigger and acquire data.

        print("Waiting for acquisition to complete...") 
        StartTime = time.perf_counter() # Reset timer for acquisition wait
        triggered = 0
        success = False


        # Wait until acquisition is done  or timeout
        while success != True and (time.perf_counter() - StartTime) <= max_acq_wait_sec:
            time.sleep(poll_interval_sec)
            try:
                acq_complete = self.scope.query(":ACQuire:COMPlete?")
                if triggered == 0:
                    # Once the triggered event register is read, it is reset to zero
                    triggered = bool(self.scope.query(":TER?"))
                run_state = self.scope.query(":RSTATE?")

                acq_complete_bool = int(acq_complete) == 100
                run_state_bool = run_state.strip() == "STOP"

                success = acq_complete_bool and run_state_bool and triggered

                if success:
                    break # Exit loop once acquisition is done
            except Exception as e:
                print(f"Error during completion poll: {e}")
                break

            time.sleep(poll_interval_sec)

        if success != True:
            print("Acquisition did not complete within the maximum wait time.")

        print(f"Acquisition complete: {acq_complete_bool}")
        print(f"Triggered: {triggered}")
        print(f"Run state: {run_state.strip()}")

        print("Acquisition complete. Ready to retrieve data.\n") # [14]
        return success