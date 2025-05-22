import numpy as np
import pyvisa as visa
import pandas as pd
import os
from datetime import datetime
import time
import matplotlib.pyplot as plt

# herramienta poderosa para interactuar con un osciloscopio, adquirir datos, guardarlos y visualizarlos. 
# La clase permite la adquisición de datos en diferentes configuraciones, incluyendo la adquisición basada en triggers.

class oscilloscope_manager:

    def __init__(self, scope_id='USB0::0x2A8D::0x900E::MY53450121::0::INSTR'):
        self.scope_id = scope_id

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


    @staticmethod   # Se comporta como una función independiente, pero está encapsulada dentro de la clase por razones de organización.
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


    def configure_scope(self, samp_rate=1e10, timebase_range=1e-8, centered_0=False):
        # Configurar el osciloscopio para iniciar muestreo
        print("configuring the scope settings")
        self.scope.write('ACQUIRE:MODE HRESOLUTION')
        self.scope.write(f'ACQUIRE:SRATE:ANALOG {samp_rate}')
        self.scope.write(f'TIMEBASE:RANGE {timebase_range}') 
        self.scope.write(f'TIMEBASE:POSITION {timebase_range / 2}')
        if centered_0:
            self.scope.write(f"TIMEBASE:POSITION 0") 
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
        # Set the trigger level and slope#
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
    

    def set_to_digitize(self):
        """
        Function to set the scope to digitize mode. 
        """
        # Set the scope to digitize mode
        self.scope.write(':DIGITIZE CHANNEL1 CHANNEL2')
        print("Oscilloscope set to digitize mode.")


    # Function to acquire data from a single channel
    def acquire_single_channel(self, channel, samp_rate = 1e10, timebase_range=1e-8, save_file = False):
        """
        Function to sample data from a single channel.
        WARNING: if the product of samp_rate and timebase_range is too large (>>1e3)
        then too many samples will be collected and the program may crash.

        Inputs:
         - channel (int): Channel to collect data from
         - samp_rate (float): Rate at which samples are collected
         - timebase_range (float): How long to collect samples for
         - save_file (bool): Option to save the collected data in a csv file

        Returns:
         - collected_data (pd.DataFrame): Dataframe of time and voltage values at each sample
        """

        # Set up scope to begin sampling
        self.scope.write('ACQUIRE:MODE HRESOLUTION')
        self.scope.write('ACQUIRE:SRATE:ANALOG {}'.format(samp_rate))
        self.scope.write('TIMEBASE:RANGE {}'.format(timebase_range)) 
        self.scope.write('WAVEFORM:FORMAT WORD')
        self.scope.write('WAVEFORM:STREAMING OFF')
        self.scope.write(':RUN')

        # Start collecting samples
        self.scope.write(f'DIGITIZE CHANNEL{channel}')
        self.scope.write('WAVEFORM:BYTEORDER LSBFIRST')
        self.scope.write(f'WAVEFORM:SOURCE CHANNEL{channel}')

        # Collect y (voltage) data
        y_incr = float(self.scope.query('WAVEFORM:YINCREMENT?'))
        y_orig = float(self.scope.query('WAVEFORM:YORIGIN?'))
        y_data = self.scope.query_binary_values('WAVEFORM:DATA?', datatype='h', container=np.array, is_big_endian=False)
        y_data = y_data * y_incr + y_orig  # Apply scaling

        # Collect x (time) data
        x_incr = float(self.scope.query('WAVEFORM:XINCREMENT?'))
        x_orig = float(self.scope.query('WAVEFORM:XORIGIN?'))
        num_points = int(self.scope.query('WAVEFORM:POINTS?'))
        time_data = np.linspace(x_orig, x_orig + x_incr * (num_points - 1), num_points)

        # Store x and y data in a pandas dataframe
        collected_data = pd.DataFrame({'Time (s)': time_data, 'Voltage (V)': y_data})
        
        # If the data should be saved to a csv, save the data
        if save_file:
            filename = self.save_data(collected_data, f"channel_{channel}_data")
            return collected_data, filename

                        
        # Return the dataframe of collected data
        return collected_data
    
                    
    
    def acquire_with_trigger(self, channel, samp_rate = 1e10, timebase_range=1e-8, save_file = False, window=00, centered_0=False): 
        """
        Function to sample data from a single channel when a trigger has been manually set on the oscilloscope.
        (Alternative to using the get_data_triggered function if it doesn't work).
        WARNING: if the product of samp_rate and timebase_range is too large (>>1e3)
        then too many samples will be collected and the program may crash.

        Inputs:
         - channel (int): Channel to collect data from
         - samp_rate (float): Rate at which samples are collected
         - timebase_range (float): How long to collect samples for
         - save_file (bool): Option to save the collected data in a csv file

        Returns:
         - collected_data (pd.DataFrame): Dataframe of time and voltage values at each sample
        """
        try:
            # Set up scope to begin sampling
            self.scope.write('ACQUIRE:MODE HRESOLUTION')
            self.scope.write('ACQUIRE:SRATE:ANALOG {}'.format(samp_rate))
            self.scope.write('TIMEBASE:RANGE {}'.format(timebase_range)) 
            self.scope.write(f"TIMEBASE:POSITION {timebase_range / 2}") # starts measurements at time 0
            if centered_0:
                self.scope.write(f"TIMEBASE:POSITION 0")  # centers measurements on 0 
            self.scope.write('WAVEFORM:FORMAT WORD')
            self.scope.write('WAVEFORM:STREAMING OFF')
            self.scope.write(':RUN')
            # actual_samp_rate = float(self.scope.query('ACQUIRE:SRATE?'))
            # print(f"Sample rate real: {actual_samp_rate} Hz")
            points = timebase_range * samp_rate
            self.scope.write(f'ACQUIRE:POINTS {points}')  # Needed if we want timebase range smaller than 1us

            # Start collecting samples
            self.scope.write(f'DIGITIZE CHANNEL{channel}')
            self.scope.write('WAVEFORM:BYTEORDER LSBFIRST')
            self.scope.write(f'WAVEFORM:SOURCE CHANNEL{channel}')

            # Collect y (voltage) data
            y_incr = float(self.scope.query('WAVEFORM:YINCREMENT?'))
            y_orig = float(self.scope.query('WAVEFORM:YORIGIN?'))
            y_data = self.scope.query_binary_values('WAVEFORM:DATA?', datatype='h', container=np.array, is_big_endian=False)
            y_data = y_data * y_incr + y_orig  # Apply scaling

            # Collect x (time) data
            x_incr = float(self.scope.query('WAVEFORM:XINCREMENT?'))
            x_orig = float(self.scope.query('WAVEFORM:XORIGIN?'))
            num_points = int(self.scope.query('WAVEFORM:POINTS?'))
            time_data = np.linspace(x_orig, x_orig + x_incr * (num_points - 1), num_points)

            # Store x and y data in a pandas dataframe
            collected_data = pd.DataFrame({'Time (s)': time_data, 'Voltage (V)': y_data})
            
            # If the data should be saved to a csv, save the data
            if save_file:
                filename = self.save_data(collected_data, f"channel_{channel}_data", window)
                return collected_data, filename
                            
            # Return the dataframe of collected data
            return collected_data
        
        finally:
            # Poner el osciloscopio en modo RUN después de adquirir los datos
            self.scope.write(":RUN")
            print("Osciloscopio puesto en modo RUN.")


    def acquire_with_trigger_multichannel(self, channels, save_file=False, window=00):   
        """
        Function to sample data from multiple channels when a trigger has been manually set on the oscilloscope.
        
        Inputs:
         - channels (list of int): List of channels to collect data from
         - sample_rate (float): Rate at which samples are collected
        - timebase_range (float): How long to collect samples for
        - save_file (bool): Option to save the collected data in a csv file
        - window (int): Name for saving the data
        - centered_0 (bool): if True, the time axis will be centered on 0. If false, the time axis will start at 0.

        Returns:
         - collected_data (pd.DataFrame): Datafram with time and voltage values for each channel
        """
        collected_data = None

        for channel in channels:
            self.scope.write('WAVEFORM:BYTEORDER LSBFIRST')
            self.scope.write(f'WAVEFORM:SOURCE CHANNEL{channel}')

            print(f"Collecting data from channel {channel}...")
            y_incr = float(self.scope.query('WAVEFORM:YINCREMENT?'))
            y_orig = float(self.scope.query('WAVEFORM:YORIGIN?'))
            print("reached this point")
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

        
        
        if save_file:
            channels_str = "_".join(map(str, channels))
            filename = self.save_data(collected_data, f"channels_{channels_str}_data", window)
            return collected_data, filename
    
        return collected_data
        

            

    def get_data_triggered(self, trigger_channel, data_channel, trigger_level, scope_opts=None):
        """
        Adquiere datos con trigger, esperando activamente a que se cumpla la condición de trigger.
        """
        defaults = {"data_scale": 0.1, "time_scale": 5e-7, "trigger_scale": 5e-3}
        if scope_opts:
            for key in defaults:
                if key in scope_opts:
                    defaults[key] = scope_opts[key]

        try:
            # Configurar el osciloscopio
            self.scope.write(":RUN")
            self.scope.write(f"CHANNEL{trigger_channel}:SCALE {defaults['trigger_scale']}")
            self.scope.write(f":CHANNEL{data_channel}:SCALE {defaults['data_scale']}")
            self.scope.write(f":TIMEBASE:SCALE {defaults['time_scale']}")

            # Configurar el trigger
            print(f"Configurando trigger en el canal {trigger_channel} con nivel {trigger_level} V...")
            self.scope.write(f":TRIGGER:LEVEL CHANNEL{trigger_channel} {trigger_level}")
            self.scope.write(":TRIGGER:MODE EDGE")
            self.scope.write(":TRIGGER:SWEEP TRIGGERED")
            self.scope.write(f":TRIGGER:EDGE:SOURCE CHANNEL{trigger_channel}")
            self.scope.write(":TRIGGER:EDGE:SLOPE POSITIVE")

            # Iniciar la adquisición
            print("Iniciando adquisición...")
            self.scope.write(":DIGITIZE")

            # Esperar activamente a que se cumpla el trigger
            max_attempts = 100  # Número máximo de intentos para evitar un bucle infinito
            for _ in range(max_attempts):
                trigger_state = self.scope.query(":TRIGGER:STATE?").strip()
                print(f"Estado del trigger: {trigger_state}")

                if trigger_state == "READY":
                    print("Trigger detectado. Adquiriendo datos...")
                    break
                elif trigger_state == "ARMED":
                    time.sleep(0.1)  # Esperar 100 ms antes de verificar nuevamente
                else:
                    raise ValueError(f"Estado del trigger no reconocido: {trigger_state}")
            else:
                raise TimeoutError("El trigger no se activó después de varios intentos.")

            # Adquirir los datos
            self.scope.write(f'WAVEFORM:SOURCE CHANNEL{data_channel}')
            self.scope.write('WAVEFORM:BYTEORDER LSBFIRST')

            # Leer los datos
            y_incr = float(self.scope.query('WAVEFORM:YINCREMENT?'))
            y_orig = float(self.scope.query('WAVEFORM:YORIGIN?'))
            y_data = self.scope.query_binary_values('WAVEFORM:DATA?', datatype='h', container=np.array, is_big_endian=False)
            y_data = y_data * y_incr + y_orig  # Aplicar escalado

            # Leer los datos de tiempo
            x_incr = float(self.scope.query('WAVEFORM:XINCREMENT?'))
            x_orig = float(self.scope.query('WAVEFORM:XORIGIN?'))
            num_points = int(self.scope.query('WAVEFORM:POINTS?'))
            time_data = np.linspace(x_orig, x_orig + x_incr * (num_points - 1), num_points)

            # Guardar los datos en un DataFrame
            collected_data = pd.DataFrame({'Time (s)': time_data, 'Voltage (V)': y_data})
            filename = self.save_data(collected_data, f"channel{data_channel}")

            return filename

        except visa.Error as e:
            print(f"Error durante la adquisición: {e}")
            print("Reiniciando la conexión con el osciloscopio...")
            self.scope.close()
            self.scope = self.rm.open_resource(self.scope_id)
            raise
        except (ValueError, TimeoutError) as e:
            print(e)
            return None


    def aux_triggered_multichannel(self, data_channels, trigger_level, trigger_slope="+", scope_opts=None):
        """
        Gets data from the scope, with the scope waveform based on edge triggering.

        Inputs:
         - data_channels (list of int): channels on which to collect data
         - trigger_level (float): level at which to trigger
         - trigger_slope (str): whether the slope should be positive, + or negative, -
         - scope_opts (dict): dictionary containing data about the horizontal and vertical scaling of the waveforms
        
        Returns:
         - filename (str): file path of file in which the data is stored
        """
        # adquiere datos de varios canales cuando se cumple una condición de trigger (antes solo uno)

        defaults = {"data_scale": 5e-2, "time_scale": 5e-7, "trigger_scale": 5e-3}
        # Change default values to those specified in scope_opts
        if scope_opts:
            for key in defaults:
                if key in scope_opts:
                    defaults[key] = scope_opts[key]

        # Set the scope to run mode
        self.scope.write(":RUN")

        # Set the voltage scales for the data channel, and the timebase scale
        #self.scope.write(f"CHANNEL{trigger_channel}:SCALE {defaults["trigger_scale"]}")
        for channel in data_channels:
            self.scope.write(f":CHANNEL{channel}:SCALE {defaults['data_scale']}")
        self.scope.write(f":TIMEBASE:SCALE {defaults['time_scale']}")

        # Set the scope to trigger on the rising edge of the trigger channel
        self.scope.write(f":TRIGGER:LEVEL AUX {trigger_level}")
        self.scope.write(":TRIGGER:MODE EDGE")
        self.scope.write(":TRIGGER:SWEEP TRIGGERED")
        self.scope.write(f":TRIGGER:EDGE:SOURCE AUX")
        if trigger_slope == "+":
            self.scope.write(":TRIGGER:EDGE:SLOPE POSITIVE")
        elif trigger_slope == "-":
            self.scope.write(":TRIGGER:EDGE:SLOPE NEGATIVE")
        else:
            raise ValueError(f"Invalid value for trigger_slope: {trigger_slope}")


        # Stop data collection?
        self.scope.write(":STOP")

        # Collect the data from the scope

        # Collect x (time) data
        x_incr = float(self.scope.query('WAVEFORM:XINCREMENT?'))
        x_orig = float(self.scope.query('WAVEFORM:XORIGIN?'))
        num_points = int(self.scope.query('WAVEFORM:POINTS?'))

        time_data = np.linspace(x_orig, x_orig + x_incr * (num_points - 1), num_points)

        # Store x and y data in a pandas dataframe
        collected_data = pd.DataFrame({'Time (s)': time_data})


        for channel in data_channels:
            self.scope.write(f'DIGITIZE CHANNEL{channel}')
            self.scope.write('WAVEFORM:BYTEORDER LSBFIRST')
            self.scope.write(f'WAVEFORM:SOURCE CHANNEL{channel}')

            # Collect y (voltage) data
            y_incr = float(self.scope.query('WAVEFORM:YINCREMENT?'))
            y_orig = float(self.scope.query('WAVEFORM:YORIGIN?'))
            y_data = self.scope.query_binary_values('WAVEFORM:DATA?', datatype='h', container=np.array, is_big_endian=False)
            y_data = y_data * y_incr + y_orig  # Apply scaling


            # append data to dataframe
            collected_data[f"Channel {channel} Voltage (V)"] = y_data

        
        channels = "".join([f"{i}" for i in data_channels])
        filename = self.save_data(collected_data, f"channels{channels}")
        return filename


