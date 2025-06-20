from oscilloscope_manager import OscilloscopeManager


data_chs = [1,4]
time_range = 5e-3
centred_0 = False
trig_ch = 1
trig_lvl = 1

scope = OscilloscopeManager()
scope.reset_scope()
scope.configure_scope(data_chs, timebase_range=time_range, centered_0=centred_0)
scope.configure_trigger(trig_ch, trig_lvl)
scope.set_to_run()