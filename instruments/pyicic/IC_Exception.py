# -*- coding: utf-8 -*-
#import exceptions

class IC_Exception(Exception):
    """
    An exception for the IC imaging control software. It contains a message
    property which is a string indicating what went wrong.
    
    error code -3 has multiple possible interpretations, sometimes from the same function!
    
    :param errorCode: Error code to be used to look up error message.
    """

    @property
    def message(self):
        return self._error_codes[self.warning_code]
            
    @property
    def warning_code(self):
        return self._error_code
        
    _error_codes = {    # IC errors
                        1   :   'IC SUCCESS',
                        0   :   'IC ERROR',
                       -1   :   'IC NO HANDLE',
                       -2   :   'IC NO DEVICE',
                       -3   :   'IC NOT AVAILABLE / IC NO PROPERTYSET / IC DEFAULT WINDOW SIZE SET / IC NOT IN LIVEMODE',
                       -4   :   'IC PROPERTY ITEM NOT AVAILABLE',
                       -5   :   'IC PROPERTY ELEMENT NOT AVAILABLE',
                       -6   :   'IC PROPERTY ELEMENT WRONG INTERFACE',
                       
                        # other errors
                     -100   :   'UNKNOWN ERROR',
                     -101   :   'UNKNOWN DEVICE FEATURE',
                     -102   :   'VIDEO NORM INDEX OUT OF RANGE',
                     -103   :   'VIDEO FORMAT INDEX OUT OF RANGE',
                     -104   :   'VIDEO NORM RETURNED NULL TYPE',
                     -105   :   'VIDEO FORMAT RETURNED NULL TYPE',
                     -106   :   'DEVICE NAME NOT FOUND'
                     
                   }
    
    def __init__(self, warning_code):
        # if error code does not match expected codes then assign invalid code
        if warning_code in self._error_codes:
            self._error_code = warning_code
        else:
            self._error_code = -100


