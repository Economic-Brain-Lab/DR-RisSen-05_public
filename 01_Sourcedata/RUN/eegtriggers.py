# -*- coding: utf-8 -*-
# @Author: Dragan Rangelov <uqdrange>
# @Date:   08-3-2019
# @Email:  d.rangelov@uq.edu.au
# @Last modified by:   uqdrange
# @Last modified time: 15-10-2019
# @License: CC-BY-4.0
#===============================================================================
# issue tracker
#===============================================================================
# %%
#===============================================================================
# importing libraries
#===============================================================================
from __future__ import division, print_function
# import parallel port drivers
try:
    from psychopy.parallel._inpout32 import PParallelInpOut32 as ParallelPort
except ImportError:
    from psychopy.parallel._inpout import PParallelInpOut as ParallelPort
# import USB to LPT API if it is available
try:
    import ftd2xx as ftd
    USB2LPT = False
    if ftd.listDevices():
        USB2LPT = True
except:
    USB2LPT = False
import numpy as np
import time

# NOTE: Messages should not be divisible by 256 as the trigger is then 0 which cannot be sent
def getTriggers(mssg):
    mssg += 256
    return np.packbits(np.array(np.split(np.unpackbits(np.array([mssg],
                                                       dtype = np.dtype('>u2')).view(np.uint8)),
                                         2)))
def getMessage(triggers):
    return np.packbits(np.concatenate(np.split(np.unpackbits(np.array(triggers,
                                                               dtype = np.uint8)), 2)[::-1])).view(np.uint16) - 256
class EegTriggerLPT():
    def __init__(self, portNumber = 0xD050):
        '''
        Open parallel port to communicate with the EEG system
        Params:
            - portNumber: hexadecimal number containing the port address on the
                          presentation computer
        '''
        self.port = ParallelPort(address = portNumber)
        self.sendTrigger(0)

    def sendTrigger(self, message = 0, wait = .005, bytes = 1):
        '''
        Send message to the EEG system
        **NOTE** it does not do anything on MacOS
        Params:
            - message: what number should be sent, int 0-255
            - wait: how long to wait (in sec) before resetting the trigger
            - bytes: sent one or two bytes triggers
        '''
        try:
            triggers = np.array([message])
            if bytes == 2:
                triggers = getTriggers(message)
            for idx_trig, trigger in enumerate(triggers):
                self.port.setData(int(trigger))
                time.sleep(wait)
                self.port.setData(0)
                if idx_trig > 0:
                    time.sleep(wait)
        except NotImplementedError:
            pass
class EegTriggerUSB():
    def __init__(self):
        '''
        Open parallel port to communicate with the EEG system
        Params:
            - portNumber: hexadecimal number containing the port address on the
                          presentation computer
        '''
        self.port = ftd.open()
        self.port.resetPort()
        self.port.setBitMode(0xFF, 1)
        self.sendTrigger(0)

    def sendTrigger(self, message = 0, wait = .005, bytes = 1):
        '''
        Send message to the EEG system
        **NOTE** it does not do anything on MacOS
        Params:
            - message: what number should be sent, int 0-255
            - wait: how long to wait (in sec) before resetting the trigger
            - bytes: sent one or two bytes triggers
        '''
        try:
            triggers = np.array([message])
            if bytes == 2:
                triggers = getTriggers(message)
            for idx_trig, trigger in enumerate(triggers):
                self.port.write(chr(trigger))
                time.sleep(wait)
                self.port.write(chr(0))
                if idx_trig > 0:
                    time.sleep(wait)
        except NotImplementedError:
            pass

if __name__ == '__main__':
    messages = np.arange(1, 256).astype('int')
    if USB2LPT:
        triggers = EegTriggerUSB()
    else:
        triggers = EegTriggerLPT()
    for mssg in messages:
        triggers.sendTrigger(mssg)
        time.sleep(.500)
