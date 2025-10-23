'''
Author: Andrew Mckay (Andrew.Mckay@uq.edu.au)
File Created: 2023-05-03
-----
Last Modified: 2023-05-03
Modified By: Andrew Mckay (Andrew.Mckay@uq.edu.au)
-----
Licence: Creative Commons Attribution 4.0
Copyright 2023 Andrew Mckay, The University of Queensland
'''
#===============================================================================
# %% import libraries
#===============================================================================
import sys

from numpy.core.fromnumeric import size
sys.dont_write_bytecode = True

from auxfunctions import createMonitor, sd2k, wrapTopi, wrapTo90
from datetime import datetime
import eegtriggers as eegtrigs
from eyelink import TrackerEyeLink
from experimentinfo import ExperimentInfo
import itertools
import json
import numpy as np
import pandas as pd
from pathlib import Path
import psutil
from psychopy import core, monitors, logging, event, visual, misc, tools
import random
import glob
import os
#===============================================================================
# setting process priority
#===============================================================================
ps = psutil.Process()
if sys.platform == 'win32': ps.nice(psutil.REALTIME_PRIORITY_CLASS)
else: ps.nice(0)

#========================================================================
# %% Define functions
#========================================================================
## Porting psychopy's contains function to elementArrayStim
def contains(thisElementArrayStim, x, y=None, units=None):
    """Returns True if a point x,y is inside the stimulus' border.

    Can accept variety of input options:
        + two separate args, x and y
        + one arg (list, tuple or array) containing two vals (x,y)
        + an object with a getPos() method that returns x,y, such
            as a :class:`~psychopy.event.Mouse`.

    Returns `True` if the point is within the area defined either by its
    `border` attribute (if one defined), or its `vertices` attribute if
    there is no .border. This method handles
    complex shapes, including concavities and self-crossings.

    Note that, if your stimulus uses a mask (such as a Gaussian) then
    this is not accounted for by the `contains` method; the extent of the
    stimulus is determined purely by the size, position (pos), and
    orientation (ori) settings (and by the vertices for shape stimuli).

    See Coder demos: shapeContains.py
    """
    # get the object in pixels
    if hasattr(x, 'border'):
        xy = x._borderPix  # access only once - this is a property
        units = 'pix'  # we can forget about the units
    elif hasattr(x, 'verticesPix'):
        # access only once - this is a property (slower to access)
        xy = x.verticesPix
        units = 'pix'  # we can forget about the units
    elif hasattr(x, 'getPos'):
        xy = x.getPos()
        units = x.units
    elif type(x) in [list, tuple, np.ndarray]:
        xy = np.array(x)
    else:
        xy = np.array((x, y))
    # try to work out what units x,y has
    if units is None:
        if hasattr(xy, 'units'):
            units = xy.units
        else:
            units = thisElementArrayStim.units
    if units != 'pix':
        xy = tools.monitorunittools.convertToPix(xy, pos=(0, 0), units=units, win=thisElementArrayStim.win)
    # ourself in pixels
    if hasattr(thisElementArrayStim, 'border'):
        poly = thisElementArrayStim._borderPix  # e.g., outline vertices
    else:
        poly = thisElementArrayStim.verticesPix[:, :, 0:2]  # e.g., tesselated vertices

    polyIdx = None
    foundPoly = False
    for idx, thisPoly in enumerate(poly):
        is_in = visual.helpers.pointInPolygon(xy[0], xy[1], thisPoly)
        if is_in:
            currPoly = thisPoly
            polyIdx = idx
            foundPoly = True
            break
    if not foundPoly:
        polyIdx = None

    return is_in, polyIdx
    
def main(info, params, trials, intro):
    '''
    Main experiment routine
    Params:
        - info: session specific information
        - params: parameters for the experiment
        - trials: data frame with trial parameters
        - intro: instructions
    '''
    #===========================================================================
    # setting experimental parameters
    #===========================================================================
    # GENERAL
    refreshRate = int(info.monitorRefreshRate)
    runfile = info.runfile
    monitor = monitors.Monitor(info.monitorName)

    # SESSION
    nTrialsPerBlock = 100
    nBlocks = 1
    nFramesPerTrial = int(params['TRIAL']['responseDeadline']['value'] * refreshRate)
    nFramesRespDelay = int(params['TRIAL']['responseDelay']['value'] * refreshRate)
    nFramesPerStimOn = int(params['TRIAL']['stimOn']['value'] * refreshRate)
    nFramesPerStimOff = int(params['TRIAL']['stimOff']['value'] * refreshRate)
    nFramesPerFeedback = int(0.5 * refreshRate)
    nFramesPerFixation = int(params['TRIAL']['fixationDuration']['value'] * refreshRate)

    #MISC
    #===========================================================================
    # creating objects
    #===========================================================================
    win = visual.Window(
        monitor = monitor, 
        units = 'pix', color = 'gray',
        fullscr = info.fullScreen, 
        screen = int(info.monitorNumber)
    )

    trialClock = core.Clock()

    # General Text Object
    stimText = visual.TextStim(
        win,
        height = misc.deg2pix(params['STIM']['textHeight']['value'], monitor),
        wrapWidth = misc.deg2pix(params['STIM']['textWidth']['value'], monitor),
        pos = misc.deg2pix(params['STIM']['textPosition']['value'], monitor), 
        anchorHoriz = 'center', alignText = 'center'
    )
    
    #Feedback Text
    feedText = visual.TextStim(
        win,
        height = misc.deg2pix(params['STIM']['textHeight']['value'], monitor),
        wrapWidth = misc.deg2pix(params['STIM']['textWidth']['value'], monitor),
        pos = misc.deg2pix(params['STIM']['textPosition']['value'], monitor), 
        anchorHoriz = 'center', alignText = 'center'
    )
    stimText.text = 'Loading the experiment. Please wait.'
    stimText.draw()
    win.flip()

    mouse = event.Mouse(visible=True, win=win)

    elemMax = trials['maxElements'].max()		

    # Create target (1) distractor(0)
    stimType = np.array([0] * elemMax)
    stimOris = np.array([0] * elemMax)

    # Create opacity array
    stimOpac = np.array([1] * elemMax)
    NewTrialOpac = np.array([1] * elemMax)


    # Clickable object for window
    winRec = visual.rect.Rect(win = win,
        width = win.size[0],
        height = win.size[1],
        lineColor = 'Orange')

        

    # Define the resolution
    height = int(720 * 0.75)
    width = int(1280 * 0.75)

    # Define the width of the vertical line
    line_width = width // 4

    # Create an array filled with zeros
    t0s = np.zeros((height, width), dtype=int)

    # Fill in the vertical line of the T
    start_idx = width // 2 - line_width // 2
    end_idx = start_idx + line_width
    t0s[:, start_idx:end_idx] = 1

    # Fill in the horizontal line of the T
    hori_len = (line_width // 2)
    t0s[:hori_len, :] = 1
    tArray = np.flipud(t0s)



    # Code for creating target and distractor points on the window
    # #Have points that jitter and then assing positions afterwards
    maxX = 0.8 * win.size[0]
    maxY = 0.8 * win.size[1]
    nX = 25
    nY = 12
    positions = []
    for Idx in range(nX):           
        for Jdx in range(nY):
            positions.append([Idx/(nX-1)*maxX-(maxX*0.5), Jdx/(nY-1)*maxY-(maxY*0.5)]) #Multiply by half because psychopy pixel is half neg half pos


    #Incase of dynamically changing max stim on screen (not in this experiment)
    #Generate a list of elements which will contain up to the max amount of elements needed


    eas = visual.ElementArrayStim(win = win,
        nElements = elemMax,
        fieldShape = 'sqr',
        elementMask = None,
        xys = positions,
        oris = stimOris,
        opacities = stimOpac,
        elementTex = tArray,
        sizes = 40)

    #===========================================================================
    # run trials
    #===========================================================================
    pressedKeys = [] # intialize pressed keys array
    exitTrialLoop = False # initialize trial loop break flag  

    mouse.setVisible(True)
    
    for idx_trial, trial in trials.iterrows():
        
        pressed = event.getKeys(
            keyList = ['q', 'f', 'j'], 
            timeStamped = trialClock
        )

        # abort experiment
        if 'q' in pressedKeys: 
            break


        if exitTrialLoop:
            exitTrialLoop = False
            break

        # show intro text at the beginning of experiment
        if trial['runningTrialNo'] == 0 and not info.simulate:
            stimText.pos = misc.deg2pix(
                params['STIM']['textPosition']['value'], 
                monitor
            )
            stimText.alignText = 'left'
            stimText.text = '\n'.join(intro)
            while True:
                stimText.draw()
                win.flip()
                if event.getKeys(keyList=['space']):
                    break

        #Randomise Stimuli Positions
        frame = 0
        if frame == 0:
            stimType[:] = 0
            stimType[:trials.loc[idx_trial, 'targets']] = 1
            random.shuffle(stimType)
            stimOris = np.random.choice([90, 180, 270], size = elemMax)
            stimOris[stimType != 0] = 0
            eas.setOris(stimOris)
            random.shuffle(positions)
            eas.setXYs(positions)
            stimOpac[:] = 1
            eas.setOpacities(NewTrialOpac)
    
        # trial period
        terminate_trial = False
        trialClock.reset()
        clickedStims = []
        while True:
            trialNo = trial['blockTrialNo']
            if len(clickedStims) != 0:
                stimOpac[clickedStims] = 0
                eas.setOpacities(stimOpac)
            eas.draw()
            winRec.draw()
            win.flip()

            hover, stimNo = contains(eas, mouse)
    
            if mouse.isPressedIn(winRec, buttons=[0]) and hover:
                clickedStims.append(stimNo)
                clickedStims = list(set(clickedStims))

            pressed = event.getKeys(
                keyList= ['q', 'space'],
                timeStamped = trialClock)
            if pressed:
                pressedKeys, pressedTimes = zip(*pressed)
                if 'q' in pressedKeys: 
                    terminate_trial = True
                    break
                if 'space' in pressedKeys:
                    respStim = [stimOris[i] for i in clickedStims]
                    for k in respStim:
                        if k == 0:
                            trials.loc[idx_trial, 'tCancel'] += 1
                        if k != 0:	
                            trials.loc[idx_trial, 'dCancel'] += 1
                    trials.loc[idx_trial, 'accuracy'] = int(trials.loc[idx_trial, 'tCancel'] == trials.loc[idx_trial, 'targets'])
                    # Reaction Time or more specifically time taken to feel confident to move on 
                    trials.loc[idx_trial, 'RT'] = trialClock.getTime()
                    if (trialNo > 5) and (trials.loc[idx_trial, 'accuracy'] == 0):
                        reassign = np.random.choice([0,0,1,1])
                        trial['reassigned'] = reassign
                        trials.loc[idx_trial, 'accuracy'] = reassign #75% chance to rewrite to correct if incorrect after 10 trials
                    if info.feedback:
                        feedText.text = ['Incorrect', 'Correct'][int(trials.loc[idx_trial, 'accuracy'])]
                        for frame in range(nFramesPerFeedback):
                            feedText.draw()
                            win.flip()
                    terminate_trial = True
            
            #Cancel condition if requirements met
            if (trialNo > 4) and (trials.loc[
                    idx_trial - 4 : idx_trial + 1,
                    'accuracy'
                ].sum() >= 3):
                exitTrialLoop = True
                break  
            
            if terminate_trial:
                break
            
            #Increment frame count at the end of the loop 
            frame += 1
            

    # END EXPERIMENT
    stimText.pos = misc.deg2pix(
        params['STIM']['textPosition']['value'], 
        monitor
    )
    stimText.alignText = 'center'
    stimText.text = '\n'.join([
        'End of the session.',
        f'Congratulations! You have earned {int(4000 * 0.55)} Points to wager in the main experiment.',
        'Please contact the experimenter.'
    ])
    while True:
        stimText.draw()
        win.flip()
        if event.getKeys(keyList=['space']): 
            win.close()
            break 
    
    # GET THE COLLECTED DATA
    return trials


#===============================================================================
# %% run main
#===============================================================================
if __name__ == '__main__':
    RUNPATH = Path(__file__)
    ROOTPATH = RUNPATH.parent.parent
    RUNFILE = RUNPATH.stem.split('_')[-1]
    EXPERIMENT = ROOTPATH.stem
    #===========================================================================
    # get experiment info and create monitor if necessary
    #===========================================================================
    paths = sorted(glob.glob(os.path.join(ROOTPATH, 'LOG', 'gui_info_sub*.json')), key=os.path.getctime)
    with (ROOTPATH / 'RUN' / 'exp_information.json').open('r') as f:
        dataToCollect = json.load(f)
    # Dynamically change entries
    dataToCollect['expData'][8][2] = False # eye tracking
    dataToCollect['expData'][9][2] = False # gaze contingent
    dataToCollect['expData'][10][2] = False # send triggers
    gui_data = False
    if len(paths) > 0:
        gui_data = paths[-1]
    if gui_data:
        with open(gui_data, 'r') as f:
            sub_info = json.load(f)
            dataToCollect['expData'][0][2] = sub_info['monitorName']
            dataToCollect['expData'][1][2] = sub_info['monitorRefreshRate']
            dataToCollect['expData'][3][2] = sub_info['subjectNumber']
            dataToCollect['expData'][4][2] = sub_info['subjectAgeYears']
    # GUI dialog to collect experimental info
    expInfo = ExperimentInfo(title = EXPERIMENT,
                             data = dataToCollect['expData'])
    expInfo.__dict__['path'] = ROOTPATH
    expInfo.__dict__['experiment'] = EXPERIMENT
    expInfo.__dict__['runfile'] = RUNFILE
    # create monitor if necessary
    if expInfo.monitorName not in monitors.getAllMonitors():
        createMonitor(expInfo.monitorName, dataToCollect['monData'])
    if expInfo.sendTriggers:
        if not eegtrigs.USB2LPT:
            portInfo = ExperimentInfo(title = 'Parallel port',
                                      data = dataToCollect['portAddress'])
            expInfo.__dict__['triggers'] = eegtrigs.EegTriggerLPT(
                portNumber = int(portInfo.portAddress, 16)
            )
        else:
            expInfo.__dict__['triggers'] = eegtrigs.EegTriggerUSB()
    #===========================================================================
    # set up logging
    #===========================================================================
    SNO = expInfo.subjectNumber.rjust(2,'0')
    logging.LogFile((ROOTPATH / 'LOG' / f'sub-{SNO}_{RUNFILE}.log').open('a'))
    #===========================================================================
    # get experimental parameters
    #===========================================================================
    with (ROOTPATH / 'RUN' / 'exp_parameters.json').open('r') as f:
        expParams = json.load(f)
    #===========================================================================
    # get experimental instructions
    #===========================================================================
    with (ROOTPATH / 'RUN' / 'exp_instructions.json').open('r') as f:
        expIntro = json.load(f)[RUNFILE]
    #===========================================================================
    # create trial structure
    #===========================================================================
    nTrialsPerBlock = 100
    nBlocksPerRun = 1
    nTrialsTotal = nTrialsPerBlock * nBlocksPerRun

    expTrials = pd.DataFrame()
    # compile trial parameters into a data frame
    # The left variable names are written to columns in the trial df
    #   whenever you want to adjust the names of trial[x] also change them here
    
    maxelem = np.array([300] * nTrialsTotal)

    reassigned = np.array([0] * nTrialsTotal)

    #Randomly draw the number of targets from 1 to 21 (randint is [low, high))
    targets = np.random.randint(low = 10, high = 31, size = nTrialsTotal)

    tcancel = np.array([0] * nTrialsTotal)

    distractors = maxelem - targets
    dcancel = np.array([0] * nTrialsTotal)

    expTrials = pd.concat([
        expTrials, 
        pd.DataFrame(
            dict(
                sNo = int(SNO),
                targets = targets,
                distractors = distractors,
                maxElements = maxelem,
                tCancel = tcancel,
                dCancel = dcancel,
                accuracy = np.nan,
                reassigned = reassigned,
                RT = np.nan
            )
        )
    ], ignore_index=True)

    # shuffle experimental conditions
    expTrials = expTrials.sample(frac = 1).reset_index(drop=True)

    # add trial and block numbers
    expTrials['runningTrialNo'] = np.arange(expTrials.shape[0])
    expTrials['blockTrialNo'] = expTrials['runningTrialNo'] % nTrialsPerBlock
    expTrials['blockNo'] = expTrials['runningTrialNo'] // nTrialsPerBlock
    #===========================================================================
    # run the main function
    #===========================================================================
    recordedData = main(expInfo, expParams, expTrials, expIntro)
    #===========================================================================
    # %% save the recorded behavioural data and end the experiment
    #===========================================================================
    bhvFile = ROOTPATH / 'BHV' / f'sub-{SNO}_task-{RUNFILE}_bhv.tsv.gz'
    mode = 'w'
    header = True
    if bhvFile.is_file():
        mode = 'a'
        header = False
    recordedData.to_csv(
        bhvFile,
        sep = '\t',
        na_rep='n/a',
        index = False,
        mode = mode,
        header = header
    )
    
    #Clean up for memory optimisation:
    for name in dir():
        if not name.startswith('_'):
            del globals()[name]
    
    core.quit()

