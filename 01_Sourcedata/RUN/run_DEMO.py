'''
Author: Dragan Rangelov (d.rangelov@uq.edu.au)
File Created: 2021-09-28
-----
Last Modified: 2023-04-05
Modified By: Andrew Mckay (Andrew.Mckay@uq.edu.au)
-----
Licence: Creative Commons Attribution 4.0
Copyright 2019-2021 Dragan Rangelov, The University of Queensland
'''
#===============================================================================
# %% issue tracker
#===============================================================================
# NOTE: DEMO will have eye tracking only in the last block
#===============================================================================
# %% import libraries
#===============================================================================
import sys

from matplotlib import units
sys.dont_write_bytecode = True

from auxfunctions import createMonitor, sd2k, wrapTopi, wrapTo90
from datetime import datetime
import eegtriggers as eegtrigs
from eyelink import TrackerEyeLink
from experimentinfo import ExperimentInfo
import itertools
import json
import numpy as np
import math
import pandas as pd
from pathlib import Path
import psutil
from psychopy import core, monitors, logging, event, visual, misc
from scipy import stats
from scipy.stats import chisquare
import time
#===============================================================================
# setting process priority
#===============================================================================
ps = psutil.Process()
if sys.platform == 'win32': ps.nice(psutil.REALTIME_PRIORITY_CLASS)
else: ps.nice(0)
#===============================================================================
# %% define functions
#===============================================================================
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
    contingent = info.contingent

    # SESSION
    nTrialsPerBlock = params[runfile]['nTrialsPerBlock']['value']
    nBlocks = int(params[runfile]['nBlocksPerRun']['value'])
    nFramesPerTrial = int(params['TRIAL']['responseDeadline']['value'] * refreshRate)
    nFramesRespDelay = int(params['TRIAL']['responseDelay']['value'] * refreshRate)
    nFramesPerStimOn = int(params['TRIAL']['stimOn']['value'] * refreshRate)
    nFramesPerStimOff = int(params['TRIAL']['stimOff']['value'] * refreshRate)
    nFramesPerFeedback = int(params['TRIAL']['feedbackDuration']['value'] * refreshRate)
    nFramesPerFixation = int(params['TRIAL']['fixationDuration']['value'] * refreshRate)

    # MISC
    fixRad = float(params['TRIAL']['fixationRadius']['value'])
    fixDur = int(params['TRIAL']['fixationDuration']['value'] * refreshRate)
    totalPrize = int(params['MAIN']['totalPrize']['value']) 
    #===========================================================================
    # creating objects
    #===========================================================================
    win = visual.Window(
        monitor = monitor, 
        units = 'pix', color = 'gray',
        fullscr = info.fullScreen, 
        screen = int(info.monitorNumber)
    )

    event.Mouse(visible=False)

    trialClock = core.Clock()
    
    # General Text Object
    stimText = visual.TextStim(
        win,
        height = misc.deg2pix(params['STIM']['textHeight']['value'], monitor),
        wrapWidth = misc.deg2pix(params['STIM']['textWidth']['value'], monitor),
        pos = misc.deg2pix(params['STIM']['textPosition']['value'], monitor), 
        anchorHoriz = 'center', alignText = 'center'
    )
    stimText.text = 'Loading the experiment. Please wait.'
    stimText.draw()
    win.flip()

    #Text Objects for Response
    texArray = ['Accept', "Reject"]
    leftText = visual.TextStim(
        win,
        height = misc.deg2pix(params['STIM']['textHeight']['value'], monitor),
        wrapWidth = misc.deg2pix(params['STIM']['textWidth']['value'], monitor),
        pos = misc.deg2pix(params['STIM']['leftPos']['value'], monitor),
        anchorHoriz = 'center', alignText = 'center'
    )
   
    rightText = visual.TextStim(
        win,
        height = misc.deg2pix(params['STIM']['textHeight']['value'], monitor),
        wrapWidth = misc.deg2pix(params['STIM']['textWidth']['value'], monitor),
        pos = misc.deg2pix(params['STIM']['rightPos']['value'], monitor),
        anchorHoriz = 'center', alignText = 'center'
    )

    #Click areas for value response
    leftResp = visual.rect.Rect(
        win,
        size = misc.deg2pix(params['STIM']['clickSize']['value'], monitor),
        pos = misc.deg2pix(params['STIM']['leftPos']['value'], monitor)
    )

    rightResp = visual.rect.Rect(
        win,
        size = misc.deg2pix(params['STIM']['clickSize']['value'], monitor),
        pos = misc.deg2pix(params['STIM']['rightPos']['value'], monitor)
    )

    # Visual Circle
    stimCirc = visual.Polygon(win, 
        edges = params['STIM']['stimEdges']['value'], 
        radius=misc.deg2pix(params['STIM']['stimRadius']['value'], monitor), autoDraw=False)
    vert = stimCirc.vertices
    
    stimGratings = dict([
        (
            f'stim{idx+1}', 
            visual.GratingStim(
                win,
                size = misc.deg2pix(params['STIM']['stimSize']['value'], monitor),
                tex = params['STIM']['stimTexture']['value'],
                mask = params['STIM']['stimMask']['value'],
                sf = (
                    params['STIM']['stimSF']['value'] 
                    / misc.deg2pix(params['STIM']['stimSize']['value'], monitor)
                ),
                pos = vert[idx],
                contrast = params['STIM']['stimContrast']['value']
            )
        )
        for idx in range(len(vert))
    ])    

    # Orientation reproduction stim
    demoStim = visual.GratingStim(
        win,
        size = misc.deg2pix(params['STIM']['stimSize']['value'], monitor),
        tex = params['STIM']['stimTexture']['value'],
        mask = params['STIM']['stimMask']['value'],
        sf = (
            params['STIM']['stimSF']['value'] 
            / misc.deg2pix(params['STIM']['stimSize']['value'], monitor)
        ),
        pos = [0,0],
        contrast = params['STIM']['stimContrast']['value']

    )
    # Orientation reproduction stim
    reproStim = visual.Circle(
        win = win,
        radius = misc.deg2pix(params['STIM']['reproStim']['value'], monitor),
        fillColor = 'gray',
        lineWidth = params['STIM']['stimLineWidth']['value'],
        lineColor = params['STIM']['reproLineColour']['value']    
    )

    # Line for markers in reproduction stim
    line = visual.Line(
        win = win,
        start = (0, misc.deg2pix(params['STIM']['stimLineLength']['value'], monitor)/2),
        end =(0, misc.deg2pix(-params['STIM']['stimLineLength']['value'], monitor)/2),
        lineWidth = params['STIM']['stimLineWidth']['value'],
        lineColor = params['STIM']['respLineColor']['value']
    )

    feedStim = visual.Line(
        win = win,
        start = (0, misc.deg2pix(params['STIM']['stimLineLength']['value'], monitor)/2),
        end = (0, misc.deg2pix(-params['STIM']['stimLineLength']['value'], monitor)/2),
        lineWidth = params['STIM']['stimLineWidth']['value'],
        lineColor = params['STIM']['feedLineColor']['value']
    )
    #Create mouse event for orientation trial responses
    mouse = event.Mouse(visible=False, win=win)

    #Create rectangle shape for mouseclick response in orientation trials
    # Required for mouse.isPressedIn() to work
    click_area = visual.rect.Rect(win, 
        size = (win.size[0], win.size[1]),
        opacity = 0)

    # set fixation stimulus
    innerAreaProportion = 0.25 # previously 0.25
    texRes = 256
    xys = np.indices([texRes, texRes], dtype = np.float)
    xys -= xys.max() * .5
    xys /= xys.max()
    xys_abs = np.abs(xys[0] + xys[1] * 1j)
    mask = np.array((xys_abs <= 1) & (xys_abs >= innerAreaProportion),
                    dtype = np.int)
    mask[mask == 0] = -1
    stimFix = visual.ImageStim(
        win,
        image = np.ones([texRes, texRes]),
        mask = mask,
        texRes = texRes,
        size = misc.deg2pix(params['STIM']['fixSize']['value'], monitor)
    )

    if info.trackEyes:
        ilnk = TrackerEyeLink(
            win,
            sampleRate = 1000,
            saccadeSensitivity = 0,
            calibrationType = 'HV5',
            calibrationArea = [.7, .5],
            validationArea = [.7, .5],
            target = stimFix,
            text = stimText
        )                  
    #===========================================================================
    # run trials
    #===========================================================================
    pressedKeys = [] # initialize pressed keys
    exitTrialLoop = False # initialize trial loop break flag  
    passCondMet = False   
    
    for blockNo in trials['blockNo'].unique():
        

        # NOTE: Block descriptions
        # block 0 - Fixed small variance, single stimulus, value based decision
        # block 1 - same as block 0, 12 stimuli, orientation reproduction
        # block 2 - same as block 1, 12 stimuli, value trials
        # block 3 - same as block 1, mixture of value and orientation trials,
        # block 4 - same as block 2, mixture of small and low variance trials , added response deadline

        # abort experiment
        if 'q' in pressedKeys:
            break
        
        # show intro text at the beginning of block
        if not info.simulate:
            mouse.setVisible(False)
            stimText.pos = misc.deg2pix(
                params['STIM']['textPosition']['value'], 
                monitor
            )
            stimText.alignText = 'left'
            stimText.color = 'white'
            stimText.text = '\n'.join(intro[blockNo])
            while True:
                stimText.draw()
                win.flip()
                if event.getKeys(keyList=['space']):
                    break
        if not passCondMet and blockNo <= 2 and blockNo != 0:
            stimText.alignText = 'left'
            stimText.color = 'white'
            stimText.text = f"Please call the experimenter ({int(blockNo)})"
            while True:
                stimText.draw()
                win.flip()
                if event.getKeys(keyList=['space']):
                    break
                if event.getKeys(keyList=['p']):
                    core.quit()
        
                #reset check for second block
        if blockNo == 1: 
            passCondMet = False

        # setup eye-tracker at the beginning the block
        if (
            info.trackEyes and 
            ilnk.getStatus() is 'ONLINE'
        ):
            # start recording and open an new data file
            timestamp =  datetime.now().strftime("%y%m%d%H%M%S")
            edfFileName = str(
                info.path
                / 'EYE'
                /'sub-{}_task-{}_run-{}_{}.edf'.format(
                    trials['sNo'].unique(),
                    runfile,
                    blockNo,
                    timestamp
                )
            )
            ilnk.beginRecording(
                win,
                edfFileName = edfFileName
            )  

        # run trials per block
        for idx_trial, trial in trials.loc[trials['blockNo'] == blockNo].iterrows():
            # exit trial loop
            if exitTrialLoop:
                exitTrialLoop = False
                break

            # fixation period
            frame = 0
            while frame < fixDur:
                stimFix.draw()
                win.flip()
                # get gaze position
                if info.trackEyes and ilnk.getStatus() in ['ONLINE', 'RECORDING']:
                    gazeXY = misc.pix2deg(
                        ilnk.getSample(), 
                        monitor
                    )
                    gazeDist = np.sqrt(np.sum(gazeXY ** 2))
                if (
                    # is the gaze at the fixation
                    (info.trackEyes and gazeDist <= fixRad)
                    # is the gaze irrelevant
                    or not info.trackEyes
                ):
                    frame += 1
                else:
                    # reset counter if the gaze is elsewhere
                    frame = 0
            
            # trial period
            frame = 0
            trialClock.reset()
            flips = []
            terminateTrial = False
            showCue = False
            if info.trackEyes:
                    status_msg = '\n'.join([f"Trial: {trial['blockTrialNo']}/{nTrialsPerBlock-1}",
                    f"Block: {trial['blockNo']}/{nBlocks-1}"])
                    ilnk.sendCommand("record_status_message '%s'" % status_msg)
            

            thetas = []
            mu_cond = wrapTopi(trial['tstThetaRad'])
            
            # Sample orientations
            for key_stim, val_stim in stimGratings.items():
                val_stim.phase = np.random.random()
                theta = np.random.vonmises(mu=(trial['tstThetaRad']), 
                    kappa=sd2k(trial['tstVarRad']))
                thetas.append(theta)
            thetas = np.array(thetas)
            # Perform mean centering over sampled orientations
            mu_t = np.angle(np.exp(thetas * 1j).sum())
            thetas_norm = np.angle(((
                    np.exp(thetas * 1j) 
                    / np.exp(mu_t * 1j)
                ) * np.exp(mu_cond *1j)
            ))
            
            #convert normalised angles from radians to degrees and assign them to sitmuli
            for idx, (key_stim, val_stim) in enumerate(stimGratings.items()):
                ori = round(thetas_norm[idx] * 90 / np.pi, 3)
                stimGratings[key_stim].ori = ori 
                trials.loc[idx_trial, f'{key_stim}Ori'] = val_stim.ori 

            #selecting single stimulus for block 0
            if blockNo == 0:
                rand_id = np.random.randint(1,13)

            while True:
                mouse.setVisible(False)
                if blockNo != 0:
                    for key_stim, val_stim in stimGratings.items():
                        val_stim.draw()
                        stimCirc.draw()
                    
                    if contingent:
                        stimFix.draw()

                flips += [win.flip()]

                if frame == 0:
                    logMssg = '_'.join([
                        'BLOCK:{}'.format(trial['blockNo']),
                        'TRIAL:{}'.format(trial['blockTrialNo']),
                        # Surely there is a better way and this is probably redundant but whatever
                        'STIM1:{}'.format(round(trials.loc[idx_trial, 'stim1Ori'], 2)),
                        'STIM2:{}'.format(round(trials.loc[idx_trial, 'stim2Ori'], 2)),
                        'STIM3:{}'.format(round(trials.loc[idx_trial, 'stim3Ori'], 2)),
                        'STIM4:{}'.format(round(trials.loc[idx_trial, 'stim4Ori'], 2)),
                        'STIM5:{}'.format(round(trials.loc[idx_trial, 'stim5Ori'], 2)),
                        'STIM6:{}'.format(round(trials.loc[idx_trial, 'stim6Ori'], 2)),
                        'STIM7:{}'.format(round(trials.loc[idx_trial, 'stim7Ori'], 2)),
                        'STIM8:{}'.format(round(trials.loc[idx_trial, 'stim8Ori'], 2)),
                        'STIM9:{}'.format(round(trials.loc[idx_trial, 'stim9Ori'], 2)),
                        'STIM10:{}'.format(round(trials.loc[idx_trial, 'stim10Ori'], 2)),
                        'STIM11:{}'.format(round(trials.loc[idx_trial, 'stim11Ori'], 2)),
                        'STIM12:{}'.format(round(trials.loc[idx_trial, 'stim12Ori'], 2)),
                    ])
                    logging.warning(
                        logMssg
                    )
                    logging.flush()
                    if info.trackEyes:
                        ilnk.sendMessage(logMssg)
                pressed = event.getKeys(
                    keyList = ['q', 'f', 'j', 'space', 'p'], 
                    timeStamped = trialClock
                )

                if contingent and not showCue:
                    # get gaze position
                    if info.trackEyes and ilnk.getStatus() in ['ONLINE', 'RECORDING']:
                        gazeXY = misc.pix2deg(
                            ilnk.getSample(), 
                            monitor
                        )
                        gazeDist = np.sqrt(np.sum(gazeXY ** 2))
                    # present stimuli
                    if (info.trackEyes and gazeDist >= fixRad):
                        win.flip()
                        
                        stimText.pos = misc.deg2pix(
                            params['STIM']['textPosition']['value'], 
                            monitor
                        )
                        stimText.alignText = 'center'
                        stimText.text = "Remember to keep looking at the middle circle"
                        stimText.color = 'White'
                        #logging message for fix break:
                        logMssg = f"TRIAL:{trial['runningTrialNo']}_FIX BREAK"
                        logging.warning(
                            logMssg
                        )
                        logging.flush()
                        for frame in range(nFramesPerFeedback):
                            stimText.draw()
                            win.flip()
                        break
                # Responses for block 1
                if blockNo == 0:
                    mouse.setPos()
                    leftResp.opacity = 0
                    rightResp.opacity = 0
                    mouse.setVisible(True)
                    if 'left' == trial['testStim']:
                        leftText.text = texArray[0]
                        rightText.text = texArray[1]
                    if 'right' == trial['testStim']:
                        leftText.text = texArray[1]
                        rightText.text = texArray[0]
                    demoStim.ori = trial["tstThetaDeg"]
                    mouse.clickReset() # Reset timer so that RTs make sense
                    while True:
                        demoStim.draw()
                        leftResp.draw()
                        rightResp.draw()
                        leftText.draw()
                        rightText.draw()
                        win.flip()
                        pressed = event.getKeys(
                            keyList = ['q', 'space', 'p'], 
                            timeStamped = trialClock
                            )
                        if leftResp.contains(mouse):
                            leftResp.opacity = params['STIM']['buttonOpac']['value']
                            leftResp.lineColor = params['STIM']['buttonColour']['value']
                        else:
                            leftResp.color = None
                        if rightResp.contains(mouse):
                            rightResp.opacity = params['STIM']['buttonOpac']['value']
                            rightResp.lineColor = params['STIM']['buttonColour']['value']
                        else:
                            rightResp.color = None
                        if pressed: 
                            pressedKeys, pressedTimes = zip(*pressed)
                            if 'q' in pressedKeys: 
                                terminateTrial = True
                                exitTrialLoop = True
                                break
                            if 'p' in pressedKeys: 
                                terminateTrial = True
                                exitTrialLoop = True
                                passCondMet = True
                                break
                            if 'space' in pressedKeys: 
                                terminateTrial = True
                                exitTrialLoop = True
                                break
                        if mouse.isPressedIn(leftResp, buttons=[0]):
                            _, clickTime = mouse.getPressed(getTime=True)
                            response = 'left'
                            payoff = int(trial[f'{response}Payoff'])
                            trials.loc[idx_trial, 'payoff'] = payoff   
                            trials.loc[idx_trial, 'response'] = response
                            trials.loc[idx_trial, 'RT'] = clickTime[0]
                            trials.loc[idx_trial, 'accuracy'] = int(
                                payoff == trial[['leftPayoff', 'rightPayoff']].max() 
                            )
                            mouse.setVisible(False)
                            #Adding log message for response time (to faciliate eye data analysis)
                            logMssg = '_'.join(['Response', f'{response}'])
                            logging.warning(
                                logMssg
                            )
                            logging.flush()
                            if info.trackEyes:
                                ilnk.sendMessage(logMssg)
                            if payoff > 0:
                                stimText.color = 'green'
                            if payoff < 0:
                                stimText.color = 'red'
                            if payoff == 0:
                                stimText.color = 'white'
                            if info.feedback:
                                stimText.pos = misc.deg2pix(
                                    params['STIM']['textPosition']['value'], 
                                    monitor
                                )
                                stimText.alignText = 'center'
                                stimText.text = f'{int(payoff)} Points'
                                for frame in range(nFramesPerFeedback):
                                    stimText.draw()
                                    win.flip()
                                terminateTrial = True
                                stimText.color = 'white'
                                break
                        if mouse.isPressedIn(rightResp, buttons=[0]):
                            _, clickTime = mouse.getPressed(getTime=True)
                            response = 'right'
                            payoff = int(trial[f'{response}Payoff'])
                            trials.loc[idx_trial, 'payoff'] = payoff   
                            trials.loc[idx_trial, 'response'] = response
                            trials.loc[idx_trial, 'RT'] = clickTime[0]
                            trials.loc[idx_trial, 'accuracy'] = int(
                                payoff == trial[['leftPayoff', 'rightPayoff']].max() 
                            )
                            mouse.setVisible(False)
                            #Adding log message for response time (to faciliate eye data analysis)
                            logMssg = '_'.join(['Response', f'{response}'])
                            logging.warning(
                                logMssg
                            )
                            logging.flush()
                            if info.trackEyes:
                                ilnk.sendMessage(logMssg)
                            if payoff > 0:
                                stimText.color = 'green'
                            if payoff < 0:
                                stimText.color = 'red'
                            if payoff == 0:
                                stimText.color = 'white'
                            if info.feedback:
                                stimText.pos = misc.deg2pix(
                                    params['STIM']['textPosition']['value'], 
                                    monitor
                                )
                                stimText.alignText = 'center'
                                stimText.text = f'{int(payoff)} Points'
                                for frame in range(nFramesPerFeedback):
                                    stimText.draw()
                                    win.flip()
                                terminateTrial = True
                                stimText.color = 'white'
                                break
                        elif info.simulate:
                            if frame > np.random.choice(range(nFramesPerTrial)):
                                response = np.random.choice(['left', 'right'])
                                payoff = trial[f'{response}Payoff']
                                trials.loc[idx_trial, 'response'] = response
                                trials.loc[idx_trial, 'RT'] = round(frame / refreshRate, 3)
                                trials.loc[idx_trial, 'payoff'] = payoff                    
                                trials.loc[idx_trial, 'accuracy'] = int(
                                    payoff == trial[['leftPayoff', 'rightPayoff']].max()
                                )
                                # show feedback about the payoff
                                if info.feedback:
                                    stimText.pos = misc.deg2pix(
                                        params['STIM']['textPosition']['value'], 
                                        monitor
                                    )
                                    stimText.alignText = 'center'
                                    stimText.text = f'{int(payoff)} Points'
                                    for frame in range(nFramesPerFeedback):
                                        stimText.draw()
                                        win.flip()
                                terminateTrial = True
                            break
                    
                # Move on to cue and reset frame count
                if frame >= nFramesPerStimOn and blockNo != 0:
                    showCue = True
                    frame = 0
                    logMssg = 'ArrayOff'
                    logging.warning(
                        logMssg
                    )
                    logging.flush()
                # Critical statement for getting out of nested for loop
                if pressed: 
                    pressedKeys, pressedTimes = zip(*pressed)
                    if 'q' in pressedKeys: 
                        terminateTrial = True
                        exitTrialLoop = True
                        break

                #introducing random jitter between offset and response
                if showCue:
                    win.flip()
                    jitter = np.random.uniform(0.1,0.3) #100ms - 300ms
                    time.sleep(jitter)

                if trial['trialType'] == 0 and showCue and blockNo != 0:
                    stimText.color = 'white'
                    leftResp.opacity = 0
                    rightResp.opacity = 0
                    mouse.setVisible(True)
                    if 'left' == trial['testStim']:
                        leftText.text = texArray[0]
                        rightText.text = texArray[1]
                    if 'right' == trial['testStim']:
                        leftText.text = texArray[1]
                        rightText.text = texArray[0]
                    mouse.setPos()
                    mouse.clickReset() # Reset timer so that RTs make sense
                    while True:
                        leftResp.draw()
                        rightResp.draw()
                        leftText.draw()
                        rightText.draw()
                        win.flip()
                        pressed = event.getKeys(
                            keyList = ['q', 'space'], 
                            timeStamped = trialClock
                            )
                        if leftResp.contains(mouse):
                            leftResp.opacity = params['STIM']['buttonOpac']['value']
                            leftResp.lineColor = params['STIM']['buttonColour']['value']
                        else:
                            leftResp.color = None
                        if rightResp.contains(mouse):
                            rightResp.opacity = params['STIM']['buttonOpac']['value']
                            rightResp.lineColor = params['STIM']['buttonColour']['value']
                        else:
                            rightResp.color = None
                        frame += 1  
                        if pressed: 
                            pressedKeys, pressedTimes = zip(*pressed)
                            if 'q' in pressedKeys: 
                                terminateTrial = True
                                exitTrialLoop = True
                                break
                            if 'p' in pressedKeys: 
                                terminateTrial = True
                                exitTrialLoop = True
                                passCondMet = True
                                break
                            if 'space' in pressedKeys:
                                terminateTrial = True
                                exitTrialLoop = True
                                break
                        if mouse.isPressedIn(leftResp, buttons=[0]):
                            _, clickTime = mouse.getPressed(getTime=True)
                            response = 'left'
                            payoff = int(trial[f'{response}Payoff'])
                            trials.loc[idx_trial, 'payoff'] = payoff 
                            trials.loc[idx_trial, 'response'] = response
                            trials.loc[idx_trial, 'RT'] = clickTime[0]
                            trials.loc[idx_trial, 'accuracy'] = int(
                                payoff == trial[['leftPayoff', 'rightPayoff']].max() 
                            )
                            mouse.setVisible(False)
                            #Adding log message for response time (to faciliate eye data analysis)
                            logMssg = '_'.join(['Response', f'{response}'])
                            logging.warning(
                                logMssg
                            )
                            logging.flush()
                            if info.trackEyes:
                                ilnk.sendMessage(logMssg)
                            if payoff > 0:
                                stimText.color = 'green'
                            if payoff < 0:
                                stimText.color = 'red'
                            if payoff == 0:
                                stimText.color = 'white'
                            if info.feedback:
                                stimText.pos = misc.deg2pix(
                                    params['STIM']['textPosition']['value'], 
                                    monitor
                                )
                                stimText.alignText = 'center'
                                stimText.text = f'{int(payoff)} Points'
                                for frame in range(nFramesPerFeedback):
                                    stimText.draw()
                                    win.flip()
                                terminateTrial = True
                                break
                        if mouse.isPressedIn(rightResp, buttons=[0]):
                            _, clickTime = mouse.getPressed(getTime=True)
                            response = 'right'
                            payoff = int(trial[f'{response}Payoff'])
                            trials.loc[idx_trial, 'payoff'] = payoff 
                            trials.loc[idx_trial, 'response'] = response
                            trials.loc[idx_trial, 'RT'] = clickTime[0]
                            trials.loc[idx_trial, 'accuracy'] = int(
                                payoff == trial[['leftPayoff', 'rightPayoff']].max() 
                            )
                            mouse.setVisible(False)
                            #Adding log message for response time (to faciliate eye data analysis)
                            logMssg = '_'.join(['Response', f'{response}'])
                            logging.warning(
                                logMssg
                            )
                            logging.flush()
                            if info.trackEyes:
                                ilnk.sendMessage(logMssg)
                            if payoff > 0:
                                stimText.color = 'green'
                            if payoff < 0:
                                stimText.color = 'red'
                            if payoff == 0:
                                stimText.color = 'white'
                            # show feedback about the payoff
                            if info.feedback:
                                stimText.pos = misc.deg2pix(
                                    params['STIM']['textPosition']['value'], 
                                    monitor
                                )
                                stimText.alignText = 'center'
                                stimText.text = f'{int(payoff)} Points'
                                for frame in range(nFramesPerFeedback):
                                    stimText.draw()
                                    win.flip()
                                terminateTrial = True
                                break
                        elif info.simulate:
                            if frame > np.random.choice(range(nFramesPerTrial)):
                                response = np.random.choice(['left', 'right'])
                                payoff = trial[f'{response}Payoff']
                                trials.loc[idx_trial, 'response'] = response
                                trials.loc[idx_trial, 'RT'] = round(frame / refreshRate, 3)
                                trials.loc[idx_trial, 'payoff'] = payoff                    
                                trials.loc[idx_trial, 'accuracy'] = int(
                                    payoff == trial[['leftPayoff', 'rightPayoff']].max()
                                )
                                # show feedback about the payoff
                                if info.feedback:
                                    stimText.pos = misc.deg2pix(
                                        params['STIM']['textPosition']['value'], 
                                        monitor
                                    )
                                    stimText.alignText = 'center'
                                    stimText.text = f'{int(payoff)} Points'
                                    for frame in range(nFramesPerFeedback):
                                        stimText.draw()
                                        win.flip()
                                terminateTrial = True
                            break
                        if blockNo > 2:
                            if info.deadline and frame == (nFramesPerTrial):
                                stimText.pos = misc.deg2pix(
                                    params['STIM']['textPosition']['value'], 
                                    monitor
                                )
                                stimText.alignText = 'center'
                                stimText.text = 'Try to respond faster!'
                                for frame in range(nFramesPerFeedback):
                                    stimText.draw()
                                    win.flip()
                                terminateTrial = True
                                break                    
                                    
                if trial['trialType'] == 1 and showCue:
                    stimText.color = 'white'
                    bound_radius = misc.deg2pix(params['STIM']['boundRadius']['value'], monitor)
                    # Get random x and y with given radius r
                    start_theta = np.random.uniform(0, 180)
                    x_pos  = bound_radius * math.cos(start_theta)
                    y_pos  = bound_radius * math.sin(start_theta)
                    mouse.setPos((x_pos, y_pos)) # Set position between inner and upper bound
                    mouse.clickReset() # Reset mouse timing for RTs
                    mouse.setVisible(True)
                    pressed = event.getKeys(
                        keyList = ['q', 'space'], 
                        timeStamped = trialClock
                        )
                    reproStim.lineColor = 'white'
                    oriLineOffset =  np.random.uniform(-90, 90)
                    while True:
                        mouse_x, mouse_y = mouse.getPos() # get cartesian coord
                        distance = np.sqrt(mouse_x**2 + mouse_y**2) # get the distance from the center
                        angle = math.atan2(mouse_y, -mouse_x) / np.pi * 180
                        if distance >= bound_radius:
                            new_x = bound_radius * math.cos(angle * np.pi / 180) # X coordinate of angle with distance bound_radius
                            new_y = bound_radius * math.sin(angle * np.pi / 180) # y coordinate of angle with distance bound_radius
                            mouse.setPos((-new_x, new_y))
                        angle = angle - 90 # Vertical is 0, Horizontal is -90
                        line.ori = angle
                        line.draw()
                        click_area.draw()
                        reproStim.draw()
                        win.flip()
                        pressed = event.getKeys(
                            keyList = ['q', 'space', 'p'], 
                            timeStamped = trialClock
                            ) 
                        frame += 1   
                        if pressed:
                            pressedKeys, pressedTimes = zip(*pressed)
                            if 'q' in pressedKeys: 
                                terminateTrial = True
                                exitTrialLoop = True
                                break 
                            if 'p' in pressedKeys: 
                                terminateTrial = True
                                exitTrialLoop = True
                                passCondMet = True
                                break
                            if 'space' in pressedKeys:
                                terminateTrial = True
                                exitTrialLoop = True
                                break 
                        #Mouseclick response
                        if mouse.isPressedIn(click_area, buttons=[0]):
                            mouse.setVisible(False)
                            _, clickTime = mouse.getPressed(getTime=True)
                            response = round(wrapTo90(line.ori), 0) 
                            trials.loc[idx_trial, 'response'] = response
                            trials.loc[idx_trial, 'RT'] = clickTime[0]
                            #Adding log message for response time (to faciliate eye data analysis)
                            logMssg = '_'.join(['Response', f'{response}'])
                            logging.warning(
                                logMssg
                            )
                            logging.flush()
                            if info.trackEyes:
                                ilnk.sendMessage(logMssg)
                            if info.feedback:
                                feedStim.ori = round(mu_cond * 90 / np.pi, 0)
                                reproStim.lineColor = 'black'
                                for frame in range(nFramesPerFeedback):
                                    line.draw()
                                    feedStim.draw()
                                    reproStim.draw()
                                    win.flip()
                                terminateTrial = True
                                break
                        elif info.simulate:
                            if frame > np.random.choice(range(nFramesPerTrial)):
                                response = np.random.choice(np.linspace(-90, 90))
                                trials.loc[idx_trial, 'response'] = response
                                trials.loc[idx_trial, 'RT'] = round(frame / refreshRate, 3)
                                mouse.setVisible(False)
                                if info.feedback:
                                    # feedStim.ori = round(mu_cond * 90 / np.pi, 0) # No feedback stim in main task
                                    for frame in range(nFramesPerFeedback):
                                        line.draw()
                                        # feedStim.draw()
                                        reproStim.draw()
                                        win.flip()
                                    terminateTrial = True
                                    break

                        if blockNo > 2:
                            if info.deadline and frame == (nFramesPerTrial):
                                stimText.pos = misc.deg2pix(
                                    params['STIM']['textPosition']['value'], 
                                    monitor
                                )
                                stimText.alignText = 'center'
                                stimText.text = 'Try to respond faster!'
                                for frame in range(nFramesPerFeedback):
                                    stimText.draw()
                                    win.flip()
                                terminateTrial = True
                                break    
                
                # check if criterion has been met
                
                if pressed:
                    pressedKeys, pressedTimes = zip(*pressed)
                    if (
                            ('space' in pressedKeys) and 
                            (blockNo > 1)
                        ):                             
                        exitTrialLoop = True
                        terminateTrial = True
                        break
                if (blockNo == 0) and (trials.loc[
                    idx_trial - 9 : idx_trial + 1,
                    'accuracy'
                ].sum() >= 8):
                    passCondMet = True
                    exitTrialLoop = True
                    break
                if blockNo == 1:
                    #Filter out any missing responses due to fixation
                    response_data = trials[trials['response'].notna() & trials['blockNo']==1]
                    if response_data.shape[0] >= 10:
                        response_data_ori = round((response_data.iloc[-10:]['response'].astype(float)) * np.pi / 90, 2) # Convert degrees to radians
                        cond_data = round((response_data.iloc[-10:]['tstThetaRad'].astype(float)), 2)
                        # if pd.notna(response_data).all():
                        diff = np.angle(np.exp(1j*response_data_ori) / np.exp(1j*cond_data)) # Difference in radians
                        circ_mean = stats.circmean(diff) * 90 / np.pi # Convert to degrees
                        if abs(circ_mean) <= 7.5: #is the absolute value of the mean deviation <7.5 degrees? i.e. +/- 7.5 degrees
                            passCondMet = True
                            exitTrialLoop = True
                            break
                    
                # terminate trial
                if terminateTrial:
                    flips = np.array(flips)
                    flipStat = '-'.join([
                        f'{flip}:{freq}' 
                        for flip, freq 
                        in zip(*np.unique(
                            (flips[1:] - flips[:-1]).round(3), 
                            return_counts=True
                        ))
                    ])

                    logMssg = '_'.join([
                        'BLOCK:{}'.format(trial['blockNo']),
                        'TRIAL:{}'.format(trial['blockTrialNo']),
                        'FLIPS:[{}]'.format(flipStat)
                    ])
                    logging.warning(
                        logMssg
                    )
                    logging.flush()
                    if info.trackEyes:
                        ilnk.sendMessage(logMssg)
                    break
                frame += 1
    # END EXPERIMENT
    stimText.pos = misc.deg2pix(
        params['STIM']['textPosition']['value'], 
        monitor
    )
    stimText.color = 'white'
    stimText.alignText = 'center'
    stimText.text = '\n'.join([
        'End of the session.',
        'Please contact the experimenter.'
    ])
    while True:
        stimText.draw()
        win.flip()
        if event.getKeys(keyList=['space']): 
            win.close()
            break    
    # STOP EYELINK IF RELEVANT
    if info.trackEyes and ilnk.getStatus() in ['ONLINE', 'RECORDING']:
        ilnk.closeConnection()
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
    with (ROOTPATH / 'RUN' / 'exp_information.json').open('r') as f:
        dataToCollect = json.load(f)
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
        # Write text file for gui
    output_file_path = ROOTPATH / 'LOG' /f"gui_info_sub{SNO}.json"
    with open(output_file_path, 'w') as json_file:
        exp_info_dict = expInfo.__dict__.copy()
        exp_info_dict.pop('dlg', None)
        exp_info_dict.pop('path', None)
        json.dump(exp_info_dict, json_file, indent=2)
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
    nTrialsPerBlock = expParams[RUNFILE]['nTrialsPerBlock']['value']
    nBlocksPerRun = expParams[RUNFILE]['nBlocksPerRun']['value']
    nTrialsTotal = nTrialsPerBlock * nBlocksPerRun
    # trial structure
    valueRef = np.array(expParams['TRIAL']['valueRef']['value'])
    # valueDiff = np.array(expParams['TRIAL']['valueDiff']['value'])
    thetaVars = np.array(expParams['TRIAL']['thetaVars']['value'])
    trialType = np.array(expParams['TRIAL']['trialType']['value'])
    expConds = list(itertools.product(valueRef, thetaVars, trialType))
    # expTrials = pd.DataFrame()

    # Initialize p-value and chi-squared statistic
    bin_list = []
    p_value = 0
    p_v = 0
    p_p = 0
    chi2_stat = 0
    while any(x <= 0.80 for x in [p_value, p_v, p_p]):
        expTrials = pd.DataFrame()

        for val_ref, theta_var, trial_type in expConds:
            
            #Only sample from 0 - pi, as thetasign accounts for -pi - 0
            tst_samp = np.random.uniform(low=-np.pi, high=np.pi, size=len(range((nTrialsTotal // len(expConds)))))
            
            #Set a reference value to calculate angular difference
            ref = (np.pi/2)

            # compute the direction of shift of the standard
            thetaSign = np.sign(tst_samp)

            tstVarRad = theta_var * np.pi

            # compute the angle relative to the reward values
            # counterbalance the standard across participants between 
            # vertical (i.e., 0) and horizontal (i.e. np.pi)
            refThetaRad = [0, np.pi][int(SNO) % 2] + ref * thetaSign
            refThetaDeg = refThetaRad / np.pi * 90
            tstThetaRad = [0, np.pi][int(SNO) % 2] + tst_samp 
            tstThetaDeg = tstThetaRad / np.pi * 90

            #Compute Angular Difference Between Ref (+-45, and tst)
            deltaThetaRad = np.angle(np.exp(tstThetaRad* 1j)
            /np.exp(refThetaRad * 1j)).round(3) * thetaSign
            # NOTE: Multiply by sign to account for subtracting negative numbers 
            #   if pos dosent change if neg, flips sign such that -120 - 135 = -14 (instead of 14) 
            #   which is important for payoff calculation, (i.e. -120 - 135 SHOULD be a negative payoff instead of positive)


            deltaThetaOri = (deltaThetaRad / np.pi * 90)

            #* (nTrialsTotal // len(expConds))
            refPayoff = np.array([0] *  (nTrialsTotal // len(expConds))) 
            tstPayoff = np.array((deltaThetaRad) *  (nTrialsTotal // len(expConds)))
            
            refPayoff = refPayoff
            tstPayoff = tstPayoff

            tstSide = ['left', 'left', 'right', 'right'][int(SNO) % 4]
            testStim = np.array([tstSide] *  (nTrialsTotal // len(expConds)))

            if tstSide == 'right':
                [leftPayoff, rightPayoff] = [
                    np.array(refPayoff),
                    np.array(tstPayoff)
                ]

            if tstSide == 'left':
                [leftPayoff, rightPayoff] = [
                    np.array(tstPayoff),
                    np.array(refPayoff)
                ]    
        
            [tstOri] = [tstThetaDeg]
            # Match the left and right payoffs to reference and test (Same as orientation above)
            if trial_type == 0:
                [leftPayoff, rightPayoff] = [
                    leftPayoff,
                    rightPayoff
                ]
            if trial_type == 1:
                [leftPayoff, rightPayoff] = [
                    leftPayoff,
                    rightPayoff
                ]
            trialType = trial_type

            # compile trial parameters into a data frame
            # The left variable names are written to columns in the trial df
            #   whenever you want to adjust the names of trial[x] also change them here
            expTrials = pd.concat([
                expTrials, 
                pd.DataFrame(
                    dict(
                        sNo = int(SNO),
                        refValue = np.round(refThetaRad, 3), 
                        tstValue = np.round(tst_samp, 3), 
                        thetaSign = thetaSign.round(3),
                        tstThetaRad = np.round(tstThetaRad, 3), #Values in interval [0, pi] (unwrapped)
                        tstThetaDeg = np.round(tstThetaDeg, 3), #Values in interval [0, 180] deg (unwrapped)
                        deltaThetaRad = deltaThetaRad, #Values in the interval [-pi/4, pi/4] as the difference between tst and ref (pi/2)
                        deltaThetaOri = deltaThetaOri,
                        tstVarRad = np.round(tstVarRad, 3), 
                        trialType = trialType,
                        thetaVar = theta_var,
                        testStim = testStim,
                        tstOri = tstOri, # Equivalent to tstThetaDeg
                        leftPayoff = leftPayoff,
                        rightPayoff = rightPayoff, 
                        response = np.nan,
                        accuracy = np.nan,
                        RT = np.nan,
                        payoff = np.nan,
                        stim1Ori = np.nan, #Values in interval [-90,90] (Wrapped to s.t. [-90, 90] = [-pi, pi]) 0=vertical
                        stim2Ori = np.nan,
                        stim3Ori = np.nan,
                        stim4Ori = np.nan,
                        stim5Ori = np.nan,
                        stim6Ori = np.nan,
                        stim7Ori = np.nan,
                        stim8Ori = np.nan,
                        stim9Ori = np.nan,
                        stim10Ori = np.nan,
                        stim11Ori = np.nan,
                        stim12Ori = np.nan,
                        tst_samp = tst_samp # Values sampled before transform
                    )
                )
            ], ignore_index=True)
    
        # shuffle experimental conditions
        expTrials = expTrials.sample(frac = 1).reset_index(drop=True)
        
        # add trial and block numbers
        expTrials['runningTrialNo'] = np.arange(expTrials.shape[0])
        expTrials['blockTrialNo'] = expTrials['runningTrialNo'] % nTrialsPerBlock
        expTrials['blockNo'] = expTrials['runningTrialNo'] // nTrialsPerBlock

        # Check to see if the created data conform
        bin_edges = np.linspace(-90, 90, 15)
        observed, _ = np.histogram(expTrials['tst_samp']/ np.pi * 90, bins=bin_edges)
        obs_val, _ = np.histogram(expTrials.loc[expTrials['trialType'] == 0, 'tst_samp']/ np.pi * 90, bins=bin_edges)
        obs_per, _ = np.histogram(expTrials.loc[expTrials['trialType'] == 1, 'tst_samp']/ np.pi * 90, bins=bin_edges)

        # Define null hypothesis distribution
        # expected = np.full(10, 50)
        # exp_val = np.full(10, 38)
        # exp_per = np.full(10, 13)
        expected = np.full(14, 71)
        exp_val = np.full(14, 53)
        exp_per = np.full(14, 18)
        
        for i, count in enumerate(observed):
            print(f"Bin {i+1}: {count} observations")
        # sd_across_bins = np.std(observed)
        
        # Calculate chi-squared statistic and p-value
        chi2_stat, p_value = chisquare(f_obs=observed, f_exp=expected)
        print(f"Chi-squared statistic: {chi2_stat:.4f}")
        print(f"P-value: {p_value:.4f}")

        for i, count in enumerate(obs_val):
            print(f"Bin {i+1}: {count} Value Observation")
        x2, p_v = chisquare(f_obs=obs_val, f_exp=exp_val)
        print(f"Chi-squared statistic: {x2:.4f}")
        print(f"P-value: {p_v:.4f}")


        for i, count in enumerate(obs_per):
            print(f"Bin {i+1}: {count} Percep Observation")
        x3, p_p = chisquare(f_obs=obs_per, f_exp=exp_per)
        print(f"Chi-squared statistic: {x3:.4f}")
        print(f"P-value: {p_p:.4f}")
    
    # Normalise payoffs such that max is 4000
    normPointMax = 4000 #Set the max amount of points (NOTE:55% of points given to participants so they can possibly go negative)
    nonNormPointMax = expTrials[f'{tstSide}Payoff'].abs().sum() #Get the non-normalised max points (abs for negative points)
    expTrials[f'{tstSide}Payoff'] = round((expTrials[f'{tstSide}Payoff'] / nonNormPointMax) * normPointMax, 0) #normalise payoffs to 4000

    ### customize blocks depending on block number ###
    # set small variance
    expTrials.loc[
        expTrials['blockNo'].isin([0,1,2,3]), 
        'tstVarRad'
    ] = (thetaVars * np.pi)[thetaVars != 0].min()
    # set value in block 1 and ori repro in block 2
    expTrials.loc[
        expTrials['blockNo'].isin([0,2]),
        'trialType'
    ] = 0
    expTrials.loc[
        expTrials['blockNo'] == 1,
        'trialType'
    ] = 1
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