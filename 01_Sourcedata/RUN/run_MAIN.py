'''
Author: Dragan Rangelov (d.rangelov@uq.edu.au)
File Created: 2021-09-28
-----
Last Modified: 2023-03-01
Modified By: Andrew Mckay (Andrew.Mckay@uq.edu.au)
-----
Licence: Creative Commons Attribution 4.0
Copyright 2019-2021 Dragan Rangelov, The University of Queensland
'''
#===============================================================================
# %% issue tracker
#===============================================================================
# NOTE: for iLink calibration to work the units must be in pixels
#===============================================================================
# %% import libraries
#===============================================================================
from pickle import TRUE
import sys
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
from psychopy import core, monitors, logging, event, visual, misc, parallel
import math
from scipy import stats
from scipy.stats import chisquare
import glob
import os
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
    nTrialsPerBlock = int(params[runfile]['nTrialsPerBlock']['value'])
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
    trigDur = 0.010 #10 ms triggers
    TrigDel = 0.010 #10 ms Delay

    # TRIGGER CODES: CONDITIONS & EXP EVENTS
    # 0xx - Variation condition (1 low, 2 high)
    # x0x - Trial type (0 value, 1 ori)
    FIXON = 130
    STIMON_VAL_LOW = 101
    STIMON_VAL_HIGH = 201
    STIMON_ORI_LOW = 111
    STIMON_ORI_HIGH = 211
    RESPON_VAL_LOW = 102
    RESPON_VAL_HIGH = 202
    RESPON_ORI_LOW = 113
    RESPON_ORI_HIGH = 213
    RESPSUB_VAL_LOW = 104
    RESPSUB_VAL_HIGH = 204
    RESPSUB_ORI_LOW = 115
    RESPSUB_ORI_HIGH = 215
    FEEDBCK_VAL_LOW = 106
    FEEDBCK_VAL_HIGH = 206
    FEEDBCK_ORI_LOW = 117
    FEEDBCK_ORI_HIGH = 217
    FIX_BRK = 133
    NO_RESP = 99

    Stim_Trig_List = [[STIMON_VAL_LOW, STIMON_VAL_HIGH], [STIMON_ORI_LOW, STIMON_ORI_HIGH]]

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

    rewardText = visual.TextStim(
        win,
        height = misc.deg2pix(params['STIM']['textHeight']['value'], monitor),
        wrapWidth = misc.deg2pix(params['STIM']['textWidth']['value'], monitor),
        pos = misc.deg2pix(params['STIM']['textPosition']['value'], monitor), 
        anchorHoriz = 'center', alignText = 'center'
    )

    breakText = visual.TextStim(
        win,
        height = misc.deg2pix(params['STIM']['textHeight']['value'], monitor),
        wrapWidth = misc.deg2pix(params['STIM']['textWidth']['value'], monitor),
        pos = misc.deg2pix(params['STIM']['textPosition']['value'], monitor), 
        anchorHoriz = 'center', alignText = 'center'
    )

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

    counter = visual.TextStim(
        win,
        height = misc.deg2pix(params['STIM']['textHeight']['value'], monitor),
        wrapWidth = misc.deg2pix(params['STIM']['textWidth']['value'], monitor),
        pos = misc.deg2pix([10,-10], monitor),
        anchorHoriz = 'center', alignText = 'center'
    )
    counter.text = '0'

    # feedStim = visual.Line(
    #     win = win,
    #     start = (0, misc.deg2pix(params['STIM']['stimLineLength']['value'], monitor)/2),
    #     end = (0, misc.deg2pix(-params['STIM']['stimLineLength']['value'], monitor)/2),
    #     lineWidth = params['STIM']['stimLineWidth']['value'],
    #     lineColor = params['STIM']['respLineColor']['value']
    # )

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
    
    if info.sendTriggers:
        eeg = info.__dict__['triggers']
        pport = parallel.ParallelPort(address=0xD050)

    gratingCols = [f'stim{idx+1}Ori' for idx in range(12)]

    #===========================================================================
    # run trials
    #===========================================================================
    pressedKeys = []
    repeatIdx = []
    lastTrial = False
    last = False
    blockEndSeen = False
    redo = False
    end = False
    rest= False
    currBlock = 0
    repeats = 0
    while not end:
        blockTrials = trials[trials['blockNo'] == currBlock]
        if len(repeatIdx) != 0 and blockEndSeen:
            redo = True
            blockTrials = trials.iloc[pd.DataFrame(repeatIdx).index]
            repeats += len(repeatIdx)
        #End experiment if someone has done 17 repeats in a single block (i.e. repeated the same block twice not including the initial pass)
        if repeats >= 17:
           end = True
        if (len(repeatIdx) == 0 and redo) or (blockEndSeen and rest):
            redo = False
            rest = False
            repeats = 0
            currBlock +=1
            blockTrials = trials[trials['blockNo'] == currBlock]
        for idx_trial, trial in blockTrials.iterrows():
            #TODO: REMOVE AFTER CHECKING
#            skip = np.random.choice([0,0,0,0,0,0,0,0,0,1])
            # setup eye-tracker on the first trial of the block
            if (
                trial['blockTrialNo'] == 0 and 
                info.trackEyes and
                not redo and 
                ilnk.getStatus() is 'ONLINE'
            ):
                # start recording and open an new data file
                timestamp =  datetime.now().strftime("%y%m%d%H%M%S")
                edfFileName = str(
                    info.path
                    / 'EYE'
                    /'sub-{}_task-{}_run-{}_{}.edf'.format(
                        trial['sNo'],
                        runfile,
                        trial['blockNo'],
                        timestamp
                    )
                )
                ilnk.beginRecording(
                    win,
                    edfFileName = edfFileName
                )
            # Make sure we have the index for our TRIALS df not our random other df
            if trial['runningTrialNo'] == trials['runningTrialNo'].iloc[-1]:
                lastTrial = True
            if trial['blockTrialNo'] == 0 and not redo: 
                blockEndSeen = False
                # Start of block message
                logMssg = f"Starting Block: {currBlock+1}"
                if info.trackEyes:
                        ilnk.sendMessage(logMssg)
                if info.sendTriggers:
                    trig = int(currBlock+1)
                    pport.setData(trig)
                    time.sleep(trigDur)
                    pport.setData(0)
                    time.sleep(TrigDel)
                logging.warning(
                    logMssg
                )
                logging.flush()
            # abort experiment
            if 'q' in pressedKeys:
                end = True
                break
            # show intro text at the beginning of experiment
            if trial['runningTrialNo'] == 0 and not info.simulate and not redo:
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

            #===========================================================================
            # Fixation Period
            #===========================================================================

            stimFix.pos = [0,0] #Reset after calibration
            frame = 0
            if info.trackEyes:
                logMssg = '_'.join(['Where: Fixon',
                        f"Trial: {trial['blockTrialNo']}/{nTrialsPerBlock-1}",
                        f"Block: {trial['blockNo']}/{nBlocks-1}",
                        f"Type: {trial['trialType']}",
                        f"Type: {trial['thetaVar']}"])
                ilnk.sendMessage(logMssg)
            
            if info.sendTriggers:
                trig = FIXON 
                pport.setData(trig)
                time.sleep(trigDur)
                pport.setData(0)
                time.sleep(TrigDel)

            
            logging.warning(
                "FixON"
            )
            logging.flush()
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

            #===========================================================================
            # Stimulus Presentation and Onset
            #===========================================================================
            logging.warning(
                "FixOFF"
            )
            logging.flush()
            frame = 0
            trialClock.reset()
            flips = []
            terminateTrial = False
            showCue = False
            last = False
            # Send message to exp comp to track exp progress
            if info.trackEyes:
                    status_msg = '\n'.join([f"Trial: {trial['blockTrialNo']}/{nTrialsPerBlock-1}",
                    f"Block: {trial['blockNo']}/{nBlocks-1}",
                    f"Repeats: {len(repeatIdx)}"]) # This is for eye tracking computer display - experimenter convenience
                    ilnk.sendCommand("record_status_message '%s'" % status_msg)

            if trial['blockTrialNo'] == (nTrialsPerBlock-1) or (redo and len(repeatIdx) == 0):
                last = True 

            if redo and len(repeatIdx) != 0:
                # Reset the Trial Variables to rerun
                terminateTrial = False
                repeatIdx.pop(0)
            
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


            #==============================
            # PRESENTING STIMULI
            #=============================
            while True and not showCue:
                # print(f'Mouse dva: {misc.pix2deg(mouse.getPos(),monitor).round(3)}')
                #not gaze contingent stimulus presentation (add code here if needed)
                for key_stim, val_stim in stimGratings.items():
                    val_stim.draw()
                    stimCirc.draw()
                

                if contingent:
                    stimFix.draw()

                counter.text = str(f'{len(repeatIdx)}')
                pressed = event.getKeys(
                    keyList = ['q', 'f', 'j'], 
                    timeStamped = trialClock
                )
                if pressed: 
                    pressedKeys, pressedTimes = zip(*pressed)
                    if 'q' in pressedKeys: 
                        terminateTrial = True
                        break

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
                    
                    # Message for each indiviudal stim
                    if info.trackEyes:
                        ilnk.sendMessage(logMssg)
                
                # Message and trigger for stimulus onset
                    if info.trackEyes:
                            logMssg = '_'.join(['Where: StimOn'
                                f"Trial: {trial['blockTrialNo']}/{nTrialsPerBlock-1}",
                                f"Block: {trial['blockNo']}/{nBlocks-1}",
                                f"Type: {trial['trialType']}",
                                f"Type: {trial['thetaVar']}"])
                            ilnk.sendMessage(logMssg)
                    if info.sendTriggers:
                        ttype = trial['trialType']
                        varcond = trial['thetaVar']
                        trig = Stim_Trig_List[ttype][varcond > 0.075]
                        pport.setData(trig)
                        time.sleep(trigDur)
                        pport.setData(0)
                        time.sleep(TrigDel)
                
                pressed = event.getKeys(
                    keyList = ['q', 'f', 'j'], 
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
                        repeatIdx.append(trials.iloc[idx_trial])
                        counter.text = str(f'{len(repeatIdx)}')
                        
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
                        if info.sendTriggers:
                            trig = FIX_BRK
                            time.sleep(TrigDel)
                            pport.setData(trig)
                            time.sleep(trigDur)
                            pport.setData(0)
                            time.sleep(TrigDel)
                        break
                    # Code for randomly implementing breaks
                    #TODO: REMOVE AFTER CHECKING
#                    if not info.trackEyes and (int(trial['blockTrialNo']) * skip) != 0:
#                       repeatIdx.append(trials.iloc[idx_trial])
#                       counter.text = str(f'{len(repeatIdx)}')
#                       
#                       stimText.pos = misc.deg2pix(
#                           params['STIM']['textPosition']['value'], 
#                           monitor
#                       )
#                       stimText.alignText = 'center'
#                       stimText.text = "Don't break fixation"
#                       #logging message for fix break: TODO: REMOVE FOR REAL EXP
#                       logMssg = f"TRIAL:{trial['runningTrialNo']}_FIX BREAK"
#                       logging.warning(
#                           logMssg
#                       )
#                       logging.flush()
#                       for frame in range(nFramesPerFeedback):
#                           stimText.draw()
#                           win.flip()
#                       terminateTrial = True
#                       if info.sendTriggers:
#                           time.sleep(TrigDel)
#                           trig = FIX_BRK 
#                           pport.setData(trig)
#                           time.sleep(trigDur)
#                           pport.setData(0)
#                           time.sleep(TrigDel)

                # Move on to cue and reset frame count
                #TODO: Eyelink and Trigger here for array off?
                if frame > nFramesPerStimOn:
                    showCue = True
                    frame = 0
                    logMssg = 'ArrayOff'
                    logging.warning(
                        logMssg
                    )
                    logging.flush()
                if pressed: 
                    pressedKeys, pressedTimes = zip(*pressed)
                    if 'q' in pressedKeys: 
                        break

                if showCue:
                    win.flip()
                    jitter = np.random.uniform(0.1,0.3) #100ms - 300ms
                    time.sleep(jitter)

                #===========================================================================
                # Response Screen Based on Trial Type
                #===========================================================================

                if last:
                    blockEndSeen = True

                if trial['trialType'] == 0 and showCue:
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
                    
                    # Trigger and message for response screen value
                    if info.trackEyes:
                            logMssg = '_'.join(['Where: RespScreen',
                            f"Trial: {trial['blockTrialNo']}/{nTrialsPerBlock-1}",
                            f"Block: {trial['blockNo']}/{nBlocks-1}",
                            f"Type: {trial['trialType']}",
                            f"Type: {trial['thetaVar']}"])
                            ilnk.sendMessage(logMssg)
                    
                    while True:
                        leftResp.draw()
                        rightResp.draw()
                        leftText.draw()
                        rightText.draw()
                        win.flip()
                        if info.sendTriggers and frame == 0:
                            if varcond == 0.075:
                                trig = RESPON_VAL_LOW
                            if varcond == 0.3:
                                trig = RESPON_VAL_HIGH
                            pport.setData(trig)
                            time.sleep(trigDur)
                            pport.setData(0)
                            time.sleep(TrigDel)
                        pressed = event.getKeys(
                            keyList = ['q'], 
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
                                break
                            if 'space' in pressedKeys:
                                terminateTrial = True
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
                            logMssg = '_'.join(['Where: RespSub',
                                    f"Trial: {trial['blockTrialNo']}/{nTrialsPerBlock-1}",
                                    f"Block: {trial['blockNo']}/{nBlocks-1}",
                                    f"Type: {trial['trialType']}",
                                    f"Type: {trial['thetaVar']}",
                                    f"Response: {response}"])
                            logging.warning(
                                logMssg
                            )
                            logging.flush()
                            # Trigger and message for participant response
                            if info.trackEyes:
                                    ilnk.sendMessage(logMssg)
                            if info.sendTriggers:
                                if varcond == 0.075:
                                    trig = RESPSUB_VAL_LOW
                                if varcond == 0.3:
                                    trig = RESPSUB_VAL_HIGH
                                pport.setData(trig)
                                time.sleep(trigDur)
                                pport.setData(0)
                                time.sleep(TrigDel)
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
                                # Message and trigger for value feedback
                                logMssg = '_'.join(['Where: Feedback',
                                        f"Trial: {trial['blockTrialNo']}/{nTrialsPerBlock-1}",
                                        f"Block: {trial['blockNo']}/{nBlocks-1}",
                                        f"Type: {trial['trialType']}",
                                        f"Type: {trial['thetaVar']}",
                                        f"Response: {response}"])
                                logging.warning(
                                    logMssg
                                )
                                logging.flush()
                                # Trigger and message for participant feedback
                                if info.trackEyes:
                                        ilnk.sendMessage(logMssg)
                                logging.warning(
                                    "FeedON"
                                )
                                logging.flush()
                                for frame in range(nFramesPerFeedback):
                                    stimText.draw()
                                    win.flip()
                                    if info.sendTriggers and frame == 0:
                                        if varcond == 0.075:
                                            trig = FEEDBCK_VAL_LOW
                                        if varcond == 0.3:
                                            trig = FEEDBCK_VAL_HIGH
                                        pport.setData(trig)
                                        time.sleep(trigDur)
                                        pport.setData(0)
                                        time.sleep(TrigDel)
                                logging.warning(
                                    "FeedOFF"
                                )
                                logging.flush()
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
                            logMssg = '_'.join(['Where: RespSub',
                                    f"Trial: {trial['blockTrialNo']}/{nTrialsPerBlock-1}",
                                    f"Block: {trial['blockNo']}/{nBlocks-1}",
                                    f"Type: {trial['trialType']}",
                                    f"Type: {trial['thetaVar']}",
                                    f"Response: {response}"])
                            logging.warning(
                                logMssg
                            )
                            logging.flush()
                            # Trigger and message for participant response
                            if info.trackEyes:
                                    ilnk.sendMessage(logMssg)
                            if info.sendTriggers:
                                if varcond == 0.075:
                                    trig = RESPSUB_VAL_LOW
                                if varcond == 0.3:
                                    trig = RESPSUB_VAL_HIGH
                                pport.setData(trig)
                                time.sleep(trigDur)
                                pport.setData(0)
                                time.sleep(TrigDel)
                            # show feedback about the payoff
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
                                # Message and trigger for value feedback
                                logMssg = '_'.join(['Where: Feedback',
                                        f"Trial: {trial['blockTrialNo']}/{nTrialsPerBlock-1}",
                                        f"Block: {trial['blockNo']}/{nBlocks-1}",
                                        f"Type: {trial['trialType']}",
                                        f"Type: {trial['thetaVar']}",
                                        f"Response: {response}"])
                                logging.warning(
                                    logMssg
                                )
                                logging.flush()
                                # Trigger and message for participant feedback
                                if info.trackEyes:
                                        ilnk.sendMessage(logMssg)
                                logging.warning(
                                    "FeedON"
                                )
                                logging.flush()
                                for frame in range(nFramesPerFeedback):
                                    stimText.draw()
                                    win.flip()
                                    if info.sendTriggers and frame == 0:
                                        if varcond == 0.075:
                                            trig = FEEDBCK_VAL_LOW
                                        if varcond == 0.3:
                                            trig = FEEDBCK_VAL_HIGH
                                        pport.setData(trig)
                                        time.sleep(trigDur)
                                        pport.setData(0)
                                        time.sleep(TrigDel)
                                logging.warning(
                                    "FeedOFF"
                                )
                                logging.flush()
                            terminateTrial = True
                            break
                        elif info.deadline and frame == (nFramesPerTrial):
                            logMssg = '_'.join(['Where: RespDeadline',
                                        f"Trial: {trial['blockTrialNo']}/{nTrialsPerBlock-1}",
                                        f"Block: {trial['blockNo']}/{nBlocks-1}",
                                        f"Type: {trial['trialType']}",
                                        f"Type: {trial['thetaVar']}",
                                        f"Response: N/A"])
                            if info.trackEyes:
                                    ilnk.sendMessage(logMssg)
                            if info.sendTriggers:
                                trig = NO_RESP
                                pport.setData(trig)
                                time.sleep(trigDur)
                                pport.setData(0)
                            logging.warning(
                                logMssg
                            )
                            logging.flush()

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
                        elif info.simulate:
                            if frame > np.random.choice(range(nFramesPerTrial)):
                                response = np.random.choice(['left', 'right'])
                                payoff = int(trial[f'{response}Payoff'])
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
                        frame += 1
                
                
                if trial['trialType'] == 1 and showCue:
                    bound_radius = misc.deg2pix(params['STIM']['boundRadius']['value'], monitor)
                    # Get random x and y with given radius r
                    start_theta = np.random.uniform(0, 180)
                    x_pos  = bound_radius * math.cos(start_theta)
                    y_pos  = bound_radius * math.sin(start_theta)
                    mouse.setPos((x_pos, y_pos)) # Set position between inner and upper bound
                    mouse.clickReset() # Reset mouse timing for RTs
                    mouse.setVisible(True)
                    
                    # Trigger and message for response screen orientation
                    if info.trackEyes:
                            logMssg = '_'.join(['Where: RespScreen',
                            f"Trial: {trial['blockTrialNo']}/{nTrialsPerBlock-1}",
                            f"Block: {trial['blockNo']}/{nBlocks-1}",
                            f"Type: {trial['trialType']}",
                            f"Type: {trial['thetaVar']}"])
                            ilnk.sendMessage(logMssg)
                    if info.sendTriggers and frame == 0:
                        if varcond == 0.075:
                            trig = RESPON_ORI_LOW
                        if varcond == 0.3:
                            trig = RESPON_ORI_HIGH
                        pport.setData(trig)
                        time.sleep(trigDur)
                        pport.setData(0)
                        time.sleep(TrigDel)
                    
                    while True:
                        mouse_x, mouse_y = mouse.getPos() # get cartesian coord
                        distance = np.sqrt(mouse_x**2 + mouse_y**2) # get the distance from the center
                        angle = math.atan2(mouse_y, -mouse_x) / np.pi * 180
                        if distance >= bound_radius:
                            new_x = bound_radius * math.cos(angle * np.pi / 180) # X coordinate of angle with distance bound_radius
                            new_y = bound_radius * math.sin(angle * np.pi / 180) # y coordinate of angle with distance bound_radius
                            mouse.setPos((-new_x, new_y))
                        angle = angle - 90 #To adjust 0 deg to the vertical pos is increasing cw, neg is increasing ccw 
                        line.ori = angle
                        line.draw()
                        click_area.draw()
                        reproStim.draw()
                        win.flip()
                        pressed = event.getKeys(
                            keyList = ['q'], 
                            timeStamped = trialClock
                            ) 
                        frame += 1                   
                        if pressed:
                            pressedKeys, pressedTimes = zip(*pressed)
                            if 'q' in pressedKeys: 
                                terminateTrial = True
                                break 
                            mouse.setVisible(False)
                            terminateTrial = True
                            break
                        
                        #Mouseclick response
                        if mouse.isPressedIn(click_area ,buttons=[0]):
                            mouse.setVisible(False)
                            _, clickTime = mouse.getPressed(getTime=True)
                            response = round(wrapTo90(line.ori), 0)
                            trials.loc[idx_trial, 'response'] = response
                            trials.loc[idx_trial, 'RT'] = clickTime[0]
                            #Adding log message for response time (to faciliate eye data analysis)
                            logMssg = '_'.join(['Where: RespSub',
                                    f"Trial: {trial['blockTrialNo']}/{nTrialsPerBlock-1}",
                                    f"Block: {trial['blockNo']}/{nBlocks-1}",
                                    f"Type: {trial['trialType']}",
                                    f"Type: {trial['thetaVar']}",
                                    f"Response: {response}"])
                            logging.warning(
                                logMssg
                            )
                            logging.flush()
                            
                            # Trigger and message for participant response
                            if info.trackEyes:
                                    ilnk.sendMessage(logMssg)
                            if info.sendTriggers:
                                if varcond == 0.075:
                                    trig = RESPSUB_ORI_LOW
                                if varcond == 0.3:
                                    trig = RESPSUB_ORI_HIGH
                                pport.setData(trig)
                                time.sleep(trigDur)
                                pport.setData(0)
                                time.sleep(TrigDel)
                            mouse.setVisible(False)
                            
                            if info.feedback:
                                # feedStim.ori = round(mu_cond * 90 / np.pi, 0) # No feedback stim in main task
                                logMssg = '_'.join(['Where: Feedback',
                                        f"Trial: {trial['blockTrialNo']}/{nTrialsPerBlock-1}",
                                        f"Block: {trial['blockNo']}/{nBlocks-1}",
                                        f"Type: {trial['trialType']}",
                                        f"Type: {trial['thetaVar']}",
                                        f"Response: {response}"])
                                logging.warning(
                                    logMssg
                                )
                                logging.flush()
                                # Trigger and message for participant response
                                if info.trackEyes:
                                        ilnk.sendMessage(logMssg)
                                if info.sendTriggers:
                                    if varcond == 0.075:
                                        trig = FEEDBCK_ORI_LOW
                                    if varcond == 0.3:
                                        trig = FEEDBCK_ORI_HIGH
                                    pport.setData(trig)
                                    time.sleep(trigDur)
                                    pport.setData(0)
                                logging.warning(
                                    "FeedON"
                                )
                                logging.flush()
                                for frame in range(nFramesPerFeedback):
                                    line.draw()
                                    # feedStim.draw()
                                    reproStim.draw()
                                    win.flip()
                                logging.warning(
                                    "FeedOFF"
                                )
                                logging.flush()
                                terminateTrial = True
                                break
                        
                        if info.deadline and frame == (nFramesPerTrial):
                            logMssg = '_'.join(['Where: RespDeadline',
                                        f"Trial: {trial['blockTrialNo']}/{nTrialsPerBlock-1}",
                                        f"Block: {trial['blockNo']}/{nBlocks-1}",
                                        f"Type: {trial['trialType']}",
                                        f"Type: {trial['thetaVar']}",
                                        f"Response: N/A"])
                            if info.trackEyes:
                                    ilnk.sendMessage(logMssg)
                            if info.sendTriggers:
                                trig = NO_RESP
                                pport.setData(trig)
                                time.sleep(trigDur)
                                pport.setData(0)
                            logging.warning(
                                logMssg
                            )
                            logging.flush()
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
                    

                #===========================================================================
                # Block to Block Clean-up, Inter Block Feedback, and Finishing Trials
                #===========================================================================

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
    
                    # take a break between blocks
                    stimText.color = 'white'
                    noRepsLeft = False
                    #logic for making sure we get a break only after we have atleast showed n trials in
                    if len(repeatIdx) == 0 and blockEndSeen:
                        noRepsLeft = True
                    if blockEndSeen and noRepsLeft and not info.simulate:
                        # stop recording eye data
                        if ( 
                            info.trackEyes and 
                            ilnk.getStatus() is 'RECORDING'
                        ):
                            stimText.text = 'Block finished. Saving data, please wait.'
                            stimText.pos = misc.deg2pix(
                                params['STIM']['textPosition']['value'], 
                                monitor
                            )
                            stimText.alignText = 'center'
                            stimText.draw()
                            win.flip()
                            # stop recording and save data
                            ilnk.endRecording()  
                        # present feedback
#                        blockEndSeen = False #Prepare for next block
                        pointsMax = trials[f'{tstSide}Payoff'].abs().sum().round(-3)
                        # Give participants the starting amount of points 55% (2200)
                        startPoints = pointsMax * 0.55
                        block = trial['blockNo']
                        if block == 0:
                            prevPoints = round(startPoints, 0) #Assign previous points to beginning points
                        pointsWonBlock = round(trials.loc[trials['blockNo'] == block].dropna()['payoff'].sum(), 0) # Points earned on block
                        pointsWonTotal = round(startPoints + trials.dropna()['payoff'].sum(), 0) # Points earned overall
                        if pointsWonBlock == 0:
                            rewardText.text = '\n'.join([
                                f'{int(pointsWonBlock)} points',
                                ])
                            rewardText.color = 'white'
                        if pointsWonBlock > 0:
                            rewardText.text = '\n'.join([
                                f'GAINED: {int(pointsWonBlock)} points',
                                ])
                            rewardText.color = 'green'
                        if pointsWonBlock < 0:
                            rewardText.text = '\n'.join([
                                f'LOST: {abs(int(pointsWonBlock))} points',
                                ])
                            rewardText.color = 'red'
                        stimText.text = '\n'.join([
                            'Take a short break!',
                            '',
                            f'OLD SCORE: {int(prevPoints)} points'
                        ])
                        stimText.pos = misc.deg2pix(
                            [0, 3], 
                            monitor
                        )
                        rewardText.pos = misc.deg2pix(
                            params['STIM']['textPosition']['value'], 
                            monitor
                        )
                        breakText.pos =misc.deg2pix(
                            [0, -3], 
                            monitor
                        )
                        breakText.text = '\n'.join([f' NEW SCORE: {int(pointsWonTotal)} points',
                            '',
                            'Press SPACE to continue.'])
                        stimText.alignText = 'center'
                        rewardText.alignText = 'center'
                        breakText.alignText = 'center'
                        
                        rest = True
                        
                        while True:
                            stimText.draw()
                            rewardText.draw()
                            breakText.draw()
                            win.flip()
                            if event.getKeys(keyList=['space']): 
                                break    
                        prevPoints = pointsWonTotal # Save point total for this block for feedback

                        if len(repeatIdx) == 0 and lastTrial and currBlock == (nBlocks-1):
                            end = True
                    break
                
                frame += 1
    # END EXPERIMENT
    stimText.pos = misc.deg2pix(
        params['STIM']['textPosition']['value'], 
        monitor
    )
    stimText.alignText = 'center'
    pointsMax = trials[f'{tstSide}Payoff'].abs().sum().round(-3)
    pointsWon = (pointsMax * 0.55) + trials.dropna()['payoff'].sum()
    totalPayoff = round((pointsWon / pointsMax) * totalPrize, 1)
    if totalPayoff > totalPrize:
        totalPayoff = totalPrize
    stimText.text = '\n'.join([
        'End of the session.',
        f'You have earned AUD {totalPayoff}.',
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
    paths = sorted(glob.glob(os.path.join(ROOTPATH, 'LOG', 'gui_info_sub*.json')), key=os.path.getctime)
    with (ROOTPATH / 'RUN' / 'exp_information.json').open('r') as f:
        dataToCollect = json.load(f)
    # Dynamically change entries 
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
            dataToCollect['expData'][5][2] = sub_info['subjectVisualAcuity']
            dataToCollect['expData'][6][2] = sub_info['subjectDominantHand']
            dataToCollect['expData'][7][2] = sub_info['subjectGender']
            dataToCollect['expData'][12][2] = 2200 #Fixed points 
        
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
                    np.nan,
                    np.nan
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
                        stim1Ori = np.nan, #Values in interval [-90,90] (Wrapped s.t. [-90, 90] = [-pi, pi]) 0=vertical
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
        bin_edges = np.linspace(-90, 90, 15) #Change 15 to 11 for 5 bins
        observed, _ = np.histogram(expTrials['tst_samp']/ np.pi * 90, bins=bin_edges)
        obs_val, _ = np.histogram(expTrials.loc[expTrials['trialType'] == 0, 'tst_samp']/ np.pi * 90, bins=bin_edges)
        obs_per, _ = np.histogram(expTrials.loc[expTrials['trialType'] == 1, 'tst_samp']/ np.pi * 90, bins=bin_edges)

        # Define null hypothesis distribution 5 bins
        # 150 value trials per data point
        # 50 percep trials per datapoint
        # expected = np.full(10, 100)
        # exp_val = np.full(10, 75)
        # exp_per = np.full(10, 25)
        # Define null hypothesis distribution 7 bins
        # 106 value trials per data point
        # 36 percep trials per data point
        expected = np.full(14, 71)
        exp_val = np.full(14, 53)
        exp_per = np.full(14, 18)
        
        for i, count in enumerate(observed):
            print(f"Bin {i+1}: {count} observations")
        sd_across_bins = np.std(observed)
        print(f"SD: {sd_across_bins:.4f}")
        
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

    normPointMax = 4000 #Set the max amount of points (NOTE: 55% of the points i.e. 2200 are given to the participants at the beginning )
    nonNormPointMax = expTrials[f'{tstSide}Payoff'].abs().sum() #Get the non-normalised max points (abs for negative points)
    expTrials[f'{tstSide}Payoff'] = round((expTrials[f'{tstSide}Payoff'] / nonNormPointMax) * normPointMax, 0) #normalise Payoff to 4000

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