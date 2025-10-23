# @Author: Dragan Rangelov <uqdrange>
# @Date:   30-7-2019
# @Email:  d.rangelov@uq.edu.au
# @Last modified by:   uqdrange
# @Last modified time: 06-8-2019
# @License: CC-BY-4.0
#===============================================================================
# issue tracker
#===============================================================================
# %%
#===============================================================================
# importing libraries
#===============================================================================
from __future__ import division, print_function
import sys
sys.dont_write_bytecode = True
from auxfunctions import wrapTopi
import numpy as np
from psychopy import logging
#===============================================================================
# functions to present stimuli and collect response
#===============================================================================
def collectResponse(frame,
                    trialDuration,
                    simulatedResponseTime,
                    refreshRate,
                    responseDevice):
    play_sound = False
    feedbackSound = None
    # has the response been given
    if responseDevice.responded:
        play_sound = True
        feedbackSound = 'on'

    # is there response deadline and are we out of time to respond
    elif not np.isnan(trialDuration) and frame == trialDuration:
        play_sound = True
        feedbackSound = 'off'

    # simulating response
    elif not np.isnan(simulatedResponseTime):
        play_sound = True
        feedbackSound = 'on'

    return (play_sound, feedbackSound)

def simulateBehaviour(trialDuration, simulate):
    '''
    convenience function to simulate response and response time
    '''
    simulatedResponse = wrapTopi(np.random.random_sample() * 2 * np.pi)
    if np.isnan(trialDuration):
        simulatedResponseTime = 0
    else:
        simulatedResponseTime = np.random.choice(trialDuration)
    if simulate:
        output = (simulatedResponse, simulatedResponseTime)
    else:
        output = (np.NaN, np.NaN)
    return output

def runTrial(win,
             task,
             simulate,
             trialNumber,
             trialType,
             responseDevice,
             stimDotField,
             kappas,
             radInn,
             arcPos,
             arcSpan,
             dotOffsetPerFrame,
             stimFeedback,
             signalDuration,
             trialDuration,
             motionDirections,
             refreshRate,
             feedbackDuration):

    simulatedResponse, simulatedResponseTime = simulateBehaviour(
        trialDuration
        - [feedbackDuration,0][np.isnan(feedbackDuration)],
        simulate)
    # initialize lists to store trial data
    flipTimes = []
    responseDevice.targAngle = np.random.choice(motionDirections)
    targetAngle = responseDevice.targAngle
    responseAngle = np.NaN
    responseTime = np.NaN
    dirsLeft = []
    dirsRight = []
    signalStart = signalDuration['peripheral']
    signalEnd = np.sum(list(signalDuration.values()))
    play_sound = False
    responded = False
    trialFrame = 0
    feedbackFrame = 0
    # looping over frames per trial
    while True:
        #=======================================================================
        # stop trial or take a pause
        #=======================================================================
        if responseDevice.stop or responseDevice.cont:
            break
        elif responseDevice.pause:
            responseDevice.pause = False
            while True:
                if responseDevice.pause:
                    break
                responseDevice.update()
            responseDevice.pause = False
        #=======================================================================
        # terminate trial
        #=======================================================================
        if ((not np.isnan(trialDuration) # if there is a response deadline
             and trialFrame == trialDuration) # if the deadline is reached
            or (np.isnan(trialDuration) # if there is no deadline
                and responded # and response was provided
                and (np.isnan(feedbackDuration) # if there is no feedback
                     or feedbackFrame == feedbackDuration))): # if the feedback was presented
            break
        #=======================================================================
        # update response interface
        #=======================================================================
        responseDevice.update()
        #=======================================================================
        # check responses and play auditory feedback
        #=======================================================================
        if responseDevice.waitResponse:
            play_sound, feedbackSound = collectResponse(trialFrame,
                                                        trialDuration
                                                        - [feedbackDuration,
                                                           0][np.isnan(feedbackDuration)],
                                                        simulatedResponseTime,
                                                        refreshRate,
                                                        responseDevice)
        # save data and stop waiting for responses
        if play_sound:
            responseDevice.waitResponse = False
            [targetAngle, responseAngle, responseTime] = [responseDevice.targAngle,
                                                          responseDevice.respAngle,
                                                          responseDevice.respTime]
            stimFeedback[feedbackSound].play()
            stimFeedback[feedbackSound].t = 0
            play_sound = False
            responded = True

            # logging response onset
            mssg = '_'.join(['TASK:{}'.format(task),
                             'TRIAL:{}'.format(trialNumber),
                             'RESPONSE',
                             '',
                             '',
                             '',
                             '',
                             '',
                             '',
                             ''])
            logging.warning(mssg)
            logging.flush()
        #=======================================================================
        # set stimulus properties and present stimuli
        #=======================================================================
        logStim = False
        # is it signal period
        if trialFrame == signalStart:
            responseDevice.waitResponse = True
            responseDevice.mouse.clickReset()
            sigDotLife = -1
            stimDotField['C'].setDotLife(sigDotLife)
            
            # Select the motion directions presented concurrently with the signal 
            # that vary in their predictivness of the target signal: 
            # B - both left and right side contribute 50%
            # N - neither side is informative
            # L - left side contributes 100%
            # R - righe side contributes 100%
            dirL, dirR = dict(
                B=wrapTopi(
                    targetAngle
                    + np.random.permutation([-1, 1])
                    * ((2 * np.pi) / len(motionDirections))),
                N=np.random.choice(motionDirections, 2),
                L=[targetAngle, np.random.choice(motionDirections)],
                R=[np.random.choice(motionDirections), targetAngle]
            )[trialType]
            
            dirsLeft += [dirL]
            dirsRight += [dirR]
            logStim = True
        # or noise period
        elif (trialFrame == 0 # initial buffer
              or (trialFrame >= signalEnd
                  and (trialFrame - signalEnd) % signalDuration['peripheral'] == 0)):
            sigDotLife = 1
            stimDotField['C'].setDotLife(sigDotLife)
            # randomly select three motions
            dirL, dirR = np.random.choice(motionDirections, 2)
            dirsLeft += [dirL]
            dirsRight += [dirR]
            logStim = True

        mdirs = dict((stimName, [targetAngle, dirL, dirR][idx])
                     for idx, stimName in enumerate(['C', 'L', 'R']))

        # setting motion direction, coherence level
        for stimName, stim in stimDotField.items():

            stim.setNewXYs(mu = mdirs[stimName],
                           K = kappas[stimName],
                           dist = dotOffsetPerFrame)

            stim.setArcMask(radInn = radInn[stimName],
                            arcPos = arcPos[stimName],
                            arcSpan = arcSpan[stimName])

            if ((stimName == 'C' and task == 'trainMotionDirection')
                or task not in ['trainResponse', 'trainMotionDirection']):
                stim.draw()

        # present feedback
        if (not np.isnan(feedbackDuration)
            and responded
            and feedbackSound == 'on'
            and feedbackFrame < feedbackDuration):

            feedbackFrame += 1
            responseDevice.draw(drawFeedback = True)

        else:
            responseDevice.draw()

        # saving trial info
        flipTimes += [win.flip()]

        mssg = '_'.join(['TASK:{}'.format(task),
                         'TRIAL:{}'.format(trialNumber),
                         'TRLTYPE:{}'.format(trialType),
                         'SIGDIR:{}'.format(round(mdirs['C'], 3)),
                         'SIGCOH:{}'.format(round(kappas['C'], 3)),
                         'SIGLIFE:{}'.format(sigDotLife),
                         'ARCLEFTDIR:{}'.format(round(mdirs['L'], 3)),
                         'ARCLEFTCOH:{}'.format(round(kappas['L'], 3)),
                         'ARCRIGHTDIR:{}'.format(round(mdirs['R'], 3)),
                         'ARCRIGHTCOH:{}'.format(round(kappas['R'], 3))])

        if logStim:
            logging.warning(mssg)
            logging.flush()

        trialFrame += 1

    return (
        dirsLeft, 
        dirsRight, 
        targetAngle, 
        responseAngle, 
        responseTime, 
        flipTimes
    )
