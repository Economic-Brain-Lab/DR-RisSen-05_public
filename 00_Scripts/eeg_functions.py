'''
Created on 10 May 2016

@author: Dragan Rangelov <d.rangelov@uq.edu.au>
'''
#===============================================================================
# importing libraries
#===============================================================================
import pandas
import mne
import os
import numpy as np
import eeg_fasterAlgorithm as fst
import logging
import pickle
logging.basicConfig(level = logging.INFO)
import h5py
from scipy import special
#===============================================================================
# saving processed data
#===============================================================================
def saveData(savePath,dataEEG,pStep = None, overwrite = True):
    dsets = dataEEG.keys()
    for dset in dsets:
        logging.info('Saving data set {0}.'.format(dset))

        if pStep: psteps = pStep
        else: psteps = dataEEG[dset].keys()

        for pstep in psteps:
            if pstep not in ['data','events']:
                fname = savePath + dset + '_' + pstep + '.fif.gz'
                if not os.path.exists(fname) or overwrite:
                    logging.info('Saving {} data.'.format(pstep))
                    dataEEG[dset][pstep].save(fname)

#===============================================================================
# reading pre-processed data
#===============================================================================
def loadData(savePath, dset, pstep):
    dataFile = [dfile for dfile in os.listdir(savePath) if '_'.join([dset,pstep]) in dfile][0]

    fpath = savePath + dataFile
    try:
        _, _, ftype = dataFile.split('.')
    except ValueError:
        _, ftype = dataFile.split('.')


    if pstep == 'ica': dataeeg = mne.preprocessing.read_ica(fpath)
    elif ftype == 'npy': dataeeg = np.load(fpath)
    elif ftype == 'pkl': dataeeg = pickle.load(open(fpath))
    elif pstep in ['epochs','epochsLong','epochsShort','sEpochs','AvRef','cleanICA','finalERP','finalPSD']: dataeeg = mne.epochs.read_epochs(fpath)
    else: dataeeg = mne.io.RawFIF(fpath, preload = True)

    return dataeeg

#==============================================================================
# reading in raw data
#==============================================================================
def readRawData(dset,
                dataPath, savePath, montagePath,
                dropElectrodes, eogElectrodes, renameElectrodes,
                crop = None, filePath = None):

    fnameRaw = savePath + dset + '_raw.fif.gz'

    if not os.path.exists(fnameRaw):
        if not filePath:
            filePath = dataPath + dset + '_eeg.bdf'
        raweeg = mne.io.read_raw_edf(filePath, preload=True)

        if crop:
            frst = raweeg.copy().crop(0,crop[0])
            scnd = raweeg.copy().crop(crop[1])
            mne.concatenate_raws([frst,scnd])
            raweeg = frst

        # removing flat electrodes that were not recorded
        for chnl in dropElectrodes:
            if chnl in raweeg.ch_names:
                raweeg.drop_channels(ch_names=[chnl])

        if [e for e in renameElectrodes.keys() if e in raweeg.ch_names]:
            raweeg.rename_channels(renameElectrodes)

        raweeg.set_channel_types(dict([(channel,'eog') for channel in raweeg.ch_names if channel in eogElectrodes]))

        raweeg.set_montage(mne.channels.read_montage(kind = 'biosemi64_QBI', path = montagePath))

        info = mne.create_info(ch_names=[u'hEOG',u'vEOG'], sfreq=raweeg.info['sfreq'], ch_types='eog')
        LH, RH, LV, UV = mne.pick_channels(raweeg.ch_names, ['LH', 'RH', 'LV', 'UV'])
        hEOG = raweeg[LH][0] - raweeg[RH][0]
        vEOG = raweeg[LV][0] - raweeg[UV][0]
        newEOG = mne.io.RawArray(np.concatenate([hEOG, vEOG]), info=info)
        raweeg.add_channels(add_list=[newEOG], force_update_info=True)
        raweeg.drop_channels(ch_names=eogElectrodes)

        raweeg.save(savePath + dset + '_raw.fif.gz')

    else:
        raweeg = loadData(savePath, dset, 'raw')

    return raweeg

def reReferenceData(dataeeg):
    datacopy = dataeeg.copy()
    datacopy.info['projs'] = []
    newref, _ = mne.io.set_eeg_reference(datacopy)
    newref.apply_proj(); newref.info['projs'] = []

    return newref

#===============================================================================
# read behavioural data
#===============================================================================
def readBhvData(dset, bhvPath, force = True):
    fname = bhvPath + dset + '_main_data.txt'
    databhv = pandas.read_table(fname,sep = '\t')
    return databhv

#===============================================================================
# finding and interpolating bad channels by FASTER
#===============================================================================
def findBadChannels(dataeeg):
    dataeeg.info['bads'] += fst.faster_bad_channels(dataeeg,
                                                    use_metrics=['correlation',
                                                                 'variance',
                                                                 'hurst'])

def fixBadChannels(dataeeg):
    interpolated = dataeeg.copy().interpolate_bads(mode='accurate')
    return interpolated

#===============================================================================
# selecting epochs
#===============================================================================
def findBadEpochs(epochs):
    picks = mne.pick_types(epochs.info, meg=False, eeg=True)
    bad_epochs = fst.faster_bad_epochs(epochs, picks=picks)
    return np.array(bad_epochs)

def dropBadEpochs(epochs,bad_epochs):
        epochs.drop(bad_epochs)
        return epochs

def findBadSets(sets):
    bad_epochs = fst.faster_bad_sets(sets)
    return np.array(bad_epochs)
#===============================================================================
# removing artifacts with ICA
#===============================================================================
# computing ica
def runICA(dset, eog_ch = ['vEOG', 'hEOG'], n_components = .99):
    logging.info('Running ICA for data set {0}'.format(dset))
    picks = mne.pick_types(dset.info, meg=False, eeg=True)
    random_state = np.random.RandomState(seed=1)
    ica = mne.preprocessing.run_ica(dset,
                                    n_components=n_components,
                                    max_pca_components=len(picks),
                                    random_state=random_state,
                                    picks=picks, eog_ch=eog_ch,
                                    ecg_criterion=None,
                                    skew_criterion=None,
                                    kurt_criterion=None,
                                    var_criterion=None)
    return ica

# finding bad components
def findBadICA(dset,
               ica,
               eog_correlation=3,
               kurtosis=3,
               power_gradient=3,
               hurst=3,
               median_gradient=3,
               epochs = True):
    badICA = fst.faster_bad_components(ica, dset,
                                       thres=dict(eog_correlation=eog_correlation,
                                                  kurtosis=kurtosis,
                                                  power_gradient=power_gradient,
                                                  hurst=hurst,
                                                  median_gradient=median_gradient),
                                       use_metrics=['eog_correlation',
                                                    'kurtosis',
                                                    'power_gradient',
                                                    'hurst',
                                                    'median_gradient'],
                                       epochs = epochs)
    return badICA

# removing bad components
def dropBadICA(dset, ica):
    cleanICA = ica.apply(dset.copy())
    return cleanICA

#===============================================================================
# def findBadChannelsPerEpoch
#===============================================================================
def findBadChannelsPerEpoch(epochs):
    bad_channels_per_epoch = fst.faster_bad_channels_in_epochs(epochs,
                                                               use_metrics=['amplitude',
                                                                            'deviation',
                                                                            'median_gradient',
                                                                            'variance'])
    return bad_channels_per_epoch

def fixBadChannelsPerEpoch(epochs, bad_channels_per_epoch):
    for i, b in enumerate(bad_channels_per_epoch):
        if len(b) > 0:
            epoch = epochs[i]
            epoch.info['bads'] += b
            epoch.interpolate_bads()
            epochs._data[i, :, :] = epoch._data[0, :, :]

#===============================================================================
# finding bridges
#===============================================================================
# def findBridges(dataEEG, LPcutoff = 5, LMcutoff = 10):
#     for dset in dataEEG.keys():
#         picks = mne.pick_types(dataEEG[dset]['finalERP'].info,eeg=True)
#         epochs = dataEEG[dset]['finalERP'].get_data()
#         chnlCombos = list(itertools.combinations(picks,2))
#         ED = np.array([[np.var(epochs[epoch,chnl[0]] - epochs[epoch,chnl[1]]) for epoch in range(epochs.shape[0])] for chnl in chnlCombos])
#         stdED = ED*100/np.median(ED.flatten())
#         # estimating pdf function
#         kdeED = gaussian_kde(stdED.flatten())
#         # computing probability densities for the ED range of interest
#         pdfED = kdeED(range(LPcutoff))
#         # checking if there is any local maximum in the given range
#         gradLP = np.where(pdfED[:-1] - pdfED[1:] > 0)[0]
#         if gradLP:
#             LM = range(LMcutoff)[np.where(kdeED(range(LMcutoff)) == np.min(kdeED(range(LMcutoff))))[0]]


#===============================================================================
# normalising power spectra
#===============================================================================
def normalizePSD(array):
    if len(array.shape) == 2: # normalizing average PSD
        left = array[:,:-2]
        right = array[:,2:]
        centre = array[:,1:-1]
        middle = (left + right)/2
        snr = centre/middle
    else: # normalizing epoch PSD
        left = array[:,:,:-2]
        right = array[:,:,2:]
        centre = array[:,:,1:-1]
        middle = (left + right)/2
        snr = centre/middle
    return np.log10(snr)*10

def SNR(array):
    if len(array.shape) == 2: # normalizing average PSD
        left = array[:,:-2]
        right = array[:,2:]
        centre = array[:,1:-1]
        middle = (left + right)/2
        snr = centre/middle
    else: # normalizing epoch PSD
        left = array[:,:,:-2]
        right = array[:,:,2:]
        centre = array[:,:,1:-1]
        middle = (left + right)/2
        snr = centre/middle
    return snr
#===============================================================================
# angular difference
#===============================================================================
#===============================================================================
# wrapping function
#===============================================================================
def wrap(*args):
    try: theta, boundLow, boundHigh = args
    except ValueError: theta = args[0]; boundLow = -180; boundHigh = 180

    # the angle must be an integer for the wrapping to work
    theta = int(theta)

    rangeSize = boundHigh - boundLow
    if theta < boundLow:
        theta += rangeSize * (1 + (boundLow - theta) / rangeSize)
    return boundLow + (theta - boundLow) % rangeSize
def deltaAngle(alpha,beta):
    angles = np.vstack([alpha,beta])
    delta = np.arctan2(np.sin((angles[0,:] - angles[1,:])),np.cos(angles[0,:] - angles[1,:]))
    return delta
def meanAngle(*args):
    angles = np.vstack([args])
    delta = np.arctan2(np.sin(angles).mean(0),np.cos(angles).mean(0))
    return delta
#===============================================================================
# OLE for complex numbers
#===============================================================================
# import numpy as np
# tarDir = np.deg2rad(np.random.randint(0,360,400)).round(5)
# tarDirC = np.array([np.complex(np.cos(tdir),np.sin(tdir)) for tdir in tarDir]).reshape(200,-1)
# respDirC = tarDirC[:,:2].mean(1).reshape(-1,1)
# respDir = np.deg2rad(np.random.randint(0,360,200)).round(5)
# respDirC = np.array([np.complex(np.cos(tdir),np.sin(tdir)) for rdir in respDir]).reshape(200,-1)
# tst = np.asmatrix(np.asmatrix(tarDirC).H*np.asmatrix(tarDirC)).I*(np.asmatrix(tarDirC).H*np.asmatrix(respDirC))
#
# cmplxTheta = np.vstack([map(lambda x: np.complex(np.cos(x), np.sin(x)), np.deg2rad(map(wrap,BHV.tarDirOne))),
#                         map(lambda x: np.complex(np.cos(x), np.sin(x)), np.deg2rad(map(wrap,BHV.tarDirTwo))),
#                         map(lambda x: np.complex(np.cos(x), np.sin(x)), np.deg2rad(map(wrap,BHV.disDirOne))),
#                         map(lambda x: np.complex(np.cos(x), np.sin(x)), np.deg2rad(map(wrap,BHV.disDirTwo))),
#                         map(lambda x: np.complex(np.cos(x), np.sin(x)), np.deg2rad(map(wrap,360 - BHV.dirMean))),
#                         map(lambda x: np.complex(np.cos(x), np.sin(x)), np.deg2rad(map(wrap,360 - BHV.dirResp)))]).transpose()
#
# # regression
# pred = cmplxTheta[:,:4]
# crit = cmplxTheta[:,-1].reshape((-1,1))
# coefs = np.asmatrix(np.asmatrix(pred).H*np.asmatrix(pred)).I*(np.asmatrix(pred).H*np.asmatrix(crit))
def cmplxOLS(pred, crit):
    '''
    Compute complex-valued OLS regression
    **Input**
    pred: predictor structure (N trials x M predictors) in pi radians
    crit: criterion structure (N trial x J criteria) in pi radians
    '''
    # transform values from real to complex
    pred = np.exp(pred * 1j)
    crit = np.exp(crit * 1j)
    # reformat shape if necessary
    if len(pred.shape) == 1:
        pred = pred.reshape(-1, 1)
    if len(crit.shape) == 1:
        crit = crit.reshape(-1, 1)
    # test if the N trials is the same
    if pred.shape[0] != crit.shape[0]:
        raise ValueError('Mismatch betweeh input shapes!') 
    coefs = (np.asmatrix(
        np.asmatrix(pred).H
        * np.asmatrix(pred)
    ).I * (
        np.asmatrix(pred).H
        * np.asmatrix(crit)
    ))
    return (np.abs(coefs), np.angle(coefs))

#===============================================================================
# Hochberg FWE correction
#===============================================================================
def hochFWE(pvals, alpha = .05):
    """
    function that takes a 1D array of p-values
    and returns a 1D boolean array where true values denote significant values with hochberg correction
    """
    m = len(pvals)
    cvals = np.array([alpha*i/m for i in range(1,m+1)])[::-1]
    psrtd = np.sort(pvals)[::-1]

    pcmp = psrtd <= cvals

    kcmp = np.where(pcmp)[0]

    if kcmp:
        kcmp = kcmp[0]
        indsrt = np.argsort(pvals)[::-1]
        pcorr = np.ones(len(pvals),dtype=bool)
        pcorr[indsrt[:kcmp]] = False
    else:
        pcorr = np.zeros(len(pvals),dtype=bool)

    return(pcorr)
#===============================================================================
# surface laplacian
#===============================================================================
def cosdist(elocs):
    '''
    Compute pairwise cosine distance between electrodes
    inArg: theta and phi - spherical coordinates of electrodes, radius = 1
    outArg: nElectrodes X nElectrodes array
    '''
    if len(elocs) not in [2,3]:
        return "The electrode locations were not correctly specified"

    elif len(elocs) == 2:
        theta, phi = elocs
        # spherical to cartesian transformation
        x = np.sin(theta)*np.cos(phi)
        y = np.sin(theta)*np.sin(phi)
        z = np.cos(theta)

    elif len(elocs) == 3:
        x,y,z = elocs
    # this one is from Cohen ch. 22
    cartesianDist = 1 - ((x.reshape(1,-1) - x.reshape(-1,1))**2 + (y.reshape(1,-1) - y.reshape(-1,1))**2 + (z.reshape(1,-1) - z.reshape(-1,1))**2)*.5

    # this one is from https://math.stackexchange.com/questions/833002/distance-between-two-points-in-spherical-coordinates
#     angularDist = np.sqrt(2*(1 - (np.sin(theta.reshape(1,-1))*np.sin(theta.reshape(-1,1))*np.cos(phi.reshape(1,-1) - phi.reshape(-1,1)) + np.cos(theta.reshape(1,-1))*np.cos(theta.reshape(-1,1)))))

    return cartesianDist

def legpoly(n,X):
    '''
    Compute Legendre polynomial of 0 order and all degrees between 1 and n
    inArg:
    n: degree of the polynomial
    X: data points - matrix of cosine distances
    '''
    orders = np.arange(1,n+1)
    lgp = np.stack([special.lpmv(0,order,X) for order in orders],0)

    return lgp

def matGH(elocs, m = 4, n = 10):
    '''
    Compute G and H matrices for surface laplacian
    inArgs:
    elocs: electrode locations in spherical coordinates (must be either 2xN - for spherical or 3xN - for cartesian coordinates)
    m: smoothing factor, 4 for 64 channels, 3 for >128
    n: order of the Legendre polynomial (from 1 to order)
    outArgs:
    G,H: a tuple of two containing nElectrodes X nElectrodes array for G and H respectively
    '''
    dist = cosdist(elocs)
    lgp = legpoly(n,dist)
    orders = np.arange(1,n+1).astype('float')
    twoN1 = 2*orders + 1
    gdenom = (orders*(orders+1))**m
    hdenom = (orders*(orders+1))**(m-1)

    g = (twoN1[:,np.newaxis,np.newaxis]*lgp)/gdenom[:,np.newaxis,np.newaxis]
    h = (twoN1[:,np.newaxis,np.newaxis]*lgp)/hdenom[:,np.newaxis,np.newaxis]

    G = g.sum(0)/(4*np.pi)
    H = h.sum(0)/(4*np.pi)

    return (G,H)

def surfaceLaplacian(data, elocs, *args):
    '''
    Compute surface laplacian of data
    inArgs:
    data: nElectrodes x time OR trials x nElectrodes x time
    elocs: electrode locations (2 x nElectrodes)
    *args: m = spatial smoothing parameter, n = Legendre order, l = smoothing
    outArgs:
    filtered data (original data format)
    '''
    # reshaping the array
    epochs = 0
    if len(data.shape) == 3:
        epochs, chnls, smpls = data.shape
        newdata = np.concatenate([epoch for epoch in data],-1)
    else:
        newdata = data[:]

    dimX, dimY = elocs.shape
    if dimX > dimY:
        elocs = elocs.transpose()

    if args:
        m, n, l = args
    else:
        if len(elocs) > 100:
            m, n = 3, 15
        else:
            m, n = 4, 10
        l = 1e-5


    G, H = matGH(elocs, m, n)

    Gs = G + np.eye(*G.shape)*l
    invGs = np.linalg.inv(Gs)

    GsinvS = invGs.sum(0)
    dataGs = newdata.transpose().dot(invGs)

    C = dataGs - (dataGs.sum(1)/GsinvS.sum())[:,np.newaxis]*GsinvS

    surfLapl = C.dot(H).transpose()

    # reformat data
    if epochs:
        surfLapl = np.array(np.split(surfLapl, epochs,-1))

    return surfLapl

#===============================================================================
# percentile total amplitude
#===============================================================================
def percentileAmplitude(timeseries,times,percentile, boundary, standardized):
    percentile = np.array(percentile, dtype = 'float').reshape(-1)
    if np.any(percentile > 1):
        percentile /= 100
    if boundary < 0:
        timeseries[timeseries > 0] = 0
    elif boundary > 0:
        timeseries[timeseries < 0] = 0
    if standardized:
        timeseries /= np.abs(timeseries).max()
    sumAmp = np.cumsum(timeseries,-1)
    percAmp = sumAmp[-1]*percentile
    # weighted interpolation
    tcritAll = []
    acritAll = []
    for percA in percAmp:
        # find where is the difference minimal
        idxmin = np.argmin(np.abs(sumAmp - percA))
        # checking whether we should go one interval up or down
        idx = [idxmin,idxmin+1]
        if np.abs(sumAmp[idxmin-1] - percA) < np.abs(sumAmp[idxmin+1] - percA):
            idx = [idxmin-1,idxmin]
        amin, amax = sumAmp[idx]
        tmin, tmax = times[idx]
        tcrit = tmin + (tmax-tmin)*(percA - amin)/(amax - amin)
        acrit = ((tcrit - tmin)*timeseries[idx[0]] + (tmax - tcrit)*timeseries[idx[1]])/(tmax - tmin)

        tcritAll += [tcrit]
        acritAll += [acrit]

    return np.array([acritAll,tcritAll]).flatten()

def peakAmplitude(timeseries,times,fraction):
    idxPeak = np.abs(timeseries).argmax()
    peakAmp = timeseries[idxPeak]
    acrit = peakAmp*fraction
    tcrit = times[idxPeak]
    # weighted interpolation
    if fraction < 1:
        idxmin = np.argmin(np.abs(timeseries - acrit))

        idx = [idxmin,idxmin+1]
        if np.abs(timeseries[idxmin-1] - acrit) < np.abs(timeseries[idxmin+1] - acrit):
            idx = [idxmin-1,idxmin]
        amin, amax = timeseries[idx]
        tmin, tmax = times[idx]
        tcrit = tmin + (tmax-tmin)*(acrit - amin)/(amax - amin)

    return np.array([acrit,tcrit])

def erpSlope(timeseries,times,fraction,interval):
    tcrit = peakAmplitude(timeseries,times,fraction)[-1]

    # finding time-points for which the slope should be computed
    tLow = tcrit - interval
    tHigh = tcrit + interval

    # finding indices of the time points around tLow and tHigh
    idxLow = np.abs(times-tLow).argmin()
    idxL = [idxLow, idxLow+1] # go one index up
    if np.abs(tLow - times[idxLow-1]) < np.abs(tLow - times[idxLow+1]):
        # if the index is closer to one index down, go one index down
        idxL = [idxLow - 1, idxLow]

    ampL = ((tLow - times[idxL[0]])*timeseries[idxL[0]] + (times[idxL[1]] - tLow)*timeseries[idxL[1]])/(times[idxL[1]] - times[idxL[0]])

    idxHigh = np.abs(times-tHigh).argmin()
    idxH = [idxHigh, idxHigh+1]
    if np.abs(tHigh - times[idxHigh-1]) < np.abs(tHigh - times[idxHigh+1]):
        idxH = [idxHigh - 1, idxHigh]

    ampH = ((tHigh - times[idxH[0]])*timeseries[idxH[0]] + (times[idxH[1]] - tHigh)*timeseries[idxH[1]])/(times[idxH[1]] - times[idxH[0]])

    # finding the slope
    erpslope = (ampH - ampL)/(2*interval)

    return erpslope
#===============================================================================
# decoding analyses
#===============================================================================
def smoothData(data,win,sfreq):
    '''
    Replace original data with an average of data within the window time period
    Edges are zero padded
    Accepts
    data = array: N trials x M channels x K time smpls
    window = int: the smoothing period
    sfreq = int: sampling frequency of data
    Returns
    smooth data = array: array: N trials x M channels x K time smpls
    '''
    nsamples = int(round(win*sfreq/1000.)) # how many samples to inclued in the average
    if nsamples%2 == 0: # asserting that the number of samples is odd
        nsamples -= 1
    idxWin = np.arange(nsamples) - nsamples/2
    for idx, _ in enumerate(data.transpose()):
        ssmpl = idx + idxWin
        out = (ssmpl < 0) | (ssmpl >= data.shape[-1]) # identifying samples that are not available
        data[...,idx] = np.sum(data[...,ssmpl[~out]],-1)/nsamples
    return data

def lda(data,conds):
    '''
    *data*: N trials x M sensors x T samples array
    *conds*: N trials array
    Returns:
    *weights*: M sensors x T samples array
    '''
    N = len(data)
    dataT = data.transpose()
    idxA, idxB = [conds == grp for grp in np.unique(conds)]
    weights = []
    ycrit = []
    # iterating across time samples
    for smpl in dataT:
        D = smpl[:,idxB].mean(-1) - smpl[:,idxA].mean(-1)
        sigmaX = smpl.dot(smpl.transpose())/(N-1)
        nu = np.diag(sigmaX).mean()
        W = smpl[None]*smpl[:,None]
        zW = W - W.mean(-1)[...,None]
        sigmaW = zW.sum(-1)/(N-1)
        gamma = (N/(N-1.)**2)*np.diag(sigmaW).sum()/(2*np.sum(np.tril(sigmaX)**2) + np.sum((np.diag(sigmaX) - nu)**2))
        sigmaH = (1-gamma)*sigmaX + gamma*nu*np.eye(*sigmaX.shape)
        sigmaHI = np.linalg.inv(sigmaH)

        w = sigmaHI.dot(D)/(D.transpose().dot(sigmaHI).dot(D))
        weights += [sigmaHI.dot(D)/(D.transpose().dot(sigmaHI).dot(D))]
        ycrit += [w.dot(D)]

    return (np.array(weights).transpose(),np.array(ycrit))

def decodeData(data,conds,cv,*args):
    #TODO finish this function
    if args:
        win, sfreq = args
    else: win = 30; sfreq = 256
    sdata = smoothData(data,win,sfreq) # smoothing original data
    zdata = sdata - sdata.mean(0)[np.newaxis] # demeaning across trials

    TRN = []; TST = []; W = []; Y = []; C = []; G = []
    # training set, test set, weights, decoded activity, predicted group
    # iterating across splits
    for train,test in cv.split(conds,conds):

        wtmp, crittmp = lda(zdata[train],conds[train])
        ytmp = np.sum(zdata[test]*wtmp[None],1)
        gtmp = (ytmp > crittmp[None]).astype('int')

        TRN.append(train)
        TST.append(test)
        W.append(wtmp)
        Y.append(ytmp)
        C.append(crittmp)
        G.append(gtmp)

    return (TRN,TST,W,Y,C,G)

def frwdEncoding_Kok(B,C,cv,fname):
    '''
    Fit forward encoding model and return the weight matrix and the response matrix
    Input
    =====
    B - data to fit: trials x sensors x times
    C - channel responses: trials x channels
    cv - cross-validation
    fname - file name to save data

    Output
    ======
    none
    '''
    with h5py.File(fname, 'w') as f:
        for fold, trls in enumerate(cv):
            logging.warn('Analyzing fold {}'.format(fold))
            trn, tst = trls
            n_trn = len(trn)
            n_tst = len(tst)

            trnB = B[trn].T # times x sensors x trials
            trnB -= trnB.mean(-1)[..., None] # demeaning B across trials
            trnC = C[trn].T # channels x trials
            trnC -= trnC.mean(-1)[..., None] # demeaning C across trials

            tstB = B[tst] - B[trn].mean(0)[None]

            V = [] # responses per time point
            G = [] # gamma correction
            I = []
            for Bt in trnB:
                # Bt: sensors x trials
                Vt = []
                Gt = []
                It = []

                Wt = np.asmatrix(Bt)*np.asmatrix(trnC).T*np.linalg.pinv(np.asmatrix(trnC)*np.asmatrix(trnC).T) # sensors X channels
                Et = np.asmatrix(Bt) - Wt*np.asmatrix(trnC) # residuals
                St = (Et*Et.T)/(n_trn - 1) # noise covariance between sensors

                # regularized covariance matrix
                Rt = np.array(Et[None])*np.array(Et[:,None]) # deviational cross-product
                Rt -= Rt.mean(-1)[...,None] # demeaning the deviations
                Rt **= 2 # squared deviations
                Rt = n_trn*Rt.sum(-1)/(n_trn - 1)**3 # variance of deviational cross-products

                nu = St.diagonal().mean() # average eigenvalue of the noise covariance matrix
                gamma = Rt.sum()/(2*(np.tril(St,-1)**2).sum() + (np.array(St.diagonal() - nu)**2).sum()) # correction factor for covariance matrix regularisation
                St_hat = (1 - gamma)*St + gamma*nu*np.eye(*St.shape)

                Vc = np.array((np.linalg.pinv(St_hat)*Wt)/(Wt.T*np.linalg.pinv(St_hat)*Wt).diagonal()).squeeze() # n sensors
                # inverting the model
                Vt += [Vc]
                Gt += [gamma]
                It += [np.sum(Vc[None,:,None].T*tstB,1)]

                It = np.stack(It,1)

                # saving the fitting per time point
                V += [Vt]; G += [Gt]; I += [It]

            V = np.array(V); G = np.array(G); I = np.stack(I,-1)

            grp = f.create_group('fold_{}'.format(fold))
            grp.create_dataset(name = 'TST', data = tst, compression = 'gzip', compression_opts = 9)
            grp.create_dataset(name = 'V', data = V, compression = 'gzip', compression_opts = 9)
            grp.create_dataset(name = 'G', data = G, compression = 'gzip', compression_opts = 9)
            grp.create_dataset(name = 'RSP', data = I, compression = 'gzip', compression_opts = 9)

def frwdEncoding_Myers(B,C,cv,fname):
    '''
    Fit forward encoding model and return the weight matrix and the response matrix
    Input
    =====
    B - data to fit: trials x sensors x times
    C - channel responses: trials x channels
    cv - cross-validation
    fname - file name to save data

    Output
    ======
    none
    '''
    with h5py.File(fname, 'w') as f:
        for fold, trls in enumerate(cv):
            logging.warn('Analyzing fold {}'.format(fold))
            trn, tst = trls
            n_trn = len(trn)

            trnB = B[trn].T # times x sensors x trials
            trnC = C[trn].T # channels x trials
            trnC = np.concatenate([np.ones([1,n_trn]),trnC]) # adding intercept

            tstB = B[tst]

            I = []
            for Bt in trnB:
                Wt = np.asmatrix(Bt)*np.asmatrix(trnC).T*np.linalg.pinv(np.asmatrix(trnC)*np.asmatrix(trnC).T) # sensors X channels
                Vt = np.linalg.pinv(Wt.T*Wt)*Wt.T
                I += [(np.array(Vt)[None,...,None]*tstB[:,None]).sum(-2)]

            I = np.stack(I,-1)

            grp = f.create_group('fold_{}'.format(fold))
            grp.create_dataset(name = 'TST', data = tst, compression = 'gzip', compression_opts = 9)
            grp.create_dataset(name = 'RSP', data = I, compression = 'gzip', compression_opts = 9)

def frwdEncoding_Myers_CHW(B, C, cv, fname, intercept=True, root='root'):
    '''
    Fit forward encoding model and return the weight matrix and the response matrix
    Input
    =====
    B - data to fit: trials x sensors x times
    C - channel responses: trials x channels
    cv - cross-validation
    fname - file name to save data
    root = name under which to save channel weights

    Output
    ======
    none
    '''
    with h5py.File(fname, 'a') as f:
        root = f.create_group(root)
        for fold, trls in enumerate(cv):
            logging.warn('Analyzing fold {}'.format(fold))
            trn, tst = trls
            n_trn = len(trn)

            trnB = B[trn].T # times x sensors x trials
            trnC = C[trn].T # channels x trials
            if intercept:
                trnC = np.concatenate([np.ones([1,n_trn]),trnC]) # adding intercept

            V = []
            for Bt in trnB:
                Wt = np.asmatrix(Bt)*np.asmatrix(trnC).T*np.linalg.pinv(np.asmatrix(trnC)*np.asmatrix(trnC).T) # sensors X channels
                Vt = np.linalg.pinv(Wt.T*Wt)*Wt.T
                V += [Vt]

            grp = root.create_group('fold_{}'.format(fold))
            grp.create_dataset(name = 'TST', data = tst, compression = 'gzip', compression_opts = 9)
            grp.create_dataset(name = 'CHW', data = np.array(V), compression = 'gzip', compression_opts = 9)

def decoding_Mostert(B, L, cv, fname):
    '''
    Input
    =====
    B - data to fit: trials x sensors x times
    L - condition labels
    cv - cross-validation
    fname - file name to save data

    Output
    ======
    none
    '''
    with h5py.File(fname, 'w') as f:
        for fold, trls in enumerate(cv):
            logging.warn('Analyzing fold {}'.format(fold))
            trn, tst = trls
            times = np.arange(B.shape[-1])
            trnB = B[trn]
            trnL = L[trn]
            G_one, G_two = [trnB[trnL ==  G] for  G in np.unique(trnL)]
            N_one = G_one.shape[1]
            N_two = G_two.shape[1]
            G_one -= G_one.mean(0)[None]
            G_two -= G_two.mean(0)[None]
            # times x sensors x trials
            G_one = G_one.T
            G_two = G_two.T

            V = []
            t = 0
            for t in times:
                Gt_one = G_one[t]
                Gt_two = G_two[t]
                Mt_hat = (Gt_one.mean(1) - Gt_two.mean(1)).reshape(-1,1)
                St_one = (np.asmatrix(Gt_one)*np.asmatrix(Gt_one).T)/(N_one - 1)
                St_two = (np.asmatrix(Gt_two)*np.asmatrix(Gt_two).T)/(N_two - 1)

                # regularized covariance matrix
                Rt_one = Gt_one[None]*Gt_one[:,None] # deviational cross-product
                Rt_one -= Rt_one.mean(-1)[...,None] # demeaning the deviations
                Rt_one **= 2 # squared deviations
                Rt_one = N_one*Rt_one.sum(-1)/(N_one - 1)**3 # variance of deviational cross-products

                nu_one = St_one.diagonal().mean() # average eigenvalue of the noise covariance matrix
                gamma_one = Rt_one.sum()/(2*(np.tril(St_one,-1)**2).sum() + (np.array(St_one.diagonal() - nu_one)**2).sum()) # correction factor for covariance matrix regularisation
                St_hat_one = (1 - gamma_one)*St_one + gamma_one*nu_one*np.eye(*St_one.shape)

                Rt_two = Gt_two[None]*Gt_two[:,None] # deviational cross-product
                Rt_two -= Rt_two.mean(-1)[...,None] # demeaning the deviations
                Rt_two **= 2 # squared deviations
                Rt_two = N_two*Rt_two.sum(-1)/(N_two - 1)**3 # variance of deviational cross-products

                nu_two = St_two.diagonal().mean() # average eigenvalue of the noise covariance matrix
                gamma_two = Rt_two.sum()/(2*(np.tril(St_two,-1)**2).sum() + (np.array(St_two.diagonal() - nu_two)**2).sum()) # correction factor for covariance matrix regularisation
                St_hat_two = (1 - gamma_two)*St_two + gamma_two*nu_two*np.eye(*St_two.shape)

                St_hat = np.array([St_hat_one, St_hat_two]).mean(0)
                Wt = np.linalg.pinv(St_hat)*np.matrix(Mt_hat)
                Vt = Wt/(np.asmatrix(Mt_hat).T*Wt)
                V += [Vt]

            grp = f.create_group('fold_{}'.format(fold))
            grp.create_dataset(name = 'TST', data = tst, compression = 'gzip', compression_opts = 9)
            grp.create_dataset(name = 'SNW', data = np.array(V), compression = 'gzip', compression_opts = 9)
