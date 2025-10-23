'''
Author: Dragan Rangelov (d.rangelov@uq.edu.au)
File Created: 2024-08-29
-----
Last Modified: 2024-08-29
Modified By: Dragan Rangelov (d.rangelov@uq.edu.au)
-----
Licence: Creative Commons Attribution 4.0 International
Copyright 2019-2024 Dragan Rangelov, The University of Queensland
'''
#===============================================================================
# %% issue tracker
#===============================================================================
# NOTE: 
# sub-35 has not finished the testing, dropped from analyses 
# sub-02 had only 92 "clean epochs", dropped from aggregates 
# sub-03 had only 467 epochs
# sub-10 had only 682 epochs
# sub-11 had only 659 epochs
# sub-16 had 896 epochs
#===============================================================================
# %% set up plotting
#===============================================================================
import matplotlib as mpl
mpl.use('qt5agg')
import matplotlib.pyplot as plt
plt.ion()
#===============================================================================
# %% set up logging
#===============================================================================
import logging
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s %(levelname)s %(message)s',
    datefmt = '%d-%b-%y %H:%M'
)
#===============================================================================
# %% import libraries
#===============================================================================
import eeg_functions as eegfun
import h5py as hdf
import itertools
import json
import mmodel as mm
import mne
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
# from pymer4 import Lmer, Lm, models
from scipy import signal
from scipy import stats as sps
import seaborn as sbn
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import svm
from skimage import measure
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats import multitest
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sys
import TunMah_woutTempGen as dmah
#===============================================================================
# %% setting paths
#===============================================================================
ROOTPATH = Path.cwd().parent
#===============================================================================
# %% preprocessing raw EEG
#===============================================================================
rawData = sorted(ROOTPATH.rglob('**/**/*eeg-raw.fif.gz'))

bads = {}
for dpath in rawData:
    subID, sesID, _ = dpath.stem.split('_')
    # drop sub-35
    if subID == 'sub-35': continue
    raw = mne.io.read_raw(dpath, preload = True)
    bads[subID] = {}

    # counts = {subID:{}}

    # rename exogenous channels and drop unnecessary ones
    info = mne.create_info( 
        ch_names=[u'hEOG',u'vEOG'], 
        sfreq=raw.info['sfreq'], 
        ch_types='eog'
    )
    LH, RH, LV, UV = mne.pick_channels(
        raw.ch_names, 
        ['EXG3', 'EXG4', 'EXG5', 'EXG6']
    )
    hEOG = raw[LH][0] - raw[RH][0]
    vEOG = raw[LV][0] - raw[UV][0]
    newEOG = mne.io.RawArray(np.concatenate([hEOG, vEOG]), info=info)
    raw.add_channels(add_list=[newEOG], force_update_info=True)
    raw.drop_channels(ch_names=[f'EXG{i}' for i in range(1,9)]) 

    # re-referencing 
    avRef, _ = mne.set_eeg_reference(
        raw.copy(),
        projection = False
    )
    del raw

    # low-pass filtering eeg electrodes
    filtered = avRef.copy()
    picks = mne.pick_types(filtered.info, eeg=True)
    filtered.filter(l_freq=.1, h_freq=100, picks=picks) 
    filtered.notch_filter(50, picks)
    del avRef

    # band-pasds filtering eog electrodes
    picks = mne.pick_types(filtered.info, eog=True)
    filtered.filter(l_freq=1, h_freq=10, picks=picks)

    # detection and interpolating bad channels
    eegfun.findBadChannels(filtered)
    bads[subID]['channels'] = filtered.info['bads']
    interpolated = filtered.copy().interpolate_bads(verbose = False)
    del filtered

    # re-referencing channels
    interpolated, _ = mne.set_eeg_reference(
        interpolated,
        projection = False
    )

    # run ICA artefact detection and rejection
    dwnsmpld_eeg = interpolated.copy().resample(256) 
    ica = mne.preprocessing.ICA(
        random_state=int(subID.split('-')[-1])
    ).fit(dwnsmpld_eeg)
    ica.exclude = eegfun.findBadICA(
        dwnsmpld_eeg,
        ica,
        epochs = False,
        eog_correlation=2.5
    )
    bads[subID]['ICA'] = ica.exclude
    clean = eegfun.dropBadICA(
        interpolated,
        ica
    )
    del interpolated
    del dwnsmpld_eeg

    # epoching
    events = mne.find_events(clean)
    smpls, _, trigs = events.T

    eventsStim = {
        'var-Low/task-Val': 101, 
        'var-High/task-Val': 201, 
        'var-Low/task-Ori': 111, 
        'var-High/task-Ori': 211
    }
    # eventsRespVar = {'lowRespVal': 104, 'highRespVal': 204, 'lowRespOri': 115, 'highRespOri': 215}
    # eventsRespPlay = {'lowPlay': 301, 'highPlay': 401, 'lowReject': 302, 'highReject': 402}
    # eventsRespAcc = {'lowCor': 303, 'highCor': 403, 'lowIncor': 304, 'highIncor': 404}

    epochs = mne.Epochs(
        clean,
        events, # Note this is the cleaned events list
        event_id = eventsStim, # Note this is the event dictionary above
        # tmin = -0.1, tmax = 1, # 1 second is the maximum exposure
        tmin = -0.5, tmax = 1, # longer pre-stimulus window for spectral analyses 
        baseline = (-0.1, 0), # Correct from 100ms before onset
        detrend = 1, #Linear detrending
        preload = True,
        reject = None, flat = None,
        reject_by_annotation = None 
    ).resample(256) #resample to 256Hz after epoching 

    # add metadata to epochs structure
    # NOTE: 'FIX BREAK' is the log message for breaking fixation
    epochAttrs = np.split(
        epochs.annotations.description,
        [
            idx 
            for idx, val in enumerate(epochs.annotations.description)
            # select stimulus onset messages
            if 'BLOCK' in val and 'FLIPS' not in val
        ]
    )[1:]
    assert epochs.get_data().shape[0] == len(epochAttrs)
    metadata = pd.DataFrame(
        data = [[
            float(mssg.split(':')[1])
            for mssg in epoch[0].strip('\n').split('_')
        ] for epoch in epochAttrs],
        columns = ['blockNo', 'trialNo'] + [f'stim_{stim}' for stim in range(1, 13)]
    )
    for col in ['blockNo', 'trialNo']:
        metadata[col] = metadata[col].astype('int')
    # find and drop trials that were aborted
    idx_fixBreak = [
        idx_epoch for idx_epoch, val_epoch in enumerate(epochAttrs)
        if 'FIX BREAK' in '\n'.join(val_epoch)
    ]
    bads[subID]['fixBreak'] = idx_fixBreak
    metadata.drop(idx_fixBreak, inplace = True)
    epochs.drop(idx_fixBreak)
    
    BHVPATH = (
        ROOTPATH 
        / '02_Rawdata' 
        / subID 
        / 'beh' 
        / f'{subID}_task-MAIN_bhv.tsv'
    )
    bhv_df = pd.read_csv(BHVPATH, sep = '\t')
    bhv_df.rename(columns = dict(blockTrialNo = 'trialNo'), inplace = True)
    metadata = metadata.merge(
        bhv_df, 
        on = ['blockNo', 'trialNo'],
        validate = '1:1'
    )
    # validate that the correct trial info was merged
    assert (
        metadata[[
            f'stim_{stim}' 
            for stim in range(1,13)
        ]].round(2).values 
        - metadata[[
            f'stim{stim}Ori' 
            for stim in range(1,13)
        ]].round(2).values
    ).sum() == 0
    # add metadata to epochs structure
    epochs.metadata = metadata
    # find bad epochs
    bads[subID]['epochs'] = [
        int(epoch) 
        for epoch in eegfun.findBadEpochs(epochs)
    ]
    epochs.drop(bads[subID]['epochs'])
    # Spatial filtering and saving
    SAVEPATH = (
        ROOTPATH 
        / '03_Derivatives' 
        / 'eeg_LongPreStimWin' 
        # / 'eeg' 
        / subID
    )
    SAVEPATH.mkdir(exist_ok = True, parents = True)
    epochs_csd = mne.preprocessing.compute_current_source_density(epochs)
    
    epochs_csd.save(
        SAVEPATH / f'{subID}_ses-MAIN_lck-STIM_csd-epo.fif.gz',
        overwrite = True
    )
    with open(SAVEPATH / f'{subID}_ses-MAIN_lck-STIM_bads.json', 'w') as f:
        json.dump(bads[subID], f, indent=2)
# # characterise bads
# bads = sorted(ROOTPATH.rglob('**/eeg/**/*_ses-MAIN_lck-STIM_bads.json'))
# data_bads = []
# for sub_bad in bads:
#     with sub_bad.open('r') as f:
#         data = json.load(f)
#         sub = sub_bad.stem.split('_')[0]
#         nBadChannels = len(data['channels'])
#         nBadICA = len(data['ICA'])
#         nFixBreak = len(data['fixBreak'])
#         nBadEpochs = len(data['epochs'])
#         data_bads += [[sub, nBadChannels, nBadICA, nFixBreak, nBadEpochs]]
# agg_bads = pd.DataFrame(
#     data_bads, 
#     columns = ['subNo', 'nBadChannels', 'nBadICA', 'nFixBreak', 'nBadEpochs']
# )
# agg_bads.loc[
#     ~agg_bads['subNo'].isin(['sub-02','sub-35']),
# ].describe()
#===============================================================================
# %% ERP analyses
# NOTE: full random model with random slopes per participant fails
#===============================================================================
epochPaths = sorted(ROOTPATH.rglob("*ses-MAIN_lck-STIM_csd-epo.fif.gz"))
dropSUB = ['sub-02']
epochData = dict([(
    dpath.stem.split('_')[0], 
    mne.read_epochs(dpath, preload = True)
) for dpath in epochPaths
if dpath.stem.split('_')[0] not in dropSUB])
times = list(epochData.values())[0].times
# %% find outlier participants
# NOTE: no outlier participantss
badSubs = eegfun.findBadSets([val.average().data for val in epochData.values()])
# %%aggregation across trials
aggCSD, bhv_df = zip(*[
    (sub.pick('csd').get_data(), sub.metadata)
    for sub in epochData.values()
])
bhv_df = pd.concat(bhv_df)
# which side was the test side
bhv_df = bhv_df.merge(
    bhv_df.groupby(['sNo']).apply(
        lambda x: ['right' if x['rightPayoff'].unique().size > 2 else 'left'][0]
    ).reset_index().rename(columns = {0:'tstSide'}),
    on = ['sNo']
)
# what was the payoff of the risky choice
bhv_df['tstPayoff'] = (
    bhv_df['deltaThetaRad'] 
    * 7 # this is a scaling factor to normalize maximum absolute payoff to 11
).round(0)
# code task
bhv_df['task'] = 'risk'
bhv_df.loc[bhv_df['trialType'] == 1, 'task'] = 'perc'
# did they choose the risky option
bhv_df['riskyChoice'] = np.nan
bhv_df.loc[
    (bhv_df['task'] == 'risk')
    & (bhv_df['response'] == bhv_df['tstSide']),
    'riskyChoice'
] = 1
bhv_df.loc[
    (bhv_df['task'] == 'risk')
    & (bhv_df['response'] != bhv_df['tstSide']),
    'riskyChoice'
] = 0
bhv_df['payoffSign'] = np.sign(bhv_df['tstPayoff'])
bhv_df['payoffValue'] = np.abs(bhv_df['tstPayoff'])
#%% binning trials relative to angles
thetas = np.angle(np.exp(bhv_df['tstThetaRad'] * 1j))
NCHANS = 6
channels = np.sort(np.angle(
    np.exp(np.arange(
        np.pi/NCHANS,
        2*np.pi+np.pi/NCHANS,
        2*np.pi/NCHANS
    ) * 1j) 
))
theta_bins = pd.cut(thetas, channels)
theta_mids = np.array([np.pi] + list(theta_bins.categories.mid.values)).round(3)
# add bin codes (1 to 6) to the data frame 
bhv_df['theta_bin'] = (theta_bins.codes + 1)
bhv_df['bin_angle'] = theta_mids[bhv_df['theta_bin'].values]
bhv_df['bin_value'] = (np.angle(
    np.exp(
        np.abs(np.angle(
            np.exp(bhv_df['bin_angle'] * 1j) 
            # rotate the bin angles depending on the participant value-mapping
            * np.exp(np.array([0, np.pi])[bhv_df['sNo'] % 2] * 1j)
        )) * 1j
    ) 
    / np.exp(.5 * np.pi * 1j)
) * 7).round(0)
#%% bining tst payoff
payoff_bins = pd.cut(bhv_df['tstPayoff'].values, np.linspace(-11,11,8).round(2))
payoff_mids = payoff_bins.categories.mid.values.round(3)
payoff_codes = payoff_bins.codes.copy()
payoff_codes[payoff_codes == -1] = 0
bhv_df['payoff_bins'] =  payoff_codes
bhv_df['payoff_vals'] = payoff_mids[payoff_codes]
# %% concatenate csd matrices and smooth them
aggCSD = np.concatenate(aggCSD)
sfreq = epochData['sub-01'].info['sfreq']
SD = int(sfreq * .016)
gwin = signal.windows.gaussian(
    # the window width is 16 SD (-8 to +8). That should be enough.
    16 * SD,
    SD
)
gwin /= gwin.sum()
padWidth = int(sfreq * .5)
aggCSD = signal.convolve(
    np.pad(
        aggCSD, 
        pad_width = (
            (0, 0),
            (0, 0), 
            (padWidth, padWidth)
        ),
        mode = 'reflect'
    ), 
    gwin[None, None], 
    'same'
)[..., padWidth:-padWidth]
# # %% ERPs
# avVAR = np.array([
#     [
#             aggCSD[
#                 (bhv_df['tstVarRad'] == var)
#                 & (bhv_df['sNo'] == sub)
#             ].mean(0)
#             for sub in bhv_df['sNo'].unique()
#     ]
#     for var in bhv_df['tstVarRad'].unique()
# ])
# t_val_var, p_val_var = sps.ttest_rel(*avVAR)
# p_adj_var = multitest.fdrcorrection(p_val_var.flatten())[1].reshape(*p_val_var.shape)
# t_val_var[p_adj_var >= .05] = 0
# # cntr_var = measure.find_contours(
# #     -np.log(p_adj_var), -np.log(.05)
# # )
# # %% plotting stimulus-locked channel responses
# gavTot = mne.grand_average([sub.average('csd') for sub in epochData.values()])

# ch_names = mne.channels.make_standard_montage('biosemi64').ch_names
# info = mne.create_info(ch_names, 256, 'eeg')
# info.set_montage('biosemi64')
# pos, outlines = mne.viz.topomap._get_pos_outlines(
#     info,
#     mne.pick_types(info, eeg = True), 
#     sphere = (0,0,0, 1), to_sphere = True
# )
# idx_sensor = pos[:, -1].argsort()
# sbn.set_style('ticks')
# sbn.set_palette('husl', 75, desat = .60)
# fsmall, fmed, flarge = [8, 10, 11]
# fsmall, fmed, flarge = [10, 11, 11]
# plt.rc('font', size=fmed)          # controls default text sizes
# plt.rc('font', family='Calibri')
# plt.rc('axes', facecolor=(0, 0, 0, 0))
# plt.rc('axes', titlesize=fsmall)     # fontsize of the axes title
# plt.rc('axes', labelsize=fsmall)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=fsmall)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=fsmall)    # fontsize of the tick labels
# plt.rc('legend', fontsize=fsmall)    # legend fontsize
# plt.rc('figure', titlesize=fmed)

# fig = plt.figure(figsize = (6.6, 4.4))

# axERP = fig.add_subplot(211)
# axERP.plot(gavTot.times, gavTot.data[idx_sensor[::-1]].T)
# axERP.xaxis.set_visible(False)
# axERP.set_yticks([-.002,0,.002])
# axERP.set_yticklabels(['-2.0','0.0','2.0'])
# axERP.spines['left'].set_bounds(-.002,.002)
# axERP.tick_params(direction = 'inout', length = 5)
# axERP.set_ylabel('mV/m$^{2}$')
# for spine in ['top','right', 'bottom']:
#     axERP.spines[spine].set_visible(False)

# axTopo = axERP.inset_axes([.75,.75,.5,.5])
# axTopo.set_aspect('equal')
# mne.viz.topomap._draw_outlines(axTopo, outlines)
# for ln in axTopo.get_children()[:4]:
#     ln.set_lw(.5)
# axTopo.scatter(pos[idx_sensor[::-1], 0]*12.5, pos[idx_sensor[::-1], 1]*12.5,
#                picker=True, s=12.5,
#                color = sbn.husl_palette(75, s = .60)[:64],
#                edgecolor='None', linewidth=2, clip_on=False)
# sbn.despine(ax = axTopo, left=True, bottom=True)
# axTopo.xaxis.set_visible(False)
# axTopo.yaxis.set_visible(False)

# axVAR = fig.add_subplot(212)
# # im = axVAR.imshow(
# #     erpVAR[idx_sensor], aspect = 'auto', cmap = 'coolwarm',
# #     vmin = -.0005, vmax = .0005,
# #     origin = 'lower'
# # )
# im = axVAR.imshow(
#     t_val_var[idx_sensor], aspect = 'auto', cmap = 'coolwarm',
#     vmin = -6, vmax = 6,
#     origin = 'lower'
# )
# # cntr_erp = measure.find_contours(
# #     -np.log(p_adj_erp[idx_sensor]), -np.log(.05)
# # )
# # for ln in cntr_erp:
# #     axVAR.plot(
# #         ln[:, 1], ln[:, 0],
# #         lw = 1, color = 'white'
# #     )
# cbar = plt.colorbar(
#     im, location = 'top',
#     fraction = .05
# )
# # cbar.set_ticks([-.0005,0,.0005])
# cbar.set_ticks([-6,0,6])
# cbar.set_ticklabels(['-6',' 0', ' 6'])
# # cbar.set_ticklabels([-0.5,0,0.5])
# # cbar.set_label('$High - Low$ $[mV/m^2]$')
# cbar.set_label('Low Noise - High [t]')
# cbar.ax.tick_params(direction = 'inout', length = 5)
# axVAR.tick_params(direction = 'inout', length = 5)
# idx_ch, val_ch = zip(*[
#     (idx_ch, val_ch) 
#     for idx_ch, val_ch in enumerate(np.array(gavTot.ch_names)[idx_sensor]) 
#     if val_ch in ['AFz', 'Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'POz']
# ])
# axVAR.set_yticks(idx_ch)
# axVAR.set_yticklabels(val_ch, fontsize = 8)
# axVAR.set_ylabel('')
# axVAR.spines['left'].set_bounds(min(idx_ch), max(idx_ch))
# times = np.linspace(0,1,5)
# axVAR.set_xticks(gavTot.time_as_index(times))
# axVAR.set_xticklabels(['0 ms', '250', '500', '750', '1,000'])
# axVAR.spines['bottom'].set_bounds(*gavTot.time_as_index(times)[[0,-1]])
# axVAR.set_xlim(gavTot.time_as_index(axERP.get_xlim()))
# for spine in ['top','right']:
#     axVAR.spines[spine].set_visible(False)

# fig.tight_layout()
# fig.savefig(ROOTPATH / '05_Exports' / 'gavERP_map_t.png', dpi = 600)
#===============================================================================
# %% correlations with value and choice
#===============================================================================
ROI = ['CP1', 'CPz', 'CP2', 'P1', 'Pz', 'P2']
picks = mne.pick_channels(epochData['sub-01'].ch_names, ROI)
times = epochData['sub-01'].times
aggROI = aggCSD[:,picks].mean(1)
# avROI = np.array([
#     [
#             aggROI[
#                 (bhv_df['tstVarRad'] == var)
#                 & (bhv_df['sNo'] == sub)
#             ].mean(0)
#             for sub in bhv_df['sNo'].unique()
#     ]
#     for var in bhv_df['tstVarRad'].unique()
# ])
# %% Mixed GLM, fully random design
mdf = []
for idx_tsmpl, tsmpl in enumerate(aggROI.T):
    print(f'time sample {idx_tsmpl} our of {aggROI.shape[-1]}')
    tmp_df = bhv_df.copy()
    tmp_df['thetaVar_fac'] = tmp_df['thetaVar'].astype('category')
    tmp_df['payoffSign_fac'] = np.sign(bhv_df['bin_value']).astype('category')
    tmp_df['payoffMagn_fac'] = np.abs(bhv_df['bin_value']).astype('category')
    tmp_df['erp'] = tsmpl * 1000
    mdf += [smf.mixedlm(
        "erp ~ thetaVar_fac * payoffSign_fac * payoffMagn_fac", 
        data = tmp_df, 
        groups = tmp_df['sNo']
    ).fit()]
# %% 
coef, sem = zip(*np.array([mdl.summary().tables[1].loc[
    'Intercept', 
    ['Coef.', 'Std.Err.']
].values.astype('float') for mdl in mdf]))
coef = np.array(coef); sem = np.array(sem)
p_vals = np.stack([mdl.pvalues[:8] for mdl in mdf])
t_vals = np.array([mdl.tvalues[:8] for mdl in mdf])
p_adj = multitest.fdrcorrection(p_vals.flatten())[1].reshape(*p_vals.shape)
t_vals[p_adj >= .05] = 0
# %% compute coefficients
idx_crit = epochData['sub-01'].time_as_index(.900)[0]
agg_mdl_par, agg_mdl_cov = np.stack([
    mdl.summary().tables[1].values[:-1].astype('float')[:,:2] 
    for mdl in mdf[idx_crit:]
]).T
av_mdl_par = agg_mdl_par.mean(1)
agg_mdl_par[p_adj[idx_crit:].T >= .05] = 0
mdl_par = agg_mdl_par.mean(1)
mdl_cov = np.stack([
    mdl.cov_params().values[:-1,:-1] 
    for mdl in mdf[idx_crit:]
]).mean(0)
idx_par = np.array([
    [1, 0, 0, 1, 0, 0, 0, 0], # LowLossLarge
    [1, 0, 0, 0, 0, 0, 0, 0], # LowLossSmall

    [1, 0, 1, 0, 0, 0, 0, 0], # LowGainSmall
    [1, 0, 1, 1, 0, 0, 1, 0], # LowGainLarge

    [1, 1, 0, 1, 0, 1, 0, 0], # HighLossLarge
    [1, 1, 0, 0, 0, 0, 0, 0], # HighLossSmall

    [1, 1, 1, 0, 1, 0, 0, 0], # HighGainSmall
    [1, 1, 1, 1, 1, 1, 1, 1]  # HighGainLarge
]).astype('bool')
pred_cov = np.stack(np.split(np.array([
    np.sqrt(av_mdl_par[idx][None] @ mdl_cov[idx,:][:, idx] @ av_mdl_par[idx][:, None]).flatten() 
    for idx in idx_par
]), 2))
pred_coef = np.stack(np.split(np.array([mdl_par[idx].sum() for idx in idx_par]), 2))
# %% mixed-effects analyses
# NOTE: 256 is index of t = 900 ms
bhv_df['CPP'] = aggROI[..., 256:].mean(1)
bhv_df['thetaVar_fac'] = bhv_df['thetaVar'].astype('str')
bhv_risk = bhv_df.loc[bhv_df['task'] == 'risk'].copy()

mdl = Lmer(
    "riskyChoice ~ CPP * thetaVar_fac + (CPP | sNo)",
    data = bhv_risk, family = 'binomial'
)
mdl.fit()
# # %%
# plt.figure()
# plt.imshow(t_vals.T, aspect = 'auto', cmap = 'viridis')
# plt.gca().set_yticks(range(mdf[0].tvalues.keys()[:8].size))
# ylbls = [
#     'Low / Loss / Small', 
#     'High Noise', 
#     'Gain', 
#     'Large Value',
#     'High / Gain',
#     'High / Large',
#     'Gain / Large',
#     'High / Gain / Large'
# ]
# plt.gca().set_yticklabels(mdf[0].tvalues.keys()[:8])
# plt.gca().set_yticklabels(ylbls)
# # %% plot coefficients
# plt.figure()
# for idx_noise, val_noise in enumerate(pred_coef):
#     plt.bar(
#         np.arange(4) + [0,4][idx_noise], 
#         val_noise, 
#         yerr = pred_cov[idx_noise].flatten(), 
#         label = ['Low Noise', 'High'][idx_noise]
#     )
# plt.gca().axhline(0, color = 'k', lw = .75)
# plt.legend(frameon=False, loc = 'upper left')
# plt.gca().set_xticks(range(8))
# plt.gca().set_xticklabels(['-11','-4','4','11'] * 2)
# plt.gca().set_xlabel('Bin Value')
# %% CPP
gavTot = mne.grand_average([sub.average('csd') for sub in epochData.values()])
n_chns, n_smpls = gavTot.data.shape
topo_mask = np.zeros([n_chns, n_smpls], dtype = 'bool')
topo_times = np.linspace(.1,.9,5)
which_times = gavTot.time_as_index(topo_times)
which_chnls = picks
for _time in which_times:
    topo_mask[which_chnls,_time] =True

sbn.set_style('ticks')
sbn.set_palette('husl', 75, desat = .60)
fsmall, fmed, flarge = [8, 10, 11]
fsmall, fmed, flarge = [10, 11, 11]
plt.rc('font', size=fmed)          # controls default text sizes
plt.rc('font', family='Calibri')
plt.rc('axes', facecolor=(0, 0, 0, 0))
plt.rc('axes', titlesize=fsmall)     # fontsize of the axes title
plt.rc('axes', labelsize=fsmall)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=fsmall)    # fontsize of the tick labels
plt.rc('ytick', labelsize=fsmall)    # fontsize of the tick labels
plt.rc('legend', fontsize=fsmall)    # legend fontsize
plt.rc('figure', titlesize=fmed)

# %% plot topographies
fig = plt.figure(figsize = (6.6, 6.6))

axTopo = [
    plt.subplot2grid(
        (30,32), 
        (0, idx_ax * 6),
        rowspan = 10,
        colspan = 6,
        fig = fig
    )
    for idx_ax in range(5)
]
for idx_ax, val_ax in enumerate(axTopo):
    im = mne.viz.plot_topomap(
        data = gavTot.data[:,which_times[idx_ax]],
        pos = gavTot.info, 
        vlim = (-2e-3,2e-3), 
        sphere = .07, 
        mask = topo_mask[:, which_times[idx_ax]], 
        mask_params = dict(markersize = 4, markeredgewidth = .75, linewidth = 0),
        cmap = 'coolwarm',
        axes = val_ax
    )
for idx_ax, val_ax in enumerate(axTopo):
    lbl = [f'{t:.1f}' for t in topo_times][idx_ax]
    if idx_ax == 4:
        lbl += ' s'
    val_ax.set_title(lbl)

axCBar = plt.subplot2grid(
    (30,32), 
    (2, 31),
    rowspan = 6,
    colspan = 1,
    fig = fig
)
plt.colorbar(mappable = im[0], orientation = 'vertical', cax = axCBar, shrink = .05)

axCBar.tick_params('x', direction = 'inout', length = 5)
axCBar.set_yticks([-2e-3, 0, 2e-3])
axCBar.set_yticklabels(['-2', ' 0 $\dfrac{mV}{m^2}$', ' 2'])
axCBar.tick_params(direction = 'inout', length = 5)

# plot LME t-values

axMLE = plt.subplot2grid(
    (30,32), 
    (10, 0),
    rowspan = 10,
    colspan = 30,
    fig = fig
)
im = axMLE.imshow(t_vals.T, aspect = 'auto', cmap = 'coolwarm', vmin = -8, vmax = 8)
axMLE.set_yticks(range(mdf[0].tvalues.keys()[:8].size))
ylbls = [
    'Const.', 
    'Noise',
    'Sign',
    'Magn.', 
    r'N$\times$S',
    r'N$\times$M',
    r'S$\times$M',
    r'N$\times$S$\times$M'
]
axMLE.set_yticklabels(ylbls)
axMLE.set_xticks(gavTot.time_as_index(np.linspace(0,1,6)))
axMLE.set_xticklabels([
    f"{val_lbl:.1f}"
    if idx_lbl < 4
    else f"{val_lbl:.1f}" + " s"
    for idx_lbl, val_lbl in enumerate(np.linspace(0,1,6))
])
axMLE.tick_params(direction = 'in', length = 3)
axCBar = plt.subplot2grid(
    (30,32), 
    (12, 31),
    rowspan = 6,
    colspan = 1,
    fig = fig
)
plt.colorbar(mappable = im, orientation = 'vertical', cax = axCBar, shrink = .05)
axCBar.tick_params('x', direction = 'inout', length = 5)
axCBar.set_yticks([-8,0,8])
axCBar.set_yticklabels(['-8', ' 0 $t$', ' 8'])
axCBar.tick_params(direction = 'inout', length = 5)

# # plot intercept time-course
# axCPP = plt.subplot2grid(
#     (30,32), 
#     (21, 0),
#     rowspan = 9,
#     colspan = 16,
#     fig = fig
# )
# axCPP.fill_between(
#     times, coef - sem, coef + sem,
#     color = 'darkgrey', alpha = .3
# )
# axCPP.plot(times, coef, color = 'darkgrey', lw = 1.5)

# axCPP.set_yticks([-.5,0,.5])
# axCPP.spines['left'].set_bounds(-.5, .5)
# axCPP.set_ylabel('Const. [$\dfrac{mV}{m^2}$]')
# axCPP.set_yticklabels(['-0.5','0.0','0.5'])
# axCPP.set_xticks(np.linspace(0, 1, 6))
# axCPP.set_xticklabels([
#     f"{val_lbl:.1f}"
#     if idx_lbl < 4
#     else f"{val_lbl:.1f}" + " s"
#     for idx_lbl, val_lbl in enumerate(np.linspace(0,1,6))
# ])
# axCPP.spines['bottom'].set_bounds(0,1)
# for spine in ['top', 'right']:
#     axCPP.spines[spine].set_visible(False)
# axCPP.tick_params(direction = 'in', length = 3)
# axCPP.set_xlabel('')

# plot coefs
# axCoef = plt.subplot2grid(
#     (30,32), 
#     (21, 16),
#     rowspan = 9,
#     colspan = 16,
#     fig = fig
# )
axCoef = plt.subplot2grid(
    (30,32), 
    (21, 0),
    rowspan = 9,
    colspan = 16,
    fig = fig
)

cols = np.array(sbn.color_palette('muted'))[[2,4]]
vals = np.array([-11,-4,4,11])
for idx_noise, val_noise in enumerate(pred_coef):
    axCoef.fill_between(
        vals + [-1,1][idx_noise], 
        val_noise - pred_cov[idx_noise].flatten(),
        val_noise + pred_cov[idx_noise].flatten(),
        color = cols[idx_noise], alpha = .3
    )
    axCoef.plot(
        vals + [-1,1][idx_noise],
        val_noise,
        marker = ['v', '^'][idx_noise],
        color = cols[idx_noise],
        label = ['Low Noise', 'High'][idx_noise]
    )

axCoef.legend(frameon= False, loc = 'lower center')
axCoef.set_xticks(vals)
axCoef.set_xticklabels([f'{val}' for val in vals])
axCoef.set_xlabel('Bin Value')
axCoef.tick_params(direction = 'in', length = 3)
axCoef.spines['bottom'].set_bounds(-11,11)
# axCoef.yaxis.tick_right()
# axCoef.yaxis.set_label_position('right')
axCoef.yaxis.tick_left()
axCoef.yaxis.set_label_position('left')
axCoef.set_ylabel('Fitted data')
axCoef.set_yticks([0, -.1, -.2, -.3])
axCoef.set_yticklabels([' 0.0', '-0.1', '-0.2', '-0.3'])
# axCoef.spines['right'].set_bounds(0, -.3)
axCoef.spines['left'].set_bounds(0, -.3)
# for spine in ['left', 'top']:
for spine in ['right', 'top']:    
    axCoef.spines[spine].set_visible(False)

# %% plot LOG
axLog = plt.subplot2grid(
    (30,32), 
    (21, 16),
    rowspan = 9,
    colspan = 16,
    fig = fig
)

for idx_tht, val_tht in enumerate(bhv_risk['thetaVar_fac'].unique()):
    idx = bhv_risk['thetaVar_fac'] == val_tht
    axLog.plot(
        bhv_risk.loc[idx, 'CPP'],
        np.array(mdl.fits)[idx],
        '.', mec = cols[::-1][idx_tht], mfc = 'none', alpha = .3
    )

axLog.set_ylim(.3, .7)
axLog.set_yticks(np.linspace(.3,.7,5))
axLog.spines['right'].set_bounds(.3,.7)
axLog.set_ylabel('p(Risky choice)')
axLog.yaxis.tick_right()
axLog.yaxis.set_label_position('right')

axLog.set_xlim(-.04, .04)
axLog.set_xticks(np.linspace(-.03,.03, 5)[[0,1,3,4]])
axLog.set_xticklabels([
    str(i).rjust(2,'0') 
    for i in np.linspace(-3,3,5)[[0,1,3,4]]
])
axLog.spines['bottom'].set_bounds(-.03, .03)
axLog.set_xlabel('CPP')

for spine in ['left','top']:
    axLog.spines[spine].set_visible(False)
axLog.tick_params(direction = 'in', length = 3)

fig.subplots_adjust(.12,.06,.92,1)
fig.savefig(ROOTPATH / '05_Exports' / 'gavCPP_v03.png', dpi = 600)
# #===============================================================================
# # %% Mahalanobis distance
# # TODO: temporal generalisation matrix
# # TODO: distances for representational similarity analysis
# #===============================================================================
# epochPaths = sorted(ROOTPATH.rglob("*ses-MAIN_lck-STIM_csd-epo.fif.gz"))
# # DONE: comment this out when submitting jobs to cluster
# # IDX_JOB = 0
# IDX_JOB = int(sys.argv[1])
# dpath = epochPaths[IDX_JOB]
# subID = dpath.stem.split('_')[0]
# logging.info(f'Analysing {subID}')
# epochData = mne.read_epochs(dpath, preload = True)
# bhv_df = epochData.metadata.copy()
# thetas = np.angle(np.exp(bhv_df['tstThetaRad'] * 1j))
# csd = epochData.copy().pick('csd').get_data()
# # smoothing eeg data
# SD = int(epochData.info['sfreq'] * .016)
# gwin = signal.windows.gaussian(
#     # the window width is 16 SD (-8 to +8). That should be enough.
#     16 * SD,
#     SD
# )
# padWidth = int(epochData.info['sfreq'] * .5)
# csd = signal.convolve(
#     np.pad(
#         csd, 
#         pad_width = (
#             (0, 0), 
#             (0, 0), 
#             (padWidth, padWidth)
#         ),
#         mode = 'reflect'
#     ), 
#     gwin[None, None], 
#     'same'
# )[..., padWidth:-padWidth]
# NCHANS = 16
# channels = np.sort(np.angle(
#     np.exp(np.arange(
#         np.pi/NCHANS,
#         2*np.pi+np.pi/NCHANS,
#         2*np.pi/NCHANS
#     ) * 1j) 
# ))
# bin_width = .5 * np.pi
# # orientation decoding per variance condition
# idx = []
# dist = []
# for cond in bhv_df['thetaVar'].unique():
#     idx_cond, = np.where(bhv_df['thetaVar'] == cond)
#     cv = list(LeaveOneOut().split(csd[idx_cond]))
#     dist_cond = dmah.compute_MahDist(
#         csd[idx_cond], 
#         thetas[idx_cond], 
#         cv, 
#         channels, 
#         bin_width
#     )
#     idx += [idx_cond]
#     dist += [dist_cond]
# # sort trials
# dist = np.concatenate(dist)[np.concatenate(idx).argsort()]
# tun = -(dist - dist.mean(1)[:,None])
# # save tuning data
# tun_info = mne.create_info(
#     ch_names = [str(ch) for ch in channels.round(2)], 
#     sfreq = epochData.info['sfreq'],
# )
# tun_info['description'] = 'Similarity matrices computed using Mahalanobis distance.'
# epochTun = mne.EpochsArray(
#     tun,
#     tun_info,
#     tmin = epochData.times.min(),
#     events = epochData.events, event_id = epochData.event_id,
#     metadata = epochData.metadata
# )
# SAVEPATH = ROOTPATH / '03_Derivatives' / 'tun'
# SAVEPATH.mkdir(exist_ok = True, parents = True)
# epochTun.save(
#     SAVEPATH / f'{subID}_ses-MAIN_lck-STIM_perVar_tun-epo.fif.gz',
#     overwrite = True
# )
#===============================================================================
# %% aggregating orientation tuning
#===============================================================================
tunPaths = sorted(ROOTPATH.rglob('*_ses-MAIN_lck-STIM_perVar_tun-epo.fif.gz'))
# tunPaths = sorted(ROOTPATH.rglob('*_ses-MAIN_lck-STIM_tun-epo.fif.gz'))
dropSUB = ['sub-02']
tunData = dict([
    (
        dpath.stem.split('_')[0],
        mne.read_epochs(dpath, preload = True)
    )
    for dpath in tunPaths
    if dpath.stem.split('_')[0] not in dropSUB
])
# %%aggregation across trials
channels = np.array(tunData['sub-01'].ch_names).astype('float')
times = tunData['sub-01'].times
aggSIM, bhv_df = zip(*[
    (sub.get_data(), sub.metadata)
    for sub in tunData.values()
])
aggSIM = np.concatenate(aggSIM)
bhv_df = pd.concat(bhv_df)
# which side was the test side
bhv_df = bhv_df.merge(
    bhv_df.groupby(['sNo']).apply(
        lambda x: ['right' if x['rightPayoff'].unique().size > 2 else 'left'][0]
    ).reset_index().rename(columns = {0:'tstSide'}),
    on = ['sNo']
)
# what was the payoff of the risky choice
bhv_df['tstPayoff'] = (
    bhv_df['deltaThetaRad'] 
    * 7 # this is a scaling factor to normalize maximum absolute payoff to 11
).round(0)
# code task
bhv_df['task'] = 'risk'
bhv_df.loc[bhv_df['trialType'] == 1, 'task'] = 'perc'
# did they choose the risky option
bhv_df['riskyChoice'] = np.nan
bhv_df.loc[
    (bhv_df['task'] == 'risk')
    & (bhv_df['response'] == bhv_df['tstSide']),
    'riskyChoice'
] = 1
bhv_df.loc[
    (bhv_df['task'] == 'risk')
    & (bhv_df['response'] != bhv_df['tstSide']),
    'riskyChoice'
] = 0
# # did they choose the risky option
# bhv_df['riskyChoice'] = 'no'
# bhv_df.loc[
#     bhv_df['response'] == bhv_df['tstSide'],
#     'riskyChoice'
# ] = 'yes'
bhv_df['payoffSign'] = np.sign(bhv_df['tstPayoff'])
bhv_df['payoffValue'] = np.abs(bhv_df['tstPayoff'])
#%% binning trials relative to angles
thetas = np.angle(np.exp(bhv_df['tstThetaRad'] * 1j))
NCHANS = 6
channels = np.sort(np.angle(
    np.exp(np.arange(
        np.pi/NCHANS,
        2*np.pi+np.pi/NCHANS,
        2*np.pi/NCHANS
    ) * 1j) 
))
theta_bins = pd.cut(thetas, channels)
theta_mids = np.array([np.pi] + list(theta_bins.categories.mid.values)).round(3)
# add bin codes (1 to 6) to the data frame 
bhv_df['theta_bin'] = (theta_bins.codes + 1)
bhv_df['bin_angle'] = theta_mids[bhv_df['theta_bin'].values]
bhv_df['bin_value'] = (np.angle(
    np.exp(
        np.abs(np.angle(
            np.exp(bhv_df['bin_angle'] * 1j) 
            # rotate the bin angles repending on the participant value-mapping
            * np.exp(np.array([0, np.pi])[bhv_df['sNo'] % 2] * 1j)
        )) * 1j
    ) 
    / np.exp(.5 * np.pi * 1j)
) * 7).round(0)
#%% bining tst payoff
payoff_bins = pd.cut(bhv_df['tstPayoff'].values, np.linspace(-11,11,8).round(2))
payoff_mids = payoff_bins.categories.mid.values.round(3)
payoff_codes = payoff_bins.codes.copy()
payoff_codes[payoff_codes == -1] = 0
bhv_df['payoff_bins'] =  payoff_codes
bhv_df['payoff_vals'] = payoff_mids[payoff_codes]# %% concatenate similarity matrices and smooth them
# %% aggregating and smoothing similarity matrix
# aggSIM = np.concatenate(aggSIM)
SD = int(tunData['sub-01'].info['sfreq'] * .016)
gwin = signal.windows.gaussian(
    # the window width is 16 SD (-8 to +8). That should be enough.
    16 * SD,
    SD
)
gwin /= gwin.sum()
padWidth = int(tunData['sub-01'].info['sfreq'] * .5)
aggSIM = signal.convolve(
    np.pad(
        aggSIM, 
        pad_width = (
            (0, 0),
            (0, 0), 
            (padWidth, padWidth)
        ),
        mode = 'reflect'
    ), 
    gwin[None, None], 
    'same'
)[..., padWidth:-padWidth]
#%% time-resolved similarities
avSIM = []
for sub in bhv_df['sNo'].unique():
    idx = (bhv_df['sNo'] == sub)
    avSIM += [[sub, aggSIM[idx].mean(0)]]
subID, avSIM = zip(*avSIM)
avSIM = np.array(avSIM)
t_val_sim, p_val_sim = sps.ttest_1samp(
    avSIM, 0
)
p_adj_sim = multitest.fdrcorrection(p_val_sim.flatten())[1].reshape(*p_val_sim.shape)
cntr_sim = measure.find_contours(
    -np.log(p_adj_sim), -np.log(.05)
)
gavSIM = np.mean(avSIM, 0)
# %% effect of variance
avVAR = []
for var, sub in itertools.product(bhv_df['tstVarRad'].unique(), bhv_df['sNo'].unique()):
    idx = (bhv_df['tstVarRad'] == var) & (bhv_df['sNo'] == sub)
    avVAR += [[var, sub, aggSIM[idx].mean(0)]]
varID, subID, avVAR = zip(*avVAR)
avVAR = np.array(avVAR)
t_val_var, p_val_var = sps.ttest_rel(
    *np.stack([
        np.array(avVAR)[np.array(varID) == var] 
        for var in np.unique(varID)
    ])
)
p_adj_var = multitest.fdrcorrection(p_val_var.flatten())[1].reshape(*p_val_var.shape)
cntr_var = measure.find_contours(
    -np.log(p_adj_var), -np.log(.05)
)
gavVAR = np.array([np.array(avVAR)[
    np.array(varID) == var].mean(0) for var in  np.unique(varID)
])
# %%
channels = np.array(tunData['sub-01'].ch_names).astype('float')
avTUN = (
    np.stack([
        np.array(avVAR)[np.array(varID) == var] 
        for var in np.unique(varID)
    ])
    * np.cos(channels)[None,None,:, None]
).sum(-2)
# aggTheta = (aggCond * np.exp(channels * 1j)[None, None, :, None]).sum(-2)
# aggTun = np.abs(aggTheta) * np.cos(np.angle(aggTheta))
t_val_tun, p_val_tun = sps.ttest_rel(*avTUN, axis = 0)
p_adj_tun = multitest.fdrcorrection(p_val_tun)[1]
avTUN_z = avTUN - avTUN.mean(0)[None] + avTUN.mean(0).mean(0)[None,None]
seTUN = np.sqrt((np.var(avTUN_z, axis = 1, ddof = 1) * 2) / avTUN_z.shape[1])
gavTUN = avTUN.mean(1)
# %% decoding motion
channels = np.array(tunData['sub-01'].ch_names).astype('float')
aggTHT = (aggSIM * np.exp(channels * 1j)[None,:,None]).sum(-2)
aggERR = np.angle(aggTHT)
aggTUN = np.abs(aggTHT) * np.cos(aggERR)
decodedAngle = np.angle(
    np.exp(bhv_df['tstThetaRad'].values[:,None] * 1j) 
    * np.exp(aggERR * 1j)
)
aggVAL =  (np.angle(
    np.exp(
        np.abs(np.angle(
            np.exp(decodedAngle * 1j) 
            # rotate the bin angles repending on the participant value-mapping
            * np.exp(np.array([0, np.pi])[bhv_df['sNo'] % 2] * 1j)[:,None]
        )) * 1j
    ) 
    / np.exp(.5 * np.pi * 1j)
) * 7).round(0)
# %% relating behaviour and neural representation of value
# these are indices of time-samples where decoded payoff was correlated with 
# experimental conditions
idx_crit = [
    162, 163, 164, 165, 166, 167, 168, 169, 
    170, 173, 176, 177, 178,179
]
bhv_df['eegPayoff'] = aggVAL[..., idx_crit].mean(-1)
bhv_df['eegPayoff_sgn'] = np.sign(bhv_df['eegPayoff']).astype('str')
bhv_df['eegPayoff_val'] = np.abs(bhv_df['eegPayoff'])

bhv_df['tstPayoff_sgn'] = np.sign(bhv_df['tstPayoff']).astype('str')
bhv_df['tstPayoff_val'] = np.abs(bhv_df['tstPayoff'])
# %%
# mdl_med = Lmer(
#     'eegPayoff ~ tstPayoff + (tstPayoff | sNo)',
#     data = bhv_df, family = 'gaussian'
# )
# mdl_med.fit()
# bhv_df['eegPayoff_res'] = mdl_med.residuals

bhv_risk = bhv_df.loc[bhv_df['task'] == 'risk'].copy()
bhv_risk['thetaVar_fac'] = bhv_risk['thetaVar'].astype('str')

# mdl_tst = Lmer(
#     "riskyChoice ~ tstPayoff * thetaVar_fac + (1 | sNo)",
#     data = bhv_risk, 
#     family = 'binomial'
# )
# mdl_tst.fit()

mdl_eeg = Lmer(
    "riskyChoice ~ eegPayoff * thetaVar_fac + (eegPayoff | sNo)",
    data = bhv_risk, 
    family = 'binomial'
)
mdl_eeg.fit()

# mdl_eeg_res = Lmer(
#     "riskyChoice ~ eegPayoff_res * thetaVar_fac + ((eegPayoff_res * thetaVar_fac) | sNo)",
#     data = bhv_risk, 
#     family = 'binomial'
# )
# mdl_eeg_res.fit()

# %%
bhv_df['decodedVal'] = aggVAL[:,idx_crit].mean(-1)
av_val = bhv_df.groupby([
    'sNo','thetaVar','bin_value'
])['decodedVal'].mean().reset_index().merge(
     bhv_df.groupby([
        'sNo'
    ])['decodedVal'].mean().reset_index().rename(columns = dict(
        decodedVal = 'decodedVal_sub'
    )),
    on = ['sNo']
)
av_val['decodedVal_z'] = (
    av_val['decodedVal'] 
    - av_val['decodedVal_sub'] 
    + av_val['decodedVal'].mean()
) 
gav_val = av_val.groupby([
    'thetaVar','bin_value'
])[['decodedVal','decodedVal_z']].mean().reset_index()
gav_val['sem_z'] = av_val.groupby(['thetaVar','bin_value'])['decodedVal_z'].apply(
    lambda x: np.sqrt(np.var(x, ddof = 1) * (8 / 7) / len(x))
).reset_index()['decodedVal_z']

# %% model effects of payoff on decoding precision
aggFID = -np.log10(np.abs(aggERR)/np.pi)
aggERR_p = np.abs(aggERR)/np.pi
mdf = []
for idx_tsmpl, val_tsmpl in enumerate(aggERR_p.T):
    print(f'time sample {idx_tsmpl}')
    tmp_df = bhv_df.copy()
    tmp_df['thetaVar_fac'] = tmp_df['thetaVar'].astype('str')
    tmp_df['payoffSign_fac'] = np.sign(bhv_df['bin_value']).astype('category')
    tmp_df['payoffMagn_fac'] = np.abs(bhv_df['bin_value']).astype('category')
    tmp_df['erp'] = val_tsmpl
    mdf += [smf.mixedlm(
        "erp ~ thetaVar_fac * payoffSign_fac * payoffMagn_fac", 
        data = tmp_df, 
        groups = tmp_df['sNo']
    ).fit()]
# %% 
coef, sem = zip(*np.array([mdl.summary().tables[1].loc[
    'Intercept', 
    ['Coef.', 'Std.Err.']
].values.astype('float') for mdl in mdf]))
coef = np.array(coef); sem = np.array(sem)
p_vals = np.stack([mdl.pvalues[:8] for mdl in mdf])
t_vals = np.array([mdl.tvalues[:8] for mdl in mdf])
p_adj = multitest.fdrcorrection(p_vals.flatten())[1].reshape(*p_vals.shape)
t_vals[p_adj >= .05] = 0
# %% compute coefficients
idx_crit = np.where(p_adj[:,-1] < .05)
agg_mdl_par, agg_mdl_cov = np.stack([
    mdl.summary().tables[1].values[:-1].astype('float')[:,:2] 
    for mdl in np.array(mdf)[idx_crit]
]).T
av_mdl_par = agg_mdl_par.mean(1)
agg_mdl_par[p_adj[idx_crit].T >= .05] = 0
mdl_par = agg_mdl_par.mean(1)
mdl_cov = np.stack([
    mdl.cov_params().values[:-1,:-1] 
    for mdl in np.array(mdf)[idx_crit]
]).mean(0)
idx_par = np.array([
    [1, 0, 0, 1, 0, 0, 0, 0], # LowLossLarge
    [1, 0, 0, 0, 0, 0, 0, 0], # LowLossSmall

    [1, 0, 1, 0, 0, 0, 0, 0], # LowGainSmall
    [1, 0, 1, 1, 0, 0, 1, 0], # LowGainLarge

    [1, 1, 0, 1, 0, 1, 0, 0], # HighLossLarge
    [1, 1, 0, 0, 0, 0, 0, 0], # HighLossSmall

    [1, 1, 1, 0, 1, 0, 0, 0], # HighGainSmall
    [1, 1, 1, 1, 1, 1, 1, 1]  # HighGainLarge
]).astype('bool')
pred_cov = np.stack(np.split(np.array([
    np.sqrt(av_mdl_par[idx][None] @ mdl_cov[idx,:][:, idx] @ av_mdl_par[idx][:, None]).flatten() 
    for idx in idx_par
]), 2))
pred_coef = np.stack(np.split(np.array([mdl_par[idx].sum() for idx in idx_par]), 2))
# # %%
# plt.figure()
# plt.imshow(t_vals.T, aspect = 'auto', cmap = 'viridis')
# plt.gca().set_yticks(range(mdf[0].tvalues.keys()[:8].size))
# # ylbls = [
# #     'Low / Loss / Small', 
# #     'High Noise', 
# #     'Gain', 
# #     'Large Value',
# #     'High / Gain',
# #     'High / Large',
# #     'Gain / Large',
# #     'High / Gain / Large'
# # ]
# ylbls = [
#     'Const.', 
#     'Noise',
#     'Sign',
#     'Magn.', 
#     r'N$\times$S',
#     r'N$\times$M',
#     r'S$\times$M',
#     r'N$\times$S$\times$M'
# ]
# plt.gca().set_yticklabels(mdf[0].tvalues.keys()[:8])
# plt.gca().set_yticklabels(ylbls)
# %% Plotting tuning analyses
sbn.set_style('ticks')
sbn.set_palette('husl', 75, desat = .60)
fsmall, fmed, flarge = [8, 10, 11]
fsmall, fmed, flarge = [10, 11, 11]
plt.rc('font', size=fmed)          # controls default text sizes
plt.rc('font', family='Calibri')
plt.rc('axes', facecolor=(0, 0, 0, 0))
plt.rc('axes', titlesize=fsmall)     # fontsize of the axes title
plt.rc('axes', labelsize=fsmall)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=fsmall)    # fontsize of the tick labels
plt.rc('ytick', labelsize=fsmall)    # fontsize of the tick labels
plt.rc('legend', fontsize=fsmall)    # legend fontsize
plt.rc('figure', titlesize=fmed)

# %% plot similarities

fig = plt.figure(figsize = (6.6, 6.6))

axSIM = plt.subplot2grid(
    (30,32), 
    (0, 0),
    rowspan = 10,
    colspan = 30,
    fig = fig
)

im = axSIM.imshow(
    gavSIM, aspect = 'auto', cmap = 'viridis',
    vmin = -.004, vmax = .004,
    origin = 'lower'
)
for ln in cntr_sim:
    axSIM.plot(
        ln[:, 1], ln[:, 0],
        lw = 1, color = 'white'
    )

axSIM.tick_params(direction = 'in', length = 3)
axSIM.text(290,13, r'$\times10^{-3}$', fontsize = 8)
axSIM.set_yticks([0,7.5,15])
axSIM.set_yticklabels([r"-$\pi$",r"$ 0$",r"$\pi$"])
axSIM.set_ylabel('Bin distance')
times = np.linspace(0,1,6)
axSIM.set_xticks(tunData[f'sub-{sub}'].time_as_index(times))
axSIM.set_xticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0 s'])
axSIM.xaxis.set_visible(False)

axCBar = plt.subplot2grid(
    (30,32), 
    (2, 31),
    rowspan = 6,
    colspan = 1,
    fig = fig
)
plt.colorbar(mappable = im, orientation = 'vertical', cax = axCBar, shrink = .05)
axCBar.set_yticks([-.004,0,.004])
axCBar.set_yticklabels(['-4',' 0 a.u.',' 4'])
axCBar.tick_params(direction = 'inout', length = 5)

# plot LME t-values

axMLE = plt.subplot2grid(
    (30,32), 
    (10, 0),
    rowspan = 10,
    colspan = 30,
    fig = fig
)
im = axMLE.imshow(t_vals.T, aspect = 'auto', cmap = 'coolwarm', vmin = -5, vmax = 5)
axMLE.set_yticks(range(mdf[0].tvalues.keys()[:8].size))
# ylbls = [
#     'Lw_Ls_Sm', 
#     'Hi', 
#     'Gn', 
#     'Lg',
#     'Hi_Gn',
#     'Hi_Lg',
#     'Gn_Lg',
#     'Hi_Gn_Lg'
# ]
ylbls = [
    'Const.', 
    'Noise',
    'Sign',
    'Magn.', 
    r'N$\times$S',
    r'N$\times$M',
    r'S$\times$M',
    r'N$\times$S$\times$M'
]
axMLE.set_yticklabels(ylbls)
axMLE.set_xticks(tunData[f'sub-{sub}'].time_as_index(np.linspace(0,1,6)))
axMLE.set_xticklabels([
    f"{val_lbl:.1f}"
    if idx_lbl < 4
    else f"{val_lbl:.1f}" + " s"
    for idx_lbl, val_lbl in enumerate(np.linspace(0,1,6))
])
axMLE.tick_params(direction = 'in', length = 3)
axCBar = plt.subplot2grid(
    (30,32), 
    (12, 31),
    rowspan = 6,
    colspan = 1,
    fig = fig
)
plt.colorbar(mappable = im, orientation = 'vertical', cax = axCBar, shrink = .05)
axCBar.set_yticks([-5,0,5])
axCBar.set_yticklabels(['-5', ' 0 $t$', ' 5'])
axCBar.tick_params(direction = 'inout', length = 5)

# # plot coefs
# axCoef = plt.subplot2grid(
#     (30,32), 
#     (21, 0),
#     rowspan = 9,
#     colspan = 16,
#     fig = fig
# )
# cols = np.array(sbn.color_palette('muted'))[[2,4]]
# vals = np.array([-11,-4,4,11])
# for idx_noise, val_noise in enumerate(pred_coef):
#     axCoef.fill_between(
#         vals + [-1,1][idx_noise], 
#         val_noise - pred_cov[idx_noise].flatten(),
#         val_noise + pred_cov[idx_noise].flatten(),
#         color = cols[idx_noise], alpha = .3
#     )
#     axCoef.plot(
#         vals + [-1,1][idx_noise],
#         val_noise,
#         marker = ['v', '^'][idx_noise],
#         color = cols[idx_noise],
#         label = ['Low Noise', 'High'][idx_noise]
#     )

# axCoef.legend(frameon= False, loc = 'upper left')
# axCoef.set_xticks(vals)
# axCoef.set_xticklabels([f'{val}' for val in vals])
# axCoef.set_xlabel('Bin Value')
# axCoef.tick_params(direction = 'in', length = 3)
# axCoef.spines['bottom'].set_bounds(-11,11)
# axCoef.set_ylim(.46,.52)
# axCoef.set_ylabel('Decoding error')
# axCoef.set_yticks([.46, .48, .50, .52])
# axCoef.spines['left'].set_bounds(.46,.52)
# for spine in ['top', 'right']:
#     axCoef.spines[spine].set_visible(False)

# plot decoded value
# axVAL = plt.subplot2grid(
#     (30,32), 
#     (21, 16),
#     rowspan = 9,
#     colspan = 16,
#     fig = fig
# )

axVAL = plt.subplot2grid(
    (30,32), 
    (21, 0),
    rowspan = 9,
    colspan = 16,
    fig = fig
)

cols = np.array(sbn.color_palette('muted'))[[2,4]]
vals = np.array([-11,-4,4,11])
for idx_noise, val_noise in enumerate(np.split(gav_val['decodedVal'],2)):
    axVAL.fill_between(
        vals + [-1,1][idx_noise], 
        val_noise - np.split(gav_val['sem_z'],2)[idx_noise],
        val_noise + np.split(gav_val['sem_z'],2)[idx_noise],
        color = cols[idx_noise], alpha = .3
    )
    axVAL.plot(
        vals + [-1,1][idx_noise],
        val_noise,
        marker = ['v', '^'][idx_noise],
        color = cols[idx_noise],
        label = ['Low Noise', 'High'][idx_noise]
    )

axVAL.set_xticks(vals)
axVAL.set_xticklabels([f'{val}' for val in vals])
axVAL.set_xlabel('Bin Value')
axVAL.tick_params(direction = 'in', length = 3)
axVAL.spines['bottom'].set_bounds(-11,11)
# axVAL.yaxis.tick_right()
# axVAL.yaxis.set_label_position('right')
axVAL.set_ylabel('Decoded value')
axVAL.set_yticks([-.5, 0, .5])
axVAL.set_yticklabels(['-0.5', '0.0', ' 0.5'])
axVAL.spines['left'].set_bounds(-.5,.5)
for spine in ['right', 'top']:
    axVAL.spines[spine].set_visible(False)

# plot LOG
axLog = plt.subplot2grid(
    (30,32), 
    (21, 16),
    rowspan = 9,
    colspan = 16,
    fig = fig
)

for idx_tht, val_tht in enumerate(bhv_risk['thetaVar_fac'].unique()):
    idx = bhv_risk['thetaVar_fac'] == val_tht
    axLog.plot(
        bhv_risk.loc[idx,'eegPayoff'],
        np.array(mdl_eeg.fits)[idx],
        '.', mec = cols[::-1][idx_tht], mfc = 'none', alpha = .3
    )
axLog.set_ylim(.3, .7)
axLog.set_yticks(np.linspace(.3,.7,5))
axLog.spines['right'].set_bounds(.3,.7)
axLog.set_ylabel('p(Risky choice)')
axLog.yaxis.tick_right()
axLog.yaxis.set_label_position('right')

axLog.set_xticks(vals)
axLog.set_xticklabels([f'{val}' for val in vals])
axLog.spines['bottom'].set_bounds(-11, 11)
axLog.set_xlabel('Decoded value')

for spine in ['left','top']:
    axLog.spines[spine].set_visible(False)
axLog.tick_params(direction = 'in', length = 3)

# %%
fig.subplots_adjust(.12,.06,.92,.98, hspace = 12)
fig.savefig(ROOTPATH / '05_Exports' / 'gavFID_v03.png', dpi = 600)

# %% Plotting tunning analyses
sbn.set_style('ticks')
sbn.set_palette('husl', 75, desat = .60)
fsmall, fmed, flarge = [8, 10, 11]
fsmall, fmed, flarge = [10, 11, 11]
plt.rc('font', size=fmed)          # controls default text sizes
plt.rc('font', family='Calibri')
plt.rc('axes', facecolor=(0, 0, 0, 0))
plt.rc('axes', titlesize=fsmall)     # fontsize of the axes title
plt.rc('axes', labelsize=fsmall)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=fsmall)    # fontsize of the tick labels
plt.rc('ytick', labelsize=fsmall)    # fontsize of the tick labels
plt.rc('legend', fontsize=fsmall)    # legend fontsize
plt.rc('figure', titlesize=fmed)

fig = plt.figure(figsize = (6.6, 4.4))

axSIM = fig.add_subplot(211)
im = axSIM.imshow(
    gavSIM, aspect = 'auto', cmap = 'viridis',
    vmin = -.004, vmax = .004,
    origin = 'lower'
)
for ln in cntr_sim:
    axSIM.plot(
        ln[:, 1], ln[:, 0],
        lw = 1, color = 'white'
    )
cbar = plt.colorbar(
    im, location = 'right',
    fraction = .05
)
cbar.set_ticks([-.004,-.002,0,.002,.004])
cbar.set_ticklabels([-4,-2,0,2,4])
cbar.set_label('Similarity [a.u.]')
cbar.ax.tick_params(direction = 'inout', length = 5)
axSIM.tick_params(direction = 'inout', length = 5)
axSIM.text(290,16, r'$\times10^{-3}$', fontsize = 8)
axSIM.set_yticks([0,7.5,15])
axSIM.set_yticklabels([r"-$\pi$",r"$ 0$",r"$\pi$"])
axSIM.set_ylabel('Bin distance')
times = np.linspace(0,1,5)
axSIM.set_xticks(gavSim.time_as_index(times))
axSIM.set_xticklabels(['0 ms', '250', '500', '750', '1,000'])
axSIM.xaxis.set_visible(False)
# for spine in ['top','right', 'bottom']:
#     axSIM.spines[spine].set_visible(False)
# axSIM.spines['left'].set_bounds(0, 15)

axVAR = fig.add_subplot(212)
im = axVAR.imshow(
    gavVAR[1] - gavVAR[0], aspect = 'auto', cmap = 'coolwarm',
    vmin = -.004, vmax = .004,
    origin = 'lower'
)
for ln in cntr_var:
    axVAR.plot(
        ln[:, 1], ln[:, 0],
        lw = 1, color = 'white'
    )
cbar_var = plt.colorbar(
    im, location = 'right',
    fraction = .05
)
cbar_var.set_ticks([-.004,-.002,0,.002,.004])
cbar_var.set_ticklabels([-4,-2,0,2,4])
cbar_var.set_label('High - Low [a.u.]')
cbar_var.ax.tick_params(direction = 'inout', length = 5)
axVAR.tick_params(direction = 'inout', length = 5)
axVAR.text(290,16, r'$\times10^{-3}$', fontsize = 8)
axVAR.set_yticks([0,7.5,15])
axVAR.set_yticklabels([r"-$\pi$",r"$ 0$",r"$\pi$"])
axVAR.set_ylabel('Bin distance')
times = np.linspace(0,1,5)
axVAR.set_xticks(tunData['sub-01'].time_as_index(times))
axVAR.set_xticklabels(['0 ms', '250', '500', '750', '1,000'])

# axTUN = fig.add_subplot(212)
# for cond in range(2):
#     col = sbn.color_palette('Set2').as_hex()[cond + 1]
#     axTUN.fill_between(
#         gavSim.times, 
#         gavTun[cond] - seTun[cond], 
#         gavTun[cond] + seTun[cond],
#         color = col, alpha = .3
#     )
#     axTUN.plot(
#         gavSim.times, gavTun[cond], 
#         color = col,
#         label = ['Low noise', 'High'][cond]
#     )
# idx_sig = p_adj_tun < 0.05
# axTUN.plot(
#     gavSim.times[idx_sig], 
#     [0] * idx_sig.sum(),
#     '*', color = 'darkgrey', ms = 3
# )
# axTUN.set_xticks(np.linspace(0, 1, 5))
# axTUN.set_yticks([0, .01, .02])
# axTUN.set_yticklabels([' 0', ' 1', ' 2'])
# axTUN.spines['left'].set_bounds(0,.02)
# axTUN.tick_params(direction = 'inout', length = 5)
# axTUN.set_ylabel('Orientation tuning [a.u.]')
# for spine in ['top','right']:
#     axTUN.spines[spine].set_visible(False)
# axTUN.set_xticklabels(['0 ms', '250', '500', '750', '1,000'])
# axTUN.spines['bottom'].set_bounds(0,1)
# axTUN.legend(frameon = False, loc = 'upper left')
# axSIM.set_xlim(gavSim.time_as_index(axTUN.get_xlim()))

fig.tight_layout()
fig.savefig(ROOTPATH / '05_Exports' / 'gavTUN_map_smooth.png', dpi = 600)
#===============================================================================
# %% correlations with value and choice
#===============================================================================
# average over 300 ms interval during which the difference was significant
idx_min, idx_max = tunData['sub-01'].time_as_index([.35,.65])
ssSIM = aggSIM[..., idx_min:idx_max].mean(-1)
# %% aggregate similarity matrices
thetas = np.angle(np.exp(bhv_df['tstThetaRad'] * 1j))
# compute neural estimate of the average angle
aggTHETA = np.angle((
    ssSIM 
    * (
        # rotate the channels so that the zero bin is over the actual angle
        np.exp(channels * 1j)[None] 
        * np.exp(thetas[:,None] * 1j)
    )
).sum(-1))
# compute neural neural error of estimation
aggERR = np.angle(np.exp(thetas * 1j) / np.exp(aggTHETA * 1j))
avCSTD = np.array([
    [
        mm.cstd(aggERR[(bhv_df['tstVarRad'] == var) & (bhv_df['sNo'] == sub)])
        for sub in bhv_df['sNo'].unique()
    ]
    for var in bhv_df['tstVarRad'].unique()
])
cmeanERR = []
for var, sub in itertools.product(bhv_df['tstVarRad'].unique(), bhv_df['sNo'].unique()):
    idx = (bhv_df['tstVarRad'] == var) & (bhv_df['sNo'] == sub)
    tsmplERR = []
    for tsmpl in aggERR[idx].T:
        # tsmplERR += [np.cos(mm.cmean(tsmpl))]
        tsmplERR += [mm.sd2k(mm.cstd(tsmpl))]
        # tsmplERR += [mm.sd2k(mm.cstd(tsmpl)) * np.cos(mm.cmean(tsmpl))]
    cmeanERR += [[var, sub, tsmplERR]]
varID, subID, tsmplERR = zip(*cmeanERR)
t_val, p_val = sps.ttest_rel(
    *np.stack([
        np.array(tsmplERR)[np.array(varID) == var] 
        for var in np.unique(varID)
    ])
)
p_adj = multitest.fdrcorrection(p_val)[1]
gavERR = [np.array(tsmplERR)[
    np.array(varID) == var].mean(0) for var in  np.unique(varID)
]
plt.figure()
plt.plot(np.array(gavERR).T)
# %%
erpTUN = (erpSIM * np.cos(channels)[None,:,None]).sum(1)
erpFEA = (
    erpSIM[:, np.abs(channels) == .2].mean(1) 
    - erpSIM[:, np.abs(channels) == 2.95].mean(1)
)
# %% select just the value task
idx_keep = (bhv_df['payoffSign'] != 0) & (~bhv_df['tstPayoff'].isna()) 
bhv_df = bhv_df.loc[idx_keep]
bhv_df['payoffSign_fac'] = bhv_df['payoffSign'].astype('str')
bhv_df['thetaVar_fac'] = bhv_df['thetaVar'].astype('str')

mdf = []
for tsmpl in erpTUN.T:
    tmp_df = bhv_df.copy()
    tmp_df['erp'] = tsmpl
    mdf += [smf.mixedlm(
        # "erp ~ thetaVar_fac * riskyChoice * payoffSign_fac * payoffValue", 
        "erp ~ riskyChoice", 
        data = tmp_df, 
        groups = tmp_df['sNo']
    ).fit()]
# %% 
p_vals = np.stack([mdl.pvalues[1:-1] for mdl in mdf])
t_vals = np.array([mdl.tvalues[1:-1] for mdl in mdf])
p_adj = multitest.fdrcorrection(p_vals.flatten())[1].reshape(*p_vals.shape)
t_vals[p_adj >= .05] = 0
# %%
plt.figure()
plt.imshow(t_vals.T, aspect = 'auto', cmap = 'viridis')
plt.gca().set_yticks(range(mdf[0].tvalues.keys()[1:-1].size))
plt.gca().set_yticklabels(mdf[0].tvalues.keys()[1:-1])
# NOTE: there is no effect of any of the predictors on the tuning strength, 
#       no matter how evaluated (vector length, peak to through, peak)
# #===============================================================================
# # %% RSA analyses
# # TODO: check how payoff covaries with angle: this should be non-linear function
# #===============================================================================
# epochPaths = sorted(ROOTPATH.rglob("*ses-MAIN_lck-STIM_csd-epo.fif.gz"))
# dropSUB = ['sub-02']
# epochData = dict([(
#     dpath.stem.split('_')[0], 
#     mne.read_epochs(dpath, preload = True)
# ) for dpath in epochPaths
# if dpath.stem.split('_')[0] not in dropSUB])
# csd, bhv_df = zip(*[
#     (sub.pick('csd').get_data(), sub.metadata)
#     for sub in epochData.values()
# ])
# #%% preparing behavioural data
# bhv_df = pd.concat(bhv_df)
# bhv_df = bhv_df.merge(
#     # which side was test side
#     bhv_df.groupby(['sNo']).apply(
#         lambda x: ['right' if x['rightPayoff'].unique().size > 2 else 'left'][0]
#     ).reset_index().rename(columns = {0:'tstSide'}),
#     on = ['sNo']
# )
# bhv_df['tstPayoff'] = (
#     bhv_df['deltaThetaRad'] 
#     * 7 # this is a scaling factor to normalize maximum absolute payoff to 11
# ).round(0)
# bhv_df['riskyChoice'] = 'no'
# bhv_df.loc[
#     bhv_df['response'] == bhv_df['tstSide'],
#     'riskyChoice'
# ] = 'yes'
# bhv_df['payoffSign'] = np.sign(bhv_df['tstPayoff'])
# bhv_df['payoffValue'] = np.abs(bhv_df['tstPayoff'])
# # binning trials relative to angles
# thetas = np.angle(np.exp(bhv_df['tstThetaRad'] * 1j))
# NCHANS = 6
# channels = np.sort(np.angle(
#     np.exp(np.arange(
#         np.pi/NCHANS,
#         2*np.pi+np.pi/NCHANS,
#         2*np.pi/NCHANS
#     ) * 1j) 
# ))
# bins = pd.cut(thetas, channels)
# # NOTE: the values that do not fit into bins are coded as -1
# bin_codes = bins.codes
# bin_mids = np.array([np.pi] + list(bins.categories.mid.values)).round(3)
# bin_vals = np.array([
#     bhv_df.loc[bin_codes == bin, 'tstPayoff'].mean() 
#     for bin in np.unique(bin_codes)
# ])
# #%% preparing EEG data
# csd = np.concatenate(csd)
# # smoothing eeg data
# sfreq = epochData['sub-01'].info['sfreq']
# SD = int(sfreq * .016)
# gwin = signal.windows.gaussian(
#     # the window width is 16 SD (-8 to +8). That should be enough.
#     16 * SD,
#     SD
# )
# gwin /= gwin.sum()
# padWidth = int(sfreq * .5)
# csd = signal.convolve(
#     np.pad(
#         csd, 
#         pad_width = (
#             (0, 0), 
#             (0, 0), 
#             (padWidth, padWidth)
#         ),
#         mode = 'reflect'
#     ), 
#     gwin[None, None], 
#     'same'
# )[..., padWidth:-padWidth]
# # %% computing distance matrices per condition
# out = []
# bin_vals = []
# for var, sub in itertools.product(
#     bhv_df['tstVarRad'].unique(),
#     bhv_df['sNo'].unique()
# ):
#     idx = (bhv_df['tstVarRad'] == var) & (bhv_df['sNo'] == sub)
#     ss_csd = csd[idx]
#     ss_bhv = bhv_df.loc[idx]
#     ss_bin_codes = bin_codes[idx]
#     bin_vals += [np.array([
#         ss_bhv.loc[ss_bin_codes == bin, 'tstPayoff'].mean() 
#         for bin in np.unique(ss_bin_codes)
#     ])]
#     erpPerBin = np.array([
#         ss_csd[ss_bin_codes == bin].mean(0) 
#         for bin in np.unique(ss_bin_codes)
#     ])
#     covERP = np.array([dmah.regcov(tsmpl.T) for tsmpl in ss_csd.T])
#     out += [[
#         var, sub, np.stack([
#             dmah.cdist(
#                 val_tsmpl.T,
#                 val_tsmpl.T,
#                 'mahalanobis',
#                 VI = np.linalg.pinv(covERP[idx_tsmpl])
#             ) for idx_tsmpl, val_tsmpl in enumerate(erpPerBin.T)
#         ], -1)
#     ]]
# varID, subID, mahDIST = zip(*out)
# bin_vals = np.array(bin_vals).shape
# bin_mid_dist = dmah.cdist(
#     bin_mids[:, None], 
#     bin_mids[:, None], 
#     lambda x, y: 
#         np.abs(np.angle(np.exp(x * 1j) / np.exp(y * 1j)))
# ).round(3)
# bin_val_dist = np.array([
#     dmah.cdist(val[:,None], val[:,None])
#     for val in bin_vals
# ]).round(3)
# # %%
# idx_x, idx_y = np.triu_indices(6, k = 1)
# csd_dist = []
# val_dist = []
# bin_dist = []
# sub = []
# var = []
# for idx, val in enumerate(mahDIST):
#     csd_dist += [val[idx_x, idx_y]]
#     val_dist += [bin_val_dist[idx, idx_x, idx_y]]
#     bin_dist += [bin_mid_dist[idx_x, idx_y]]
#     sub += [[subID[idx]] * idx_x.size]
#     var += [[varID[idx]] * idx_x.size]
# csd_dist = np.concatenate(csd_dist)
# val_dist = np.concatenate(val_dist)
# bin_dist = np.concatenate(bin_dist)
# sub = np.concatenate(sub)
# var = np.concatenate(var)
# # %%
# mdf = []
# for tsmpl in csd_dist.T:
#     tmp_df = pd.DataFrame(
#         dict(
#             sub = sub.astype('str'),
#             var = var.astype('str'),
#             bin_dist = bin_dist,
#             val_dist = val_dist,
#             csd = tsmpl
#         )
#     )
#     mdf += [smf.mixedlm(
#         "csd ~ var * bin_dist * val_dist", 
#         data = tmp_df, 
#         groups = tmp_df['sub']
#     ).fit()]
# # %% 
# p_vals = np.stack([mdl.pvalues[1:-1] for mdl in mdf])
# t_vals = np.array([mdl.tvalues[1:-1] for mdl in mdf])
# p_adj = multitest.fdrcorrection(p_vals.flatten())[1].reshape(*p_vals.shape)
# t_vals[p_adj >= .05] = 0
# # %%
# for idx_tsmpl, val_tsmpl in enumerate(csd_dist.T):
#     tmp_df = pd.DataFrame(
#         dict(
#             sub = sub.astype('str'),
#             var = var.astype('str'),
#             bin_dist = bin_dist,
#             val_dist = val_dist,
#             csd = val_tsmpl
#         )
#     )
#     if idx_tsmpl == 0:
#         cor_bin = tmp_df.groupby(['sub', 'var']).apply(
#             lambda x: sps.spearmanr(x['csd'], x['bin_dist'])[0]
#         ).reset_index().rename(columns = {0:f'rho_{idx_tsmpl}'})
#         cor_val = tmp_df.groupby(['sub', 'var']).apply(
#             lambda x: sps.spearmanr(x['csd'], x['val_dist'])[0]
#         ).reset_index().rename(columns = {0:f'rho_{idx_tsmpl}'})
#     else:
#         cor_bin = cor_bin.merge(tmp_df.groupby(['sub', 'var']).apply(
#             lambda x: sps.spearmanr(x['csd'], x['bin_dist'])[0]
#         ).reset_index().rename(columns = {0:f'rho_{idx_tsmpl}'}),
#         on = ['sub', 'var'])
#         cor_val = cor_bin.merge(tmp_df.groupby(['sub', 'var']).apply(
#             lambda x: sps.spearmanr(x['csd'], x['val_dist'])[0]
#         ).reset_index().rename(columns = {0:f'rho_{idx_tsmpl}'}),
#         on = ['sub', 'var'])
# # %% 
# p_vals = np.stack([mdl.pvalues[1:-1] for mdl in mdf])
# t_vals = np.array([mdl.tvalues[1:-1] for mdl in mdf])
# p_adj = multitest.fdrcorrection(p_vals.flatten())[1].reshape(*p_vals.shape)
# t_vals[p_adj >= .05] = 0
# # %%
# plt.figure()
# plt.imshow(t_vals.T, aspect = 'auto', cmap = 'viridis')
# plt.gca().set_yticks(range(mdf[0].tvalues.keys()[1:-1].size))
# plt.gca().set_yticklabels(mdf[0].tvalues.keys()[1:-1])

# # %%
# adist = np.array([
#     [bin_mids[a], bin_mids[b], np.angle(np.exp(bin_mids[a] * 1j) / np.exp(bin_mids[b] * 1j))]
#     for a, b in contrasts_idx
# ]).round(2)
# mdist = sps.zscore(np.concatenate([
#     dmah.cdist(
#         csd[bin_codes == a, :, 0].mean(0)[None],
#         csd[bin_codes == b, :, 0].mean(0)[None],
#         'mahalanobis',
#         VI = np.linalg.pinv(covERP)
#     )[0]
#     for a, b in contrasts
# ])).round(2)[:, None]
# np.concatenate([adist, mdist], axis = 1)
# adist.shape

# for cond in bhv_df['thetaVar'].unique():
#     idx_cond, = np.where(bhv_df['thetaVar'] == cond)
#     cv = list(LeaveOneOut().split(csd[idx_cond]))
#     dist_cond = dmah.compute_MahDist(
#         csd[idx_cond], 
#         thetas[idx_cond], 
#         cv, 
#         channels, 
#         bin_width
#     )
#     idx += [idx_cond]
#     dist += [dist_cond]
# # sort trials
# dist = np.concatenate(dist)[np.concatenate(idx).argsort()]
# tun = -(dist - dist.mean(1)[:,None])
# # save tuning data
# tun_info = mne.create_info(
#     ch_names = [str(ch) for ch in channels.round(2)], 
#     sfreq = epochData.info['sfreq'],
# )
# tun_info['description'] = 'Similarity matrices computed using Mahalanobis distance.'
# epochTun = mne.EpochsArray(
#     tun,
#     tun_info,
#     tmin = epochData.times.min(),
#     events = epochData.events, event_id = epochData.event_id,
#     metadata = epochData.metadata
# )
# SAVEPATH = ROOTPATH / '03_Derivatives' / 'tun'
# SAVEPATH.mkdir(exist_ok = True, parents = True)
# epochTun.save(
#     SAVEPATH / f'{subID}_ses-MAIN_lck-STIM_perVar_tun-epo.fif.gz',
#     overwrite = True
# )
# #===============================================================================
# # %% MVPA - SVM
# #===============================================================================
# SAVEPATH = ROOTPATH / '03_Derivatives' / 'svm'
# SAVEPATH.mkdir(exist_ok = True, parents = True)
# IDX_JOB = int(sys.argv[1])

# sNo = bhv_df['sNo'].unique()[IDX_JOB]
# # sNo = 1clearclear
# subID = f"sub-{str(sNo).rjust(2, '0')}"
# idx_sNo = bhv_df['sNo'] == sNo

# cv = StratifiedKFold(n_splits = 10, shuffle = True, random_state = sNo)

# for feature in ['payoff_bins', 'theta_bin']:
#     pred_ovo_var = []
#     pred_pairs_var = []

#     for var in bhv_df['thetaVar'].unique():

#         idx_var = bhv_df['thetaVar'] == var
#         bhv_ss = bhv_df.loc[idx_sNo & idx_var]
#         csd_ss = aggCSD[idx_sNo & idx_var]
#         n_trls, n_chnls, n_smpls = csd_ss.shape

#         agg_fold = []
#         agg_trn = []
#         agg_tst = []
#         agg_clf = []
#         pred_ovo = []
#         pred_pairs = []

#         for fold, (idx_trn, idx_tst) in enumerate(cv.split(
#             csd_ss,
#             bhv_ss[feature].values
#         )):    
#             y = bhv_ss[feature].values[idx_trn]
#             n_classes = np.unique(y).size
#             X = csd_ss[idx_trn]
#             clf = [
#                 svm.SVC(decision_function_shape = 'ovo').fit(X[..., tsmpl], y) 
#                 for tsmpl in range(n_smpls)
#             ]
#             # predict class in pairwise comparisons (one-versus-one)
#             pred_ovo += [np.stack([
#                 fit_clf.predict(csd_ss[idx_tst,:, idx_clf]) 
#                 for idx_clf, fit_clf in enumerate(clf)
#             ], -1)]
#             pred_pairs += [np.stack([
#                 np.stack([
#                     np.array([   
#                         i if tst_trl[p] > 0 else j
#                         # iterate across pairwise comparisons
#                         for p, (i,j) in enumerate(itertools.combinations(range(n_classes),2))
#                     ])
#                     # iterate across test trials
#                     for tst_trl in fit_clf.decision_function(csd_ss[idx_tst,:, idx_clf]) 
#                 ], 0)
#                 # iterate across time samples 
#                 for idx_clf, fit_clf in enumerate(clf)
#             ], -1)]
#             agg_fold += [fold]
#             agg_trn += [idx_trn]
#             agg_tst += [idx_tst]

#         # save sorted predictions
#         pred_ovo = np.concatenate(pred_ovo)[np.concatenate(agg_tst).argsort()]
#         pred_pairs = np.concatenate(pred_pairs)[np.concatenate(agg_tst).argsort()]

#         # save prediction per variance
#         pred_ovo_var += [pred_ovo]
#         pred_pairs_var += [pred_pairs]
    
#     # sort trials per variance condition so that all trials can be concatenated
#     # in the original order
#     idx_sort = np.concatenate([
#         # compute indices of trials per variance type
#         np.where(bhv_df.loc[idx_sNo, 'thetaVar'] == var)[0] 
#         for var in bhv_df['thetaVar'].unique()
#     ]).argsort()
#     pred_ovo_var = np.concatenate(pred_ovo_var)[idx_sort]
#     pred_pairs_var = np.concatenate(pred_pairs_var)[idx_sort]

#     SAVEFILE = SAVEPATH / f'{subID}_ses-MAIN_lck-STIM_perVAR_svm-epo.hdf'
#     with hdf.File(SAVEFILE, 'a') as f:
#         f[f'/{feature.split("_")[0]}/pred_ovo'] = pred_ovo_var
#         f[f'/{feature.split("_")[0]}/pred_pairs'] = pred_pairs_var
#===============================================================================
# %% analyse SVM results
#===============================================================================
svmData = sorted(ROOTPATH.rglob('*perVAR_svm-epo*'))
aggPayoff = []
aggTheta = []
for svm_path in svmData:
    with hdf.File(svm_path, 'r') as spth:
        aggPayoff += [spth['payoff/pred_ovo'][:]]
        aggTheta += [spth['theta/pred_ovo'][:]]
aggPayoff = np.concatenate(aggPayoff)
aggTheta = np.concatenate(aggTheta)
# %% 
accPayoff = (aggPayoff == bhv_df['payoff_bins'].values[:, None]).astype('int')
accTheta = (aggTheta == bhv_df['theta_bin'].values[:, None]).astype('int')
avPayoff = np.stack([
    [
        accPayoff[
            (bhv_df['sNo'] == sub)
            & (bhv_df['thetaVar'] == var)
        ].mean(0) 
        for var in bhv_df['thetaVar'].unique()
    ]
    for sub in bhv_df['sNo'].unique()
], 0)
avPayoff = signal.convolve(
    np.pad(
        avPayoff, 
        pad_width = (
            (0, 0),
            (0, 0), 
            (padWidth, padWidth)
        ),
        mode = 'reflect'
    ), 
    gwin[None, None], 
    'same'
)[..., padWidth:-padWidth]
avPayoff = np.swapaxes(avPayoff, 0, 1)
t_var, p_var = sps.ttest_rel(*avPayoff)
p_var_adj = multitest.fdrcorrection(p_var)[1]
avTheta = np.stack([
    [
        accTheta[
            (bhv_df['sNo'] == sub)
            & (bhv_df['thetaVar'] == var)
        ].mean(0) 
        for var in bhv_df['thetaVar'].unique()
    ]
    for sub in bhv_df['sNo'].unique()
], 0)
avTheta = signal.convolve(
    np.pad(
        avTheta, 
        pad_width = (
            (0, 0),
            (0, 0), 
            (padWidth, padWidth)
        ),
        mode = 'reflect'
    ), 
    gwin[None, None], 
    'same'
)[..., padWidth:-padWidth]
# %%
