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
# %% Issue tracker:
#===============================================================================
# TODO: 5.  analyse effects of noise level and response type on the four fitted 
#           spectral parameters using spatial cluster-based permutation testing
#       6.  analyse relationship b/w fitted parameters and average test payoffs
#           for the clusters of identified electrodes that are sensitive to IVs
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
from fooof import FOOOFGroup, fit_fooof_3d
from fooof.bands import Bands
from fooof.analysis import get_band_peak_fg
from fooof.plts.spectra import plot_spectra
from fooof.objs.utils import average_fg, combine_fooofs, compare_info
import h5py as hdf
import itertools
import json
import mmodel as mm
import mne
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import pingouin as pg
from pymer4 import models
import polars as pl
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
# %% define functions
#===============================================================================
def check_nans(data, nan_policy='zero'):
    """Check an array for nan values, and replace, based on policy."""

    # Find where there are nan values in the data
    nan_inds = np.where(np.isnan(data))

    # Apply desired nan policy to data
    if nan_policy == 'zero':
        data[nan_inds] = 0
    elif nan_policy == 'mean':
        data[nan_inds] = np.nanmean(data)
    else:
        raise ValueError('Nan policy not understood.')

    return data
#===============================================================================
# %% setting paths
#===============================================================================
ROOTPATH = Path.cwd().parent
# #===============================================================================
# # %% PSD analyses
# #===============================================================================
# epochPaths = sorted(ROOTPATH.rglob(
#     "*LongPreStimWin/**/*ses-MAIN_lck-STIM_csd-epo.fif.gz"
# ))
# dropSUB = ['sub-02']
# epochData = dict([(
#     dpath.stem.split('_')[0], 
#     mne.read_epochs(dpath, preload = True)
# ) for dpath in epochPaths
# if dpath.stem.split('_')[0] not in dropSUB])
# times = list(epochData.values())[0].times
# #===============================================================================
# # %% compute PSD
# # NOTE: window length 1,024 @ 256 Hz fs = 4 s period, f_range = 2-40 Hz
# #===============================================================================
# N_fft = 1024
# N_per_seg = 128
# fmin = 2
# fmax = 40
# tmin = -.5
# tmax = 0
# EXPORTPATH = ROOTPATH / '03_Derivatives' / 'psd_PreStim'
# EXPORTPATH.mkdir(exist_ok = True, parents = True)
# for sub in epochData:
#     tmp = epochData[sub].compute_psd(
#         method = 'welch', 
#         tmin = tmin, tmax = tmax, 
#         fmin = fmin, fmax = fmax,
#         n_fft = N_fft, n_per_seg = N_per_seg
#     )
#     tmp.save(
#         EXPORTPATH / f'{sub}_ses-MAIN_lck-PRESTIM_psd-epo.h5',
#         overwrite = True)
#===============================================================================
# %% load PSD analyses
#===============================================================================
psdPaths = sorted(ROOTPATH.rglob(
    '*psd-epo*'
))
psdData = dict([(
    dpath.stem.split('_')[0], 
    mne.time_frequency.read_spectrum(dpath)
) for dpath in psdPaths])
freqs = list(psdData.values())[0].freqs
ch_names = list(psdData.values())[0].ch_names
# %% aggregation across trials
aggPSD, bhv_df = zip(*[
    (sub.get_data(), sub.metadata)
    for sub in psdData.values()
])
aggPSD = np.concatenate(aggPSD)
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
#===============================================================================
# %% average across trials
#===============================================================================
idx_risk = bhv_df['task'] == 'risk'
avBHV = bhv_df.loc[idx_risk].groupby([
    'sNo', 'thetaVar', 'riskyChoice'
])['tstPayoff'].mean().reset_index()
avBHV = avBHV.merge(
    avBHV.groupby(['sNo'])['tstPayoff'].mean().reset_index().rename(
        columns = dict(tstPayoff = 'tstPayoff_sub')
    ),
    on = ['sNo']
).assign(**dict(
    tstPayoff_tot = avBHV['tstPayoff'].mean()
))
# to plot distributions of payoffs
avBHV['tstPayoff_z'] = (
    avBHV['tstPayoff'] 
    - avBHV['tstPayoff_sub'] 
    + avBHV['tstPayoff_tot']
)

# extract spectral power
avPSD = np.stack([
    aggPSD[(
        (bhv_df['task'] == 'risk') 
        & (bhv_df['sNo'] == sno) 
        & (bhv_df['thetaVar'] == var)
        & (bhv_df['riskyChoice'] == choice)
    )].mean(0)
    for sno, var, choice in avBHV[['sNo','thetaVar','riskyChoice']].values
])
# extract alpha band
idx_alpha = (freqs >= 7) & (freqs <= 14)
alphaPSD = avPSD[..., idx_alpha].mean(-1)
# extract electrode roi
roi_alpha = [
    'P7', 'P8',
    'PO7', 'PO3', 'POz', 'PO4', 'PO8',
    'O1', 'Oz', 'O2'
]
roi_cpp = ['CP1', 'CPz', 'CP2', 'P1', 'Pz', 'P2']
avBHV['alpha'] = alphaPSD[:, [ch_names.index(ch) for ch in roi_alpha]].mean(-1)*1e6
avBHV['cpp'] = alphaPSD[:, [ch_names.index(ch) for ch in roi_cpp]].mean(-1)*1e6
avBHV['choice'] = avBHV['riskyChoice'].astype('str')
avBHV['noise'] = avBHV['thetaVar'].astype('str')
avBHV['sno'] = avBHV['sNo'].astype('str')

# compute gavs
gavBHV = avBHV.groupby(['thetaVar','riskyChoice'])['tstPayoff'].mean().reset_index().rename(
    columns = dict(tstPayoff = 'mPayoff')
).merge(
    avBHV.groupby(['thetaVar','riskyChoice'])['tstPayoff'].std().reset_index().rename(
        columns = dict(tstPayoff = 'sdPayoff')
    ),
    on = ['thetaVar','riskyChoice']
).merge(
    avBHV.groupby(['thetaVar','riskyChoice'])['alpha'].mean().reset_index().rename(
        columns = dict(alpha = 'mAlpha')
    ),
    on = ['thetaVar', 'riskyChoice']
).merge(
    avBHV.groupby(['thetaVar','riskyChoice'])['alpha'].std().reset_index().rename(
        columns = dict(alpha = 'sdAlpha')
    ),
    on = ['thetaVar', 'riskyChoice']
).merge(
    avBHV.groupby(['thetaVar','riskyChoice'])['cpp'].mean().reset_index().rename(
        columns = dict(cpp = 'mCPP')
    ),
    on = ['thetaVar', 'riskyChoice']
).merge(
    avBHV.groupby(['thetaVar','riskyChoice'])['cpp'].std().reset_index().rename(
        columns = dict(cpp = 'sdCPP')
    ),
    on = ['thetaVar', 'riskyChoice']
)
gavBHV['semPayoff'] = gavBHV['sdPayoff'] / np.sqrt(avBHV['sNo'].unique().size - 1)
gavBHV['semAlpha'] = gavBHV['sdAlpha'] / np.sqrt(avBHV['sNo'].unique().size - 1)
gavBHV['semCPP'] = gavBHV['sdCPP'] / np.sqrt(avBHV['sNo'].unique().size - 1)
#===============================================================================
# %% analyses of the BHV data frame
#===============================================================================
lmer_payoff = models.lmer(
    'tstPayoff ~ choice * noise + (1 | sNo)',
    data = pl.DataFrame(avBHV)
)
lmer_payoff.fit()
aov_payoff = models.lmer(
    'tstPayoff ~ choice * noise + (1 | sno)',
    data = pl.DataFrame(avBHV)
)
aov_payoff.anova()

lmer_alpha = models.lmer(
    'alpha ~ choice * noise + (1 | sno)',
    data = pl.DataFrame(avBHV)
)
lmer_alpha.fit()
aov_alpha = models.lmer(
    'alpha ~ choice * noise + (1 | sno)',
    data = pl.DataFrame(avBHV)
)
aov_alpha.anova()

lmer_cpp = models.lmer(
    'cpp ~ choice * noise + (1 | sno)',
    data = pl.DataFrame(avBHV)
)
lmer_cpp.fit()
aov_cpp = models.lmer(
    'cpp ~ choice * noise + (1 | sno)',
    data = pl.DataFrame(avBHV)
)
aov_cpp.anova()
# #===============================================================================
# # %% estimate spectrum coefficients
# #===============================================================================
# # initialize a model which will fit all conditions
# mdl = FOOOFGroup(
#     peak_width_limits=[.5, 9.0],
#     max_n_peaks = 4,
#     min_peak_height= 0,
#     peak_threshold = 2,
#     aperiodic_mode = 'fixed'
# )
# # fit the model
# fgs = fit_fooof_3d(mdl, freqs, avPSD)
# # save the data
# EXPORTPATH = ROOTPATH / '03_Derivatives' / 'psd_par'
# EXPORTPATH.mkdir(exist_ok = True, parents = True)
# for idx, (sno, var, choice) in enumerate(avBHV[[
#     'sNo','thetaVar','riskyChoice'
# ]].values):
#     EXPORTFILE = f'sub-{str(sno).rjust(2,"0")}_var-{str(var)}_choice-{str(choice)}_psd-par.json'
#     with (EXPORTPATH / EXPORTFILE).open('w') as f:
#         fgs[idx].save(
#             f, save_results = True, save_settings = True, save_data = True
#         )
#===============================================================================
# %% load spectral parameters
#===============================================================================
parPaths = sorted(ROOTPATH.rglob('*psd-par*'))
allPar = []
for parp in parPaths:
    EXPORTFILE = parp.name
    EXPORTPATH = parp.parent
    sub, var, choice = [
        float(i.split('-')[-1]) 
        for i in EXPORTFILE.split('_')[:3]
    ]
    tmp = FOOOFGroup()
    tmp.load(
        EXPORTFILE, EXPORTPATH
    )
    allPar += [[[sub,var,choice], tmp]]
vars, pars = zip(*allPar) 
sno, var, choice = zip(*vars) 
agg_psd = np.stack([
    np.stack([p.get_fooof(idx).get_data() for idx in range(len(p))]).mean(0)
    for p in pars
])
av_psd = np.stack([
    agg_psd[sno == s].mean(0)
    for s in np.unique(sno)
])
gav_psd = av_psd.mean(0)
sem_psd = av_psd.std(0, ddof = 1) / np.sqrt(av_psd.shape[0] - 1)
roi_band = Bands(dict(
    alpha = [7, 14]
))
## export aggregated data
# ch_names = psdData['sub-01'].ch_names[:]
# par_df = []
# for idx_par, val_par in enumerate(pars):
#     par_df += [val_par.to_df(roi_band).assign(**dict(
#         chname = ch_names,
#         sNo = vars[idx_par][0],
#         thetaVar = vars[idx_par][1],
#         riskyChoice = vars[idx_par][2]
#     ))]
# par_df = pd.concat(par_df)
# par_df['sNo'] = par_df['sNo'].astype('int')
# par_df = par_df.merge(
#     avBHV,
#     on = ['sNo','thetaVar','riskyChoice']
# )
# par_cols = list(par_df.columns)
# par_cols = par_cols[-4:] + par_cols[-5:-4] + par_cols[:-5]
# par_df = par_df[par_cols].sort_values(by = 'sNo')
# par_df = par_df.reset_index(drop = True)
# # %% save the psd par data
# par_df.to_csv(
#     ROOTPATH / '04_Aggregates' / 'agg_psd_pars.tsv.gz',
#     sep = '\t', na_rep = 'n/a', index = False
# )
# %% load psd par data
par_df = pd.read_csv(
    ROOTPATH / '04_Aggregates' / 'agg_psd_pars.tsv.gz',
    sep = '\t', na_values = 'n/a'
)
par_df = par_df.fillna(0)
par_df['choice'] = par_df['riskyChoice'].astype('str')
par_df['var'] = par_df['thetaVar'].astype('str')
par_df['sno'] = par_df['sNo'].astype('str')
# aov_payoff = pg.rm_anova(
#     dv = 'tstPayoff',
#     within = ['choice','var'],
#     subject = 'sno',
#     data = payoff_df,
#     effsize = 'np2'
# )
# average parameters across electrodes
par_ss = par_df.loc[
    par_df['chname'].isin(roi_alpha)
].groupby([
    'sno','var','choice'
])[['offset','exponent','alpha_pw','alpha_bw']].mean().reset_index().merge(
    par_df.loc[
        par_df['chname'].isin(roi_cpp)
    ].groupby([
        'sno','var','choice'
    ])[['offset','exponent','alpha_pw','alpha_bw']].mean().reset_index(),
    on = ['sno','var','choice'],
    suffixes = ['_post','_cpp']
)
#===============================================================================
# %% analysing spectral data
#===============================================================================
# average model fits across electrodes and participants
agg_pars = combine_fooofs(pars)
gav_alpha = average_fg(agg_pars, roi_band)
gav_alpha.plot()
periodic, aperiodic = plt.gca().get_lines()
plt.close('all')
# %% running statistical inference across fitted parameters
par_test = {}
for roi in ['post','cpp']:
    par_test[roi] = {}
    for dv in ['offset','exponent','alpha_pw', 'alpha_bw']:
        mdl = models.lmer(
            f'{dv}_{roi} ~ choice * var + (1 | sno)',
            data = pl.DataFrame(par_ss)
        )
        mdl.anova()
        par_test[roi][dv] = mdl
# #===============================================================================
# # %% cluster-bases permutation tests
# # NOTE: no significant clusters for no DV and no main effect or interaction
# #===============================================================================
# info = psdData['sub-01'].info.copy()
# ch_adjcy, ch_names = mne.channels.find_ch_adjacency(info, None)
# par_tests = {}
# for dv in ['offset', 'exponent', 'alpha_pw', 'alpha_bw']:
#     # collect results
#     par_tests[dv] = {}

#     # main effect of noise
#     main_var = par_df.loc[
#         par_df['thetaVar'] == 0.075
#     ].groupby([
#         'sNo',
#         'chname'
#     ])[[dv]].mean().reset_index().rename(columns = {dv : f'{dv}_low'}).merge(
#         par_df.loc[
#             par_df['thetaVar'] == 0.3
#         ].groupby([
#             'sNo',
#             'chname'
#         ])[[dv]].mean().reset_index().rename(columns = {dv : f'{dv}_high'}),
#         on = ['sNo','chname']
#     )
#     main_var['diff'] = main_var[f'{dv}_low'] - main_var[f'{dv}_high']
#     main_var['chname_fac'] = main_var['chname'].astype('category').cat.reorder_categories(ch_names)
#     main_var = main_var.sort_values(by=['sNo', 'chname_fac'])
#     par_tests[dv]['var'] = mne.stats.permutation_cluster_1samp_test(
#         main_var['diff'].values.reshape(-1, 64),
#         adjacency = ch_adjcy,
#         seed = 0
#     )

#     # main effect of choice
#     main_choice = par_df.loc[
#         par_df['riskyChoice'] == 1
#     ].groupby([
#         'sNo',
#         'chname'
#     ])[[dv]].mean().reset_index().rename(columns = {dv : f'{dv}_risk'}).merge(
#         par_df.loc[
#             par_df['riskyChoice'] == 0
#         ].groupby([
#             'sNo',
#             'chname'
#         ])[[dv]].mean().reset_index().rename(columns = {dv : f'{dv}_safe'}),
#         on = ['sNo','chname']
#     )
#     main_choice['diff'] = main_choice[f'{dv}_risk'] - main_choice[f'{dv}_safe']
#     main_choice['chname_fac'] = main_choice['chname'].astype('category').cat.reorder_categories(ch_names)
#     main_choice = main_choice.sort_values(by=['sNo', 'chname_fac'])
#     par_tests[dv]['choice'] = mne.stats.permutation_cluster_1samp_test(
#         main_choice['diff'].values.reshape(-1, 64),
#         adjacency = ch_adjcy,
#         seed = 0
#     )
        
#     # interaction b/w choice and noise
#     choice_var = par_df.loc[
#         (par_df['riskyChoice'] == 1) & (par_df['thetaVar'] == 0.075),
#         ['sNo', 'chname', dv]
#     ].rename(columns = {dv : f'{dv}_risk_low'}).merge(
#         par_df.loc[
#             (par_df['riskyChoice'] == 0) & (par_df['thetaVar'] == 0.075),
#         ['sNo', 'chname', dv]
#         ].rename(columns = {dv : f'{dv}_safe_low'}), 
#         on = ['sNo','chname']
#     ).merge(
#         par_df.loc[
#             (par_df['riskyChoice'] == 1) & (par_df['thetaVar'] == 0.3),
#         ['sNo', 'chname', dv]
#         ].rename(columns = {dv : f'{dv}_risk_high'}), 
#         on = ['sNo','chname']
#     ).merge(
#         par_df.loc[
#             (par_df['riskyChoice'] == 0) & (par_df['thetaVar'] == 0.3),
#         ['sNo', 'chname', dv]
#         ].rename(columns = {dv : f'{dv}_safe_high'}), 
#         on = ['sNo','chname']
#     )
#     choice_var['diff'] = (
#         (choice_var[f'{dv}_risk_low'] - choice_var[f'{dv}_safe_low'])
#         - (choice_var[f'{dv}_risk_high'] - choice_var[f'{dv}_safe_high'])
#     )
#     choice_var['chname_fac'] = choice_var['chname'].astype('category').cat.reorder_categories(ch_names)
#     choice_var = choice_var.sort_values(by=['sNo', 'chname_fac'])
#     par_tests[dv]['choice_var'] = mne.stats.permutation_cluster_1samp_test(
#         choice_var['diff'].values.reshape(-1, 64),
#         adjacency = ch_adjcy,
#         seed = 0
#     )
#===============================================================================
# %% Fig 2
#===============================================================================
info = psdData['sub-01'].info.copy()
# set plotting params
sbn.set_style('ticks')
cols = np.array(sbn.color_palette('muted'))[[2,4]]
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
# %% 
fig = plt.figure(figsize = (6.6, 2.2))

axBHV = plt.subplot2grid(
    (10,30), 
    (0, 0),
    rowspan = 10,
    colspan = 10,
    fig = fig
)
sbn.violinplot(
    x = 'riskyChoice', y = 'tstPayoff_z', hue = 'thetaVar',
    data = avBHV, split = True, axes = axBHV, inner = 'quart', 
    palette = cols,
    legend = True
)
axBHV.set_xticklabels([
    'Safe choice',
    'Risky'
])
axBHV.set_ylabel('Test payoff')
axBHV.set_yticks(np.linspace(-6,6,5))
for spine in ['top','right']:
    axBHV.spines[spine].set_visible(False)
axBHV.spines['left'].set_bounds(-6,6)
axBHV.spines['bottom'].set_bounds(0,1)
axBHV.tick_params('both', direction = 'in', length = 3)

hdls, lbls = axBHV.get_legend_handles_labels() 
axBHV.legend(
    hdls, ['Low Noise', 'High'], ncol = 2, frameon = False,
    handlelength = 1.75, handletextpad = .5, columnspacing = 1,
    bbox_to_anchor = (.5, 1.2), bbox_transform = axBHV.transAxes, loc = 'upper center'
)

axTOPO = plt.subplot2grid(
    (10,30), 
    (0, 10),
    rowspan = 10,
    colspan = 10,
    fig = fig
)
idx_alpha = (freqs >= 7) & (freqs <= 14)
im,_ = mne.viz.plot_topomap(
    avPSD.mean(0)[:, idx_alpha].mean(-1), info, axes = axTOPO,
    mask = np.array([
        True if ch in roi_alpha + roi_cpp else False 
        for ch in ch_names
    ])
)
axTOPO.set_title('Mean power 7-14 Hz')
axTOPO.lines[0].set_markersize(5)
x, y = axTOPO.lines[0].get_data()
mask_ch = [ch for ch in ch_names if ch in roi_alpha + roi_cpp]
idx_ss = [mask_ch.index(ch) for ch in mask_ch if ch in roi_alpha]
x_rest = np.delete(x, idx_ss)
y_rest = np.delete(y, idx_ss)
axTOPO.lines[0].set_data(x_rest, y_rest)
axTOPO.plot(x[idx_ss],y[idx_ss], 's', mfc = 'white', mec = 'black', ms = 4)

axPSD = plt.subplot2grid(
    (10,30), 
    (0, 22),
    rowspan = 10,
    colspan = 8,
    fig = fig
)

axPSD.fill_between(
    freqs,
    gav_psd - sem_psd * 2.58,
    gav_psd + sem_psd * 2.58,
    color = 'black', alpha = .3
)
cols_psd = np.array(sbn.color_palette('muted'))[[3,0]]
axPSD.plot(freqs, gav_psd, color = 'black', label = 'Observed', lw = 2)
axPSD.plot(*periodic.get_data(), color = cols_psd[0], label = 'Full fit', lw = 2)
axPSD.plot(*aperiodic.get_data(), color = cols_psd[1], label = 'Aperiodic', lw = 2)
hdls, lbls = axPSD.get_legend_handles_labels()
axPSD.legend(
    hdls, lbls, ncol = 1, frameon = False,
    handlelength = 1.75, handletextpad = .5, columnspacing = 1,
    bbox_to_anchor = (1, 1.2), bbox_transform = axPSD.transAxes, loc = 'upper right'
)
axPSD.set_ylabel('Power (dB)')
axPSD.set_ylim(-7.5,-5.5)
axPSD.set_yticks([-7,-6])
axPSD.spines['left'].set_bounds(-7,-6)

axPSD.set_xticks([2,40])
axPSD.set_xticklabels(['2 Hz', '40'])
axPSD.spines['bottom'].set_bounds([2,40])

for spine in ['top', 'right']:
    axPSD.spines[spine].set_visible(False)

axPSD.tick_params('both', direction = 'in', length = 3)

fig.subplots_adjust(.08,.09,.975,.85)
fig.savefig(ROOTPATH / '05_Exports' / 'Fig02_v01.png', dpi = 600)
plt.close(fig)
#===============================================================================
# %% Fig 03
#===============================================================================
info = psdData['sub-01'].info.copy()
gav_par = par_df.groupby('chname')[[
    'offset','exponent','alpha_cf','alpha_pw','alpha_bw'
]].mean().reset_index()
gav_par['chname'] = gav_par['chname'].astype('category').cat.reorder_categories(ch_names)
gav_par = gav_par.sort_values(by = 'chname')
# set plotting params
sbn.set_style('ticks')
cols = np.array(sbn.color_palette('muted'))[[2,4]]
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
# %% 
fig = plt.figure(figsize = (4.4, 4.4))
for dv_idx, dv_val in enumerate(['offset','exponent','alpha_pw','alpha_bw']):
    ax = plt.subplot2grid(
        (20,20), 
        (
            (dv_idx // 2) * 10, 
            (dv_idx % 2) * 10
        ),
        rowspan = 10,
        colspan = 10,
        fig = fig
    )
    ax.set_title([
        'Offset [dB]',
        'Exponent [dB]',
        'Alpha Power [dB]',
        'Alpha Bandwth. [Hz]'
    ][dv_idx])
    vlim = [
        (-6,-5),
        (.5, 1.5),
        (0, 1),
        (0, 4)
    ][dv_idx]
    im, cm = mne.viz.plot_topomap(
        gav_par[dv_val], 
        info, 
        cmap='viridis', 
        # cmap = 'coolwarm',
        contours=0, 
        axes=ax, vlim = vlim,
        mask = np.array([
            True if ch in roi_alpha + roi_cpp else False 
            for ch in ch_names
        ])
    )
    ax.lines[0].set_markersize(4)
    ax.lines[0].set_markerfacecolor('none')
    ax.lines[0].set_markeredgecolor('white')
    ax.lines[0].set_markeredgewidth(1)
    x, y = ax.lines[0].get_data()
    mask_ch = [ch for ch in ch_names if ch in roi_alpha + roi_cpp]
    idx_ss = [mask_ch.index(ch) for ch in mask_ch if ch in roi_alpha]
    x_rest = np.delete(x, idx_ss)
    y_rest = np.delete(y, idx_ss)
    ax.lines[0].set_data(x_rest, y_rest)
    ax.plot(x[idx_ss],y[idx_ss], 's', mfc = 'none', mec = 'white', ms = 4, mew = 1)

    cbar = plt.colorbar(im, ax = ax, orientation = 'vertical', shrink = .75)
    cbar.set_ticks(vlim)
# %%
fig.subplots_adjust(.0,.0,.95,1, hspace = 0, wspace = 0)
fig.savefig(ROOTPATH / '05_Exports' / 'Fig03_v01.png', dpi = 600)
plt.close(fig)

# %%
fg = fgs[0]
# Plot the topographies across different frequency bands
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ind, (label, band_def) in enumerate(roi_band):

    # Get the power values across channels for the current band
    band_power = check_nans(get_band_peak_fg(fg, band_def)[:, 1])

    # Create a topomap for the current oscillation band
    mne.viz.plot_topomap(band_power, info, cmap='viridis', contours=0, axes=axes[ind])

    # Set the plot title
    axes[ind].set_title(label + ' power', {'fontsize' : 20})

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
for ind, (label, band_def) in enumerate(roi_band):

    # Get the power values across channels for the current band
    band_power = check_nans(get_band_peak_fg(fg, band_def)[:, 1])

    # Extracted and plot the power spectrum model with the most band power
    fg.get_fooof(np.argmax(band_power)).plot(ax=axes[ind], add_legend=False)

    # Set some plot aesthetics & plot title
    axes[ind].yaxis.set_ticklabels([])
    axes[ind].set_title('biggest ' + label + ' peak', {'fontsize' : 16})

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
