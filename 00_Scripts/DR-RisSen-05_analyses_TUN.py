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
# TODO: 2.  collate neural tuning to individual grating as a function of 
#           location and values
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
import contextlib
import eeg_functions as eegfun
import gzip
import h5py as hdf
import io
import itertools
import json
import mmodel as mm
import mne
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import pingouin as pg
from scipy import signal
from scipy import stats as sps
import seaborn as sbn
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import svm
from skimage import measure
from statsmodels.stats import multitest
from statsmodels.stats.anova import AnovaRM
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sys
import TunMah_woutTempGen as dmah
from pymer4 import Lmer, Lm
#===============================================================================
# %% setting paths
#===============================================================================
ROOTPATH = Path.cwd().parent
# #===============================================================================
# # %% Mahalanobis distance
# # Analyses of individual gratings
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
# # orientation decoding of individual stimuli
# for idx_stim, val_stim in enumerate(range(1, 13)):
#     col_stim = f'stim{val_stim}Ori'
#     thetas = (bhv_df[col_stim].values / 90) * np.pi
#     cv = list(LeaveOneOut().split(csd))
#     dist = dmah.compute_MahDist(
#         csd, 
#         thetas, 
#         cv, 
#         channels, 
#         bin_width
#     )
#     dist = np.array(dist)
#     tun = -(dist - dist.mean(1)[:,None])

#     # save tuning data
#     tun_info = mne.create_info(
#         ch_names = [str(ch) for ch in channels.round(2)], 
#         sfreq = epochData.info['sfreq'],
#     )
#     tun_info['description'] = 'Similarity matrices computed using Mahalanobis distance.'
#     epochTun = mne.EpochsArray(
#         tun,
#         tun_info,
#         tmin = epochData.times.min(),
#         events = epochData.events, event_id = epochData.event_id,
#         metadata = epochData.metadata
#     )
#     SAVEPATH = ROOTPATH / '03_Derivatives' / 'tun'
#     SAVEPATH.mkdir(exist_ok = True, parents = True)
#     epochTun.save(
#         SAVEPATH / f"{subID}_ses-MAIN_lck-STIM_stim-{f'{val_stim}'.rjust(2, '0')}_tun-epo.fif.gz",
#         overwrite = True
#     )
#===============================================================================
# %% Analysing tuning
#===============================================================================
tunPaths = sorted(ROOTPATH.rglob('**/*_ses-MAIN_lck-STIM_stim-*_tun-epo.fif.gz'))
dropSUB = ['sub-02']
tunData = dict([
    (
        f"{dpath.stem.split('_')[0]}_{dpath.stem.split('_')[-2]}",
        mne.read_epochs(dpath, preload = True)
    )
    for dpath in tunPaths
    if dpath.stem.split('_')[0] not in dropSUB
])
# %% aggregation across trials
channels = np.array(tunData['sub-01_stim-01'].ch_names).astype('float')
times = tunData['sub-01_stim-01'].times
info = tunData['sub-01_stim-01'].info
time_as_index = tunData['sub-01_stim-01'].time_as_index
# configurations for smoothing
SD = int(info['sfreq'] * .016)
gwin = signal.windows.gaussian(
    # the window width is 16 SD (-8 to +8)
    16 * SD,
    SD
)
gwin /= gwin.sum()
padWidth = int(info['sfreq'] * .5)
# without temporal smoothing
# aggSIM, bhv_df = zip(*[
#     (sub.get_data(), sub.metadata)
#     for sub in tunData.values()
# ])
# with temporal smoothing
aggSIM, bhv_df = zip(*[
    (
        signal.convolve(
            np.pad(
                sub.get_data(),
                pad_width = (
                    (0, 0),
                    (0, 0), 
                    (padWidth, padWidth)
                ),
                mode = 'reflect'
            ), 
            gwin[None, None], 
            'same'
        )[..., padWidth:-padWidth],
        sub.metadata
    )
    for sub in tunData.values()
])
# get values of sub and stim per list entry
subID, stimID = zip(*[
    [
        int(key.split('_')[0].split('-')[-1]),
        int(key.split('_')[1].split('-')[-1])
    ] for key in tunData.keys()
])
del tunData # clean up memory
# reshape similarity data to N subject x 12 stimuli x 16 channels x M times
aggSIM = np.concatenate([
    np.stack(list(itertools.compress(aggSIM, np.array(subID) == sub)), 1)
    for sub in np.unique(subID)
], 0)
# just select data frame once, and drop same data frames for other stimuli
bhv_df = pd.concat([
    val_df
    for idx_df, val_df in enumerate(bhv_df)
    if stimID[idx_df] == 1 
])
# which side was the test side
bhv_df = bhv_df.merge(
    bhv_df.groupby(['sNo']).apply(
        lambda x: ['right' if x['rightPayoff'].unique().size > 2 else 'left'][0]
    ).reset_index().rename(columns = {0:'tstSide'}),
    on = ['sNo']
)
# what was the payoff of the risky choice
bhv_df['tstPayoff'] = (np.angle(
    np.exp(
        np.abs(np.angle(
            np.exp(bhv_df['tstThetaRad'] * 1j) 
            # rotate the angles repending on the participant value-mapping
            * np.exp(np.array([0, np.pi])[bhv_df['sNo'] % 2] * 1j)
        )) * 1j
    ) 
    / np.exp(.5 * np.pi * 1j)
) * 7).round(0)

# collect individual orientations and payoffs
for stim in range(1, 13):
    bhv_df[f"stim{str(stim).rjust(2,'0')}ThetaRad"] = np.angle(np.exp(
        bhv_df[f'stim{stim}Ori'] / 90 * np.pi * 1j
    ))
    bhv_df[f"stim{str(stim).rjust(2,'0')}Payoff"] = (np.angle(
        np.exp(
            np.abs(np.angle(
                np.exp(bhv_df[f"stim{str(stim).rjust(2,'0')}ThetaRad"] * 1j) 
                # rotate the angles repending on the participant value-mapping
                * np.exp(np.array([0, np.pi])[bhv_df['sNo'] % 2] * 1j)
            )) * 1j
        ) 
        / np.exp(.5 * np.pi * 1j)
    ) * 7).round(0)
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
# compute relative payoffs
bhv_df = bhv_df.assign(**dict([
    (f'stim{str(idx_stim + 1).rjust(2,"0")}RelVal', val_stim)
    for idx_stim, val_stim in enumerate((bhv_df.loc[
        :, [f'stim{str(stim).rjust(2,"0")}Payoff' for stim in range(1,13)]
    ].values - bhv_df.loc[:, 'tstPayoff'].values[:,None]).T)
]))
#===============================================================================
# %% analyse tuning data
#===============================================================================
avSIM = np.stack([
    aggSIM[bhv_df['sNo'] == sno].mean(0)
    for sno in bhv_df['sNo'].unique()
])
gavSIM = avSIM.mean(0)
t_val_sim, p_val_sim = sps.ttest_1samp(
    avSIM, 0
)
p_adj_sim = multitest.fdrcorrection(p_val_sim.flatten())[1].reshape(*p_val_sim.shape)
cntr_sim = [measure.find_contours(
    -np.log(stim), -np.log(.05)
) for stim in p_adj_sim]
# # aggregate similarity across stimuli
# t_val_agg, p_val_agg = sps.ttest_1samp(
#     avSIM.mean(1), 0
# )
# p_adj_agg = multitest.fdrcorrection(p_val_agg.flatten())[1].reshape(*p_val_agg.shape)
# ssSIM = avSIM[..., p_adj_agg.mean(0) < .05].mean(-1)
# ssTUN = (ssSIM * np.cos(channels[None, None])).sum(-1)
#===============================================================================
# %% model decision weights
# estimate decision weights per task, variance condition and participant
# TODO: 2. complex-valued regression for perceptual task
# A. Stimuli sorted by location
# B. Stimuli sorted by payoff signal
# NOTE: trialType = 0
# NOTE: swapping intercept condition to High Noise removes the effect 
#===============================================================================
# decision weights for risky choice task
w_spatial = []
w_payoff = []
# w_paydev = []
subNo = []
thetaVar = []
pred = [
    f"stim{str(stim).rjust(2,'0')}Payoff" 
    for stim in range(1,13)
]
for sub in bhv_df['sNo'].unique():
    for var in bhv_df['thetaVar'].unique():
        idx = (
            (bhv_df['task'] == 'risk')
            & (bhv_df['sNo'] == sub) 
            & (bhv_df['thetaVar'] == var)
        )
        df_ss = bhv_df.loc[idx].copy()
        # df_ss.loc[:, pred] -= df_ss.loc[:, 'tstPayoff'].values[:,None]
        print(f"sub = {str(sub).rjust(2,'0')}; var = {str(var).rjust(2,'0')}")
        with contextlib.redirect_stdout(io.StringIO()):
            # estimate spatial weights
            w_spatial += [Lm(
                f"riskyChoice ~ {' + '.join(['tstPayoff'] + pred)}",
                data = df_ss, 
                family = 'binomial'
            ).fit().Estimate[-12:]]
            # estimate payoff weights
            df_ss_payoff = df_ss.copy()
            df_ss_payoff.loc[:, pred] = np.sort(
                df_ss_payoff.loc[:, pred].values,
                1
            )
            w_payoff += [Lm(
                f"riskyChoice ~ {' + '.join(['tstPayoff'] + pred)}",
                data = df_ss_payoff, 
                family = 'binomial'
            ).fit().Estimate[-12:]]
            # # estimate payoff deviations b/w single gratings and the average
            # df_ss_paydev = df_ss.copy()
            # df_ss_paydev.loc[:, pred] = np.sort(
            #     df_ss_paydev.loc[:, pred].values 
            #     - df_ss_paydev.loc[:, 'tstPayoff'].values[:,None],
            #     1
            # )
            # w_paydev += [Lm(
            #     f"riskyChoice ~ {' + '.join(pred)}",
            #     data = df_ss_paydev, 
            #     family = 'binomial'
            # ).fit().Estimate[1:]]
# %%
dw_risk_df = pd.DataFrame(
    data = dict(
        [
            (['sNo', 'thetaVar', 'stim'][idx_col],val_col) 
            for idx_col, val_col in enumerate(zip(*itertools.product(
                bhv_df['sNo'].unique(),
                bhv_df['thetaVar'].unique(),
                range(1,13))))
        ] 
        + [('w_spatial', np.concatenate(w_spatial))]
        + [('w_payoff', np.concatenate(w_payoff))]
        # + [('w_paydev', np.concatenate(w_paydev))]
    )
)
# x,y coordinates for different locations (0-11)
vert = np.array([
    [ 0.00000000e+00,  5.00000000e-01],
    [ 2.50000000e-01,  4.33012702e-01],
    [ 4.33012702e-01,  2.50000000e-01],
    [ 5.00000000e-01,  3.06161700e-17],
    [ 4.33012702e-01, -2.50000000e-01],
    [ 2.50000000e-01, -4.33012702e-01],
    [ 6.12323400e-17, -5.00000000e-01],
    [-2.50000000e-01, -4.33012702e-01],
    [-4.33012702e-01, -2.50000000e-01],
    [-5.00000000e-01, -9.18485099e-17],
    [-4.33012702e-01,  2.50000000e-01],
    [-2.50000000e-01,  4.33012702e-01]
])
stimRad = np.arctan2(
    *vert[:,::-1].T
).round(2)
stimVal = np.sort(
    bhv_df.loc[:, pred].values
    - bhv_df.loc[:, 'tstPayoff'].values[:,None],
    1
).mean(0).round(1)
dw_risk_df['stim_fac'] = dw_risk_df['stim'].astype('str').str.pad(2, 'left', '0')
dw_risk_df['var_fac'] = dw_risk_df['thetaVar'].astype('str')
dw_risk_df = dw_risk_df.merge(
    pd.DataFrame(dict(
        stim = range(1,13),
        stimRad = stimRad
    )),
    on = ['stim']
).merge(pd.DataFrame(dict(
        stim = range(1,13),
        stimVal = stimVal
    )),
    on = ['stim']
)
dw_risk_df['stimSin'] = np.sin(dw_risk_df['stimRad'])
dw_risk_df['stimCos'] = np.cos(dw_risk_df['stimRad'])
dw_risk_df['valAbs'] = np.abs(dw_risk_df['stimVal'])
dw_risk_df['valSgn'] = np.sign(dw_risk_df['stimVal']).astype('str')
# smoothing weights
sfreq = 12
SD = int(sfreq * .1)
gwin = signal.windows.gaussian(
    12 * SD,
    SD
)
padWidth = int(sfreq * .5)

df_smth = pd.DataFrame(
    data = list(itertools.product(
        dw_risk_df['sNo'].unique(),
        dw_risk_df['var_fac'].unique(),
        dw_risk_df['stim_fac'].unique()
    )),
    columns = ['sNo', 'var_fac', 'stim_fac']
)
for dv in ['w_spatial','w_payoff']:
    df_smth[f'{dv}_smth'] = np.concatenate(dw_risk_df.groupby([
        'sNo', 'var_fac'
    ])[dv].apply(
        lambda x:
            signal.convolve(
                np.pad(
                    x,
                    pad_width = ((padWidth, padWidth)),
                    mode = 'reflect'
                ),
                gwin, 
                mode = 'same'
            )[padWidth:-padWidth]
    ).reset_index()[dv].values)

dw_risk_df = dw_risk_df.merge(
    df_smth,
    on = ['sNo','var_fac', 'stim_fac']
)
# save aggregated decision weights
dw_risk_df.to_csv(
    ROOTPATH / '04_Aggregates' / 'agg_dw.tsv.gz', 
    sep = '\t', index = False
)

# normalise data
for dv in ['spatial', 'payoff']:
    dw_risk_df = dw_risk_df.merge(
        dw_risk_df.groupby([
            'sNo'
        ])[f'w_{dv}_smth'].mean().reset_index(),
        on = ['sNo'], suffixes = ['', '_sub']
    ).assign(**{
        f'w_{dv}_smth_tot' : dw_risk_df[f'w_{dv}_smth'].mean()
    })
    dw_risk_df[f'w_{dv}_smth_z'] = (
        dw_risk_df[f'w_{dv}_smth'] 
        - dw_risk_df[f'w_{dv}_smth_sub']
        + dw_risk_df[f'w_{dv}_smth_tot']
    )
    
gav_dw = dw_risk_df.groupby([
    'stim_fac',
    'var_fac'
])[['w_spatial_smth_z','w_payoff_smth_z']].mean().reset_index().merge(
    dw_risk_df.groupby([
        'stim_fac',
        'var_fac'
    ])[['w_spatial_smth_z','w_payoff_smth_z']].apply(
        lambda x:
            # standard error
            np.sqrt((
                # variance
                x.var(ddof = 1) 
                # correction for score normalisation
                * ((12 * 2) / ((12 * 2) - 1))) 
                # divide by N
                / (x.shape[0] - 1)
            )
    ).reset_index(),
    on = ['stim_fac','var_fac'], suffixes = ['','_sem']
)
# %% comparing weights
mdl_spatial = Lmer(
    # 'w_spatial_smth ~ (stimSin + stimCos) * var_fac + (1 | sNo)', 
    'w_spatial_smth ~ stimSin * stimCos * var_fac + (1 | sNo)', 
    data = dw_risk_df
).fit()
print(mdl_spatial)
mdl_payoff = Lmer(
    # 'w_spatial_smth ~ (stimSin + stimCos) * var_fac + (1 | sNo)', 
    'w_payoff_smth ~ valAbs * valSgn * var_fac + (1 | sNo)', 
    data = dw_risk_df
).fit()
print(mdl_payoff)
#===============================================================================
# %% plotting behavioural results
#===============================================================================
sbn.set_style('ticks')
sbn.set_palette('husl', 75, desat = .60)
fsmall, fmed, flarge = [8, 10, 11]
fsmall, fmed, flarge = [10, 11, 11]
plt.rc('font', size=fmed)          # controls default text sizes
# plt.rc('font', family='Ubuntu')
plt.rc('axes', facecolor=(0, 0, 0, 0))
plt.rc('axes', titlesize=fsmall)     # fontsize of the axes title
plt.rc('axes', labelsize=fsmall)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=fsmall)    # fontsize of the tick labels
plt.rc('ytick', labelsize=fsmall)    # fontsize of the tick labels
plt.rc('legend', fontsize=fsmall)    # legend fontsize
plt.rc('figure', titlesize=fmed)
cols = np.array(sbn.color_palette('muted'))[[2,4]]
# %% plot perceptual task
fig = plt.figure(figsize = (6.6, 2.2))

axSPC = plt.subplot2grid(
    (1,2), 
    (0, 0),
    rowspan = 1,
    colspan = 1,
    fig = fig
)
mids = np.sort(stimRad)
shift = np.unique((mids[1:] - mids[:-1]).round(1))[0] * .2
for idx_var, val_var in enumerate(gav_dw['var_fac'].unique()):
    idx = gav_dw['var_fac'] == val_var
    axSPC.plot(
        mids + shift * [-1,1][idx_var],
        gav_dw.loc[
            idx, 'w_spatial_smth_z'
        ].values[np.argsort(stimRad)],
        color = cols[idx_var],
        marker = ['v','^'][idx_var],
        linewidth = 1,
        label = ['Low Noise', 'High'][idx_var]
    )
    axSPC.errorbar(
        mids + shift * [-1,1][idx_var],
        gav_dw.loc[
            idx, 'w_spatial_smth_z'
        ].values[np.argsort(stimRad)],
        yerr = gav_dw.loc[
            idx, 'w_spatial_smth_z_sem'
        ].values[np.argsort(stimRad)],
        color = cols[idx_var],
        linewidth = .5
    )
axSPC.set_xticks(np.linspace(-np.pi, np.pi,3))
axSPC.set_xticklabels([r'$-\pi$', r'$0$', r'$\pi$'])
axSPC.tick_params('both', direction = 'in', length = 3)
axSPC.set_xlim(-np.pi - (np.pi - stimRad[-1])*.5, np.pi)
axSPC.set_ylim(.014,.186)
for spine in ['top','right']:
    axSPC.spines[spine].set_visible(False)
axSPC.spines['left'].set_bounds(0.05,.15)
axSPC.spines['bottom'].set_bounds(-np.pi, np.pi)
axSPC.set_ylabel('Decision weight [a.u.]')
axSPC.set_xlabel('Grating position [rad]')

axPAY = plt.subplot2grid(
    (1,2), 
    (0, 1),
    rowspan = 1,
    colspan = 1,
    fig = fig
)
mids = np.sort(stimVal)
shift = ((mids[1:] - mids[:-1]).round(1)).min() * .2
for idx_var, val_var in enumerate(gav_dw['var_fac'].unique()):
    idx = gav_dw['var_fac'] == val_var
    idx = gav_dw['var_fac'] == val_var
    axPAY.plot(
        mids + shift * [-1,1][idx_var],
        gav_dw.loc[
            idx, 'w_payoff_smth_z'
        ].values,
        color = cols[idx_var],
        marker = ['v','^'][idx_var],
        linewidth = 1,
        label = ['Low Noise', 'High'][idx_var]
    )
    axPAY.errorbar(
        mids + shift * [-1,1][idx_var],
        gav_dw.loc[
            idx, 'w_payoff_smth_z'
        ].values,
        yerr = gav_dw.loc[
            idx, 'w_payoff_smth_z_sem'
        ].values,
        color = cols[idx_var],
        linewidth = .5
    )
axPAY.set_xticks(np.linspace(-5, 5,3))
axPAY.set_yticks(np.linspace(.0, .3,3))
axPAY.tick_params('both', direction = 'in', length = 3)
for spine in ['top','right']:
    axPAY.spines[spine].set_visible(False)
axPAY.spines['left'].set_bounds(0,.3)
axPAY.spines['bottom'].set_bounds(-5, 5)
# axPAY.set_ylabel('Decision weight [a.u.]')
axPAY.set_xlabel('Grating value [points]')
axPAY.set_ylim(-.108,.408)


hndls, lbls = axSPC.get_legend_handles_labels()
axSPC.legend(
    hndls, lbls,
    frameon = False,
    ncol = 2, loc = 'upper center',
    bbox_to_anchor = (.5,1),
    bbox_transform = fig.transFigure
)

fig.subplots_adjust(.09,.18,.98,.95, hspace = .5) 
fig.savefig(ROOTPATH / '05_Exports' / 'gavDW_v00.png', dpi = 600)
#===============================================================================
# %% analysing tuning results per grating
# DONE:large GLMs of decoding precision as a function of sin/cos and noise
# with participant as random factor
#===============================================================================
aggERR = np.angle((aggSIM * np.exp(channels[None,None,:,None] * 1j)).sum(2))
aggERR_p = np.abs(aggERR) / np.pi
del aggERR
avERR_p = []
for sno in bhv_df['sNo'].unique():
    err_sno = []
    for var in bhv_df['thetaVar'].unique():
        idx = (bhv_df['sNo'] == sno) & (bhv_df['thetaVar'] == var)
        err_sno += [aggERR_p[idx].mean(0)]
    avERR_p += [err_sno]
avERR_p = np.array(avERR_p)
gavERR_p = avERR_p.mean(0)

# aggTUN = (aggSIM * np.cos(channels)[None, None,:,None]).sum(-2)
# avTUN = []
# for sno in bhv_df['sNo'].unique():
#     tun_sno = []
#     for var in bhv_df['thetaVar'].unique():
#         idx = (bhv_df['sNo'] == sno) & (bhv_df['thetaVar'] == var)
#         tun_sno += [aggTUN[idx].mean(0)]
#     avTUN += [tun_sno]
# avTUN = np.array(avTUN)
# gavTUN = avTUN.mean(0)
# x,y coordinates for different locations (0-11)
vert = np.array([
    [ 0.00000000e+00,  5.00000000e-01],
    [ 2.50000000e-01,  4.33012702e-01],
    [ 4.33012702e-01,  2.50000000e-01],
    [ 5.00000000e-01,  3.06161700e-17],
    [ 4.33012702e-01, -2.50000000e-01],
    [ 2.50000000e-01, -4.33012702e-01],
    [ 6.12323400e-17, -5.00000000e-01],
    [-2.50000000e-01, -4.33012702e-01],
    [-4.33012702e-01, -2.50000000e-01],
    [-5.00000000e-01, -9.18485099e-17],
    [-4.33012702e-01,  2.50000000e-01],
    [-2.50000000e-01,  4.33012702e-01]
])
stimRad = np.arctan2(
    *vert[:,::-1].T
).round(2)
# add stimulus location
bhv_df = bhv_df.assign(**dict([
    (f'stim{str(idx_stim + 1).rjust(2,"0")}LocRad', stim)
    for idx_stim, stim in enumerate(stimRad)
]))
# tmp_df = pd.melt(
#     bhv_df, 
#     id_vars = ['blockNo','trialNo','sNo','thetaVar','task'], 
#     value_vars=[
#         f'stim{str(stim).rjust(2,"0")}LocRad' 
#         for stim in range(1,13)
#     ], 
#     var_name = 'stim',
#     value_name = 'LocRad'
# )
# tmp_df['cosLoc'] = np.cos(tmp_df['LocRad'])
# tmp_df['sinLoc'] = np.sin(tmp_df['LocRad'])
tmp_df = pd.melt(
    bhv_df, 
    id_vars = ['blockNo','trialNo','sNo','thetaVar','task'], 
    value_vars=[
        f'stim{str(stim).rjust(2,"0")}RelVal' 
        for stim in range(1,13)
    ], 
    var_name = 'stim',
    value_name = 'LocVal'
)
tmp_df['absVal'] = np.abs(tmp_df['LocVal'])
tmp_df['sgnVal'] = np.sign(tmp_df['LocVal'])
idx_null = tmp_df['sgnVal'] == 0
tmp_df = tmp_df.loc[~idx_null]
tmp_df['sgnVal_fac'] = tmp_df['sgnVal'].astype('category')
tmp_df['thetaVar_fac'] = tmp_df['thetaVar'].astype('category')
# # info for sorting neural data per location value
# pred = [col for col in bhv_df.columns if ('Payoff' in col) and ('stim' in col)]
# stimVal = np.sort(
#     bhv_df.loc[:, pred].values
#     - bhv_df.loc[:, 'tstPayoff'].values[:,None],
#     1
# ).mean(0).round(1)
# valSort = bhv_df.loc[:, pred].values.argsort(1)
# make the neural error magnitudes array match dimensionality with
# the behavioural data frame
tmpERR = np.concatenate([aggERR_p[:, stim] for stim in range(12)])
tmpERR = tmpERR[~idx_null]
# tmpTUN = np.concatenate([aggTUN[:, stim] for stim in range(12)])
# # tunning sorted by grating value
# tmpTUN = np.concatenate([
#     aggTUN[np.arange(bhv_df.shape[0])[:,None],valSort][:,stim] 
#     for stim in range(12)]
# )
# %% fitting models per time sample with location as predictor
# mdf = []
# for idx_tsmpl, val_tsmpl in enumerate(tmpERR.T):
#     print(f'time sample {idx_tsmpl}')
#     tmp_df['err'] = val_tsmpl
#     mdf += [smf.mixedlm(
#         "err ~ sinLoc * cosLoc * thetaVar_fac", 
#         data = tmp_df, 
#         groups = tmp_df['sNo']
#     ).fit()]
# for idx_tsmpl, val_tsmpl in enumerate(tmpTUN.T):
#     print(f'time sample {idx_tsmpl}')
#     tmp_df['tun'] = val_tsmpl
#     mdf += [smf.mixedlm(
#         "tun ~ sinLoc * cosLoc * thetaVar_fac", 
#         data = tmp_df, 
#         groups = tmp_df['sNo']
#     ).fit()]
# # %% fitting models per time sample with value as predictor
# mdf = []
# for idx_tsmpl, val_tsmpl in enumerate(tmpERR.T):
#     print(f'time sample {idx_tsmpl}')
#     tmp_df['err'] = val_tsmpl
#     mdf += [smf.mixedlm(
#         "err ~ absVal * sgnVal_fac * thetaVar_fac", 
#         data = tmp_df, 
#         groups = tmp_df['sNo']
#     ).fit()]
# for idx_tsmpl, val_tsmpl in enumerate(tmpTUN.T):
#     print(f'time sample {idx_tsmpl}')
#     tmp_df['tun'] = val_tsmpl
#     mdf += [smf.mixedlm(
#         "tun ~ sinLoc * cosLoc * thetaVar_fac", 
#         data = tmp_df, 
#         groups = tmp_df['sNo']
#     ).fit()]
# %% pickle the models
# EXPORTPATH = ROOTPATH / '04_Aggregates' / 'GLM_NErrByLoc'
# EXPORTPATH = ROOTPATH / '04_Aggregates' / 'GLM_NTunByLoc'
# EXPORTPATH = ROOTPATH / '04_Aggregates' / 'GLM_NErrByVal'
# EXPORTPATH.mkdir(exist_ok=True, parents = True)
# for idx, val in enumerate(mdf):
#     with (
#         EXPORTPATH 
#         / f'tsample_{str(idx).rjust(3,"0")}.pkl.gz'
#     ).open('wb') as f:
#         pickle.dump(val, f)
# %% load models
EXPORTPATH = ROOTPATH / '04_Aggregates' / 'GLM_NErrByLoc'
fs = sorted(EXPORTPATH.rglob('tsample_*'))
mdf_loc = []
for f in fs:
    with f.open('rb') as ts:
        mdf_loc += [pickle.load(ts)]
EXPORTPATH = ROOTPATH / '04_Aggregates' / 'GLM_NErrByVal'
fs = sorted(EXPORTPATH.rglob('tsample_*'))
mdf_val = []
for f in fs:
    with f.open('rb') as ts:
        mdf_val += [pickle.load(ts)]
# %% 
[
    coef_loc, 
    sem_loc, 
    zval_loc, 
    pval_loc
] = np.stack([mdl.summary().tables[1][[
    'Coef.','Std.Err.','z','P>|z|'
]][:-1].values.T for mdl in mdf_loc], -1).astype('float')
padj_loc = multitest.fdrcorrection(pval_loc.flatten())[1].reshape(*pval_loc.shape)
zval_loc[padj_loc >= .05] = 0
efct_loc = mdf_loc[0].summary().tables[1].index.values[:-1]

[
    coef_val, 
    sem_val, 
    zval_val, 
    pval_val
] = np.stack([mdl.summary().tables[1][[
    'Coef.','Std.Err.','z','P>|z|'
]][:-1].values.T for mdl in mdf_val], -1).astype('float')
padj_val = multitest.fdrcorrection(pval_val.flatten())[1].reshape(*pval_val.shape)
zval_val[padj_val >= .05] = 0
efct_val = mdf_val[0].summary().tables[1].index.values[:-1]
# # %% compute coefficients
# idx_crit = np.where(p_adj[:,-1] < .05)
# agg_mdl_par, agg_mdl_cov = np.stack([
#     mdl.summary().tables[1].values[:-1].astype('float')[:,:2] 
#     for mdl in np.array(mdf)[idx_crit]
# ]).T
# av_mdl_par = agg_mdl_par.mean(1)
# agg_mdl_par[p_adj[idx_crit].T >= .05] = 0
# mdl_par = agg_mdl_par.mean(1)
# mdl_cov = np.stack([
#     mdl.cov_params().values[:-1,:-1] 
#     for mdl in np.array(mdf)[idx_crit]
# ]).mean(0)
# idx_par = np.array([
#     [1, 0, 0, 0, 0, 0, 0, 0], # Low
#     [1, 1, 0, 0, 0, 0, 0, 0], # High

#     [1, 0, 1, 0, 0, 0, 0, 0], # LowSin
#     [1, 1, 1, 1, 0, 0, 0, 0], # HighSin

#     [1, 0, 0, 0, 1, 0, 0, 0], # LowCos
#     [1, 1, 0, 0, 1, 1, 0, 0], # HighCos

#     [1, 0, 1, 0, 1, 0, 1, 0], # LowSinCos
#     [1, 1, 1, 1, 1, 1, 1, 1]  # HighSinCos
# ]).astype('bool')
# pred_cov = np.stack(np.split(np.array([
#     np.sqrt(av_mdl_par[idx][None] @ mdl_cov[idx,:][:, idx] @ av_mdl_par[idx][:, None]).flatten() 
#     for idx in idx_par
# ]), 2))
# pred_coef = np.stack(np.split(np.array([mdl_par[idx].sum() for idx in idx_par]), 2))
#===============================================================================
# %% plot TUNING analyses
#===============================================================================
sbn.set_style('ticks')
# sbn.set_palette('husl', 75, desat = .60)
fsmall, fmed, flarge = [8, 10, 11]
fsmall, fmed, flarge = [10, 11, 11]
plt.rc('font', size=fmed)          # controls default text sizes
plt.rc('font', family='Dejavu Sans')
plt.rc('axes', facecolor=(0, 0, 0, 0))
plt.rc('axes', titlesize=fsmall)     # fontsize of the axes title
plt.rc('axes', labelsize=fsmall)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=fsmall)    # fontsize of the tick labels
plt.rc('ytick', labelsize=fsmall)    # fontsize of the tick labels
plt.rc('legend', fontsize=fsmall)    # legend fontsize
plt.rc('figure', titlesize=fmed)

# %% plot similarities
fig = plt.figure(figsize = (6.6, 6.6))

for stim in range(12):
    axSTIM = plt.subplot2grid(
        (30,32),
        ((stim // 6) * 5 , (stim % 6) * 5),
        rowspan = 5,
        colspan = 5,
        fig = fig
    )
    im = axSTIM.imshow(
        gavSIM[stim], aspect = 'auto', cmap = 'viridis',
        vmin = -.004, vmax = .004,
        origin = 'lower'
    )
    for ln in cntr_sim[stim]:
        axSTIM.plot(
            ln[:, 1], ln[:, 0],
            lw = 1, color = 'white'
        )
    axSTIM.text(
        250, 14, ([12] + list(range(1,12)))[stim],
        fontsize = fsmall, transform = axSTIM.transData,
        backgroundcolor = 'white',
        ha = 'right', va = 'top',
        bbox = dict(fc = 'white', boxstyle = 'round', alpha = .7)
    )
for idx_ax, val_ax in enumerate(fig.axes):
    val_ax.tick_params('x', direction = 'in', length = 3)
    val_ax.set_xticks(time_as_index(np.linspace(0,.8,2)))
    if idx_ax in [0,6]:
        val_ax.tick_params('y', direction = 'in', length = 3)
        val_ax.set_yticks([0,7.5,15])
        val_ax.set_yticklabels([r"-$\pi$",r"$ 0$",r"$\pi$"])
        if idx_ax == 0:
            val_ax.set_ylabel('Bin distance')
            val_ax.yaxis.set_label_coords(-.35, -.1, transform = val_ax.transAxes)
    else:
        val_ax.yaxis.set_visible(False)
    if idx_ax > 5:
        val_ax.set_xticklabels(['0.0', '0.8'])
    else: 
        val_ax.set_xticklabels([''] * 2)

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
axCBar.text(-.25,1.1, r'$\times10^{-3}$', fontsize = 8, transform = axCBar.transAxes)

fig.subplots_adjust(wspace = .5, hspace = .5)
#%% plot LME t-values
axMLE_loc = plt.subplot2grid(
    (30,32), 
    (10, 0),
    rowspan = 10,
    colspan = 30,
    fig = fig
)
im = axMLE_loc.imshow(
    zval_loc[[0,1,2,4,3,5,6,7]], 
    aspect = 'auto', cmap = 'coolwarm', 
    vmin = -5, vmax = 5
)
ylbls = [
    'Const.', 
    'Noise',
    'Sine',
    'Cosine', 
    r'N$\times$S',
    r'N$\times$C',
    r'S$\times$C',
    r'N$\times$S$\times$C'
]
axMLE_loc.set_yticks(range(len(ylbls)))
axMLE_loc.set_yticklabels(ylbls)
axMLE_loc.set_xticks(time_as_index(np.linspace(0,.8,6)))
axMLE_loc.set_xticklabels([''] * 6)
axMLE_loc.tick_params(direction = 'in', length = 3)

axMLE_val = plt.subplot2grid(
    (30,32), 
    (20, 0),
    rowspan = 10,
    colspan = 30,
    fig = fig
)
im = axMLE_val.imshow(
    zval_val[[0,2,1,4,3,6,5,7]], 
    aspect = 'auto', cmap = 'coolwarm', 
    vmin = -5, vmax = 5
)
ylbls = [
    'Const.', 
    'Noise',
    'Sign',
    'Value', 
    r'N$\times$S',
    r'N$\times$V',
    r'S$\times$V',
    r'N$\times$S$\times$V'
]
axMLE_val.set_yticks(range(len(ylbls)))
axMLE_val.set_yticklabels(ylbls)
axMLE_val.set_xticks(time_as_index(np.linspace(0,.8,6)))
axMLE_val.set_xticklabels([
    f"{val_lbl:.1f}"
    if idx_lbl < 4
    else f"{val_lbl:.1f}" + " s"
    for idx_lbl, val_lbl in enumerate(np.linspace(0,.8,6))
])
axMLE_val.tick_params(direction = 'in', length = 3)

axCBar = plt.subplot2grid(
    (30,32), 
    (17, 31),
    rowspan = 6,
    colspan = 1,
    fig = fig
)
plt.colorbar(mappable = im, orientation = 'vertical', cax = axCBar, shrink = .05)
axCBar.set_yticks([-5,0,5])
axCBar.set_yticklabels(['-5', ' 0 $z$', ' 5'])
axCBar.tick_params(direction = 'out', length = 3)
# %%
fig.subplots_adjust(.1,.03,.92,.98, hspace = 5, wspace = .5)
fig.savefig(ROOTPATH / '05_Exports' / 'gavFID_perStim.png', dpi = 600)
