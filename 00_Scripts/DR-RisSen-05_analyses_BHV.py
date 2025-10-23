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
# TODO: plot performance in the perceptual task and in the gamble task
# TODO: code signed error magnitudes
# TODO: mixture distribution modelling of the perceptual task
# TODO: ALD distribution parameter estimation
# NOTE: drop sub-02 and sub-35 due to insufficient data
#===============================================================================
# %% set up plotting
#===============================================================================
import matplotlib as mpl
mpl.use('qt5agg')
import matplotlib.pyplot as plt
plt.ion()
import seaborn as sbn
sbn.set_theme('paper')
#===============================================================================
# %% import libraries
#===============================================================================
import json
import mmodel as mm
import numpy as np
import scipy.stats as sps
import pandas as pd
from pathlib import Path
# import pingouin as pg
from scipy.stats import laplace_asymmetric, binom
from scipy.optimize import minimize, curve_fit
# import statsmodels.api as sm
import statsmodels.formula.api as smf
#===============================================================================
# %% define functions
#===============================================================================
def nll_CAL(params, x):
    '''
    Loss function for the MLE of kappa, location, and scale of the cumulatice asymmetric laplace

    Takes Argurments:
        params: Starting guess paramters for the MLE fitting process (kappa, loc, scale, <- in this order)
        x: A dataframe which houses columns named ['valdiff', 'play'] for the purposes of this experiment
    
    returns a single value which is the sum negative log likelihood for that subject over each trial
    '''
    kappa, loc, scale = params
    # Model the probability of accepting as the CDF evaluated at value/orientation difference 
    prob_accept = laplace_asymmetric.cdf(x['tstPayoff'], kappa=kappa, loc=loc, scale=scale)
    #Jitter for log evals at 1 and 0
    prob_accept_rc = np.where(prob_accept == 1, 1 - np.finfo(float).eps,
                            np.where(prob_accept == 0, np.finfo(float).eps, 
                                    prob_accept))
    
    #Cacluate the log-likelihood per trial
    llt = x['play'] * np.log(prob_accept_rc) + (1 - x['play']) * np.log(1 - prob_accept_rc)
    
    #Return the sum of the per trial log likelihoods
    return -sum(llt)
#===============================================================================
# %% get the data
#===============================================================================
ROOTPATH = Path().cwd().parent
bhvFiles = sorted(ROOTPATH.rglob('*Rawdata/**/*task-MAIN_bhv*'))
agg_bhv = pd.concat([
    pd.read_csv(dfile, sep = '\t') 
    for dfile in bhvFiles
    if dfile.stem.split('_')[0] not in ['sub-02', 'sub-35']
])
agg_bhv['missing'] = agg_bhv['response'].isna().astype('int')
# drop missing trials
agg_bhv = agg_bhv.loc[agg_bhv['missing'] == 0]

val_bhv, per_bhv = [
    agg_bhv.loc[agg_bhv['trialType'] == ttype].copy()
    for ttype in [0,1]
]
# %% analyse demographics
gui_paths = sorted(ROOTPATH.rglob("gui_info*"))
sub_info = []
for gpath in gui_paths:
    with gpath.open('r') as gp:
        sub_info += [pd.DataFrame(dict([
            (key, [val]) 
            for key, val in json.load(gp).items()
        ]))]
sub_info = pd.concat(sub_info)
#===============================================================================
# %% analysing VAL trials
#===============================================================================
val_bhv['play'] = (val_bhv['testStim'] == val_bhv['response']).astype('int')

val_bhv['tstPayoff'] = val_bhv.apply(lambda x: x[f"{x['testStim']}Payoff"], axis = 1)

val_bhv['oriBin'] = pd.cut(
    val_bhv['deltaThetaOri'].values, 
    bins = np.histogram(val_bhv['deltaThetaOri'], bins=7)[1].round(0),
    labels = np.linspace(-45,45,7).round(0)
)

val_bhv['valBin'] = pd.cut(
    val_bhv['tstPayoff'], 
    bins = np.histogram(val_bhv['tstPayoff'], range=(-12,12),bins=7)[1].round(0), 
    labels = np.linspace(-10, 10, 7).round(0)
)

av_val = val_bhv.groupby([
    'sNo', 'thetaVar', 'valBin'
])['play'].mean().reset_index(name = 'pPlay')
av_val = av_val.merge(
    av_val.groupby(['sNo'])['pPlay'].mean().reset_index(),
    on = ['sNo'],
    suffixes = ['', '_sub']
)
av_val['pPlay_n'] = (
    av_val['pPlay'] 
    - av_val['pPlay_sub']
    + av_val['pPlay'].mean()
)
gav_val = av_val.groupby([
    'thetaVar', 'valBin'
])['pPlay'].mean().reset_index()
ncond = gav_val['thetaVar'].unique().size * gav_val['valBin'].unique().size
nsub = av_val['sNo'].unique().size
gav_val['sem_n'] = np.sqrt(
    av_val.groupby([
        'thetaVar', 'valBin'
    ])['pPlay_n'].var().reset_index()['pPlay_n'] 
    * (ncond / (ncond - 1))
    / (nsub - 1)
)
gav_val['sem'] = np.sqrt(
    av_val.groupby([
        'thetaVar', 'valBin'
    ])['pPlay'].var().reset_index()['pPlay']
    / (nsub - 1)
)
# # %% fit cummulative asymmetric laplacian distribution
# res_df = pd.DataFrame(columns=['sNo', 'convergence', 'thetaVar', 'kappa', 'loc', 'scale', 'LogLike', 'MSE'])
# est_df = pd.DataFrame(columns=['sNo', 'thetaVar','x', 'yEst'])

# for sno in val_bhv['sNo'].unique():
#     sub = val_bhv.loc[val_bhv['sNo'] == sno].copy()
#     for var in [0.075, 0.3]:
#         tmp = sub.loc[sub['thetaVar'] == var][['thetaVar','tstPayoff','play','valBin']].copy()

#         # Inital guesses for params
#         init_parms = [1,1,1]
#         bounds = ([(0.001, np.inf), (-np.inf, np.inf), (0.001, np.inf)])

#         result = minimize(nll_CAL, init_parms, args=(tmp), method='L-BFGS-B', bounds=bounds, options = {'disp': True, 'ftol': 1e-9})
#         k, l, s = result.x
#         log_like = result.fun
#         convergence = result.status
#         vd_cond = np.sort(tmp['tstPayoff'])
#         vd_condUni = np.sort(tmp['valBin'].unique())
        
#         # Generate estimated CDF using estimated parameters (and also mse)
#         xs = np.linspace(-10, 10, 150)
#         yEsts = laplace_asymmetric.cdf(xs, kappa=k, loc=l, scale=s)

#         median = laplace_asymmetric.median(kappa=k, loc=l, scale=s)
        
#         # append results to DataFrame
#         res_df = pd.concat([res_df, pd.DataFrame({
#             'sNo': [sno],
#             'convergence': [convergence], 
#             'thetaVar': [var], 
#             'kappa': [k], 
#             'loc': [l],
#             'scale': [s],
#             'median': [median],
#             'LogLike': [log_like]
#         })], ignore_index=True)
#         # print(result)
        
#         est_df = pd.concat([est_df, pd.DataFrame({
#             'sNo': [sno]*150,
#             'thetaVar': [var]*150,
#             'x':xs,
#             'yEst': yEsts
#         })], ignore_index=True)
        
#         # # Plot results
#         # y_obs = av_val.loc[(av_val['sNo'] == sno) & (av_val['thetaVar'] == var), 'pPlay'].copy()
#         # plt.figure()
#         # plt.plot(vd_condUni, y_obs, '.', label='Observed data')
#         # plt.plot(xs, yEsts, label='Estimated CDF')
#         # plt.ylim(-0.2,1.2)
#         # plt.title(f'sno: {sno} | thetavar: {var}')
#         # plt.legend()
# # %%
# [
#     res_df['mean'], 
#     res_df['var'], 
#     res_df['skew'], 
#     res_df['kurtosis']
# ] = laplace_asymmetric.stats(
#     kappa = res_df['kappa'], 
#     loc = res_df['loc'], 
#     scale = res_df['scale'], 
#     moments='mvsk'
# )

# res_df['logk'] = np.log10(res_df['kappa'])
# [
#     res_df['logk_z'],
#     res_df['loc_z'],
#     res_df['scale_z']
# ] = res_df[['logk','loc','scale']].apply(lambda x: sps.zscore(x)).values.T
# res_df = res_df.merge(
#     res_df.groupby('sNo')[['logk_z','loc_z','scale_z']].mean().reset_index(),
#     on = ['sNo'],
#     suffixes = ['', '_sub']
# )
# [
#     res_df['logk_n'],
#     res_df['loc_n'],
#     res_df['scale_n']
# ] = (
#     res_df[['logk_z', 'loc_z', 'scale_z']].values 
#     - res_df[['logk_z_sub', 'loc_z_sub', 'scale_z_sub']].values
#     + res_df[['logk_z','loc_z', 'scale_z']].mean().values[None]
# ).T
# res_df['thetaVar_fac'] = res_df['thetaVar'].astype('category')
# res_df['x'] = 1
# res_df['loc_n_inv'] = -res_df['loc_n']
# res_df['scale_n_inv'] = -res_df['scale_n']
# # save fitted data
# SAVEPATH = ROOTPATH / '04_Aggregates'
# res_df.to_csv(SAVEPATH  / 'agg_ald_res.tsv.gz', sep = '\t')
# est_df.to_csv(SAVEPATH  / 'agg_ald_est.tsv.gz', sep = '\t')
# %%
SAVEPATH = ROOTPATH / '04_Aggregates'
res_df = pd.read_csv(SAVEPATH  / 'agg_ald_res.tsv.gz', sep = '\t')
est_df = pd.read_csv(SAVEPATH  / 'agg_ald_est.tsv.gz', sep = '\t')

gav_adl = res_df.groupby(['thetaVar'])[['logk_z','loc_z','scale_z']].mean().reset_index()
t_var_adl = dict([
    (
        dv,
        sps.ttest_rel(
            *[
                res_df.loc[res_df['thetaVar'] == tvar, dv].values
                for tvar in res_df['thetaVar'].unique()
            ]
        ) 
    )
    for dv in ['logk', 'loc', 'scale']
])
t_var_adl_null = dict([
    (
        dv,
        dict([
            (
                tvar,
                sps.ttest_1samp(
                    res_df.loc[res_df['thetaVar'] == tvar, dv].values,
                    0
                )
            )
            for tvar in res_df['thetaVar'].unique()
        ]) 
    )
    for dv in ['logk', 'loc']
])
t_var_adl_z = dict([
    (
        dv,
        sps.ttest_rel(
            *[
                res_df.loc[res_df['thetaVar'] == tvar, dv].values
                for tvar in res_df['thetaVar'].unique()
            ]
        ) 
    )
    for dv in ['logk_z', 'loc_z', 'scale_z']
])
#===============================================================================
# %% analysing PERception trials
#===============================================================================
per_bhv['rspRad'] = np.angle(np.exp((
    per_bhv['response'].astype('float')  * np.pi / 90
) * 1j))
per_bhv['tarRad'] = np.angle(np.exp((per_bhv['tstOri']  * np.pi / 90) * 1j))
# what is the loss orientation [0 or pi] and, by implication, gain orientation
per_bhv['lossRad'] = np.array([0, np.pi])[per_bhv['sNo'] % 2]
per_bhv['errRad'] = np.angle(
    np.exp(per_bhv['tarRad'] * 1j) 
    / np.exp(per_bhv['rspRad'] * 1j)
)
per_bhv['sftRad'] = np.angle(
    np.exp(per_bhv['tarRad'] * 1j) 
    / np.exp(per_bhv['lossRad'] * 1j)
) 
idx_flip = per_bhv['sftRad'] > 0
per_bhv.loc[idx_flip, 'errRad'] *= -1
av_per = per_bhv.groupby([
    'sNo', 
    'thetaVar'
])['errRad'].apply(mm.cmean).reset_index().rename(columns = dict(
    errRad = 'cm_err'
)).merge(per_bhv.groupby([
    'sNo', 
    'thetaVar'
])['errRad'].apply(mm.cstd).reset_index().rename(columns = dict(
    errRad = 'cstd_err'
)), on = ['sNo', 'thetaVar'])
per_bhv['absErr'] = per_bhv['errRad'].abs()
av_bhv = per_bhv.groupby(['sNo','thetaVar'])['absErr'].mean().reset_index()
gav_bhv = av_bhv.groupby(['thetaVar']).mean().reset_index()
sps.ttest_rel(*[
    av_bhv.loc[av_bhv['thetaVar'] == var, 'absErr']
    for var in [.075, .3]
])
# gav_per = av_per.groupby(
#     ['thetaVar']
# )['cm_err'].apply(mm.cmean).reset_index().merge(
#     av_per.groupby(['thetaVar'])['cstd_err'].mean().reset_index(),
#     on = ['thetaVar']
# )
# t_cm = sps.ttest_rel(*[
#     av_per.loc[av_per['thetaVar'] == var, 'cm_err'].values
#     for var in av_per['thetaVar'].unique()
# ])
# t_cstd = sps.ttest_rel(*[
#     av_per.loc[av_per['thetaVar'] == var, 'cstd_err'].values
#     for var in av_per['thetaVar'].unique()
# ])
# t_cm_null = dict([
#     (
#         var,
#         sps.ttest_1samp(av_per.loc[av_per['thetaVar'] == var, 'cm_err'].values, 0)
#     )
#     for var in av_per['thetaVar'].unique()
# ])
# t_cstd_null = dict([
#     (
#         var,
#         sps.ttest_1samp(av_per.loc[av_per['thetaVar'] == var, 'cstd_err'].values, 0)
#     )
#     for var in av_per['thetaVar'].unique()
# ])
# # %% fit Mixture distribution model with shift in target distribution
# mmfit = [   
#     [
#         sub, 
#         var, 
#         mm.mmfit(
#             per_bhv.loc[
#                 (per_bhv['sNo'] == sub)
#                 & (per_bhv['thetaVar'] == var),
#                 'errRad'
#             ],
#             random_seed = sub
#         )        
#     ]
#     for sub in per_bhv['sNo'].unique()
#     for var in per_bhv['thetaVar'].unique()
# ]
# # %% compile params into one data frame
# sub, var, mm_par = zip(*mmfit)
# B, LL = zip(*mm_par)
# mu, K, Pt = zip(*B)
# mm_df = pd.DataFrame(dict(
#     sNo = sub,
#     thetaVar = var,
#     mu = mu,
#     K = K,
#     Pt = Pt,
#     LL = LL
# ))
# SAVEPATH = ROOTPATH / '04_Aggregates' / 'agg_mmodel.tsv.gz'
# mm_df.to_csv(SAVEPATH, sep = '\t', index = False)
# %%
SAVEPATH = ROOTPATH / '04_Aggregates' / 'agg_mmodel.tsv.gz'
mm_df = pd.read_csv(SAVEPATH, sep = '\t')
mm_df['Pg'] = 1 - mm_df['Pt']
[
    mm_df['mu_z'],
    mm_df['K_z'],
    mm_df['Pg_z']
] = mm_df[['mu','K','Pg']].apply(sps.zscore, axis = 0).values.T
mm_df = mm_df.merge(
    mm_df.groupby(['sNo'])[['mu_z','K_z','Pg_z']].mean().reset_index(),
    on = ['sNo'],
    suffixes = ['', '_sub']
)
[
    mm_df['mu_n'],
    mm_df['K_n'],
    mm_df['Pg_n']
] = (
    mm_df[['mu_z','K_z','Pg_z']].values 
    - mm_df[['mu_z_sub','K_z_sub','Pg_z_sub']].values
    + mm_df[['mu_z','K_z','Pg_z']].values.mean(0)[None]
).T
gav_mm = mm_df.groupby(['thetaVar'])[['mu','K', 'Pg']].mean().reset_index()
t_var_mm = dict([
    dv,
    sps.ttest_rel(*[
        mm_df.loc[mm_df['thetaVar'] == var, dv].values
        for var in mm_df['thetaVar'].unique()
    ])
] for dv in ['mu', 'K', 'Pt'])
t_var_mm_null = dict([
    (
        dv,
        dict([
            (
                tvar,
                sps.ttest_1samp(
                    mm_df.loc[res_df['thetaVar'] == tvar, dv].values,
                    0
                )
            )
            for tvar in mm_df['thetaVar'].unique()
        ]) 
    )
    for dv in ['mu']
])
#===============================================================================
# %% plotting behavioural results
#===============================================================================
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
cols = np.array(sbn.color_palette('muted'))[[2,4]]
# %% plot perceptual task

fig = plt.figure(figsize = (6.6, 4.4))

axERR = plt.subplot2grid(
    (2,30), 
    (0, 0),
    rowspan = 1,
    colspan = 15,
    fig = fig
)
bins = np.linspace(-np.pi, np.pi, 12)
bin_width = np.unique((bins[1:] - bins[:-1]).round(2))[0]
mids = pd.cut(per_bhv['errRad'].values, bins).categories.mid.values
per_bhv['errBin'] = pd.cut(per_bhv['errRad'].values, bins)
per_bhv['errBin_fac'] = per_bhv['errBin'].cat.codes.astype('category')
av_bin = per_bhv.groupby([
    'sNo', 'thetaVar', 'errBin_fac'
]).size().reset_index().rename(columns = {0 : 'bin_n'})
av_bin = av_bin.merge(
    per_bhv.groupby([
        'sNo', 'thetaVar'
    ]).size().reset_index().rename(columns = {0 : 'sub_n'}),
    on = ['sNo', 'thetaVar']
)
av_bin['bin_p'] = av_bin['bin_n'] / av_bin['sub_n']
gav_bin = av_bin.groupby([
    'thetaVar',
    'errBin_fac'
])['bin_p'].mean().reset_index()

for idx_var, val_var in enumerate(gav_bin['thetaVar'].unique()):
    plt.bar(
        mids,
        gav_bin.loc[gav_bin['thetaVar'] == val_var, 'bin_p'],
        width = bin_width * .9,
        color = cols[idx_var],
        edgecolor = 'black',
        alpha = .3,
        label = ['Low Noise', 'High'][idx_var]
    )

axERR.set_xticks([-np.pi, 0, np.pi])
axERR.set_xticklabels(['$-\pi$', '$0$', '$\pi$'])
axERR.set_xlabel('Error bin')
axERR.spines['bottom'].set_bounds(-np.pi, np.pi)
axERR.set_ylim(0,.5)
axERR.set_ylabel('Proportion of trials')
axERR.tick_params(direction = 'in', length = 3)
for spine in ('top','right'):
    axERR.spines[spine].set_visible(False)
axERR.text(
    -np.pi/2, .5, 'Loss',
    ha = 'center', va = 'top', fontsize = 10
)
axERR.text(
    np.pi/2, .5, 'Gain',
    ha = 'center', va = 'top', fontsize = 10
)
# plotting parameters
sbn.set_palette(cols)
for idx_dv, val_dv in enumerate(['Pg_n', 'K_n', 'mu_n']):
    ax = plt.subplot2grid(
        (2,30), 
        (0, idx_dv * 5 + 15),
        rowspan = 1,
        colspan = 5,
        fig = fig
    )
    sbn.violinplot(
        data = mm_df,
        y = val_dv,
        x = [1] * mm_df.shape[0],
        hue = 'thetaVar',
        colors = cols,
        linewidth = .75,
        cut = 0, inner = None,
        split = True,
        legend = False
    )
    ax.legend().set_visible(False)

for idx_ax, val_ax in enumerate(fig.axes[1:]):
    
    val_ax.set_title(['Guessing', 'Precision', 'Bias'][idx_ax])
    val_ax.tick_params(direction = 'in', length = 3)
    val_ax.xaxis.set_visible(False)
    
    val_ax.set_ylim(-2,2)
    val_ax.set_yticklabels(['-2','-1', ' 0', ' 1', ' 2'])
    val_ax.set_ylabel('z-score')
    val_ax.yaxis.set_label_position('right')
    val_ax.yaxis.set_ticks_position('right')
    
    for spine in ['top','left','bottom']:
        val_ax.spines[spine].set_visible(False)
    if idx_ax < 2:
        val_ax.yaxis.set_visible(False)
        val_ax.spines['right'].set_visible(False)

ax = fig.axes[2]
hndls, _ = ax.get_legend_handles_labels()
for idx_hndl, val_hndl in enumerate(hndls):
    val_hndl.set_facecolor(cols[idx_hndl])
ax.legend(
    hndls, ['Low Noise', 'High'], 
    ncol = 2,
    frameon = False,
    loc = 'upper center',
    bbox_to_anchor = (.5,0),
    bbox_transform = ax.transAxes
)

# ALD plots

axALD = plt.subplot2grid(
    (2,30), 
    (1, 0),
    rowspan = 1,
    colspan = 15,
    fig = fig
)

for idx_var, val_var in enumerate(gav_val['thetaVar'].unique()):
    idx_ss = gav_val['thetaVar'] == val_var
    axALD.errorbar(
        gav_val.loc[idx_ss, 'valBin'].astype('float') + [-.25,.25][idx_var],
        gav_val.loc[idx_ss, 'pPlay'],
        fmt = ['v','^'][idx_var],
        markersize = 6,
        label = ['Low Noise', 'High'][idx_var],
        yerr = gav_val.loc[idx_ss, 'sem']
    )
    fit_df = est_df.loc[est_df['thetaVar'] == val_var].groupby(
        ['sNo','x']
    )['yEst'].mean().reset_index().groupby('x')['yEst'].mean().reset_index()
    axALD.plot(
        fit_df['x'], fit_df['yEst'],
        color = cols[idx_var],
        lw = 1.5,
        label = ['Low Noise', 'High'][idx_var]
    )

axALD.tick_params(direction = 'in', length = 3)
axALD.set_ylim(-0.025,1.025)
axALD.set_yticks(np.linspace(0,1,6))
axALD.spines['left'].set_bounds(0,1)
axALD.set_ylabel('Proportion of risky choices')
axALD.spines['bottom'].set_bounds(-10,10)
axALD.set_xlabel('Value of risky choice')
for spine in ['top','right']:
    axALD.spines[spine].set_visible(False)
hndls, lbls = axALD.get_legend_handles_labels()
axALD.legend(hndls[:2], lbls[:2], frameon = False, loc = 'upper left')
# plotting parameters
sbn.set_palette(cols)
for idx_dv, val_dv in enumerate(['scale_n_inv','loc_n_inv', 'logk_n']):
    ax = plt.subplot2grid(
        (2,30), 
        (1, idx_dv * 5 + 15),
        rowspan = 1,
        colspan = 5,
        fig = fig
    )
    sbn.violinplot(
        data = res_df,
        y = val_dv,
        x = [1] * res_df.shape[0],
        hue = 'thetaVar',
        colors = cols,
        linewidth = .75,
        cut = 0,
        split = True, inner = None,
        legend = False
    )
    ax.legend().set_visible(False)

for idx_ax, val_ax in enumerate(fig.axes[5:]):
    
    val_ax.set_title([
        'Value\n sensitivity', 'Risk\n aversion', 'Loss\n aversion'
    ][idx_ax])
    val_ax.tick_params(direction = 'in', length = 3)
    val_ax.xaxis.set_visible(False)
    
    val_ax.set_ylim(-2,2)
    val_ax.set_yticklabels(['-2','-1', ' 0', ' 1', ' 2'])
    val_ax.set_ylabel('z-score')
    val_ax.yaxis.set_label_position('right')
    val_ax.yaxis.set_ticks_position('right')
    
    for spine in ['top','left','bottom']:
        val_ax.spines[spine].set_visible(False)
    if idx_ax < 2:
        val_ax.yaxis.set_visible(False)
        val_ax.spines['right'].set_visible(False)


fig.subplots_adjust(.075,.09,.92,.95, hspace = .5) 
fig.savefig(ROOTPATH / '05_Exports' / 'gavBHV_v01.png', dpi = 600)
# %%
