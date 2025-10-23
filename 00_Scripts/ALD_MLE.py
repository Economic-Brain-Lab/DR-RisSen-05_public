#%% import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

# For curve fitting
from scipy.stats import laplace_asymmetric, binom
from scipy.optimize import minimize, curve_fit
# For plotting
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import seaborn as sns
# For stats
import pingouin as pg
from statsmodels.stats.weightstats import ztest as ztest
from statsmodels.stats.weightstats import zconfint as zconfint
import statsmodels.api as sm
import statsmodels.formula.api as smf
#%% Load and concatonate all of the data and segment it into two dataframes
# val_data = Value Based Trial Data
# per_data = Orientation Reproduction (Perceptual) Trial Data
# paths = sorted(glob.glob(
#     'C:\\Phd\\Experiments\\DR-RisSen-05\\00_SourceData\\BHV\\*MAIN*.tsv', recursive=True
# ))
# print('Data Paths:', '\n', f'{paths}')

# UQ Laptop
# paths = sorted(glob.glob(
#     'C:\\PhD\\Experiments\\DR-RisSen-05-Pilot\\DR-RisSen-05\\01_RawData\\*MAIN*.tsv', recursive=True
# ))
# print('Data Paths:', '\n', f'{paths}')

paths = sorted(glob.glob(
    'D:\\DR-RisSen-05\\\BHV2\\*MAIN*.tsv', recursive=True
))
print('Data Paths:', '\n', f'{paths}')

# List of participants to exclude
exclude_participants = ['sub-35', 'sub-02', 'sub-03']

# Filter out paths of participants to exclude
filtered_paths = [path for path in paths if not any(exclude in path for exclude in exclude_participants)]


# Concatonate the data
all_df = []
for idx, path in enumerate(filtered_paths):
    tmp_df = pd.read_csv(
        path,
        sep = '\t',
        na_values = 'n/a'
    )
    print(f' Participant {idx} \n', f'DF shape: {tmp_df.shape}')
    all_df += [tmp_df]

all_df = pd.concat(all_df, ignore_index=True)
# split into val_data and per_data
val_data = all_df.loc[all_df['trialType'] == 0].copy()
print(val_data.shape)
# missing data
print('Missing value data: \n', 
    f"{val_data['response'].isnull().groupby(val_data['sNo']).sum().reset_index(name='Missing')}")

val_data = val_data[val_data['response'].notna()] #Remove NA

per_data = all_df.loc[all_df['trialType'] == 1].copy()
print(per_data.shape)
# missing data 
print('Missing value data: \n', 
    f"{per_data['response'].isnull().groupby(per_data['sNo']).sum().reset_index(name='Missing')}")

#%% Create columns needed for analysing value trials
val_data.loc[(val_data['testStim'] == val_data['response']), 'play'] = 1 # Chose to play
val_data.loc[(val_data['testStim'] != val_data['response']), 'play'] = 0 # Chose to not play

val_data.loc[(val_data['testStim'] == 'left', 'tstPayoff')] = val_data['leftPayoff']
val_data.loc[(val_data['testStim'] == 'right', 'tstPayoff')] = val_data['rightPayoff']

oriBins = np.histogram(val_data['deltaThetaOri'], bins=7)[1].round(0)
val_data['oriBin'] = pd.cut(val_data['deltaThetaOri'], bins=oriBins, labels=np.linspace(-45, 45, 7))

valBins = np.histogram(val_data['tstPayoff'], range=(-12,12),bins=7)[1].round(0)
val_data['valBin'] = pd.cut(val_data['tstPayoff'], bins=valBins, labels=np.linspace(-10, 10, 7).round(0))

# Grand Average dataframe per participant per condition for curve fitting
gav_data = val_data.groupby(['sNo', 'thetaVar', 'valBin'])['play'].mean().reset_index(name = 'propPlay')

#%%
############################
# MLE WITH BERNOULLI 
# FUNCTION DEFINTIONS -> CURVE FITTING -> PLOTTING
############################

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

# create an empty DataFrame with desired columns
results_df = pd.DataFrame(columns=['sNo', 'convergence', 'thetaVar', 'kappa', 'loc', 'scale', 'LogLike', 'MSE'])
estimated_df = pd.DataFrame(columns=['sNo', 'thetaVar','x', 'yEst'])

for sno in val_data['sNo'].unique():
    sub = val_data.loc[val_data['sNo'] == sno].copy()
    for var in [0.075, 0.3]:
        tmp = sub.loc[sub['thetaVar'] == var][['thetaVar','tstPayoff','play','valBin']].copy()

        # Inital guesses for params
        init_parms = [1,1,1]
        bounds = ([(0.001, np.inf), (-np.inf, np.inf), (0.001, np.inf)])

        result = minimize(nll_CAL, init_parms, args=(tmp), method='L-BFGS-B', bounds=bounds, options = {'disp': True, 'ftol': 1e-9})
        k, l, s = result.x
        log_like = result.fun
        convergence = result.status
        vd_cond = np.sort(tmp['tstPayoff'])
        vd_condUni = np.sort(tmp['valBin'].unique())
        
        # Generate estimated CDF using estimated parameters (and also mse)
        xs = np.linspace(-10, 10, 150)
        yEsts = laplace_asymmetric.cdf(xs, kappa=k, loc=l, scale=s)

        median = laplace_asymmetric.median(kappa=k, loc=l, scale=s)
        # append results to DataFrame
        results_df = results_df.append({'sNo': sno,
            'convergence': convergence, 
            'thetaVar': var, 
            'kappa': k, 
            'loc': l,
            'scale': s,
            'median': median,
            'LogLike': log_like,}, ignore_index=True)
        print(result)

        y_obs = gav_data.loc[(gav_data['sNo'] == sno) & (gav_data['thetaVar'] == var), 'propPlay'].copy()
        
        estimated_df = pd.concat([estimated_df, pd.DataFrame(
            {
            'sNo': [sno]*150,
            'thetaVar': [var]*150,
            'x':xs,
            'yEst': yEsts})], ignore_index=True)
        # Plot results
        plt.figure()
        plt.plot(vd_condUni, y_obs, '.', label='Observed data')
        plt.plot(xs, yEsts, label='Estimated CDF')
        plt.ylim(-0.2,1.2)
        plt.title(f'sno: {sno} | thetavar: {var}')
        plt.legend()

results_df['mean'], results_df['var'], results_df['skew'], results_df['kurtosis'] = laplace_asymmetric.stats(kappa = results_df['kappa'], 
    loc = results_df['loc'], 
    scale = results_df['scale'], 
    moments='mvsk')

results_df['logk'] = np.log(results_df['kappa'])
#%% Run stats over fitted params
#kappa
kappaMix = smf.mixedlm("logk ~ C(thetaVar)", data=results_df, groups=results_df['sNo'])
kappaMixFit = kappaMix.fit()
print(kappaMixFit.summary())

#%%
# Location
locMix = smf.mixedlm("median ~ C(thetaVar)", data=results_df, groups=results_df['sNo'])
locMixFit = locMix.fit()
print(locMixFit.summary())

#%%
# Scale
scaleMix = smf.mixedlm("scale ~ C(thetaVar)", data=results_df, groups=results_df['sNo'])
scaleMixFit = scaleMix.fit()
print(scaleMixFit.summary())

# %% Get the descriptives for plots
((results_df.groupby(['sNo', 'thetaVar'])['logk'].mean() - 
    results_df.groupby('sNo')['logk'].mean() + 
    results_df['logk'].mean()).reset_index().groupby('thetaVar')['logk'].sem() * np.sqrt(2/1)).reset_index()['logk']

#Get the mean and std of kappa for each perceptual uncertainty
kappaVarStats = results_df.groupby('thetaVar')['logk'].mean().reset_index()
kappaVarStats['kappaSEM'] = ((results_df.groupby(['sNo', 'thetaVar'])['logk'].mean() - 
    results_df.groupby('sNo')['logk'].mean() + 
    results_df['logk'].mean()).reset_index().groupby('thetaVar')['logk'].sem() * np.sqrt(2/1)).reset_index()['logk']

#Get the mean and std of loc for each perceptual uncertainty
locVarStats = results_df.groupby('thetaVar')['loc'].mean().reset_index()
locVarStats['locSEM'] = ((results_df.groupby(['sNo', 'thetaVar'])['loc'].mean() - 
    results_df.groupby('sNo')['loc'].mean() + 
    results_df['loc'].mean()).reset_index().groupby('thetaVar')['loc'].sem() * np.sqrt(2/1)).reset_index()['loc']

#PLotting 
#Get the mean and std of scale for each perceptual uncertainty
scaleVarStats = results_df.groupby('thetaVar')['scale'].mean().reset_index()
scaleVarStats['scaleSEM'] = ((results_df.groupby(['sNo', 'thetaVar'])['scale'].mean() - 
    results_df.groupby('sNo')['scale'].mean() + 
    results_df['scale'].mean()).reset_index().groupby('thetaVar')['scale'].sem() * np.sqrt(2/1)).reset_index()['scale']

#Get the mean and std of scale for each perceptual uncertainty
medianVarStats = results_df.groupby('thetaVar')['median'].mean().reset_index()
medianVarStats['medianSEM'] = ((results_df.groupby(['sNo', 'thetaVar'])['median'].mean() - 
    results_df.groupby('sNo')['median'].mean() + 
    results_df['median'].mean()).reset_index().groupby('thetaVar')['median'].sem() * np.sqrt(2/1)).reset_index()['median']

avg_param = results_df.groupby('thetaVar')[['kappa', 'loc', 'scale']].mean().reset_index()
sno_gav = gav_data.groupby(['thetaVar', 'sNo', 'valBin'])['propPlay'].mean().reset_index()
tgav = sno_gav.groupby(['thetaVar', 'valBin'])['propPlay'].mean().reset_index()

varDict = { 0.075: 'Low',
            0.30: 'High'}
gav_data.replace({"thetaVar": varDict},inplace=True)
estimated_df.replace({"thetaVar": varDict},inplace=True)
tgav.replace({"thetaVar": varDict},inplace=True)



biasBeh = gav_data.groupby(['thetaVar'])['propPlay'].mean().reset_index()
biasBeh['semNorm'] = ((gav_data.groupby(['sNo', 'thetaVar'])['propPlay'].mean() 
    - gav_data.groupby(['sNo'])['propPlay'].mean() 
    +  gav_data['propPlay'].mean()).reset_index().groupby('thetaVar')['propPlay'].sem() * np.sqrt(2/1)).reset_index()['propPlay']


#Split by sign
behLoss = gav_data[gav_data['valBin'] < 0].copy()
behLossStats = behLoss.groupby(['thetaVar'])['propPlay'].mean().reset_index()
behLossStats['semNorm'] = ((behLoss.groupby(['sNo', 'thetaVar'])['propPlay'].mean() 
    - behLoss.groupby(['sNo'])['propPlay'].mean() 
    +  behLoss['propPlay'].mean()).reset_index().groupby('thetaVar')['propPlay'].sem() * np.sqrt(2/1)).reset_index()['propPlay']

behGain = gav_data[gav_data['valBin'] >= 0].copy()
behGainStats = behGain.groupby(['thetaVar'])['propPlay'].mean().reset_index()
behGainStats['semNorm'] = ((behGain.groupby(['sNo', 'thetaVar'])['propPlay'].mean() 
    - behGain.groupby(['sNo'])['propPlay'].mean() 
    +  behGain['propPlay'].mean()).reset_index().groupby('thetaVar')['propPlay'].sem() * np.sqrt(2/1)).reset_index()['propPlay']


curveSEM = ((estimated_df.groupby(['sNo', 'thetaVar','x'])['yEst'].mean() - 
    estimated_df.groupby('sNo')['yEst'].mean() + 
    estimated_df['yEst'].mean()).reset_index().groupby(['thetaVar','x'])['yEst'].sem() * np.sqrt(2/1)).reset_index()


lowSEM = curveSEM.loc[curveSEM['thetaVar'] == 'Low']

highSEM = curveSEM.loc[curveSEM['thetaVar'] == 'High']

#%% saving dataframes as csvs
# val_data.to_csv('04-allValBeh.tsv', sep='\t')
# results_df.to_csv('04-paramEstimate.tsv', sep='\t')

#%% Actually Plot
fig = plt.figure( figsize=(16,9), dpi=600)
# mpl.rcParams['font.size'] = 18

gs = fig.add_gridspec(2,6)

ax = []
ax.append(fig.add_subplot(gs[0, 0:3]))
ax.append(fig.add_subplot(gs[0, 3:6]))
ax.append(fig.add_subplot(gs[1, 0:2]))
ax.append(fig.add_subplot(gs[1, 2:4]))
ax.append(fig.add_subplot(gs[1, 4:6]))
# ax.append(fig.add_subplot(gs[1, 0]))




# Create a colormap object
pastel = sns.color_palette('pastel', as_cmap=True)
# Generate a list of unique colors
colors = [pastel[0], pastel[4], pastel[6]]
#Pastels
colos = ['#BBCC33', '#AAAA00', '#77AADD','#7EC384', '#EEDD88', '#FFAABB', '#99DDFF', pastel[0], '#DDDDDD']

# iterate over each thetaVar and plot a line
lowdf = estimated_df.loc[estimated_df['thetaVar'] == 'Low']
for sub in lowdf['sNo'].unique():
    subdf = lowdf.loc[lowdf['sNo'] == sub]
    ax[0].plot(subdf['x'], subdf['yEst'], '--', color='gray', alpha=0.1)

df = estimated_df.loc[estimated_df['thetaVar'] == 'Low']
dat = tgav[tgav['thetaVar'] == 'Low']
ax[0].plot(dat['valBin'], dat['propPlay'], 'o', color = colos[7])
ax[0].plot(xs, df.groupby('x')['yEst'].mean(), color=colos[7], label='Low')
ax[0].fill_between(xs, y1=df.groupby('x')['yEst'].mean().values + lowSEM['yEst'], y2=df.groupby('x')['yEst'].mean().values - lowSEM['yEst'], alpha=0.2, color=colos[7])
ax[0].scatter(dat['valBin'], dat['propPlay'], marker = 'o', edgecolors='gray', color = colos[7], s=60)

# Hide the top and right spines
ax[0].spines['top'].set_visible(False)
ax[0].spines['right'].set_visible(False)
ax[0].set_ylim(0,1)
# ax[4].set_xticks([-0.3, -0.15, 0.15, 0.3])
ax[0].set_xlabel("Value Difference (Points)")
ax[0].set_ylabel("Proportion Accept Gamble")


# iterate over each thetaVar and plot a line
highdf = estimated_df.loc[estimated_df['thetaVar'] == 'High']
for sub in highdf['sNo'].unique():
    subdf = highdf.loc[highdf['sNo'] == sub]
    ax[1].plot(subdf['x'], subdf['yEst'], '--', color='gray', alpha=0.1)

df = estimated_df.loc[estimated_df['thetaVar'] == 'High']
dat = tgav[tgav['thetaVar'] == 'High']
ax[1].plot(dat['valBin'], dat['propPlay'], 'o', color = colos[3])
ax[1].plot(xs, df.groupby('x')['yEst'].mean(), color=colos[3], label='High')
ax[1].fill_between(xs, y1=df.groupby('x')['yEst'].mean().values + highSEM['yEst'], y2=df.groupby('x')['yEst'].mean().values - highSEM['yEst'], alpha=0.2, color=colos[3])
ax[1].scatter(dat['valBin'], dat['propPlay'], marker = 'o', edgecolors='gray', color = colos[3], s=60)

# Hide the top and right spines
ax[1].spines['top'].set_visible(False)
ax[1].spines['right'].set_visible(False)
ax[1].set_xlabel("Value Difference (Points)")

ax[1].set_ylim(0,1)
ax[1].sharey(ax[0])


#set bar width
wid = 0.7

ax[2].bar(['Low', 'High'], kappaVarStats['logk'], color = [colos[7], colos[3]], width = wid)
ax[2].errorbar(['Low', 'High'], kappaVarStats['logk'], yerr= kappaVarStats['kappaSEM'], 
    ls='none', ecolor = 'black', capsize= 10)


# Hide the top and right spines
ax[2].spines['top'].set_visible(False)
ax[2].spines['right'].set_visible(False)

# Offset the left and bottom spines by 10 points
ax[2].spines['left'].set_position(('outward', 10))
ax[2].spines['bottom'].set_position(('outward', 10))
ax[2].spines['bottom'].set_capstyle('butt')
ax[2].set_ylim(-0.5, 0.5)

#Titles and such
ax[2].set_title('Mean Log Kappa')
ax[2].set_ylabel('Estimated Log Kappa (a.u.)')
ax[2].axhline(0, ls = '--', color = 'red', alpha = 0.8, linewidth=2)

#Start Axis 3
ax[3].bar(['Low', 'High'], medianVarStats['median'], color = [colos[7], colos[3]], width = wid)
ax[3].errorbar(['Low', 'High'], medianVarStats['median'], yerr= medianVarStats['medianSEM'], 
    ls='none', ecolor = 'black', capsize= 10)


# Hide the top and right spines
ax[3].spines['top'].set_visible(False)
ax[3].spines['right'].set_visible(False)

# Offset the left and bottom spines by 10 points
ax[3].spines['left'].set_position(('outward', 10))
ax[3].spines['bottom'].set_position(('outward', 10))
ax[3].spines['bottom'].set_capstyle('butt')
ax[3].set_ylim(-1, 1)

#Titles and such
ax[3].set_title('Average Median')
ax[3].set_ylabel('Estimated Median (Points)')

#Start Axis 4
ax[4].bar(['Low', 'High'], scaleVarStats['scale'], color = [colos[7], colos[3]], width = wid)
ax[4].errorbar(['Low', 'High'], scaleVarStats['scale'], yerr= scaleVarStats['scaleSEM'], 
    ls='none', ecolor = 'black', capsize= 10)


# Hide the top and right spines
ax[4].spines['top'].set_visible(False)
ax[4].spines['right'].set_visible(False)

# Offset the left and bottom spines by 10 points
ax[4].spines['left'].set_position(('outward', 10))
ax[4].spines['bottom'].set_position(('outward', 10))
ax[4].spines['bottom'].set_capstyle('butt')
ax[4].set_ylim(0, 5)

#Titles and such
ax[4].set_title('Mean Scale')
ax[4].set_ylabel('Estimated Scale (a.u.)')
ax[4].set_ylim(0, 8)

fig.tight_layout()
# plt.savefig("C:\\Users\\amcka\Desktop\\05-Results\\ALD_wBars.png", dpi = 600, bbox_inches='tight')
plt.show()
# %%
