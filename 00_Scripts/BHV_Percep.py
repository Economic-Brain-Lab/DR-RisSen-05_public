#%% Import Libraries needed for basic data manipulation and plotting
import numpy as np
import pandas as pd
import scipy.stats as ss
import pingouin as pg
from statsmodels.stats.weightstats import ztest as ztest
from statsmodels.stats.weightstats import zconfint as zconfint
import statsmodels.api as sm
import statsmodels.formula.api as smf
def wrapTopi(theta):
    '''
    Wrap array of angles in pi radians from -pi to pi
    Params:
    theta: array of angles in pi radians
    Returns:
    wrapped thetas
    '''
    return (theta + np.pi) % (2 * np.pi) - np.pi
import matplotlib.pyplot as plt
import seaborn as sns
import glob #for paths

#%% Load and concatonate all of the data and segment it into two dataframes
# val_data = Value Based Trial Data
# per_data = Orientation Reproduction (Perceptual) Trial Data
paths = sorted(glob.glob(
    'D:\\DR-RisSen-05\\\BHV2\\*MAIN*.tsv', recursive=True
))
print('Data Paths:', '\n', f'{paths}')

# List of participants to exclude
exclude_participants = ['sub-35', 'sub-02', 'sub-03']

# Filter out paths of participants to exclude
filtered_paths = [path for path in paths if not any(exclude in path for exclude in exclude_participants)]



# paths = sorted(glob.glob(
#     'C:\\PhD\\Experiments\\DR-RisSen-05-Pilot\\DR-RisSen-05\\01_RawData\\*MAIN*.tsv', recursive=True
# ))
# print('Data Paths:', '\n', f'{paths}')

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

per_data = all_df.loc[all_df['trialType'] == 1].copy()
print(per_data.shape)
# missing data 
print('Missing value data: \n', 
    f"{per_data['response'].isnull().groupby(per_data['sNo']).sum().reset_index(name='Missing')}")

per_data = per_data[per_data['response'].notna()] #Remove NA

#%% Create columns needed for analysing value trials
val_data.loc[(val_data['testStim'] == 'left', 'tstPayoff')] = val_data['leftPayoff']
val_data.loc[(val_data['testStim'] == 'right', 'tstPayoff')] = val_data['rightPayoff']

valBins = np.histogram(val_data['tstPayoff'], range=(-12,12),bins=7)[1].round(0)
val_data['valBin'] = pd.cut(val_data['tstPayoff'], bins=valBins, labels=np.linspace(-10, 10, 7).round(0))

val_data.loc[(val_data['testStim'] == val_data['response']), 'play'] = 1 # Chose to play
val_data.loc[(val_data['testStim'] != val_data['response']), 'play'] = 0 # Chose to not play
#Do the same for perceptual trials
per_data.loc[(per_data['testStim'] == 'left', 'tstPayoff')] = per_data['leftPayoff']
per_data.loc[(per_data['testStim'] == 'right', 'tstPayoff')] = per_data['rightPayoff']

oriBins = np.histogram(per_data['deltaThetaRad'], bins=7)[1].round(3)
per_data['oriBin'] = pd.cut(per_data['deltaThetaRad'], bins=oriBins, labels=np.linspace(-1.58, 1.58, 7))
#Wrap tst to pi to pi
per_data['tstThetaWrap'] = wrapTopi(per_data['tstThetaRad'])
per_data.loc[(per_data['testStim'] == per_data['response']), 'play'] = 1 # Chose to play
per_data.loc[(per_data['testStim'] != per_data['response']), 'play'] = 0 # Chose to not play

## Another grouping for errors 
perBins = np.histogram(per_data['tstThetaWrap'], range=(-np.pi, np.pi), bins=6)[1]
per_data['perBin'] = pd.cut(per_data['tstThetaWrap'], bins=perBins, labels=[-2.62, -1.57, -0.52, 0.52, 1.57, 2.62])
plt.scatter(per_data['tstThetaWrap'], per_data['perBin'])
plt.show()

## TODO: put this in later for segregation after signed error is calced
# df = per_data.loc[(per_data['perBin'] == -1.57) | (per_data['perBin'] == 1.57)]

# Get RT
per_data['rtms'] = per_data['RT'] * 1000
#%% Creating play chose data and plots
gav_data = val_data.groupby(['sNo', 'thetaVar', 'valBin'])['play'].mean().round(2).reset_index(name = 'propPlay')
gav_data['sem'] = val_data.groupby(['sNo', 'thetaVar', 'valBin'])['play'].sem().round(2).reset_index(name = 'sem')['sem']
#%% Creating nominal accuracy data and plots
gav2_data = val_data.groupby(['sNo', 'thetaVar', 'valBin'])['accuracy'].mean().round(2).reset_index(name = 'Accuracy')
gav2_data['sem'] = val_data.groupby(['sNo', 'thetaVar', 'valBin'])['accuracy'].sem().round(2).reset_index(name = 'sem')['sem']

# %% Analysing circular errors
per_data['error'] = np.angle(
    np.exp((per_data['tstThetaWrap']) * 1j) / 
    np.exp((per_data['response'].astype('float') * np.pi / 90) * 1j) )

per_data['signedError'] = per_data['error'].copy()
# Multiply error by reference sign. if resulting sign is +ve then error towards V if -ve then error towards H
per_data['signedError'] = per_data['signedError'] * np.sign(per_data['tstThetaWrap'])
# Swap signs for participants in which H is +ve and V is -ve
per_data.loc[per_data['sNo'] % 2 == 1, 'signedError'] *= -1

bin_centers = np.mean([
    np.linspace(-np.pi, np.pi, 12)[1:],
    np.linspace(-np.pi, np.pi, 12)[:-1]
],0)
bin_centers = np.linspace(-np.pi, np.pi, 12)


# Create a colormap object
pastel = sns.color_palette('pastel', as_cmap=True)
# Generate a list of unique colors
colors = [pastel[0], pastel[4], pastel[6]]
colos = ['#BBCC33', '#AAAA00', '#77AADD', '#7EC384', '#EEDD88', '#FFAABB', '#99DDFF', pastel[0], '#DDDDDD']
#%% 
# per_data.to_csv('04-allperBeh.tsv', sep='\t')
# %%

fig, axes = plt.subplots(nrows=8, ncols=5, sharex=True)
plt.subplots_adjust(hspace=0.8)
axesFlat = axes.flatten()
for id, ax in enumerate(axesFlat):
    #get participant data
    sub = per_data[per_data['sNo'] == (id+1)]
    #plot the hist on the axis
    ax.hist(sub['signedError'], bins = bin_centers)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
        if (id+1) not in [1,6,11,16,21,26,31,36]:
            ax.set_yticks([])
            for spine in ['left']:
                ax.spines[spine].set_visible(False)
    #sub 25 looking weird was due to ylim not being the same for all hists
    ax.set_ylim(0, 150)
    ax.set_xticks([-3.14, 3.14])
    ax.set_xticklabels(["-$\pi$", "$\pi$"])
    ax.set_title(f"sub-{id+1}")

# plt.savefig("C:\\Users\\amcka\Desktop\\05-Results\\subErrorHist.png", dpi = 600)
plt.show()
#%%
participant_hists = []
for idx, data in per_data[(per_data['thetaVar'] == 0.075)].groupby('sNo'):
    data = data.reset_index()
    hist, edges = np.histogram(data[data['thetaVar'] == 0.075]['signedError'], bins=11, range=(-np.pi, np.pi))
    
    participant_hists.append(hist/np.sum(hist)) # Normalised Hists for Each participant

gav_hist = np.sum(participant_hists, axis=0) #axis = 0 sums each histogram point together
norm_gav_hist1 = gav_hist / np.sum(gav_hist)

participant_hists = []
for idx, data in per_data[(per_data['thetaVar'] == 0.30)].groupby('sNo'):
    data = data.reset_index()
    hist, edges = np.histogram(data[data['thetaVar'] == 0.30]['signedError'], bins=11, range=(-np.pi, np.pi))
    
    participant_hists.append(hist/np.sum(hist)) # Normalised Hists for Each participant

gav_hist = np.sum(participant_hists, axis=0) #axis = 0 sums each histogram point together
norm_gav_hist2 = gav_hist / np.sum(gav_hist)

#%% plots
meanUncertStats = per_data.groupby(['thetaVar'])['signedError'].apply(ss.circmean, low=-np.pi, high=np.pi).reset_index(name = 'meanError')
meanUncertStats['meanSEM'] = ((per_data.groupby(['sNo', 'thetaVar'])['signedError'].apply(ss.circmean, low=-np.pi, high=np.pi) - 
                                per_data.groupby('sNo')['signedError'].apply(ss.circmean, low=-np.pi, high=np.pi) +
                                per_data.groupby('thetaVar')['signedError'].apply(ss.circmean, low=-np.pi, high=np.pi)).reset_index().groupby('thetaVar')['signedError'].sem() * np.sqrt(2/1)).reset_index()['signedError']

sdUncertStats = per_data.groupby(['thetaVar'])['signedError'].apply(ss.circstd, low=-np.pi, high=np.pi).reset_index(name = 'sdError')
sdSub = per_data.groupby(['sNo','thetaVar'])['signedError'].apply(ss.circstd, low=-np.pi, high=np.pi).reset_index(name = 'sdError')
sdUncertStats['meanSEM'] = ((sdSub.groupby(['sNo', 'thetaVar'])['sdError'].mean() - 
                                sdSub.groupby('sNo')['sdError'].mean() +
                                sdSub.groupby('thetaVar')['sdError'].mean()).reset_index().groupby('thetaVar')['sdError'].sem() * np.sqrt(2/1)).reset_index()['sdError']
# create circular mean of meanError and sdError per variance condition

fig, axz = plt.subplots(nrows=1, ncols=2, figsize=(14,8), sharex=True, dpi=600)
axF = axz.flatten()
axF[0].hist(edges[:-1], edges, weights=norm_gav_hist1, color=colos[7], label='Low', edgecolor = 'black')
axF[0].set_xticks([-3.14, 0,3.14])
axF[0].set_xticklabels(["-$\pi$", 0, "$\pi$"])
axF[0].set_ylabel('Proportion in Bin', fontsize=22)
axF[0].set_ylim(0, 0.5)
axF[0].set_yticks([0, 0.25, 0.5])
axF[0].set_title('Low')
axF[0].vlines(meanUncertStats['meanError'][0], 0, norm_gav_hist1.max(), color = 'black', ls='--')

for spine in ['top', 'right']:
    axF[0].spines[spine].set_visible(False)

axF[0].spines['left'].set_position(('outward', 10))
axF[0].spines['bottom'].set_position(('outward', 10))

#Start High
axF[1].hist(edges[:-1], edges, weights=norm_gav_hist2, color=colos[3], label='High', edgecolor = 'black')
axF[1].set_xticks([-3.14, 0,3.14])
axF[1].set_xticklabels(["-$\pi$", 0, "$\pi$"])
axF[1].set_yticks([])
axF[1].set_ylim(0, 0.5)
axF[1].set_title('High')
axF[1].vlines(meanUncertStats['meanError'][1], 0, norm_gav_hist2.max(), color = 'black', ls='--')


for spine in ['top', 'right', 'left']:
    axF[1].spines[spine].set_visible(False)

axF[1].spines['bottom'].set_position(('outward', 10))
print(fig.get_size_inches()*fig.dpi)
fig.text(0.55, 0.01, 'Signed Angular Error (rad)', ha='center', fontsize=22)
# plt.subplots_adjust(bottom=.8)
# plt.tight_layout()
# plt.savefig("C:\\Users\\amcka\Desktop\\05-Results\\SignedError_Gav.png", dpi = 600, bbox_inches="tight")
plt.show()

#%%
for var in per_data['thetaVar'].unique():
    plt.hist(per_data.loc[
        per_data['thetaVar'] == var,
        'signedError'
    ], bins = bin_centers, alpha = .3)
plt.legend(per_data['thetaVar'].unique())
# plt.savefig("C:\\Users\\amcka\Desktop\\05-Results\\SignedError_Overlap.png", dpi = 600, bbox_inches="tight")
plt.show()

#%% Stats
#Mean error
testME = per_data.groupby(['sNo', 'thetaVar'])['signedError'].apply(ss.circmean, low=-np.pi, high=np.pi).reset_index()
pg.ttest(testME[testME['thetaVar'] == 0.075]['signedError'], testME[testME['thetaVar'] == 0.30]['signedError'], paired=True)

#%%
#sd error
pg.ttest(sdSub[sdSub['thetaVar'] == 0.075]['sdError'], sdSub[sdSub['thetaVar'] == 0.30]['sdError'], paired=True)
#%%
#Colours
colos = ['#BBCC33', '#AAAA00', '#77AADD', '#7EC384', '#EEDD88', '#FFAABB', '#99DDFF', pastel[0], '#DDDDDD']
fig, ax = plt.subplots(nrows = 1, ncols=2, figsize = (14,4), dpi=600)
pos = [0, 0.4]
ax[0].bar(pos, meanUncertStats['meanError'], color = [colos[7], colos[3]], width=0.2, edgecolor = 'black')
ax[0].errorbar(pos, meanUncertStats['meanError'], yerr= meanUncertStats['meanSEM'], 
    ls='none', ecolor = 'black', capsize= 10)
for spine in ['top', 'right']:
    ax[0].spines[spine].set_visible(False)

ax[0].spines['left'].set_position(('outward', 10))
ax[0].spines['bottom'].set_position(('outward', 20))
ax[0].spines['bottom'].set_capstyle('butt')
ax[0].set_xlim(-0.2, 0.6)
ax[0].set_ylim(-0.25, 0.25)
ax[0].set_xticks(pos, ['Low', 'High'])
# ax[0].set_yticks([-0.10, 0, 0.10])


ax[0].set_title('Average Signed Error')
ax[0].set_ylabel("Error (rad)")
# plt.tight_layout()
# plt.savefig("C:\\Users\\amcka\\Desktop\\Image\\circAveUncert.png", dpi = 600)
# plt.show()

# fig, ax = plt.subplots(figsize = (6,4))
ax[1].bar(pos, sdUncertStats['sdError'], color = [colos[7], colos[3]], width=0.2, edgecolor = 'black')
ax[1].errorbar(pos, sdUncertStats['sdError'], yerr= sdUncertStats['meanSEM'], 
    ls='none', ecolor = 'black', capsize= 10)
for spine in ['top', 'right']:
    ax[1].spines[spine].set_visible(False)

ax[1].spines['left'].set_position(('outward', 10))
ax[1].spines['bottom'].set_position(('outward', 20))
ax[1].spines['bottom'].set_capstyle('butt')
ax[1].set_xlim(-0.2, 0.6)
ax[1].set_ylim(0, 1.2)
ax[1].set_yticks([0, 0.5, 1])

ax[1].set_xticks(pos, ['Low', 'High'])


ax[1].set_title('Average Circular $\it{SD}$')
ax[1].set_ylabel("$\it{SD}$ (rad)")
plt.tight_layout()
# plt.savefig("C:\\Users\\amcka\Desktop\\05-Results\\AvePercep_ErrorSD.png", dpi = 600)
plt.show()

# %%
