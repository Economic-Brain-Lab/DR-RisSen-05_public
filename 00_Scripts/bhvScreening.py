#%% import packages
import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import pingouin as pg
import os
import glob
def wrapTopi(theta):
    '''
    Wrap array of angles in pi radians from -pi to pi
    Params:
    theta: array of angles in pi radians
    Returns:
    wrapped thetas
    '''
    return (theta + np.pi) % (2 * np.pi) - np.pi
#%%

#%% Load and concatonate all of the data and segment it into two dataframes
# val_data = Value Based Trial Data
# per_data = Orientation Reproduction (Perceptual) Trial Data
# temp = 'C:\\Phd\\Experiments\\DR-RisSen-05\\00_SourceData\\BHV\\*MAIN*.tsv'

paths = sorted(glob.glob(
    'D:\\DR-RisSen-05\\\BHV2\\*MAIN*.tsv', recursive=True
))
print('Data Paths:', '\n', f'{paths}')
# paths = sorted(glob.glob(
#     'C:\\PhD\\Experiments\\DR-RisSen-05-Pilot\\DR-RisSen-05\\01_RawData\\*MAIN*.tsv', recursive=True
# ))
# print('Data Paths:', '\n', f'{paths}')

# Concatonate the data
all_df = []
for idx, path in enumerate(paths):
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

per_data = per_data[per_data['response'].notna()] #Remove NA

#%% functions for binning data
def create_bins(data, column, bin_count):
    if data['trialType'].iloc[0] == 0:
        min_val = -11
        max_val = 11
    elif data['trialType'].iloc[0] == 1:
        min_val = -3.14
        max_val = 3.14
    bins = np.linspace(min_val, max_val, bin_count + 1)
    labels = [f'Bin{i}' for i in range(1, len(bins))]
    midpoints = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
    
    data[f'{column}_bin_num'] = pd.cut(data[column], bins=bins, include_lowest=True)
    data[f'{column}_bin_midpoint'] = pd.cut(data[column], bins=bins, labels=midpoints, include_lowest=True)
    
    return data

def bin_dataframe(data, column, min_bins=5, max_bins=7):
    bin_counts = np.arange(min_bins, max_bins + 1)
    bin_counts = bin_counts[(bin_counts >= min_bins) & (bin_counts <= max_bins)]
    
    # Create a copy of the dataframe to avoid modifying the original data
    binned_data = data.copy()
    
    for bin_count in bin_counts:
        binned_data = create_bins(binned_data, column, bin_count)
        print(f'Data binned into {bin_count} bins:')
        print(binned_data)
    
    return binned_data
#%%

#%% Value-based trials analysis
# Create columns needed for analysing value trials
val_data.loc[(val_data['testStim'] == 'left', 'tstPayoff')] = val_data['leftPayoff']
val_data.loc[(val_data['testStim'] == 'right', 'tstPayoff')] = val_data['rightPayoff']
val_data.loc[(val_data['testStim'] == val_data['response']), 'play'] = 1 # Chose to play
val_data.loc[(val_data['testStim'] != val_data['response']), 'play'] = 0 # Chose to not play

column = 'tstPayoff'
tst_df = bin_dataframe(val_data, column)

# Plotting
unique_bins = tst_df[f'{column}_bin_num'].cat.categories
participants = tst_df['sNo'].unique()
n_participants = len(participants)
nrows = 8
ncols = 5
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 15))

for i, participant in enumerate(participants):
    # Determine the row and column index
    row = i // ncols
    col = i % ncols
    
    # Select the participant's data
    participant_data = tst_df[tst_df['sNo'] == participant]
    hist_data = participant_data.groupby(f'{column}_bin_midpoint')['play'].mean().reset_index()
    
    # Plotting histogram using midpoints and weights, aligning the bins to the center
    axes[row, col].hist(hist_data[f'{column}_bin_midpoint'], bins=len(unique_bins), range=(-10, 10), 
                        weights=hist_data['play'], edgecolor='black', alpha=0.7, align='mid')
    
    # Setting titles and labels
    axes[row, col].set_title(f'sub_{participant}')
    if col == 0:
        axes[row, col].set_ylabel('Play (Proportion)')
    if row == nrows - 1:
        axes[row, col].set_xlabel(f'Value')


    # Set y-axis limits
    axes[row, col].set_ylim(0, 1)
    axes[row, col].set_xlim(-10, 10)

    axes[row, col].set_xticks([-10, 0, 10])
    
    # Remove top and right spines
    axes[row, col].spines['top'].set_visible(False)
    axes[row, col].spines['right'].set_visible(False)

    axes[row, col].spines['bottom'].set_position(('outward', 10))
    axes[row, col].spines['left'].set_position(('outward', 10))

# Adjust layout
plt.tight_layout()
plt.show()
#%%

#%% Perceptual trials analysis
# Create columns needed for analysing perceptual trials
per_data.loc[(per_data['testStim'] == 'left', 'tstPayoff')] = per_data['leftPayoff']
per_data.loc[(per_data['testStim'] == 'right', 'tstPayoff')] = per_data['rightPayoff']

# Wrap tstThetaRad to -pi to pi
per_data['tstThetaWrap'] = wrapTopi(per_data['tstThetaRad'])
per_data.loc[(per_data['testStim'] == per_data['response']), 'play'] = 1 # Chose to play
per_data.loc[(per_data['testStim'] != per_data['response']), 'play'] = 0 # Chose to not play

per_data['error'] = np.angle(
    np.exp((per_data['tstThetaWrap']) * 1j) / 
    np.exp((per_data['response'].astype('float') * np.pi / 90) * 1j)
)

per_data['signedError'] = per_data['error'].copy()
# Multiply error by reference sign. if resulting sign is +ve then error towards V if -ve then error towards H
per_data['signedError'] = per_data['signedError'] * np.sign(per_data['tstThetaWrap'])
# Swap signs for participants in which H is +ve and V is -ve
per_data.loc[per_data['sNo'] % 2 == 1, 'signedError'] *= -1

column = 'signedError'
tst_df = bin_dataframe(per_data, column)

#%%
# Plotting
unique_bins = tst_df[f'{column}_bin_num'].cat.categories
midpoints = [float(mid) for mid in tst_df[f'{column}_bin_midpoint'].cat.categories]
participants = tst_df['sNo'].unique()
n_participants = len(participants)
nrows = 8
ncols = 5
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 15))

for i, participant in enumerate(participants):
    # Determine the row and column index
    row = i // ncols
    col = i % ncols
    
    # Select the participant's data
    participant_data = tst_df[tst_df['sNo'] == participant]
    hist_data = participant_data.groupby(f'{column}_bin_midpoint')['signedError'].count().reset_index()
    
    # Plotting histogram using midpoints and weights, aligning the bins to the center
    axes[row, col].hist(hist_data[f'{column}_bin_midpoint'], bins=len(unique_bins), range=(-3.14, 3.14), 
                        weights=hist_data['signedError'], edgecolor='black', alpha=0.7, align='mid', density=True)
    
    # Setting titles and labels
    axes[row, col].set_title(f'sub_{participant}')
    if col == 0:
        axes[row, col].set_ylabel('Frequency')
    if row == nrows - 1:
        axes[row, col].set_xlabel('Signed Error')

    
    # Set y-axis limits
    axes[row, col].set_ylim(0, 1)
    
    axes[row, col].set_xticks([-3.14, 3.14])
    axes[row, col].set_xticklabels(["-$\pi$", "$\pi$"])
    # Remove top and right spines
    axes[row, col].spines['top'].set_visible(False)
    axes[row, col].spines['right'].set_visible(False)
    
    axes[row, col].spines['bottom'].set_position(('outward', 10))
    axes[row, col].spines['left'].set_position(('outward', 10))

# Adjust layout
plt.tight_layout()
plt.show()
# %% Perform chi-square test of uniformity on each participant
results = pd.DataFrame(columns=['Participant', 'Task', 'Chi2', 'P-value'])

for data in [val_data, per_data]:
    for participant in data['sNo'].unique():
        # Filter data for the current participant and trial type
        participant_data = data[(data['sNo'] == participant)].copy()
        
        if participant_data.empty:
            continue
        
        if participant_data['trialType'].iloc[0] == 0:
            column = 'tstPayoff'
            taskType = 'Value'
        if participant_data['trialType'].iloc[0] == 1:
            column = 'signedError'
            taskType = 'Perceptual'
        
        binned_data = bin_dataframe(participant_data, column, min_bins=7)
        
        # Calculate observed frequencies
        if participant_data['trialType'].iloc[0] == 0:
           observed_freq = binned_data[binned_data['play'] == 1].groupby(f'{column}_bin_num')['play'].count().values
           total_observations = participant_data.shape[0]
           expected_mean = binned_data[binned_data['play'] == 1].groupby(f'{column}_bin_num')['play'].count().values.sum()
           

        if participant_data['trialType'].iloc[0] == 1:
            observed_freq = binned_data[f'{column}_bin_num'].value_counts().sort_index().values
            total_observations = sum(observed_freq)
            expected_mean = sum(observed_freq)

        # Calculate expected frequencies with variability
        expected_freq = np.full(7, expected_mean // 7)
        remainder = expected_mean % 7
        expected_freq[:remainder] += 1
        
        # Perform chi-square test with observed_freq and expected_freq
        chi2, p = ss.chisquare(observed_freq, f_exp=expected_freq)
        
        # Append the results to the results dataframe
        results = results.append({'Participant': participant, 'Task': taskType, 'Chi2': chi2, 'P-value': p.round(4), 'Obs': total_observations}, ignore_index=True)

results['Accept'] = results['P-value'] < 0.05
print(results)
# %%
results[results['Accept'] == 0]
#%%
#EEG Euclidean distance testing
df = pd.read_csv('R:\\DR-RisSen-05\\03_Derivatives\\Outputs\\distance_dataframe.csv')
df = df.sort_values('subject_number').reset_index(drop=True)
mean = df['euclidean_distance'].mean()
sd = df['euclidean_distance'].std()
threshold = 3 * sd
df['outside_3std'] = (df['euclidean_distance'] - mean).abs() > threshold

# %%
