#===============================================================================
# %% importing libraries
#===============================================================================
from eeg_fasterAlgorithm import faster_bad_epochs
import mne
import logging
logging.basicConfig(level = logging.INFO,
                    format = '%(asctime)s %(levelname)s %(message)s',
                    datefmt = '%d-%b-%y %H:%M')
import numpy as np
import gzip
import json
import os
from pathlib import Path
import pandas as pd
import h5py
dt = h5py.special_dtype(vlen = np.dtype('float'))
import eeg_functions as eegfun
#===============================================================================
# %% setting paths
#===============================================================================
# UQ Laptop
# ROOTPATH = Path('C:\\PhD\\Experiments\\DR-RisSen-05-Pilot\\DR-RisSen-05\\00_SourceData\\EEG')
# SAVEPATH = Path('C:\\PhD\\Experiments\\DR-RisSen-05-Pilot\\DR-RisSen-05\\01_RawData')
# RDM
ROOTPATH = Path('/QRISdata/Q4364/DR-RisSen-05/01_RawData/')
SAVEPATH = Path('/QRISdata/Q4364/DR-RisSen-05')
# Home PC
# ROOTPATH = Path('D:\\Honours\\DR-RisSen-05\\')
# SAVEPATH = Path('D:\\Honours\\DR-RisSen-05\\01_RawData')
# load data
srcDataEEG = sorted(ROOTPATH.glob('**/*.bdf'))
srcDataLOG = sorted(ROOTPATH.glob('**/*MAIN*.log'))
srcDataBHV = sorted(ROOTPATH.glob('**/*MAIN*.tsv'))

job_array_index = int(os.getenv('SLURM_ARRAY_TASK_ID', 0))
print(job_array_index)
print(srcDataEEG)

dpath = srcDataEEG[job_array_index]

logfile = srcDataLOG[job_array_index]

bhvfile = srcDataBHV[job_array_index]

#%% LOAD IN LOG AND BEHAVIOURAL DATA FOR SYNCING
#behavioural 
beh_df = pd.read_csv(
    bhvfile,
    sep = '\t',
    na_values = 'n/a'
)

trials = []
trig_numbers = []
blckN = []
counter = 1

with open(logfile, 'r') as file:
    for line in file:
        if "Where: RespSub_Trial:" in line:
            parts = line.split("Where: RespSub_Trial:")[1]
            trial_number = parts.split('/')[0].strip()
            block_number = parts.split('/')[1].split(':')[1].strip()
            trials.append(int(trial_number))
            blckN.append(int(block_number))
            trig_numbers.append(counter)
            counter += 1

log_data = {'blockTrialNo': trials, "blockNo": blckN, 'respTrigN': trig_numbers}
log_df = pd.DataFrame(log_data)

log_df_sorted = log_df.sort_values(by=['blockNo', 'blockTrialNo']).reset_index(drop=True)
bhv_df_trigs = pd.merge(log_df_sorted, beh_df, on=['blockNo', 'blockTrialNo'])
#===============================================================================
# %% running eeg processing
#===============================================================================
montage = mne.channels.make_standard_montage('biosemi64')


subID, session, task = dpath.stem.split('_')
session = session.split('-')[-1]

dataEEG = {subID: {}}
bads = {subID: {}}
counts = {subID:{}}

# loading raw data
logging.info(f'Loading raw data for {subID}')
dataEEG[subID]['raw'] = mne.io.read_raw_bdf(str(dpath), preload = True)
dataEEG[subID]['raw'].set_montage('biosemi64', on_missing='ignore')

# rename exogenous channels and drop unnecessary ones
info = mne.create_info( 
    ch_names=[u'hEOG',u'vEOG'], 
    sfreq=dataEEG[subID]['raw'].info['sfreq'], 
    ch_types='eog'
)
LH, RH, LV, UV = mne.pick_channels(
    dataEEG[subID]['raw'].ch_names, 
    ['EXG3', 'EXG4', 'EXG5', 'EXG6'])
hEOG = dataEEG[subID]['raw'][LH][0] - dataEEG[subID]['raw'][RH][0]
vEOG = dataEEG[subID]['raw'][LV][0] - dataEEG[subID]['raw'][UV][0]
newEOG = mne.io.RawArray(np.concatenate([hEOG, vEOG]), info=info)
dataEEG[subID]['raw'].add_channels(add_list=[newEOG], force_update_info=True)
dataEEG[subID]['raw'].drop_channels(ch_names=[f'EXG{i}' for i in range(1,9)]) 

# re-referencing 
logging.info(f'Re-referencing for {subID}') 
dataEEG[subID]['avRef'], _ = mne.set_eeg_reference(
    dataEEG[subID]['raw'].copy(),
    projection = False
)
del dataEEG[subID]['raw']

# low-pass filtering eeg electrodes
logging.info(f'band-pass filtering eeg electrodes for {subID}')
dataEEG[subID]['filtered'] = dataEEG[subID]['avRef'].copy()
picks = mne.pick_types(dataEEG[subID]['filtered'].info, meg=False, eeg=True)
dataEEG[subID]['filtered'].filter(l_freq=.1, h_freq=100, picks=picks) 
dataEEG[subID]['filtered'].notch_filter(50, picks)
del dataEEG[subID]['avRef']

# band-pass filtering eog electrodes
logging.info(f'band-pass filtering eog electrodes for {subID}')
picks = mne.pick_types(dataEEG[subID]['filtered'].info, meg=False, eog=True)
dataEEG[subID]['filtered'].filter(l_freq=1, h_freq=10, picks=picks)

# detection and interpolating bad channels
logging.info(f'interpolating bad channels for {subID}')
eegfun.findBadChannels(dataEEG[subID]['filtered'])
bads[subID]['channels'] = dataEEG[subID]['filtered'].info['bads']
dataEEG[subID]['interpolated'] = dataEEG[subID]['filtered'].copy().interpolate_bads(verbose = False)
del dataEEG[subID]['filtered']

# re-referencing channels
logging.info(f're-referencing channels for {subID}')
dataEEG[subID]['interpolated'], _ = mne.set_eeg_reference(
    dataEEG[subID]['interpolated'],
    projection = False
)

# run ICA artefact detection and rejection
logging.info(f'ICA artefact detection and rejection for {subID}')
dwnsmpld_eeg = dataEEG[subID]['interpolated'].copy().resample(256) 
dataEEG[subID]['ica'] = mne.preprocessing.ICA(
    random_state=int(subID.split('-')[-1])
).fit(dwnsmpld_eeg)
dataEEG[subID]['ica'].exclude = eegfun.findBadICA(
    dwnsmpld_eeg,
    dataEEG[subID]['ica'],
    epochs = False,
    eog_correlation=2.5
)
bads[subID]['ICA'] = dataEEG[subID]['ica'].exclude
dataEEG[subID]['clean'] = eegfun.dropBadICA(
    dataEEG[subID]['interpolated'],
    dataEEG[subID]['ica']
)
del dataEEG[subID]['interpolated']
del dwnsmpld_eeg

#%%
# Annotating Accuray and value responses
# Identify the block start trigger in the EEG data

events = mne.find_events(dataEEG[subID]['clean'], initial_event= False, consecutive= True, output = 'onset', shortest_event = 1)

logging.info('Looking for starting block numbers')
block_start_triggers = list(range(1, 21))
block_start_index = None
block_start_event_idx = None

for i, event in enumerate(events):
    if event[2] in block_start_triggers:
        block_start_index = event[0]
        block_start_event_idx = i
        block_start_trigger = event[2]
        break

if block_start_index is None:
    raise ValueError("No block start trigger found in the EEG data.")

logging.info(f'Starting block is {block_start_trigger}')

# Clip eeg to block start Go back 1/5th a second to make sure block trigger is still included
eegCrop = dataEEG[subID]['clean'].copy()
eegCrop.crop(tmin=(block_start_index - 200) / dataEEG[subID]['clean'].info['sfreq'])

# Finding events
events = mne.find_events(eegCrop, initial_event= False, consecutive= True, output = 'onset', shortest_event = 1)
events_df = pd.DataFrame(events)
events_df.index.name = 'index'

#Make triggers for accuracy and accept
bhv_df_trigs.loc[(bhv_df_trigs['testStim'] == bhv_df_trigs['response']), 'play'] = 1 # Chose to play
bhv_df_trigs.loc[(bhv_df_trigs['testStim'] != bhv_df_trigs['response']), 'play'] = 0 # Chose to not play

# Function to generate trigPlay
def generate_trigPlay(row):
    if row['trialType'] == 0:
        if row['thetaVar'] == 0.075:
            return 301 if row['play'] == 1 else 302
        elif row['thetaVar'] == 0.300:
            return 401 if row['play'] == 1 else 402
    return None

# Function to generate trigAcc
def generate_trigAcc(row):
    if row['trialType'] == 0:
        if row['thetaVar'] == 0.075:
            return 303 if row['accuracy'] == 1 else 304
        elif row['thetaVar'] == 0.300:
            return 403 if row['accuracy'] == 1 else 404
    return None

# Apply the functions to create new columns
bhv_df_trigs['trigPlay'] = bhv_df_trigs.apply(generate_trigPlay, axis=1)
bhv_df_trigs['trigAcc'] = bhv_df_trigs.apply(generate_trigAcc, axis=1)


# Adjust the response trigger numbers in the log data to match the EEG recording
bhv_df_trigs_filt = bhv_df_trigs[bhv_df_trigs['blockNo'] >= (block_start_trigger - 1)].reset_index(drop=True)
bhv_df_trigs_filt = bhv_df_trigs_filt.sort_values('respTrigN')

eeg_resp_trigs = events_df[events_df[2].isin([104, 204, 115,215])][2].values
eeg_resp_onsets = events_df[events_df[2].isin([104, 204, 115,215])][0].values


bhv_df_trigs_filt['trigOnsets'] = eeg_resp_onsets
bhv_df_trigs_filt['Trigs'] = eeg_resp_trigs

#need to make ndarray a list to append  values
events_list = events.tolist()

# Function to add events to the list
def add_events(df, col_name, event_list):
    for index, row in df.iterrows():
        if not pd.isna(row[col_name]):
            event_list.append([int(row['trigOnsets']), 0, int(row[col_name])])


# Add trigPlay and trigAcc to the events list
add_events(bhv_df_trigs_filt, 'trigPlay', events_list)
add_events(bhv_df_trigs_filt, 'trigAcc', events_list)

events_array = np.array(events_list)
events_df = pd.DataFrame(events_array)
events_df[events_df[2].isin([104, 204, 115, 215, 301, 302, 303, 304, 401, 402, 403, 404])]

#%%
# removing events for which there was a fixation break
rmv_df = pd.DataFrame(columns=events_df.columns)

# Indicies where a fixation break occured
indices_133 = np.where(events_df[2] == 133)[0]

#Remove the two preceeding triggers
rows_to_remove = []
for idx in indices_133:
    rows_to_remove.extend([idx-2, idx-1, idx])

#Keep it within the data frames bounds
rows_to_remove = [idx for idx in rows_to_remove if 0 <= idx < len(events_df)]
rows_to_remove = sorted(set(rows_to_remove))

#Copy the rows we will remove, and then remove them from the dataframe proper
rmv_df = events_df.loc[rows_to_remove]

events_df = events_df.drop(rows_to_remove)
events_df.reset_index(drop=True)
rmv_df = rmv_df.reset_index(drop=True)

print("Updated DataFrame:")
print(events_df.shape)

print("\nRemoved DataFrame:")
print(rmv_df.shape)

print("Removed indicies:")
print(f'rows: {int(len(rows_to_remove))} breaks:{int(len(indices_133))}' )

#Convert event_df back into numpy array
events = events_df.to_numpy()
print(events)

# Save cleaned crop back to clean
dataEEG[subID]['clean'] = eegCrop.copy()
del eegCrop

#Now we can save the file as a .fif
logging.info(f'saving data for {subID}')
EXPORTPATH = SAVEPATH / '03_Derivatives' / f'{subID}' / 'eeg'
EXPORTPATH.mkdir(parents=True, exist_ok=False)
dataEEG[subID]['clean'].save(str(EXPORTPATH/ f'{subID}_task-MAIN_Raw-Prepro_eeg.fif'))

#%%
#EVENT DICTIONARIES FOR EPOCHING
eventsStim = {'lowOnVal': 101, 'highOnVal': 102, 'lowOnOri': 111, 'highOnOri': 211}
eventsRespVar = {'lowRespVal': 104, 'highRespVal': 204, 'lowRespOri': 115, 'highRespOri': 215}
eventsRespPlay = {'lowPlay': 301, 'highPlay': 401, 'lowReject': 302, 'highReject': 402}
eventsRespAcc = {'lowCor': 303, 'highCor': 403, 'lowIncor': 304, 'highIncor': 404}

#Get the epochs
dataEEG[subID]['epochsStimLock'] = mne.Epochs(
    dataEEG[subID]['clean'],
    events, # Note this is the cleaned events list
    event_id = eventsStim, # Note this is the event dictionary above
    tmin = -0.1, tmax = 1, # 1 second is the maximum exposure
    baseline = (-0.1, 0), # Correct from 100ms before onset
    detrend = 1, #Linear detrending
    preload = True,
    reject = None, flat = None,
    reject_by_annotation = None 
).resample(256) #resample to 256Hz after epoching 

dataEEG[subID]['epochsRespLockVar'] = mne.Epochs(
    dataEEG[subID]['clean'],
    events, # Note this is the cleaned events list
    event_id = eventsRespVar, # Note this is the event dictionary above
    tmin = -0.3, tmax = 1, # 1 second is the maximum exposure
    baseline = (-0.3, -0.2), # Correct from 300-200ms before onset
    detrend = 1, #Linear detrending
    preload = True,
    reject = None, flat = None,
    reject_by_annotation = None 
).resample(256) #resample to 256Hz after epoching 

dataEEG[subID]['epochsRespLockPlay'] = mne.Epochs(
    dataEEG[subID]['clean'],
    events, # Note this is the cleaned events list
    event_id = eventsRespPlay, # Note this is the event dictionary above
    tmin = -0.3, tmax = 1, # 1 second is the maximum exposure
    baseline = (-0.3, -0.2), # Correct from 300-200ms before onset
    detrend = 1, #Linear detrending
    preload = True,
    reject = None, flat = None,
    reject_by_annotation = None 
).resample(256) #resample to 256Hz after epoching 

dataEEG[subID]['epochsRespLockAcc'] = mne.Epochs(
    dataEEG[subID]['clean'],
    events, # Note this is the cleaned events list
    event_id = eventsRespAcc, # Note this is the event dictionary above
    tmin = -0.3, tmax = 1, # 1 second is the maximum exposure
    baseline = (-0.3, -0.2), # Correct from 300-200ms before onset
    detrend = 1, #Linear detrending
    preload = True,
    reject = None, flat = None,
    reject_by_annotation = None 
).resample(256) #resample to 256Hz after epoching 

#%%
#Find bad epochs
logging.info(f'finding bad epochs for {subID}') 
badEpochs = {subID: {}}
badEpochs[subID]['epochsStimLock'] = faster_bad_epochs(dataEEG[subID]['epochsStimLock'])
badEpochs[subID]['epochsRespLockVar'] = faster_bad_epochs(dataEEG[subID]['epochsRespLockVar'])
badEpochs[subID]['epochsRespLockPlay'] = faster_bad_epochs(dataEEG[subID]['epochsRespLockPlay'] )
badEpochs[subID]['epochsRespLockAcc'] = faster_bad_epochs(dataEEG[subID]['epochsRespLockAcc']
)
# dropping bad epochs
logging.info(f'dropping bad epochs for {subID}')

#Get indices
badsArray_StimLock = np.array(badEpochs[subID]['epochsStimLock'])
badsArray_RespLockVar = np.array(badEpochs[subID]['epochsRespLockVar'])
badsArray_RespLockPlay = np.array(badEpochs[subID]['epochsRespLockPlay'])
badsArray_RespLockAcc = np.array(badEpochs[subID]['epochsRespLockAcc'])

#Copy data to drop
dataEEG[subID]['epochsStimLock_drpd'] = dataEEG[subID]['epochsStimLock'].copy()
dataEEG[subID]['epochsRespLockVar_drpd'] = dataEEG[subID]['epochsRespLockVar'].copy()
dataEEG[subID]['epochsRespLockPlay_drpd'] = dataEEG[subID]['epochsRespLockPlay'].copy()
dataEEG[subID]['epochsRespLockAcc_drpd'] = dataEEG[subID]['epochsRespLockAcc'].copy()

#drop the epochs
dataEEG[subID]['epochsStimLock_drpd'].drop(badsArray_StimLock)
dataEEG[subID]['epochsRespLockVar_drpd'].drop(badsArray_RespLockVar)
dataEEG[subID]['epochsRespLockPlay_drpd'].drop(badsArray_RespLockPlay)
dataEEG[subID]['epochsRespLockAcc_drpd'].drop(badsArray_RespLockAcc)

# compute current source density
dataEEG[subID]['epochsStimLock_csd'] = mne.preprocessing.compute_current_source_density(dataEEG[subID]['epochsStimLock_drpd'])
dataEEG[subID]['epochsRespLockVar_csd'] = mne.preprocessing.compute_current_source_density(dataEEG[subID]['epochsRespLockVar_drpd'])
dataEEG[subID]['epochsRespLockPlay_csd'] = mne.preprocessing.compute_current_source_density(dataEEG[subID]['epochsRespLockPlay_drpd'] )
dataEEG[subID]['epochsRespLockAcc_csd'] = mne.preprocessing.compute_current_source_density(dataEEG[subID]['epochsRespLockAcc_drpd'])

# save data
logging.info(f'saving data epochs for {subID}')
EXPORTPATH = SAVEPATH / '03_Derivatives' / f'{subID}' / 'eeg'
EXPORTPATH.mkdir(parents=True, exist_ok=True)
dataEEG[subID]['epochsStimLock_drpd'].save(str(EXPORTPATH/ f'{subID}_task-MAIN_stimLocked-epo.fif'))
dataEEG[subID]['epochsRespLockVar_drpd'].save(str(EXPORTPATH/ f'{subID}_task-MAIN_epochsRespLockVar-epo.fif'))
dataEEG[subID]['epochsRespLockPlay_drpd'].save(str(EXPORTPATH/ f'{subID}_task-MAIN_epochsRespLockPlay-epo.fif'))
dataEEG[subID]['epochsRespLockAcc_drpd'].save(str(EXPORTPATH/ f'{subID}_task-MAIN_epochsRespLockAcc-epo.fif'))

# Saving channels that were interpolated and epochs that were dropped
with open(EXPORTPATH / f'{subID}_badEpochs.json', 'a+') as badsEpochsFile:
    json.dump(badEpochs[subID], badsEpochsFile, indent=2)
with open(EXPORTPATH / f'{subID}_bads.json', 'a+') as badsFile:
    json.dump(bads[subID], badsFile, indent=2)
mne.write_events(EXPORTPATH / f'{subID}_events.txt', events)
# %%
