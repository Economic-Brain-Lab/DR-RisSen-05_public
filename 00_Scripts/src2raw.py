'''
Author: Dragan Rangelov (d.rangelov@uq.edu.au)
File Created: 2024-10-13
-----
Last Modified: 2024-10-13
Modified By: Dragan Rangelov (d.rangelov@uq.edu.au)
-----
Licence: Creative Commons Attribution 4.0 International
Copyright 2019-2024 Dragan Rangelov, The University of Queensland
'''
#===============================================================================
# %% issue tracker
#===============================================================================
# TODO: load eeg files, add log annotations, and save as fif.gz files Rawdata
# NOTE: trigger values
# FIXON = 130
# STIMON_VAL_LOW = 101
# STIMON_VAL_HIGH = 201
# STIMON_ORI_LOW = 111
# STIMON_ORI_HIGH = 211
# RESPON_VAL_LOW = 102
# RESPON_VAL_HIGH = 202
# RESPON_ORI_LOW = 113
# RESPON_ORI_HIGH = 213
# RESPSUB_VAL_LOW = 104
# RESPSUB_VAL_HIGH = 204
# RESPSUB_ORI_LOW = 115
# RESPSUB_ORI_HIGH = 215
# FEEDBCK_VAL_LOW = 106
# FEEDBCK_VAL_HIGH = 206
# FEEDBCK_ORI_LOW = 117
# FEEDBCK_ORI_HIGH = 217
# FIX_BRK = 133
# NO_RESP = 99
# NOTE: sub-01 EEG recording started from trial 74, 
# for other subs there is a matching number of trials b/w logs and eeg data
#===============================================================================
# %% set up plotting
#===============================================================================
import matplotlib as mpl
mpl.use('qt5agg')
import matplotlib.pyplot as plt
plt.ion()
#===============================================================================
# %% importing libraries
#===============================================================================
import mne
import numpy as np
import pandas as pd
from pathlib import Path
#===============================================================================
# %% inspect files
#===============================================================================
ROOTPATH = Path().cwd().parent
src_eeg = sorted(ROOTPATH.rglob('**/01_Sourcedata/**/*task-MAIN_eeg.bdf'))
src_log = sorted(ROOTPATH.rglob('**/01_Sourcedata/**/*MAIN.log'))
subs = [log_file.stem.split('_')[0] for log_file in src_log]
# %%
# idx_sub = 0
# data_counts = []
for idx_sub in range(len(src_log)):
    sub_id = subs[idx_sub]
    eeg_file = src_eeg[idx_sub]
    log_file = src_log[idx_sub]

    eeg_data = mne.io.read_raw_bdf(eeg_file, preload = True)
    eeg_data.set_montage('biosemi64', on_missing='ignore')
    eeg_events = mne.find_events(eeg_data)
    # identify non-trigger events and drop them from the array
    idx_nonTrig, = np.where(eeg_events[:, 2] > 255)
    # NOTE: there could have been breaks b/w recordings
    nStartEEG = idx_nonTrig.size
    # if idx_nonTrig.size:
    #     eeg_events = np.delete(eeg_events, idx_nonTrig, 0)
    idxEEG_FixON, = np.where(eeg_events[:,2] == 130)
    # trlEEG = np.split(eeg_events, idxEEG_FixON)
    with log_file.open('r') as f:
        log_data = f.readlines()
    idxLOG_FixON = [
        idx_log 
        for idx_log, val_log in enumerate(log_data) 
        if 'FixON' in val_log
    ]
    # data_counts += [sub_id, nStartEEG, idxEEG_FixON.size, len(idxLOG_FixON)]
    trlON_eeg = np.array([
        eeg_events[idxEEG_FixON, 0] 
        / eeg_data.info['sfreq']
    ]).flatten()
    trlON_log = [float(log_data[idx].split('\t')[0]) for idx in idxLOG_FixON]
    trls_log = np.split(log_data, idxLOG_FixON)[1:]
    # drop trials that were not recorded in the EEG
    if sub_id == 'sub-01':
        idxLOG_FixON = idxLOG_FixON[74:]
        trlON_log = trlON_log[74:]
        trls_log = trls_log[74:]
    
    log_onsets = [
        float(log_line.split('\t')[0]) - trlON_log[trl_idx] + trl_onset 
        for trl_idx, trl_onset in enumerate(trlON_eeg)
        for log_line in trls_log[trl_idx]
    ]

    log_entries = [
        log_line.split('\t')[2]
        for trl_idx, trl_onset in enumerate(trlON_eeg)
        for log_line in trls_log[trl_idx]
    ]
    eeg_data.set_annotations(
        mne.Annotations(log_onsets, .001, log_entries)
    )
    exportPath = ROOTPATH / '02_Rawdata' / sub_id / 'eeg'
    exportPath.mkdir(parents = True, exist_ok = True)
    eeg_data.save(
        exportPath / f'{sub_id}_ses-MAIN_eeg-raw.fif.gz',
        overwrite = True
    )
# %%