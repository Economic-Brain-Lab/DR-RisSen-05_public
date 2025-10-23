'''
Author: Dragan Rangelov (d.rangelov@uq.edu.au)
File Created: 2021-05-21
-----
Last Modified: 2021-05-21
Modified By: Dragan Rangelov (d.rangelov@uq.edu.au)
-----
Licence: Creative Commons Attribution 4.0
Copyright 2019-2021 Dragan Rangelov, The University of Queensland
'''
#===============================================================================
# issue tracker
#===============================================================================
#===============================================================================
# importing libraries
#===============================================================================
# # DONE: comment this out prior to submitting the job
# # %% import functions for plotting
# import matplotlib as mpl
# from scipy import signal
# mpl.use('qt5agg')
# mpl.interactive(True)
# import matplotlib.pyplot as plt
# plt.ion()
# import seaborn as sbn
# sbn.set()
# %% import libraries
import logging
logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s %(levelname)s %(message)s',
    datefmt = '%d-%b-%y %H:%M'
)
import EEG_functions as eegfun
import h5py
import itertools
import mne
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal
from sklearn.model_selection import LeaveOneOut, KFold
import sys
import TunMah_woutTempGen as mdist
#===============================================================================
# %% unpack input arguments
#===============================================================================
sub_idx, tmpdir, trn_epochs, tst_epochs = sys.argv[1:5]
sub_idx = int(sub_idx)
#===============================================================================
# %% Encoding of concurrent central and peripheral motion directions
#===============================================================================
ROOTPATH = Path('/home/uqdrange/scratch/DR-MulFea-26')
logging.info(f'SUB_IDX:{sub_idx}, TRN:{trn_epochs}, TST:{tst_epochs}')
dpaths = [
    sorted(ROOTPATH.rglob(f'*{trn_epochs}-epo.fif.gz'))[sub_idx],
    sorted(ROOTPATH.rglob(f'*{tst_epochs}-epo.fif.gz'))[sub_idx]
]

# prepare data for decoding
epochs_all = []
events_dfs = []
for dpath in dpaths:
    sub, ses, modality = dpath.stem.split('.')[0].split('_')
    logging.info(f'Analysing {modality} for {sub}, {ses}')

    epochs = mne.read_epochs(dpath, preload = True).resample(256)
    epochs.pick_types(csd = True);
    picks = mne.pick_types(epochs.info, csd = True)
    times = epochs.times

    # filter bad epochs
    badEpochs = eegfun.findBadEpochs(epochs, picks)
    epochs.drop(badEpochs);

    # swap between event_ids/log messages and trigger values in the events structure
    event_id_swap = dict([
        (val, key) for key, val in epochs.event_id.items()
    ])
    # compile list of log messages per event
    logs = [
        event_id_swap[id]
        for id in epochs.events[:, -1]
    ]

    # create event descriptors
    events_df = pd.DataFrame(
        [
            [sub] + 
            [
                i.split(':')[-1] 
                for i in log.split('_')
            ]
            for log in logs
            
        ],
        columns = [
            'sub', 'task', 'trialNo',
            'sigDir', 'sigCoh', 'sigLife',
            'arcLeftDir', 'arcLeftCoh', 'arcRightDir', 'arcRightCoh'
        ]
    )
    events_df = events_df.astype(
        dict(
            sub='str', task='str', trialNo='int',
            sigDir='float', sigCoh='float', sigLife='int',
            arcLeftDir='float', arcLeftCoh='float', 
            arcRightDir='float', arcRightCoh='float'
        )
    )

    assert events_df.shape[0] == epochs.get_data().shape[0]

    # smoothing EEG epochs
    epochs_smth = epochs.copy().get_data()[:, picks]
    SD = int(.016 * epochs.info['sfreq'])
    win = signal.gaussian(epochs_smth.shape[-1], SD)
    win /= win.sum()
    epochs_smth = signal.convolve(
        epochs_smth,
        win[None, None],
        mode = 'same'
    )

    epochs_all += [epochs_smth]
    events_dfs += [events_df]

# prepare data for encoding
NCHANS = 16
channels = np.sort(np.angle(
    # get the equidistantly spaced angles for channel centers
    np.exp(np.arange(0, 2 * np.pi, 2 * np.pi / NCHANS) * 1j) 
    # rotate them so that they would sum up to 0
    * np.exp((np.pi / NCHANS) * 1j)
).round(3))
bin_width = .5 * np.pi
#===============================================================================
# %% MAHALANOBIS DISTANCE ENCODING
#===============================================================================
np.random.seed(int(sub.split('-')[-1]))
SAVEPATH = Path(tmpdir) / '03_Derivatives' / f'{sub}'
SAVEPATH.mkdir(exist_ok = True, parents = True)
SAVEFNAME = f'{sub}_{ses}_trn-{trn_epochs}_tst-{tst_epochs}_mdist.hdf'
# create predictor vectors
trnLoc, tstLoc = [
    [
        ['arcLeftDir', 'arcRightDir'],
        ['arcLeftDir', 'sigDir', 'arcRightDir']
    ][idx]
    for idx in [
        ['periph', 'centre'].index(e)
        for e in [trn_epochs, tst_epochs]
    ]
]
# combine training and testing directions
trn_tst_combos = list(filter(
    lambda x:
        # remove generalization b/w central and peripheral signals
        not (
            (x[0] != 'sigDir' and x[1] == 'sigDir')
            or (x[0] == 'sigDir' and x[1] != 'sigDir')
        ), 
    list(itertools.product(trnLoc, tstLoc))
))
signalRenaming = dict(
    arcLeftDir = 'thetaLeft',
    arcRightDir = 'thetaRight',
    sigDir = 'thetaCentre'
)
theta_combos = dict([
    (
        f'{signalRenaming[trn_dir]}_{signalRenaming[tst_dir]}',
        [
            np.angle(np.exp(events_dfs[0][trn_dir] * 1j)).round(3),
            np.angle(np.exp(events_dfs[1][tst_dir] * 1j)).round(3)
        ]
    )
    for trn_dir, tst_dir in trn_tst_combos
])
# compute and save the distances per predictor
for combo, thetas in theta_combos.items():
    trn_thetas, tst_thetas = thetas
    trn_sig, tst_sig = combo.split('_')
    logging.info(f'TRN-{trn_sig}_TST-{tst_sig}')
    
    cv = [
        [
            np.arange(trn_thetas.size),
            np.array(trl)
        ]
        for trl in np.arange(tst_thetas.size)
    ]
    if trn_epochs == tst_epochs:
        for idx, [trn_trls, tst_trl] in enumerate(cv):
            cv[idx][0] = trn_trls[trn_trls != tst_trl]
 
    with h5py.File(
        SAVEPATH / SAVEFNAME,
        'a'
    ) as f:
        dset_name = f'/TRN-{trn_sig}/TST-{tst_sig}'
        # skip computations if already done
        if dset_name in f:
            continue
        dist = mdist.compute_MahDist_DiffTrnTstEpochs(
            epochs_all, 
            thetas, 
            cv, 
            channels, 
            bin_width
        )
        f.create_dataset(
            name = dset_name,
            # data = [],
            data = dist,
            compression = 9
        )
        f[dset_name].attrs['channels'] = channels
        f[dset_name].attrs['bin_width'] = bin_width
        f[dset_name].attrs['times'] = times
        for colname in events_df.columns:
            f[dset_name].attrs[colname] = events_df[colname].values
        # cleanup
        del dist
# %%
