import numpy as np
from scipy.spatial.distance import cdist
import logging as log

def regcov(X):
    '''
    Compute regularized covariance matrix between features
    Input
    =====
    X : n_trials x n_features
    Output
    ======
    regcov : n_features x n_features
    '''
    # de-mean data
    n_trials, n_features = X.shape
    X -= X.mean(0)[None]
    # sample co-variance
    S = X.T.dot(X)/n_trials
    # compute prior
    P = np.diag(np.diag(S))
    # compute shrinkage
    d = np.linalg.norm(S - P)**2/n_features
    Y = X**2
    r2 = 1./n_features/n_trials**2*np.sum(Y.T.dot(Y)) - 1./n_features/n_trials*np.sum(S**2)
    l = np.max([0, np.min([1, r2/d])])

    S_hat = l * P + (1 - l) * S
    return S_hat

def compute_MahDist(X, y, cv, channels, bin_width):
    '''
    Compute Mahalanobis distance for a time series
    Input
    =====
    X : main data matrix, n_trials x n_features x n_samples
    y : angles in pi_rad, n_trials
    cv : indices of training and test trials
    channels : means of channels to be modelled
    bin_width: how broad channels should be
    Output
    ======
    D : distance between the trial and channels, n_channels x n_samples
    '''
    n_trials, _, n_samples = X.shape
    # channel distances, n_trials x n_channels x n_samples
    D = []
    # looping through test trials
    q = np.linspace(10, 90, 9)
    percentiles = np.percentile(np.arange(n_trials),
                                q).astype('int')
    for trn, tst in cv:
        percentileDone = percentiles == tst
        if percentileDone.any():
            log.info('{}% done.'.format(q[percentileDone][0]))
        # subsetting trials
        X_tst = X[tst]
        X_trn = X[trn]
        y_trn = y[trn]
        y_tst = y[tst]
        # grouping of trials  that belong to different channels
        # n_channels x n_trials
        trn_trls = np.abs(np.angle(np.exp(1j*y_trn)[None]
                                   / np.exp(1j*(y_tst - channels))[:,None])
                          ) < bin_width*.5
        # mean of training trials per channel
        # n_channels x n_sensors x n_samples
        M = np.array([X_trn[trls].mean(0) for trls in trn_trls])
        # distances between channel patterns and the trial
        dist = []
        for trn_sample in range(n_samples):
            covX = regcov(X_trn[..., trn_sample])
            dist += [
                [
                    cdist(
                        M[..., trn_sample],
                        X_tst[..., tst_sample],
                        'mahalanobis',
                        VI = np.linalg.pinv(covX)
                    ).squeeze()
                    for tst_sample in range(n_samples)
                ]
            ]
        D += [
            # rearrange order of axes to I channels x J training smpls x K test
            np.moveaxis(np.array(dist), -1, 0)
        ]
    D = np.array(D).squeeze()
    return D
