'''
Created on 29Jun.,2017

@author: uqdrange
'''
import sys
sys.dont_write_bytecode = True
import numpy as np
from scipy import special,stats,optimize
#===============================================================================
# Mixture model Bays (www.paulbays.com)
#===============================================================================
# auxilliary functions
def wrap(theta, bound = np.pi):
    """
    compute wrapped values around the bound
    theta: a scalar or a sequence
    bound: the value around which to wrap, defaults to pi
    """
    theta = np.array(theta)
    wrappedT = (theta+bound)%(bound*2) - bound
    return wrappedT

def cmean(theta):
    theta = np.array(theta)
    if np.any(np.abs(theta) > np.pi):
        print('The range of values must be between -pi and pi')
        return np.nan
    else:
        thetaHat = np.arctan2(np.sin(theta).sum(),np.cos(theta).sum())
        return thetaHat

def cstd(theta):
    theta = np.array(theta)
    if np.any(np.abs(theta) > np.pi):
        print('The range of values must be between -pi and pi')
        return np.nan
    else:
        R = np.sqrt(np.sin(theta).sum()**2+np.cos(theta).sum()**2)/theta.size
        thetaSD = np.sqrt(-2*np.log(R))
        return thetaSD

def k2sd(K):
    if K == 0:
        S = np.Inf
    elif np.isinf(K):
        S = 0
    else:
        S = np.sqrt(-2*np.log(special.iv(1,K)/special.iv(0,K)))
    return S

def sd2k(S):
    R = np.exp(-S**2/2.)
    if R < .85:
        K = -.4 + 1.39*R + .43/(1 - R)
    elif R < .53:
        K = 2*R + R**3 + (5*R**5)/6.
    else:
        K = 1./(R**3 - 4*R**2 + 3*R)
    return(K)

def A1inv(R):
    if 0 <= R < 0.53:
        K = 2 * R + R**3 + (5 * R**5)/6;
    elif R < 0.85:
        K = -0.4 + 1.39 * R + 0.43/(1 - R);
    else:
        K = 1./(R**3 - 4 * R**2 + 3 * R)
    return K
# %% no reversals
def llhood(B, E):
    '''
    Compute maximum likelihood for:
    1. the given set of parameters (B),
    2. the vector of target errors (E),
    3. the vector of non-target errors (NE)
    '''
    n = E.shape[0]
    mu, K, Pt = B
    Pu = 1 - Pt # guess probability

    if Pu < 0:
        # penalize negative guess rates
        LL = -1e10 
    else:
        Wt = Pt * stats.vonmises.pdf(E, kappa = K, loc = mu)
        Wu = Pu * np.ones([n, 1]) / (2 * np.pi)

        W = np.nansum(
            np.hstack([Wt,Wu]),
            axis = 1).reshape(-1,1)

        LL = np.log(W).sum()

    return -LL

def mmfit(E = [np.nan], nruns = 100, maxiter = 15000, random_seed = 0):
    '''
    Function to fit mixture distribution model to observed error magnitudes
    relative to the target (E) and non-targets (NE)
    '''
    print('Fitting model W/OUT reversals for sub-{}'.format(random_seed))
    np.random.seed(random_seed)
    E = np.array(E)
    if np.isnan(E).any():
        raise ValueError('NaN values in the input arrays')
    E = np.angle(np.exp(E * 1j))
    # reformatting arrays so that they would be 2D
    if len(E.shape) == 1:
        E = E.reshape(-1,1)
    # initialize cost function and best parameters
    LL = np.inf
    B = [np.nan] * 3
    # starting parameters
    mu = np.angle(np.exp(2 * np.pi * np.random.random(size = (nruns, 1)) * 1))
    K = 0 + 10 * np.random.random(size = (nruns, 1))
    Pt = np.random.random(size = (nruns, 1))
 
    startB = np.hstack([mu, K, Pt])
    boundMu = [-np.pi, np.pi]
    boundK = [0, None]
    boundPT = [0, 1]
    bounds = np.array([
        boundMu,
        boundK,
        boundPT
    ])
    percDone = np.linspace(10, 100, 10).astype('int')
    for run in range(nruns):
        percentile = (run + 1) * 100. / nruns
        if percentile in percDone:
            print('{}% done'.format(percentile))
            percDone = np.array(percDone[1:])
        start = startB[run]
        fit = optimize.minimize(llhood, start,
                                args=(E),
                                bounds = bounds,
                                method = 'L-BFGS-B',
                                options = dict(
                                    disp = True,
                                    maxiter = maxiter
                                ))
        if fit['success'] and fit['fun'] < LL:
            LL = fit['fun']
            B = fit['x']

    return (B, LL)
# # %% reversals
# def llhood(B, E, NE):
#     '''
#     Compute maximum likelihood for:
#     1. the given set of parameters (B),
#     2. the vector of target errors (E),
#     3. the vector of non-target errors (NE)
#     '''
#     n = E.shape[0]
#     K, Pt, Pn = B
#     Pno = Pt * Pn * Pn # no motions reversed
#     Ple = Pt * (1 - Pn) * Pn # left motion reversed
#     Pri = Pt * Pn * (1 - Pn) # right motion reversed
#     Pbo = Pt * (1 - Pn) * (1 - Pn) # both motions reversed
#     Pu = 1 - Pt # guess probability

#     if Pu < 0:
#         # penalize negative guess rates
#         LL = -1e10 
#     else:
#         Wt = Pno * stats.vonmises.pdf(E, K)
#         Wu = Pu * np.ones([n, 1]) / (2 * np.pi)
#         Wn = np.array([Ple, Pri, Pbo])[None] * stats.vonmises.pdf(NE, K)

#         W = np.nansum(
#             np.hstack([Wt,Wn,Wu]),
#             axis = 1).reshape(-1,1)

#         LL = np.log(W).sum()
#         if LL > 0:
#             # penalize overfitting
#             LL = -1e10

#     return -LL

# def mmfit(
#     E = [np.nan], 
#     NE = [np.nan], 
#     nruns = 100, 
#     maxiter = 15000, 
#     random_seed = 0,
#     **kwargs
# ):
#     '''
#     Function to fit mixture distribution model to observed error magnitudes
#     relative to the target (E) and non-targets (NE)
#     '''
#     print('Model fitting for sub-{}'.format(random_seed))
#     np.random.seed(random_seed)
#     E = np.array(E)
#     NE = np.array(NE)
#     if np.isnan(E).any():
#         raise ValueError('NaN values in the input arrays')
#     E = np.angle(np.exp(E * 1j))
#     NE = np.angle(np.exp(NE * 1j))
#     # reformatting arrays so that they would be 2D
#     if len(E.shape) == 1:
#         E = E.reshape(-1,1)
#     if len(NE.shape) == 1:
#         NE = NE.reshape(-1,1)
#     # initialize cost function and best parameters
#     LL = np.inf
#     B = [np.nan] * 3
#     # starting parameters
#     K = 0 + 10 * np.random.random(size = (nruns, 1))
#     Pt, Pn = np.random.random(size = (nruns, 2)).T
#     if 'Pt' in kwargs.keys(): Pt = np.ones(nruns) * kwargs['Pt']
#     if 'Pn' in kwargs.keys(): Pn = np.ones(nruns) * kwargs['Pn']
#     startB = np.hstack([K, Pt[:, None], Pn[:, None]])
#     # parameter bounds
#     boundK = [0, None]
#     boundPT = [0, 1]
#     if 'Pt' in kwargs.keys(): boundPT = [kwargs['Pt']] * 2
#     boundPN = [0, 1]
#     if 'Pn' in kwargs.keys(): boundPN = [kwargs['Pn']] * 2
#     bounds = np.array([
#         boundK,
#         boundPT,
#         boundPN
#     ])
#     percDone = np.linspace(10, 100, 10).astype('int')
#     for run in range(nruns):
#         percentile = (run + 1) * 100. / nruns
#         if percentile in percDone:
#             print('{}% done'.format(percentile))
#             percDone = np.array(percDone[1:])
#         start = startB[run]
#         fit = optimize.minimize(llhood, start,
#                                 args=(E, NE),
#                                 bounds = bounds,
#                                 method = 'L-BFGS-B',
#                                 options = dict(
#                                     disp = True,
#                                     maxiter = maxiter
#                                 ))
#         if fit['success'] and fit['fun'] < LL:
#             LL = fit['fun']
#             B = fit['x']

#     return (B, LL)