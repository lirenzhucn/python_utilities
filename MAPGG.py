#!/usr/bin/env python3

import warnings
import numpy as np
from time import time


class MatrixOperator:
    def __init__(self, M):
        self._M = M

    def __call__(self, x):
        return np.dot(self._M, x.flatten())


def indexMin(vals):
    return min(range(len(vals)), key=vals.__getitem__)


def mapggStepSize(UA, v, w, gamma, UG, x, xx, d, t):
    dd = np.square(d)
    xd = x * d  # element-wise product
    ug_dd = UA(dd) + gamma*UG(dd)
    ug_xd = UA(xd) + gamma*UG(xd)
    # flatten everything
    dd = dd.flatten()
    xd = xd.flatten()
    ug_dd = ug_dd.flatten()
    ug_xd = ug_xd.flatten()
    xx = xx.flatten()
    t = t.flatten()
    v = v.flatten()
    q4 = np.dot(dd, ug_dd)
    q3 = 4*np.dot(dd, ug_xd)
    q2 = 4*np.dot(xd, ug_xd) + 2*np.dot(dd, t)
    q1 = 4*np.dot(xd, t)
    q0 = np.dot(xx, t - v) + w
    if q4 <= 0.0:
        warnings.warn('Negative q_4')
    alphas = np.roots((4*q4, 3*q3, 2*q2, q1)).real
    vals = np.polyval((q4, q3, q2, q1, q0), alphas)
    return alphas[indexMin(vals)]


def objFun(UA, v, w, gamma, UG, xx):
    return (np.dot(xx, UA(xx)) + gamma*np.dot(xx, UG(xx))
            + 2*np.dot(xx, v) + w)


def MAPGG_solver(y, A, At, gamma, **kwargs):
    """
    Usage x, objective, times = MAPGG_solver(y, A, At, gamma, **kwargs)

    This is a Python MAPGG solver for the following optimization problem

        arg min_x { || y - Ax ||_2^2 + gamma || Gx ||_2^2

    with the constraint of x being non-negative. A is a generic matrix
    and G is a regularization operator

    ===== Required inputs =====

    y: 1D vector or 2D array (image) of observation

    A: a callable acts as the forward operator

    At: a callable acts as the Hermitian adjoint operator

    gamma: regularization parameter, usually a non-negative real number
           of the objective function

    ===== Optional inputs =====

    G: a callable acts as the regularization operator

    stopCriterion: type of stopping criterion to use
                   1: relative change in objective function
                   2: norm of difference between two consecutive
                      estimates
                   3: objective function itself

    tolA: stopping threshold
          default = 0.01

    maxIterA: maximum number of iterations
              default = 1000

    minIterA: minimum number of iterations
              default = 5

    x0: initialization being {0, 1, 2, array}
        0 = initialize with zeros
        1 = initialize with random array
        2 = initialize with np.dot(At, y)
        array = user provided initial guess
        default = 0

    verbose: 0: silent, 1: progress, 2: information

    ===== Outputs =====

    x: solution

    objective: sequence of values of the objective function

    times: CPU time after each iteration
    """
    # default arguments
    UG = None
    stopCriterion = 1
    tolA = 1e-5
    maxIterA = 1000
    minIterA = 5
    x0 = 1
    verbose = 2
    for k in kwargs:
        if k == 'UG':
            UG = kwargs[k]
        elif k == 'stopCriterion':
            stopCriterion = kwargs[k]
        elif k == 'tolA':
            tolA = kwargs[k]
        elif k == 'maxIterA':
            maxIterA = kwargs[k]
        elif k == 'minIterA':
            minIterA = kwargs[k]
        elif k == 'x0':
            x0 = kwargs[k]
        elif k == 'verbose':
            verbose = kwargs[k]
        else:
            warnings.warn('Unknown keyword \'{:}\''.format(k))
    # objective function and time spent on each iteration
    times = []
    objective = []
    # build operators
    if hasattr(A, '__call__') and hasattr(At, '__call__'):
        At = At

        def UA(x):
            return At(A(x))
    elif hasattr(A, '__len__') and hasattr(At, '__len__'):
        At = MatrixOperator(At)
        UA = MatrixOperator(np.dot(At, A))
    else:
        raise ValueError('Having problem with either A or At')
    if hasattr(UG, '__call__'):
        UG = UG
    elif hasattr(UG, '__len__'):
        UG = MatrixOperator(UG)
    elif UG is None:
        def UG(x):
            return x
    else:
        raise ValueError('Having problem with UG')
    v = At(y)
    w = np.dot(y, y)
    # checking parameters
    if stopCriterion not in [1, 2, 3]:
        raise ValueError('Unknow stop criterion \'{:d}\''.format(stopCriterion))
    if x0 is 0:
        raise ValueError('Cannot initialize to zeros')
    elif x0 is 1:
        xt = At(np.zeros(y.shape))
        x0 = np.random.random(xt.shape)
    elif x0 is 2:
        x0 = At(y)
    # pre-processing
    startTime = time()
    x = np.sqrt(np.abs(x0))
    xx = np.square(x)
    fPrev = objFun(UA, v, w, gamma, UG, xx)
    d = np.zeros(x.shape)
    prPrev = 1.0
    xPrev = x
    times.append(time() - startTime)
    objective.append(fPrev)
    # main loop
    for k in range(maxIterA):
        t = UA(xx) + gamma*UG(xx) - v
        r = 4 * x * t
        pr = np.dot(r, r)
        d = pr/prPrev * d - r
        alpha = mapggStepSize(UA, v, w, gamma, UG, x, xx, d, t)
        # update x
        x = x + alpha * d
        # update other informations
        xx = np.square(x)
        f = objFun(UA, v, w, gamma, UG, xx)
        # compute stop criterion
        if stopCriterion == 1:
            # norm of diff
            criterion = abs((f-fPrev)/fPrev)
        elif stopCriterion == 2:
            # relative change in objective function
            criterion = np.linalg.norm(x.flatten() - xPrev.flatten()) /\
                np.linalg.norm(xPrev.flatten())
        elif stopCriterion == 3:
            # objective function itself
            criterion = f
        # update 'previous' results
        fPrev = f
        prPrev = pr
        xPrev = x
        # record results
        times.append(time() - startTime)
        objective.append(fPrev)
        # print info
        if verbose >= 2:
            message = 'Iter # {:3d}, objFun = {:9.5e}, criterion={:7.3e}'.\
                format(k, f, criterion/tolA)
            print(message)
        if (k >= minIterA) and (criterion < tolA):
            break
    # report results and return
    if verbose >= 1:
        print('\nFinished the main algorithm!\nResults:')
        print('Objective function: {:10.3e}'.format(objective[-1]))
        print('CPU time so far: {:10.3e}'.format(times[-1]))
    return (xx, objective, times)


if __name__ == '__main__':
    # a simple test
    from math import sqrt, pi
    import matplotlib.pyplot as plt
    from scipy.signal import filtfilt
    dx = 0.01
    x0 = 5.0
    sigma = 0.5
    x = np.arange(1000) * dx
    h = np.exp(- (x - x0)**2 / (2 * sigma**2)) / (sigma * sqrt(2*pi))
    y0 = np.zeros(1000)
    y0[500] = 1.0
    y = np.convolve(y0, h, mode='same')
    # add noise
    y = y + np.random.randn(1000) * 0.01
    h = h + np.random.randn(1000) * 0.001
    ht = h[::-1]
    # MAPGG solver

    def A(x):
        return np.convolve(x, h, mode='same')

    def At(x):
        return np.convolve(x, ht, mode='same')

    sigma1 = 0.01
    sigma2 = 0.02
    xt = np.arange(-6*sigma2, 6*sigma2, dx)
    """
    b = -1.0/pi/sigma0**4 * (1 - xt**2/2.0/sigma0**2) *\
        np.exp(- xt**2/2.0/sigma0**2)
    """
    b = np.exp(- xt**2 / (2 * sigma1**2)) / (sigma1 * sqrt(2*pi)) -\
        np.exp(- xt**2 / (2 * sigma2**2)) / (sigma2 * sqrt(2*pi))

    def UG(x):
        # b = np.array([1, -2, 1])
        return filtfilt(b, 1, x)

    # reconstruct with regularizer
    params = {
        'x0': 2,
        'tolA': 1e-6,
        'stopCriterion': 2,
        'verbose': 1,
        # 'minIterA': 1000,
        # 'maxIterA': 1000,
    }
    ym1, objective, times = MAPGG_solver(y, A, At, 1e-2, UG=UG, **params)

    # reconstruct without regularizer
    ym2, objective, times = MAPGG_solver(y, A, At, 0, **params)

    # display
    fig = plt.figure()
    # plt.plot(xt + x0, b/np.max(np.abs(b)), 'k')
    # plt.plot(x, y/np.max(y), 'b')
    plt.plot(x, y, 'b')
    plt.plot(x, ym1/np.max(ym1), 'r')
    plt.plot(x, ym2/np.max(ym2), 'g')
    plt.xlim([4, 6])
    plt.show()
