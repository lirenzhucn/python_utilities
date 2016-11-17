#!/usr/bin/env python3

import warnings
import numpy as np
from math import sqrt
from time import time
import scipy.ndimage as spnd

# an arbitary sentinal upper bound for maxSvd
# if maxSvd hits this during monotonic optimization, terminates the process
MAX_SVD_UPPER_BOUND = 1.0e20


def TwIST(y, A, At, tau, callback=None, **kwargs):
    """
    Usage x, objective, times = TwIST(y, A, At, tau, **keywords)

    This is a Python mock for the MATLAB TwIST_v2 toolbox, which solves
    the regularization problem

        arg min_x { 0.5*|| y - Ax ||_2^2 + tau phi(x) },

    where A is a generic matrix and phi(.) is a regularization function
    such that the solution of the denoising problem

        Psi(y, th) = arg min_x { 0.5*|| y - Ax ||_2^2 + th phi(x) },

    is known.

    ===== Required inputs =====

    y: 1D vector or 2D array (image) of observation

    A: a callable acts as the forward operator

    At: a callable acts as the Hermitian adjoint operator

    tau: regularization parameter, usually a non-negative real number
         of the objective function

    ===== Optional inputs =====

    psi: denoising function handle (callable)
         default = soft threshold

    phi: regularization term function handle (callable)
         default = ||x||_1

    lam1: lam1 paramter in the TwIST algorithm
          rule of thumb:
              lam1 = 1e-4 for severely ill-conditioned problems
              lam1 = 1e-2 for mildly ill-conditioned problems
              lam1 = 1    for unitary direct operators

          default = 0.04

    alpha: parameter alpha of TwIST (see ex. (22) of the paper)
           default = alpha(lamN=1, lam1)

    beta: paramter beta of TwIST (see ex. (23) of the paper)
          default = beta(lamN=1, lam1)

    stopCriterion: type of stopping criterion to use
                   0: change in the number of non-zero components
                      of the estimate
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

    monotone: True or False
              default = True

    sparse: True or False. accelarates the convergence rate when
            the regularizer phi(x) is sparse inducing, such as ||x||_1

    verbose: 0: silent, 1: progress, 2: information

    ===== Outputs =====

    x: solution

    objective: sequence of values of the objective function

    times: CPU time after each iteration
    """
    # default arguments
    psiFunc = soft
    phiFunc = phiL1
    stopCriterion = 1
    tolA = 0.01
    maxIterA = 1000
    minIterA = 5
    x0 = 0
    monotone = True
    sparse = False
    threshold = None
    verbose = 1
    alpha = 0
    beta = 0
    lam1 = 1e-4
    lamN = 1
    # unpack keyword arguments
    for k in kwargs:
        if k == 'psi':
            psiFunc = kwargs[k]
        elif k == 'phi':
            phiFunc = kwargs[k]
        elif k == 'lam1':
            lam1 = kwargs[k]
        elif k == 'alpha':
            alpha = kwargs[k]
        elif k == 'beta':
            beta = kwargs[k]
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
        elif k == 'monotone':
            monotone = kwargs[k]
        elif k == 'sparse':
            sparse = kwargs[k]
        elif k == 'verbose':
            verbose = kwargs[k]
        elif k == 'threshold':
            threshold = kwargs[k]
        else:
            warnings.warn('unknown keyword \'{:}\''.format(k))
    # twist parameters
    rho0 = (1-lam1/lamN)/(1+lam1/lamN)
    if alpha == 0:
        alpha = 2/(1+sqrt(1-rho0**2))
    if beta == 0:
        beta = alpha*2/(lam1+lamN)
    if stopCriterion not in [0, 1, 2, 3]:
        raise ValueError('Unknow stop criterion \'{:d}\''.format(stopCriterion))
    # ====================
    if x0 is 0:
        xt = At(np.zeros(y.shape))
        x0 = np.zeros(xt.shape)
    elif x0 is 1:
        xt = At(np.zeros(y.shape))
        x0 = np.random.random(xt.shape)
    elif x0 is 2:
        x0 = At(y)
    # initialize x estimate
    x = x0
    # precompute At * y
    # Aty = At(y)
    # non-zeros in estimate x
    nzX = np.count_nonzero(x)
    xPrev = x
    # compute and store initial value of the objective function
    resid = y - A(x)
    fPrev = 0.5*(np.dot(resid.flatten(), resid.flatten())) + tau*phiFunc(x)
    # start timer
    startTime = time()
    times = []
    times.append(time() - startTime)
    objective = []
    objective.append(fPrev)
    # continue flag
    contOuter = True
    iterNum = 1
    # invoke callback function
    if callback:
        callback(-1, x, fPrev, 'cube')
    if verbose >= 1:
        print('Initial objective = {:10.6e}, nonzeros={:7d}'.format(fPrev, nzX))
    # variables controling first and second order iterations
    itersIST = 0
    itersTwIST = 0
    # initialize
    xm2 = x
    xm1 = x
    # ============================
    # ===== TwIST iterations =====
    # ============================
    maxSvd = 1.0
    while contOuter:
        grad = At(resid)
        while True:
            # IST estimate
            temp = xm1 + grad/maxSvd
            x = psiFunc(temp, tau/maxSvd)
            if callback:
                callback(iterNum, x, 0.0, callType='cube')
            # apply constraint
            if threshold is not None:
                x[x < threshold] = 0.0
            if (itersIST >= 2) or (itersTwIST != 0):
                # set to zero the past when the present is zero
                # suitable for sparse inducing priors
                if sparse:
                    mask = (x != 0.0)
                    xm1 = xm1 * mask
                    xm2 = xm2 * mask
                # two step iteration
                xm2 = (alpha-beta)*xm1 + (1-alpha)*xm2 + beta*x
                # compute residual
                resid = y - A(xm2)
                f = 0.5*(np.dot(resid.flatten(), resid.flatten())) +\
                    tau*phiFunc(xm2)
                if (f > fPrev) and monotone:
                    itersTwIST = 0
                else:
                    itersTwIST = itersTwIST + 1
                    itersIST = 0
                    x = xm2
                    if itersTwIST % 100000 == 0:
                        maxSvd = 0.9 * maxSvd
                    break
            else:
                resid = y - A(x)
                if callback:
                    callback(iterNum, resid, 0.0, callType='resid')
                f = 0.5*(np.dot(resid.flatten(), resid.flatten())) +\
                    tau*phiFunc(x)
                if f > fPrev:
                    # if monotonicity fails here is because
                    # max eig (A'A) > 1, we increase our guess of maxSvd
                    maxSvd = 2*maxSvd
                    if maxSvd > MAX_SVD_UPPER_BOUND:
                        print('[WARNING] maxSvd upper bound reached')
                        return (x, objective, times)
                    if verbose >= 2:
                        print('Objective = {:9.5e}. Incrementing S = {:2.2e}'.format(
                            f, maxSvd))
                    itersIST = 0
                    itersTwIST = 0
                else:
                    itersTwIST = itersTwIST + 1
                    break  # break loop while
        xm2 = xm1
        xm1 = x
        # update the number of nonzero components and its variation
        nzX = np.count_nonzero(x)
        numChangesActive = x.size - np.sum(np.isclose(xPrev, x))
        # check stop criterion
        if stopCriterion == 0:
            # compute the stopping criterion based on the change
            # of the number of non-zero components of the estimate.
            criterion = numChangesActive
        elif stopCriterion == 1:
            # compute the stopping criterion based on the relative
            # variation of the objective function.
            criterion = abs(f-fPrev)/fPrev
        elif stopCriterion == 2:
            # compute the stopping criterion based on the relative
            # variation of the estimate.
            criterion = np.linalg.norm(x.flatten() - xm1.flatten()) /\
                np.linalg.norm(x.flatten())
        elif stopCriterion == 3:
            # objective function value
            criterion = f
        else:
            raise ValueError('Unknown stopping criterion')
        contOuter = ((iterNum <= maxIterA) and (criterion > tolA))
        if iterNum <= minIterA:
            contOuter = True
        iterNum = iterNum + 1
        fPrev = f
        objective.append(f)
        times.append(time() - startTime)
        # invoke callback for each iteration
        if callback:
            callback(iterNum, x, f, 'cube')
        # print out the various stopping criteria
        if verbose >= 1:
            print(('Iteration = {:d}, objective = {:9.5e}, nz = {:7d}, ' +
                   'criterion = {:7.3e}').format(iterNum, f,
                                                 nzX, criterion/tolA))
    # ============================
    # ===== end of main loop =====
    # ============================

    if verbose >= 1:
        print('\nFinished the main algorithm!\nResults:')
        print('|| A x - y ||_2 = {:10.3e}'.
              format(np.linalg.norm(resid.flatten())))
        print('||x||_1 = {:10.3e}'.format(np.sum(np.abs(x))))
        print('Objective function = {:10.3e}'.format(f))
        print('Number of non-zero componets = {:d}'.format(nzX))
        print('CPU time so far = {:10.3e}'.format(times[-1]))
    return (x, objective, times)


def modulo(x):
    R = np.sqrt(np.sum(np.square(x), axis=1))
    R = np.tile(np.reshape(R, (-1, 1)), (1, 2))
    return R


def projk(g, lam, opQ, opQt, niter):
    """
    Usage:
        function u = projk(g, lam, opQ, opQt, niter)

    Chambolle projection's algorithm from: 2004,
    "An algorithm for total variation minimization and applications"

    The algorithm solves the following problem:
        arg min_u = 0.5*|| u - g ||_2^2 + lam ||op(u)||_1     (1)

    The solution of (1) is given by:
        u = g - projk(g)

    Parameters:
        g:     noisy image (size X: ny*nx)
        lam:   lam parameter in eq. (1)
        opQ:   Functional Q(u): X -> X1*X2 (product spaces)
               The return of this function should be a matrix with
               two columns in this form:
                   [X1(:)' X2(:)'] (size: (ny*nx)*2)
        opQt:  Adjoint operator of Q(u). Qt: X1*X2 -> X
        niter: Number of iterations
    """
    tau = 0.25
    uy, ux = g.shape
    pn = np.zeros((uy*ux, 2))
    for i in range(niter):
        S = opQ(-opQt(pn)-g/lam)
        pn = (pn+tau*S)/(1+tau*modulo(S))
    u = -lam*opQt(pn)
    return u


def tvDenoise(x, lam, tvIters):
    uy, ux = x.shape
    dh = lambda x: spnd.filters.convolve1d(x, [1, -1, 0], axis=0, mode='wrap')
    dv = lambda x: spnd.filters.convolve1d(x, [1, -1, 0], axis=1, mode='wrap')
    dht = lambda x: spnd.filters.convolve1d(x, [0, -1, 1], axis=0, mode='wrap')
    dvt = lambda x: spnd.filters.convolve1d(x, [0, -1, 1], axis=1, mode='wrap')
    im2vec = lambda x: np.reshape(np.ravel(x), (-1, 1))
    vec2im = lambda x: np.reshape(x, (uy, ux))
    opQ = lambda x: np.concatenate((im2vec(dh(x)), im2vec(dv(x))), axis=1)
    opQt = lambda x: (dht(vec2im(x[:,0])) + dvt(vec2im(x[:,1])))
    return x - projk(x, lam/2, opQ, opQt, tvIters)


def tvDenoise_old(x, lam, tvIters):
    # assert len(x.shape) == 2
    dt = 0.25
    N = x.shape
    divp = np.zeros(N)
    p1 = np.zeros(N)
    p2 = np.zeros(N)
    id = list(range(1, N[0])) + [N[0]-1]
    iu = [0] + list(range(0, N[0]-1))
    ir = list(range(1, N[1])) + [N[1]-1]
    il = [0] + list(range(0, N[1]-1))
    for i in range(tvIters):
        # lastdivp = divp
        z = divp - x * lam
        z1 = z[:, ir] - z
        z2 = z[id, :] - z
        denom = 1 + dt*np.sqrt(np.square(z1) + np.square(z2))
        p1 = (p1 + dt*z1) / denom
        p2 = (p2 + dt*z2) / denom
        divp = p1 - p1[:, il] + p2 - p2[iu, :]
    return x - divp / lam


def tvNorm(x):
    # diffh = spnd.filters.prewitt(x, axis=0, mode='wrap')
    # diffv = spnd.filters.prewitt(x, axis=1, mode='wrap')
    # return np.sum(np.sqrt(np.square(diffh) + np.square(diffv)))
    diffh = np.diff(x, axis=0, n=1)
    diffv = np.diff(x, axis=1, n=1)
    return np.sum(np.sqrt(np.square(diffh[:, :-1]) + np.square(diffv[:-1, :])))


def phiL1(x):
    return np.sum(np.abs(x))


def soft(x, tau):
    y = np.maximum(np.abs(x) - tau, 0)
    y = y/(y + tau) * x
    return y


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    n = 4096
    k = 1024
    nSpikes = 160
    x = np.zeros(n)
    q = np.random.permutation(n)
    x[q[0:nSpikes]] = np.sign(np.random.randn(nSpikes))
    # measurement matrix
    print('Building measurement matrix...')
    R = np.random.randn(k, n)
    print('Finished creating matrix...')
    # hR = lambda x: np.dot(R, x)
    # hRt = lambda y: np.dot(R.T, y)

    def hR(x):
        return np.dot(R, x)

    def hRt(y):
        return np.dot(R.T, y)

    # noise variance
    sigma = 1e-2
    y = hR(x) + sigma*np.random.randn(k)
    tau = 0.1 * np.amax(np.abs(hRt(y)))
    lam1 = 0.001
    tolA = 1e-5
    x_twist, objective, times = TwIST(y, hR, hRt, tau, lam1=lam1,
                                      monotone=True, sparse=True,
                                      x0=0, stopCriterion=1, tolA=tolA,
                                      maxIterA=100, verbose=1)
    # display
    fig = plt.figure()
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    ax.plot(np.arange(n), x_twist, 'b', np.arange(n), x+2.5, 'k')
    plt.show()
