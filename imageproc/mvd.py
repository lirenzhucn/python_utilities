import numpy as np
from scipy.ndimage.filters import convolve, correlate
from scipy.ndimage.filters import gaussian_filter, laplace
from scipy.ndimage.measurements import maximum_position
from scipy.fftpack import fft2, fftn, ifftn
from scipy.signal import fftconvolve
import time
import sys


def pad_and_center_psf(psf, s):
    ss = psf.shape
    if len(ss) == 2:
        psf = np.pad(psf, ((0, s[0]-ss[0]), (0, s[1]-ss[1])),
                     mode='constant', constant_values=(0.0, 0.0))
    elif len(ss) == 3:
        psf = np.pad(psf,
                     ((0, s[0]-ss[0]), (0, s[1]-ss[1]), (0, s[2]-ss[2])),
                     mode='constant')
    mp = maximum_position(psf)
    for idx in xrange(len(ss)):
        psf = np.roll(psf, -mp[idx], axis=idx)
    return psf


def mvd_wiener(initImg, imgList, psfList, iterNum, mu, positiveOnly=True):
    if positiveOnly:
        initImg[initImg < 0.0] = 0.0
    viewNum = len(imgList)
    fftFactor = np.sqrt(initImg.shape[0]*initImg.shape[1])
    mu = mu * fftFactor
    I = np.sum(np.abs(initImg))
    e = fft2(initImg)
    e_img_old = initImg
    e_img = initImg
    if iterNum == 0:
        return e_img
    # pre-compute spectra
    ijList = [fftn(img) for img in imgList]
    pjList = [fftn(pad_and_center_psf(psf, initImg.shape)) for psf in psfList]
    for i in xrange(iterNum):
        c_all = np.zeros(e.shape, dtype=float)
        for j in xrange(viewNum):
            ij = ijList[j]
            pj = pjList[j]
            sj = e * pj
            cj = (np.conj(pj) * (ij - sj))/(np.square(np.abs(pj)) + mu**2)
            c_all = c_all + cj / float(viewNum)
        e = e + c_all
        e_img = np.real(ifftn(e))
        if positiveOnly:
            e_img[e_img < 0.0] = 0.0
        e_img = e_img / np.sum(np.abs(e_img)) * I
        e = fftn(e_img)
        print 'iter #%d, total change: %f.' %\
            (i+1, np.sum(np.abs(e_img_old-e_img))/I)
        e_img_old = e_img
    return e_img


def mvd_lr(initImg, imgList, psfList, iterNum):
    EPS = np.finfo(float).eps
    viewNum = len(imgList)
    initImg = initImg - np.amin(initImg)
    initImg = initImg / np.sum(np.abs(initImg))
    reconImg = initImg
    startTime = time.time()
    for i in xrange(iterNum):
        updateAll = np.ones(initImg.shape, dtype=float)
        for j in xrange(viewNum):
            img = imgList[j]
            psf = psfList[j]
            psf_prime = np.flipud(np.fliplr(psf))
            # update = convolve(img/(convolve(reconImg, psf)+EPS), psf_prime)
            update = fftconvolve(reconImg, psf, mode='same')
            update[update <= 0.0] = EPS
            update = img/update
            update = fftconvolve(update, psf_prime, mode='same')
            updateAll = updateAll * update
            # display progress
            progress = float(i*viewNum+j+1)/(viewNum*iterNum)
            timeElapsed = time.time() - startTime
            timeRemaining = timeElapsed/progress*(1-progress)
            sys.stdout.write('\r%.2f%%, %.2f s elapsed, %.2f s remaining' %
                             (progress*100.0, timeElapsed, timeRemaining))
            sys.stdout.flush()
        reconImg = reconImg * updateAll
        reconImg = np.abs(reconImg)
        reconImg = reconImg / np.sum(reconImg)
    sys.stdout.write('\n')
    return reconImg


def inner_product(x, y=None):
    if y is None:
        y = x
    return np.sum(x*y)


def index_min(vals):
    return min(xrange(len(vals)), key=vals.__getitem__)


def u_gamma(u, gamma, x, lap_reg):
    if lap_reg:
        return convolve(x, u) + gamma*laplace(x)
    else:
        return convolve(x, u) + gamma*x


def mapgg_step_size(x, d, u, v, w, gamma, t, t_xx, lap_reg):
    t_dd = np.square(d)
    t_xd = x * d
    ug_dd = u_gamma(u, gamma, t_dd, lap_reg)
    ug_xd = u_gamma(u, gamma, t_xd, lap_reg)
    q4 = inner_product(t_dd, ug_dd)
    q3 = 4*inner_product(t_dd, ug_xd)
    q2 = 4*inner_product(t_xd, ug_xd) + 2*inner_product(t_dd, t)
    q1 = 4*inner_product(t_xd, t)
    q0 = inner_product(t_xx, t-v) + w
    if q4 <= 0.0:
        print 'warning: negative q4'
    # find local extrema
    alphas = np.roots((4*q4, 3*q3, 2*q2, q1)).real
    vals = np.polyval((q4, q3, q2, q1, q0), alphas)
    return alphas[index_min(vals)]


def phi_eval(t_xx, t, v, w):
    return inner_product(t_xx, t-v) + w


def mvd_map_tikhonov(initImg, imgList, psfList, iterNum,
                     gamma, weights='even', sigma=None, lap_reg=False):
    viewNum = len(imgList)
    assert len(psfList) == viewNum
    if isinstance(weights, (list, tuple)):
        assert len(weights) == viewNum
        c2 = weights ** 2
    elif weights == 'even':
        c2 = [1.0] * viewNum
    # initialization ##
    print 'initializing...'
    # pre-blur inputs
    if sigma is not None:
        imgList = [gaussian_filter(img, sigma) for img in imgList]
        psfList = [gaussian_filter(psf, sigma) for psf in psfList]
    # check and process init. guess
    if iterNum == 0:
        return initImg
    initImg[initImg < 0.0] = 0.0
    x = np.sqrt(initImg)
    # initializing u, v, w
    u = np.zeros(psfList[0].shape)
    v = np.zeros(imgList[0].shape)
    w = 0.0
    for idx in xrange(viewNum):
        u = u + c2[idx] * correlate(psfList[idx], psfList[idx])
        v = v + c2[idx] * correlate(imgList[idx], psfList[idx])
        w = w + c2[idx] * inner_product(imgList[idx])
    # initializing previous gradient power
    p_rp = 1.0
    # initializing init. search direction
    d = np.zeros(imgList[0].shape)
    # start iteration ##
    print 'start iteration...'
    for k in xrange(iterNum):
        # temp. t_xx
        t_xx = np.square(x)
        # temp. t
        # t = convolve(t_xx, u) + gamma*t_xx - v
        t = u_gamma(u, gamma, t_xx, lap_reg) - v
        # gradient r
        r = 4*x*t
        # search direction
        p_r = inner_product(r)
        d = (p_r/p_rp)*d - r
        # step size
        alpha = mapgg_step_size(x, d, u, v, w, gamma, t, t_xx, lap_reg)
        # save previous power of r
        p_rp = p_r
        # update
        update = alpha*d
        x = x + update
        print 'iter #%d, residue: %f.' %\
            (k+1, phi_eval(t_xx, t, v, w)/w)
    # return result ##
    return np.square(x)
