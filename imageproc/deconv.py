from scipy.ndimage.filters import convolve1d
import numpy as np

EPS = np.finfo(float).eps


def deconv_lucy_richardson(img, psf, max_iter, axis=-1, init_img=None):
    '''1D Lucy Richardson Deconvolution'''
    assert(psf.ndim == 1)  # make sure PSF is 1D
    if init_img is None:
        u = img
    else:
        u = init_img
    psf_hat = psf[::-1]
    for i in xrange(max_iter):
        temp = convolve1d(u, psf, axis=axis)
        temp[temp == 0.0] = EPS
        u = u * convolve1d(img/temp, psf_hat, axis=axis)
    return u
