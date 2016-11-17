'''utils for image processing package'''

import numpy as np


def normalize(img):
    img = img.astype(np.float_)
    return (img - np.amin(img)) / (np.amax(img) - np.amin(img))


def imgNegativeOnly(img, threhold=0.0):
    pass


import scipy.fftpack as spfft


def envelope2DQFT(img):
    img = img.astype(np.float)
    dxy = 1/20  # in mm
    umax = 1/dxy  # in mm^-1
    # input image related parameters
    [height, width] = img.shape
    size = max(height, width)
    # pad array to make it square
    img = np.pad(img, ((0, size-height), (0, size-width)), mode='constant')
    du = 2 / (size * dxy)
    # FT and frequency range
    fimg = spfft.fftshift(spfft.fft2(img))
    uRange = np.arange(-umax, umax, du)
    [uMap, vMap] = np.meshgrid(uRange, uRange)
    uNorm = np.sqrt(uMap ** 2 + vMap ** 2)
    # extract envelope
    uNorm[uNorm == 0.0] = np.finfo(float).eps
    # 2nd order hilbert transforms
    H1x = 1j*uMap/uNorm
    H1y = 1j*vMap/uNorm
    H2xy = - uMap*vMap/(uNorm ** 2)
    fh1 = np.real(spfft.ifft2(spfft.ifftshift(fimg * H1x)))
    fh2 = np.real(spfft.ifft2(spfft.ifftshift(fimg * H1y)))
    fh = np.real(spfft.ifft2(spfft.ifftshift(fimg * H2xy)))
    fA = np.sqrt(img ** 2 + fh1 ** 2 + fh2 ** 2 + fh ** 2)
    return fA
