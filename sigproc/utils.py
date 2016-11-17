"""Utility functions for sigproc package
"""

import numpy as np
from scipy.interpolate import UnivariateSpline

def make_norm_dist(x, mean, sd):
    return 1.0/(sd*np.sqrt(2*np.pi))*np.exp(-(x-mean)**2/(2*sd**2))

def fwhm(y, x=None, return_edges=False):
    y = y - y.max()*0.5
    if x is None:
        x = np.arange(len(y))
    spline = UnivariateSpline(x, y, k=3, s=0)
    r1, r2 = spline.roots()[:2]
    if return_edges:
        return r1, r2
    else:
        return r2 - r1
