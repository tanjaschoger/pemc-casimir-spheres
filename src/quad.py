"""
quad.py calculates the nodes and weights for the Fourier-Chebyshev quadrature
and the  trapezoidal rule.

"""
from math import pi
import numpy as np


def quad_chebychev(deg):
    """
    nodes and weights for the Fourier-Chebyshev quadrature

    Parameters
    ----------
    deg : int
        defines the number of nodes

    Returns
    -------
    tuple

    References
    ----------
    The nodes and weights are defined in Eq. (10) of Ref. [Boy87]_.

    .. [Boy87]  J. P. Boyd, J. Sci. Comp. 2, 99 (1987)

    """
    n = np.arange(1, deg+1)
    tn = n*pi/(deg+1)
    j = n[:, np.newaxis]
    w_sum = np.sum(np.sin(j*tn)*(1-np.cos(j*pi))/j, axis=0)
    w = 4*np.sin(tn)/((deg+1)*(1-np.cos(tn))**2) * w_sum
    x = 1/np.tan(tn/2)**2
    return x, w


def quad_trapezoidal(deg):
    """
    nodes and weights for the trapezoidal rule

    Parameters
    ----------
    deg : int
        defines the number of nodes

    Returns
    -------
    tuple

    """
    x = np.arange(1, deg+1)/deg
    w = np.ones(deg, order='F')/deg
    return x, w
