"""
psd.py calculates the poles and weights of the Pade spectrum decomposition
(PSD) as given by Ref. [HXY10]_.

References
----------
.. [HXY10] J. Hu, R.-X. Xu, Y. Yan, J. Chem. Phys. 133, 101106 (2010)

"""
import numpy as np
from numpy import linalg as LA


def psd_freq(n):
    """
    calculates the PSD frequencies according to Eq. (14) in Ref. [HXY10]_.

    Parameters
    ----------
    n : int
        order of the PSD approximation

    Returns
    -------
    array_like
        n-dim array with PSD frequencies

    """
    k = np.arange(1, 2*n)
    entries = 1/np.sqrt((2*k+3)*(2*k+1))
    tri_matrix = np.diag(entries, k=1)
    tri_matrix += tri_matrix.T
    eig_vals, _ = LA.eig(tri_matrix)
    return 2/eig_vals[eig_vals >= 0]


def psd_weights(n, z):
    """
    calculates the PSD weights according to Eq. (6) in Ref. [HXY10]_

    Parameters
    ----------
    n : int
        order of the PSD approximation
    z : float

    Returns
    -------
    float
        value of the PSD weight

    """
    poly_a1, poly_b1, poly_c1 = 0.25, 3, 0
    poly_a2, poly_b2, poly_c2 = 1.25, 15+0.25*z, 0.25
    eta_k = poly_a2/poly_c2
    if n == 1:
        return 0.5*eta_k
    k = 3
    while k <= 2*n:
        poly_a3 = (2*k+1)*poly_a2 + 0.25*z*poly_a1
        poly_b3 = (2*k+1)*poly_b2 + 0.25*z*poly_b1
        poly_c3 = (2*k+1)*poly_c2 + 0.25*poly_b1 + 0.25*z*poly_c1

        scale = poly_a3
        k += 1
        poly_a1, poly_b1, poly_c1 = poly_a2/scale, poly_b2/scale, poly_c2/scale
        poly_a2, poly_b2, poly_c2 = poly_a3/scale, poly_b3/scale, poly_c3/scale
    return 0.5*poly_a3/poly_c3


def bose_func1(n, x):
    """
    calculates the PSD of the Bose function (see Eq. (5) Ref. [HXY10]_).

    Parameters
    ----------
    n : int
        order of the PSD approximation
    x: float
        argument of the Bose function

    Returns
    -------
    float
        Bose function

    """
    z = x**2
    phi_k = 1/12
    a_k = 5
    b_k = 5 + z/12
    k = 2
    while k <= 2*n:
        phi_k *= a_k/b_k
        k += 1
        a_k = 2*k+1 + 0.25*z/a_k
        b_k = 2*k+1 + 0.25*z/b_k
    return 1/x + 0.5 + x*phi_k


def bose_func2(n, x):
    """
    calculates the PSD of the Bose function (see Eq. (6) Ref. [HXY10]_).

    Parameters
    ----------
    n : int
        order of the PSD approximation
    x: float
        argument of the Bose function

    Returns
    -------
    float
        Bose function

    """
    xi_vals = psd_freq(n)
    eta_vals = psd_weights(n, -xi_vals**2)
    psd_sum = 2*x*np.sum(eta_vals/(x**2 + xi_vals**2))
    return 1/x + 0.5 + psd_sum


def psd(n):
    """
    returns nodes and weights of the PSD approximation for a given order `n`.

    Parameters
    ----------
    n : int
        order of the PSD approximation

    Returns
    -------
    xi_vals : array_like
        array of length `n` with PSD nodes
    eta_vals : array_like
        array of length `n` with PSD weights

    """
    xi_vals = psd_freq(n)
    eta_vals = psd_weights(n, -xi_vals**2)
    return xi_vals, eta_vals
