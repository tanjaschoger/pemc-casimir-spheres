r"""
angular_func.py calculates the angular functions associated with the Mie
scattering. The functions are defined in terms of the associated Legendre
polynomials

.. math::
    \pi_{j}(u) = \frac{P_j^1(u)}{\sqrt{u^2 - 1} } \\
        \tau_j(u) = - u\pi_j(u) + j(j+1)P_j(u)

and can be calculated by the following recurrence relations

.. math::
    j\pi_{j+1}(u) = (2j+1)u\pi_j(u) - (j+1)\pi_{j-1}(u)\\

        \tau_{j+1}(u) = (j+1)u \pi_{j+1}(u) - (j+2)\pi_{j}(u)

"""
import numpy as np
from numba import njit

from src.legendre import legendre_pl, ratio_10


@njit
def angularj(j, z, scaling=False):
    r"""
    calculates the angular functions :math:`\pi_j(\mathrm{cosh}(z))` and
    :math:`\tau_j(\mathrm{cosh}(z))` of order `j`. For `j<=1000` the angular
    functions are determined via their recurrence relations, for `j>1000` the
    definition with the associated Legendre polynomials is used.

    Notes
    -----
    If `scaling=True` the exponentially scaled angular functions are returned

    .. math::
        \pi_j(\mathrm{cosh}(z)) = e^{(j+1/2)z}
        \tilde{\pi}_j(\mathrm{cosh}(z)) \\

            \tau_j(\mathrm{cosh}(z)) = e^{(j+1/2)z}
            \tilde{\tau}_j(\mathrm{cosh}(z))


    Parameters
    ----------
    j : integer
        order of the angular functions
    z : float
        defines the argument of the angular functions

    Returns
    -------
    pi_j, tau_j :  tuple of floats
        value of the angular functions

    References
    ----------
    The definitions of the angular functions can be found in Ref. [vdH16]_

    .. [vdH16] H. C. van de Hulst, Light Scattering by small particles, Dover
        Publ. Inc. (2016)

    """
    if not scaling:
        scaling_fac = np.exp(z*(j+0.5))
    else:
        scaling_fac = 1

    pi_jm1 = 0
    pi_j = np.exp(-1.5*z)
    tau_j = 0.5*(np.exp(-0.5*z) + np.exp(-2.5*z))
    cosh_z = np.cosh(z)
    exp_mz = np.exp(-z)
    if j == 1:
        return pi_j*scaling_fac, tau_j*scaling_fac
    if j <= 1000 or z==0:
        for n in range(1, j):
            pi_jp1 = (2*n+1)*cosh_z*pi_j/n*exp_mz - (n+1)*pi_jm1/n * exp_mz**2
            pi_jm1 = pi_j
            pi_j = pi_jp1
        return (pi_jp1*scaling_fac,
                (j*cosh_z*pi_j - (j+1)*pi_jm1*exp_mz)*scaling_fac)

    # angular functions for j > 1000
    lp0 = legendre_pl(j, z)*scaling_fac
    lp1 = lp0*ratio_10(j, z)
    pi_j = lp1/abs(np.sinh(z))
    tau_j = - cosh_z*pi_j + j*(j+1)*lp0
    return pi_j, tau_j


@njit
def angularj_large_arg(j, z):
    r"""
    calculates the scaled angular functions for large arguments `z` and
    order `j`.

    Parameters
    ----------
    j : int
        order of the angular functions
    z : float
        defines argument `cosh(z)` of the angular functions

    Returns
    -------
    tuple of floats
        values of the angular functions

    Notes
    -----
    The scaled angular functions are defined by

    .. math::
        \pi_j(u) = \frac{(2j)!}{2^{j}(j-1)! j!} u^{j-1} \tilde{\pi}_j(u)\\

            \tau_j(u) = \frac{(2j)!}{2^{j}(j-1)!(j-1)!} u^j
            \tilde{\tau}_j(u)

    where the asymptotic expressions of the associated Legendre polynomials
    for large arguments was used (see Eq. (14.8.12) of Ref. [DLMF14]_)

    .. [DLMF14] https://dlmf.nist.gov/14.8#iii

    See Also
    --------
    angularj

    """
    pi_jm1 = 0
    pi_j = 1
    tau_j = 1
    if j == 1:
        return pi_j, tau_j
    sech2_z = 4*np.exp(-2*z)/(1+np.exp(-2*z))**2
    for n in range(1, j):
        pi_jp1 = pi_j - pi_jm1*sech2_z*(n**2-1)/(4*n**2 - 1)
        pi_jm1 = pi_j
        pi_j = pi_jp1
    return pi_j, pi_j - (j**2 - 1)*pi_jm1*sech2_z/(j*(2*j-1))
