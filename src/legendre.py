"""
legendre.py calculates the associated Legendre polynomials
:math:`P_j^m(cosh(x))` of order `m=0` and `m=1`.

"""
from math import pi, sqrt
from scipy.special import gammaln
import numpy as np
from numba import njit

import src.bessel as bsl


@njit
def ratio_10(j, x, eps=1e-17):
    r"""
    calculates the ratio
    :math:`P_j^1(\mathrm{cosh}(x))/P_j^0(\mathrm{cosh}(x))`
    by a continued fraction.

    Parameters
    ----------
    j : int
        order of the Legendre polynomial
    x : float
        `cosh(x)` argument of the Legendre polynomial
    eps : float, default=1e-17
        accuracy of the continued fraction value

    Returns
    -------
    float
        value of the scaled Legendre polynomial

    References
    ----------
    The continued fraction is based on the recurrence relation Eq. (14.10.6)
    in https://dlmf.nist.gov/14.10 .

    """
    n_max = 1
    n = np.arange(0, 2)
    an = (j+1+n)*(j-n)
    bn = 2*(n+1)/abs(np.tanh(x))
    gn = bn[0]
    p_n = bn[1] + an[1]/bn[0]
    q_n = bn[1]
    r = p_n/q_n
    gn *= r
    while abs(r-1) > eps:
        n_max += 1
        a_nmax = (j+1+n_max)*(j-n_max)
        b_nmax = 2*(n_max+1)/abs(np.tanh(x))
        p_n = b_nmax + a_nmax/p_n
        q_n = b_nmax + a_nmax/q_n
        r = p_n/q_n
        gn *= r
    return an[0]/gn


@njit
def f2n_coef(x):
    """
    calculates the exponentially scaled coefficients for the series expansion
    of the Legendre polynomials.

    Parameters
    ----------
    x : float
        `cosh(x)` defines the argument of the Legendre polynomial

    Returns
    -------
    list
        list of the first 7 expansion coefficients

    References
    ----------
    See Eq. (3.30) in Ref. [Bo12]_

    .. [Bo12] I. Bogaert, et al. SIAM J. Sci. Comput. 34, C83 (2012)

    """
    n = 12
    hn = []
    for k in range(n+1):
        hn.append((-x)**k * bsl.In(k, x, scaling=True))
    f0 = hn[0]
    f2 = hn[1]/8 - hn[2]/12
    f4 = 11*hn[2]/384 - 7*hn[3]/160 + hn[4]/160
    f6 = 173*hn[3]/15360 - 101*hn[4]/3584 + 671*hn[5]/80640 - 61*hn[6]/120960
    f8 = (22931*hn[4]/3440640 - 90497*hn[5]/3870720 + 217*hn[6]/20480
          - 1261*hn[7]/967680 + 1261*hn[8]/29030400)
    f10 = (1319183*hn[5]/247726080 - 10918993*hn[6]/454164480
           + 1676287*hn[7]/113541120 - 7034857*hn[8]/2554675200
           + 1501*hn[9]/8110080 - 79*hn[10]/20275200)
    f12 = (233526463*hn[6]/43599790080 - 1396004969*hn[7]/47233105920
           + 2323237523*hn[8]/101213798400 - 72836747*hn[9]/12651724800
           + 3135577*hn[10]/5367398400 - 1532789*hn[11]/61993451520
           + 66643*hn[12]/185980354560)
    return [f0, f2, f4, f6, f8, f10, f12]


@njit
def legendre_pl(j, x):
    """
    calculates the exponentially scaled Legendre polynomials of order `j` and
    argument `cosh(x)`.

    Parameters
    ----------
    j : int
        order of the Legendre polynomial
    x : float
        `cosh(x)` argument of the Legendre polynomial

    Returns
    -------
    float
        value of the scaled Legendre polynomial

    Notes
    -----
    For `j > 1000` asymptotic relations derived by Ref. [Bo12]_ and adopted by
    Ref. [Ha19]_ are used.

    .. [Ha19] M. Hartman, PhD thesis (2019)

    """
    # Eq. (3.4) in Ref. [1]
    if (j+1)*np.sinh(x) > 25:
        m_max = 20
        pl = 0
        # see Eq. 5.11.13 in https://dlmf.nist.gov/5.11
        clm = (1/sqrt(j) - 3/(8*j**(1.5))
               + 25/(128*j**(2.5)) - 105/(1024*j**(3.5))
               + 1659/(32768*j**(4.5)) - 6237/(262144*j**(5.5))
               + 50765/(4194304*j**(6.5)) - 242385/(33554432*j**(7.5)))
        for m in range(m_max):
            pl += 2**m * clm*(1 + np.exp(-(2*m+2*j+1)*x))/(1 - np.exp(-2*x))**m
            clm *= np.exp(2*np.log(m+0.5) - np.log(2) - np.log(m+1)
                          - np.log(j+m+1.5))
        pl *= 0.5*np.sqrt(2/(pi*np.sinh(x)))
        return pl

    # for (j+1)*np.sinh(x) <= 25 use Eq. (3.19) in Ref. [1]
    f2n = f2n_coef(x*(j+0.5))
    pl = 0
    for n, f2n_val in enumerate(f2n):
        pl += f2n_val*np.exp(-2*n*np.log(j+0.5))
    return pl


def legendre_pl1(j, x):
    """
    returns the exponentially scaled associated Legendre polynomials of order
    `j` and degree `m=1`.

    Parameters
    ----------
    j : int
        order of the Legendre polynomial
    x : float
        `cosh(x)` argument of the Legendre polynomial

    Returns
    -------
    float
        value of the scaled Legendre polynomial
    """
    return legendre_pl(j, x)*ratio_10(j, x)


def coeff_pl1(x):
    """
    expansion coefficients for the associated Legendre polynomials.

    Parameters
    ----------
    x : float
        defines the argument `cosh(x)` of the associated Legednre polynomial

    Returns
    -------
    list
        list of the first 6 expansion coefficients

    References
    ----------
    See Eq. (7.23) in Ref. [Sp20]_

    .. [Sp20] B. Spreng, PhD thesis (2020)

    """
    c0 = 1
    c1 = (1-x/np.tanh(x))/(8*x**2)
    c2 = (8*x**2 - 3*x**2/np.tanh(x)**2 - 18*x/np.tanh(x) + 21)/(384*x**4)
    c3 = (-3*x**3/np.tanh(x)**3 + 40*x**2 - 15*x**2/np.tanh(x)**2
          - 81*x/np.tanh(x) + 99)/(3072*x**6)
    d4 = (64*x**4 - 225*x**4/np.tanh(x)**4 - 1260*x**3/np.tanh(x)**3
          + 13200*x**2 + 30*(8*x**2 - 165)*x**2/np.tanh(x)**2
          - 25740*x/np.tanh(x) + 32175)
    c4 = d4/(1474560*x**8)
    d5 = (-105*x**5/np.tanh(x)**5 + 192*x**4 - 675*x**4/np.tanh(x)**4
          - x*(64*x**4 + 49725)/np.tanh(x) + 26000*x**2
          + 30*x**2*(24*x**2 - 325)/np.tanh(x)**2
          + 10*x**3*(16*x**2 - 273)/np.tanh(x)**3 + 62985)
    c5 = d5/(3932160*x**10)
    return [c0, c1, c2, c3, c4, c5]


def asymlegendre_l1(j, x):
    """
    calculates the associated Legendre polynomials of order `j` and degree
    `m=1` from an asymptotic expansion for large `j`.

    Parameters
    ----------
    j : int
        order of the Legendre polynomial
    x : float
        `cosh(x)` defines the argument of the Legendre polynomial

    Returns
    -------
    float
        value of the Legendre polynomial

    References
    ----------
    See Eq. (29.3.71) in Ref. [Tem14]_ and Eq. (7.20) in [Sp20]_

    .. [Tem14] N. M. Temme, Asymptotic Methods for Integrals, world scientific,
        (2014)
    """
    plm1 = 0
    for k, ck in enumerate(coeff_pl1(x)):
        bessel = bsl.In(k+1, (j+0.5)*x, scaling=True)
        plm1 += ck*np.exp(np.log(j) + np.log(j+1) + gammaln(k+1.5)
                          - gammaln(1.5) - (k+1)*np.log(j+0.5)
                          )*bessel*(2*x)**k
    return np.sqrt(x/np.sinh(x))*plm1


def asym_l1(j, x):
    """
    calculates the associated Legendre polynomials of order `j` and degree
    `m=1` from an asymptotic expansion for large `j`.

    Parameters
    ----------
    j : int
        order of the Legendre polynomial
    x : float
        `cosh(x)` defines the argument of the Legendre polynomial

    Returns
    -------
    float
        value of the Legendre polynomial

    References
    ----------
    See Eq. (1.1) in Ref. [SW88]_

    .. [SW88] P. N. Shivakumar and R. Wong, Quart. Appl. Math, 46, 3 (1988)

    """
    # Eq. (2.5) in [1]
    def phi(nu_max, z):
        nu = np.arange(0, nu_max+1)
        phi_nu = np.exp(-gammaln(nu+2))*bsl.modbesselI(nu_max, z)/(2*z)**nu
        return np.sqrt(pi*z/2)/np.sinh(z)*phi_nu

    # Eq. (2.7) in [1]
    def psi(nu_max, z):
        psi_nu = []
        psi_nu.append(1)
        for nu in range(0, nu_max):
            phi_nu = phi(nu+1, z)
            temp = 0
            for j in range(nu+1):
                temp += (0.5 - 1.5*j/(nu+1))*phi_nu[nu+1-j]*psi_nu[j]
            psi_nu.append(temp)
        return psi_nu

    # Eq. (2.14) in [1]
    def coef(nu_max, z):
        nu = np.arange(0, nu_max+1)
        return (-1)**nu*np.exp(gammaln(nu+3/2)
                               - gammaln(3/2))*(2*z)**nu*psi(nu_max, z)

    plm1 = 0
    for k, ck in enumerate(coef(9, x)):
        bessel = bsl.In(k+1, (j+0.5)*x, scaling=True)
        plm1 += ck*bessel*np.exp(-(k+1)*np.log(j+0.5))
    return j*(j+1)*np.sqrt(x/np.sinh(x))*plm1
