"""
bessel.py calculates the modified Bessel functions of the first and second kind
and of half-integer and integer order.

"""
from math import pi
from itertools import accumulate
from numba import njit
import numpy as np


def n_max(x):
    """
    computes maximal order where the Mie series is truncated.

    Parameters
    ----------
    x: float
        specifies the product of the wave number and sphere radius

    Returns
    -------
    int
        order of the Mie series

    References
    ----------
    For details see p. 1508 of Ref. [Wis80]_.

    .. [Wis80] W. J. Wiscombe, Appl. Opt. 19, 1505-1509 (1980)
        (DOI:10.1364/AO.19.001505)

    """
    abs_x = abs(x)
    if 0 <= abs_x < 0.02:
        return 5
    if 0.02 <= abs_x < 8:
        return int(round(abs_x+4.0*abs_x**0.3333 + 1))
    if 8 <= abs_x < 4200:
        return int(round(abs_x+4.05*abs_x**0.3333 + 2))
    return int(round(abs_x+4.0*abs_x**0.3333 + 2))


def cfraction(an):
    """
    recursively calculates the finite part of a continued fraction.

    Parameters
    ----------
    an : array_like
        array with coeffiecients of the continued fraction

    Returns
    -------
    float
        value of the continued fraction

    Notes
    -----
    For more details on continued fractions see e. g. §3.10 of Ref. [DLMF3]_.

    .. [DLMF3] https://dlmf.nist.gov/3.10#i

    """
    if len(an) == 1:
        return an[0]
    # Eq. (5) in [1]
    return an[0] + (1 / cfraction(an[1::]))


@njit
def ratio(nu, z, eps=1e-17):
    r"""
    calculates the ratio :math:`I_{\nu-1}(z)/I_{\nu}(z)` of consecutive
    Bessel functions of the first kind using the continued fraction method
    with fixed accuracy `eps`.

    Parameters
    ----------
    nu : int or float
        defines the order of the Bessel functions
    z : float
        argument of the Bessel functions
    eps : float, default=1e-17
        accuracy of the continued fraction value

    Returns
    -------
    float
        continued fraction

    References
    ----------
    See Eq. (14-17) of Ref. [Len73]_.

    .. [Len73] W.J. Lentz, Tech. rep., DTIC Document (1973)

    """
    a1 = 2*nu/z
    res = a1
    a2 = 2*(nu+1)/z
    p_n = a2 + 1/a1
    q_n = a2
    r = p_n/q_n
    res *= r
    nmax = 2
    while abs(r-1) > eps:
        nmax += 1
        a_nmax = (2*(nu + nmax-1))/z
        p_n = a_nmax + 1/p_n
        q_n = a_nmax + 1/q_n
        r = p_n/q_n
        res *= r
    return res


def modbesselI(nmax, z):
    r"""
    calculates the modified Bessel functions of the first kind and of half-
    integer order `n+1/2` with `n` from 0 to `nmax` with a downward recursion

    .. math::
        I_{n-3/2}(z) = \frac{2n-1}{z}I_{n-1/2}(z) + I_{n+1/2}


    Parameters
    ----------
    nmax : int
        determines the maximal order of the Bessel function
    z : float
        argument of the Bessel function

    Returns
    -------
    inu : array_like
        array of length nmax+1

    References
    ----------
    For details see chap. 6 in [Pre92]_ .

    .. [Pre92] W. H. Press, et al., Numerical Recipes in C: The Art of
        Scientific Computing (Cambridge University Press, 1992)

    """
    inu = np.empty(nmax+1, dtype=np.complex128)
    inu_0 = np.sinh(z)*np.sqrt(2/(pi*z))
    a = 1
    inu[nmax] = a
    inu[nmax-1] = ratio(nmax+1/2, z)*a
    for k in range(2, nmax+1):
        inu[nmax-k] = inu[nmax-k+2] + (2*(nmax-k)+3)*inu[nmax-k+1]/z
        if abs(inu[nmax-k]) > 1e10:
            inu *= 1e-10
    a = inu_0/inu[0]
    inu = inu*a
    return inu


def scaled_modbesselI(nmax, z):
    r"""
    calculates the scaled modified Bessel functions of the first kind and of
    half-integer order `n+1/2` with `n` from 0 to `nmax`.

    .. math::
        I_{n+1/2}(z) = \frac{z^{n+1/2} 2^{n+1} (n+1)!}{(2n+2)!}
        \tilde{I}_{n+1/2}(z)

    Parameters
    ----------
    nmax : int
        determines the maximal order of the Bessel function
    z : float
        argument of the Bessel function

    Returns
    -------
    inu : array_like
        array of length nmax+1

    References
    ----------
    See also Eq. (10.25.2) in https://dlmf.nist.gov/10.25 .

    """
    inu = np.empty(nmax+1, dtype=np.complex128)
    inu_0 = np.sinh(z)*np.sqrt(2/pi)/z
    a = 1
    inu[nmax] = a
    inu[nmax-1] = ratio(nmax+1/2, z)*a*z/(2*nmax+1)
    for k in range(2, nmax+1):
        inu[nmax-k] = inu[nmax-k+2]*z**2/(4*(nmax-k+2)**2-1) + inu[nmax-k+1]
        if abs(inu[nmax-k]) > 1e10:
            inu *= 1e-10
    a = inu_0/inu[0]
    inu = inu*a
    return inu


@njit
def arg(n, z):
    """
    defines the argument of the exponential for the modified Bessel functions.

    Parameters
    ----------
    n : int
    z : float

    Returns
    -------
    float

    References
    ----------
    See also Eq. (10.41.7) in https://dlmf.nist.gov/10.41 .

    """
    lamb = (n+0.5)/z
    return np.sqrt(1+lamb**2) - lamb*np.arcsinh(lamb)


def scaled2_modbesselI(nmax, z):
    r"""
    calculates the exponentially scaled modified Bessel functions of the first
    kind and of half-integer order `n+1/2` with `n` from 0 to `nmax`

    .. math::
        I_{\nu}(z) = e^{z(\sqrt{1+\nu^2}-\nu \mathrm{arcsinh}(\nu))}
        \tilde{I}_{\nu}(z), \quad \nu = n + 1/2

    Parameters
    ----------
    nmax : int
        determines the maximal order of the Bessel function
    z : float
        argument of the Bessel function

    Returns
    -------
    inu : array_like
        array of length nmax+1

    References
    ----------
    See also Eq. (10.41.3) in https://dlmf.nist.gov/10.41 .

    """
    inu = np.empty(nmax+1, dtype=np.complex128)
    inu_0 = 0.5*(1-np.exp(-2*z))*np.sqrt(2/(pi*z))*np.exp(-z*(arg(0, z)-1))
    seed_value = 1
    inu[nmax] = seed_value
    inu[nmax-1] = (seed_value*ratio(nmax+0.5, z)
                   * np.exp(-z*(arg(nmax-1, z) - arg(nmax, z))))
    nu = np.arange(0, nmax+1)
    scaling1 = np.exp(-z*(arg(nu, z)-arg(nu+2, z)))
    scaling2 = np.exp(-z*(arg(nu, z)-arg(nu+1, z)))

    for k in range(nmax-2, -1, -1):
        inu[k] = inu[k+2]*scaling1[k] + (2*k+3)*inu[k+1]/z * scaling2[k]

    normalization = inu_0/inu[0]
    inu = inu*normalization
    return inu


def modbesselK(nmax, z):
    r"""
    calculates the modified Bessel functions of the second kind and of half-
    integer order `n+1/2` with `n` from 0 to `nmax` with an upward recursion

    .. math::
        K_{n+1/2}(z) = \frac{2n-1}{z} K_{n-1/2}(z) + K_{n-3/2}(z)

    Parameters
    ----------
    nmax : int
        determines the maximal order of the Bessel function
    z : float
        argument of the Bessel function

    Returns
    -------
    knu : array_like
        array of length nmax+1

    References
    ----------
    See also Eq. (10.29.1) in https://dlmf.nist.gov/10.29 .

    """
    knu = []
    k0 = np.exp(-z)*np.sqrt(pi/(2*z))
    k1 = (1+1/z)*np.exp(-z)*np.sqrt(pi/(2*z))
    knu.append(k0)
    knu.append(k1)
    for k in range(2, nmax+1):
        k2 = (2*k-1)*k1/z + k0
        knu.append(k2)
        k0 = k1
        k1 = k2
    return knu


def scaled2_modbesselK(nmax, z):
    r"""
    calculates the exponentially scaled modified Bessel functions of the second
    kind and of half-integer order `n+1/2` with `n` from 0 to `nmax`

    .. math::
        K_{\nu}(z) = e^{z(\sqrt{1+\nu^2}-\nu \mathrm{arcsinh}(\nu))}
        \tilde{K}_{\nu}(z), \quad \nu = n + 1/2

    Parameters
    ----------
    nmax : int
        determines the maximal order of the Bessel function
    z : float
        argument of the Bessel function

    Returns
    -------
    knu : array_like
        array of length nmax+1

    References
    ----------
    See also Eq. (10.41.4) in https://dlmf.nist.gov/10.41 .

    """
    knu = []
    k0 = np.sqrt(pi/(2*z)) * np.exp(z*(arg(0, z)-1))
    k1 = (1+1/z)*np.sqrt(pi/(2*z)) * np.exp(z*(arg(1, z)-1))
    knu.append(k0)
    knu.append(k1)
    nu = np.arange(2, nmax+1)
    scaling1 = np.exp(z*(arg(nu, z)-arg(nu-1, z)))
    scaling2 = np.exp(z*(arg(nu, z)-arg(nu-2, z)))
    for k, (s1, s2) in enumerate(zip(scaling1, scaling2)):
        k2 = (2*k+3)*k1/z * s1 + k0 * s2
        knu.append(k2)
        k0 = k1
        k1 = k2
    return knu


def scaled_modbesselK(nmax, z):
    r"""
    calculates the scaled modified Bessel functions of the second kind and of
    half-integer order :math:`n+1/2` with `n` from 0 to `nmax`

    .. math::
        K_{n+1/2}(z) = \frac{(2n)!}{z^{n+1/2}2^n n!}\tilde{K}_{n+1/2}(z)

    Parameters
    ----------
    nmax : int
        determines the maximal order of the Bessel function
    z : float
        argument of the Bessel function

    Returns
    -------
    knu : array_like
        array of length nmax+1

    References
    ----------
    See also Eq. (10.31.1) in https://dlmf.nist.gov/10.31 .

    """
    knu = []
    k0 = np.exp(-z)*np.sqrt(pi/2)
    k1 = (z+1)*np.exp(-z)*np.sqrt(pi/2)
    knu.append(k0)
    knu.append(k1)
    for k in range(2, nmax+1):
        k2 = k1 + z**2*k0/((2*k-1)*(2*k-3))
        knu.append(k2)
        k0 = k1
        k1 = k2
    return knu


def ratioK(nmax, z):
    """
    calculates the ratio :math:`K_{n+1/2}(z)/K_{n-1/2}(z)` for `n` from 0 to
    `nmax` with an upward recursion.

    Parameters
    ----------
    nmax : int
        determines the maximal order of the Bessel function
    z : float
        argument of the Bessel function

    Returns
    -------
    rknu : array_like
        array of length `nmax+1`

    References
    ----------
    See also Eq. (10.29.1) in https://dlmf.nist.gov/10.29 .

    """
    n = np.arange(nmax+1)
    rknu = (2*n-1)/z
    rknu[0] = 1
    return np.array(list(accumulate(rknu, lambda a, b: 1/a + b)))


def invK(nmax, z):
    r"""
    calculates the inverse of the modified Bessel function of the second kind
    and of half-integer order `n+1/2` with `n` from 0 to `nmax`

    .. math::
        K_{\nu+1}^{-1} = z\left[I_{\nu}(z) + I_{\nu +1/2}
                                 \frac{K_{\nu}(z)}{K_{\nu +1}(z)}\right]

    Parameters
    ----------
    nmax : int
        determines the maximal order of the Bessel function
    z : float
        argument of the Bessel function

    Returns
    -------
    inv_besselk : array_like
        array of length `nmax+1`

    References
    ----------
    See Eq. (10.28.2) in https://dlmf.nist.gov/10.28 .

    """
    kn0 = np.exp(-z)*np.sqrt(pi/(2*z))
    ratiok = ratioK(nmax, z)
    besseli = modbesselI(nmax, z)
    inv_besselk = np.concatenate(([1/kn0],
                                  z*(besseli[:-1] + besseli[1:]/ratiok[1:])))
    return inv_besselk


@njit(fastmath=True)
def I0(z, scaling=False, eps0=1e-17):
    """
    calculates the modified Bessel function of the first kind of order zero
    of argument `z`.

    Parameters
    ----------
    z : float
        argument of the Bessel function
    scaling : bool, default=False
        if scaling=True the Bessel function scaled with `exp(-z)` is returned
    eps0 : float, default=1e-17
        defines the accuracy of the result

    Returns
    -------
    float
        value of the zero order Bessel function

    References
    ----------
    See Ref. [Am83]_

    .. [Am83] D. E. Amos, Computation of Bessel functions of complex argument.
        United States: N. p., 1983.

    """
    if z == 0:
        return 1
    bound = 1.2*17 + 2.4
    # Eq. (5.1) in Ref. [1]
    if abs(z) <= bound:
        term_ak = 1
        k_sum = term_ak
        k = 0
        while abs(term_ak) > eps0*abs(k_sum):
            term_ak *= np.exp(2*np.log(0.5*z) - 2*np.log(k+1))
            k += 1
            k_sum += term_ak
        if not scaling:
            return k_sum
        return np.exp(-z)*k_sum

    # Eq. (6.1) in Ref. [1]
    term_ak = 1/(8*z)
    k_sum = term_ak
    k = 1
    while abs(term_ak) > eps0*abs(k_sum):
        term_ak *= np.exp(2*np.log(2*k+1) - np.log(k+1))/(8*z)
        k += 1
        k_sum += term_ak
    if not scaling:
        return np.exp(z)*(1+k_sum)/np.sqrt(2*pi*z)
    return (1+k_sum)/np.sqrt(2*pi*z)


@njit(fastmath=True)
def In(nmax, z, scaling=False):
    r"""
    calculates the modified Bessel functions of the first kind and of integer
    order `nmax` and argument `z` with a downward recursion

    .. math::
        I_{n-1}(z) = \frac{2n}{z} I_{n}(z) + I_{n+1}(z)

    Parameters
    ----------
    nmax : int
        integer order of the modified Bessel function
    z : float
        argument of the Bessel function
    scaling : bool, default=False
        if scaling=True the Bessel function scaled with `exp(-z)` is returned

    Return
    ------
    ans : float
        value of the modified Bessel function

    References
    ----------
    See also Ref. [Pre92]_

    """
    if not scaling:
        besseli_0 = I0(z)
    else:
        besseli_0 = I0(z, scaling=True)
    if nmax == 0:
        return besseli_0
    ans = 2/z
    besseli_np2 = ans
    besseli_np1 = ratio(nmax, z)*ans
    if nmax == 1:
        return ans*besseli_0/besseli_np1
    for n in range(nmax-2, -1, -1):
        besseli_n = besseli_np2 + 2*(n+1)*besseli_np1/z
        besseli_np2 = besseli_np1
        besseli_np1 = besseli_n
        if abs(besseli_n) > 1e10:
            ans *= 1e-10
            besseli_np2 *= 1e-10
            besseli_np1 *= 1e-10
            besseli_n *= 1e-10
    ans *= besseli_0/besseli_n
    return ans
