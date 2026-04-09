"""
mie_coeff.py calculates the Mie scattering coefficients of a dielectric, pec,
pemc and biisotropic sphere.

"""
from cmath import pi, cosh, sinh, cos, sin, exp, tanh
from scipy.special import gammaln
import numpy as np

import src.bessel as bsl


def add_prefac(refl_coeff, nmax):
    n = np.arange(1, nmax+1)
    prefac = (-1)**n * np.exp(np.log(2*n+1)-np.log(n)-np.log(n+1))
    for rn in refl_coeff:
        rn[1:] = prefac*rn[1:]
    return refl_coeff
        

def ratiobessel(x, nmax, scaling):
    """
    returns the ratio :math:`I_{n+1/2}(x)/K_{n+1/2}(x)` of the modified
    Bessel function of the first and second kind.

    Parameters
    ----------
    x : float
        argument of the Bessel functions
    nmax : int
        maximal order of the Bessel function
    scaling : bool
        if scaling=True the scaled ratio is returned

    Returns
    -------
    array_like
        array of length `nmax+1`

    """
    if not scaling:
        return bsl.modbesselI(nmax, x)*bsl.invK(nmax, x)
    if abs(x) < 10:
        return bsl.scaled_modbesselI(nmax, x)/bsl.scaled_modbesselK(nmax, x)
    return bsl.scaled2_modbesselI(nmax, x)/bsl.scaled2_modbesselK(nmax, x)


def mie_die(N, x, nmax, scaling=False):
    """
    calculates the electric :math:`a_n(ix)` and magnetic :math:`b_n(ix)` Mie
    coeffiencts for order `n` from 0 to `nmax`.

    Parameters
    ----------
    N : float
        relative refractive index between the sphere and the surrounding medium
    x : float
        size parameter
    nmax : int
        maximal order of the Mie coefficients
    scaling : bool, default=False
        if scaling=True the scaled Mie coefficients are returned

    Returns
    -------
    an : array_like
        array of length `nmax+1` with electric Mie coefficients
    bn : array_like
        array of length `nmax+1` with magnetic Mie coefficients

    References
    ---------
    See also Eq. (4.56) and (4.57) in Ref. [BH04]_.

    """
    j = np.arange(0, nmax+1)
    ratioI_Nx = np.array([bsl.ratio(n+1/2, N*x) for n in range(0, nmax+1)])
    ratioI_x = np.array([bsl.ratio(n+1/2, x) for n in range(0, nmax+1)])

    ratioIK = ratiobessel(x, nmax, scaling)

    sj_a = (x*ratioI_x - j)
    sj_b = (N*x*ratioI_Nx - j)
    sj_c = x/bsl.ratioK(nmax, x) + j
    sj_d = N*x*ratioI_Nx - j
    pre_fac = (-1)**j * 0.5*pi*ratioIK
    an = pre_fac*(N**2*sj_a - sj_b)/(N**2*sj_c + sj_d)
    bn = pre_fac*(sj_a - sj_b)/(sj_c + sj_d)
    return an, bn


def mie_die_x0(N, x, nmax, scaling=False):
    """
    calculates the low-frequency asymptotic expression for the Mie coefficients
    of a dielectric sphere.

    """
    j = np.arange(1, nmax+1)
    if not scaling:
        pre = 0.5*(-1)**j * np.exp(np.log(j+1)+2*gammaln(j+1)-np.log(j)
                                   - np.log(2*j+1)
                                   - 2*gammaln(2*j+1))*(2*x)**(2*j+1)
    else:
        pre = 1
    an = pre*j*(N**2 - 1)/(j*N**2 + j+1)
    bn = np.zeros(nmax)
    return an, bn


def wkb_die(N, x, nmax, scaling=False):
    """
    calculates the Mie coefficients in the limit of large aspect ratios `x`.

    References
    ----------
    See p. 137 in Ref. [Sp20]_.

    """
    j = np.arange(0, nmax+1)
    lamb = (j+1/2)/x
    cos_i = np.sqrt(1+lamb**2)
    cos_t = np.sqrt(N**2 + lamb**2)
    if not scaling:
        pre = 0.5*(-1)**j*np.exp(2*x*(cos_i - lamb*np.arcsinh(lamb)))
    else:
        pre = 0.5*(-1)**j
    geo_corr = 0.25/cos_i - 5*lamb**2/(12*cos_i**3)
    r_tmtm = (N**2*cos_i-cos_t)/(N**2*cos_i+cos_t)
    r_tete = (cos_i-cos_t)/(cos_i+cos_t)
    a_corr = (geo_corr + lamb**2/cos_i**3
              + (lamb**2*(cos_t**2/cos_i**3 - N**2*cos_i/cos_t**2)
                 / ((N**2-1)*(N**2 + (N**2+1)*lamb**2)))
              )
    b_corr = geo_corr + lamb**2 * (cos_i/cos_t**2 - 1/cos_i)/(N**2-1)
    an = pre*r_tmtm*(1 + a_corr/x)
    bn = pre*r_tete*(1 + b_corr/x)
    return an, bn


def die_n0(N, x):
    """
    calculates the zero-th order of the Mie coefficients for a dielectric
    sphere.

    See Also
    --------
    mie_die

    """
    a0 = (N*cosh(x)*sinh(N*x)-sinh(x)*cosh(N*x))*exp(x)/(N*sinh(N*x)+cosh(N*x))
    b0 = (cosh(x)*sinh(N*x)-N*sinh(x)*cosh(N*x))*exp(x)/(sinh(N*x)+N*cosh(N*x))
    return a0, b0


def mie_pec(x, nmax, scaling=False):
    """
    calculates the Mie coefficients for a perfect electric conductor.

    Parameters
    ----------
    x : float
        size parameter
    nmax : int
        maximal order of the Mie coefficients
    scaling : bool, default=False
        if scaling=True the scaled Mie coefficients are returned

    Returns
    -------
    an : array_like
        electric Mie coefficients
    bn : array_like
        magnetic Mie coefficients

    """
    j = np.arange(0, nmax+1)
    ratioI = np.array([bsl.ratio(n+1/2, x) for n in range(0, nmax+1)])
    ratioIK = ratiobessel(x, nmax, scaling)
    pre_fac = (-1)**j * 0.5*pi*ratioIK

    an = pre_fac*(x*ratioI - j)/(x/bsl.ratioK(nmax, x) + j)
    bn = -pre_fac
    return an, bn


def mie_pec_x0(x, nmax, scaling=False):
    """
    calculates the low-frequency asymptotic expression for the Mie coefficents
    of a perfect electric conductor sphere.

    """
    j = np.arange(1, nmax+1)
    if not scaling:
        pre = 0.5*(-1)**j * np.exp(np.log(j+1)+2*gammaln(j+1)-np.log(j)
                                   - np.log(2*j+1)
                                   - 2*gammaln(2*j+1))*(2*x)**(2*j+1)
    else:
        pre = 1
    an = pre
    bn = pre*(-np.exp(np.log(j)-np.log(j+1)))
    return an, bn


def wkb_pec(x, nmax, scaling=False):
    """
    calculates the Mie coefficients in the limit of large aspect ratios `x`
    (see Eq. (C.16) in Ref. [Sp20]_).

    """
    j = np.arange(0, nmax+1)
    lamb = (j+1/2)/x
    if not scaling:
        cos_i = np.sqrt(1+lamb**2)
        pre = 0.5*(-1)**j*np.exp(2*x*(cos_i - lamb*np.arcsinh(lamb)))
    else:
        pre = pre = 0.5*(-1)**j
    a_corr = 0.25/np.sqrt(1+lamb**2) + 7*lamb**2/(12*(1+lamb**2)**1.5)
    b_corr = 0.25/np.sqrt(1+lamb**2) - 5*lamb**2/(12*(1+lamb**2)**1.5)
    an = pre*(1+a_corr/x)
    bn = - pre*(1+b_corr/x)
    return an, bn


def pec_n0(x):
    """
    calculates the zeroth order of the mie coefficients for a perfect electric
    conductor sphere.

    See Also
    --------
    mie_pec

    """
    a0 = cosh(x)*exp(x)
    b0 = -sinh(x)*exp(x)
    return a0, b0


def mie_pemc(theta, x, nmax, scaling=False):
    """
    calculates the Mie coefficients for a perfect electromagnetic conductor.

    Parameters
    ----------
    theta : float
        parameter in the range between 0 and pi/2 interpolating between a
        perfect electric conductor (theta=0) and a perfect magentic conductor
        (theta=pi/2)
    x : float
        size parameter
    nmax : int
        maximal order of the Mie coefficients
    scaling : bool, default=False
        if scaling=True the scaled Mie coefficients are returned

    Returns
    -------
    an : array_like
        reflection coefficients corresponding to electric polarized incomming
        wave and an electric reflected wave
    bn : array_like
        reflection coefficients corresponding to magnetic polarized incomming
        wave and a magnetic reflected wave
    cn : array_like
        reflection coefficients corresponding to magnetic polarized incomming
        wave and an electric reflected wave
    dn : array_like
        reflection coefficients corresponding to electric polarized incomming
        wave and a magnetic reflected wave

    """
    j = np.arange(0, nmax+1)
    ratioI = np.array([bsl.ratio(n+1/2, x) for n in range(0, nmax+1)])
    invratioK = 1/bsl.ratioK(nmax, x)
    ratioIK = ratiobessel(x, nmax, scaling)

    termI = x*ratioI - j
    termK = x*invratioK + j
    pre_fac = (-1)**j * 0.5*pi*ratioIK

    an = pre_fac*(cos(theta)**2 * termI
                  - sin(theta)**2 * termK)/(x*invratioK + j)
    bn = pre_fac*(-cos(theta)**2 * termK
                  + sin(theta)**2 * termI)/(x*invratioK + j)
    cn = -0.5*pre_fac*sin(2*theta)*(termI + termK)/(x*invratioK + j)
    dn = -0.5*pre_fac*sin(2*theta)*(termI + termK)/(x*invratioK + j)
    return an, bn, cn, dn


def mie_pemc_x0(theta, x, nmax, scaling=False):
    """
    calculates the Mie coefficients for a bi-isotropic sphere in the limit of
    low frequencies.

    """
    j = np.arange(1, nmax+1)
    if not scaling:
        pre = 0.5*(-1)**j * np.exp(np.log(j+1)+2*gammaln(j+1)-np.log(j)
                                   - np.log(2*j+1)
                                   - 2*gammaln(2*j+1))*(2*x)**(2*j+1)
    else:
        pre = 1
    an = pre*(cos(theta)**2-sin(theta)**2*np.exp(np.log(j)-np.log(j+1)))
    bn = pre*(sin(theta)**2-cos(theta)**2*np.exp(np.log(j)-np.log(j+1)))
    cn = -0.5*pre*sin(2*theta)*(1 + np.exp(np.log(j)-np.log(j+1)))
    dn = -0.5*pre*sin(2*theta)*(1 + np.exp(np.log(j)-np.log(j+1)))
    return an, bn, cn, dn


def wkb_pemc(theta, x, nmax, scaling=False):
    """
    calculates the Mie coefficients for a bi-isotropic sphere in the limit of
    large aspect ratios.

    """
    j = np.arange(0, nmax+1)
    lamb = (j+1/2)/x
    if not scaling:
        cos_i = np.sqrt(1+lamb**2)
        pre = 0.5*(-1)**j*np.exp(2*x*(cos_i - lamb*np.arcsinh(lamb)))
    else:
        pre = pre = 0.5*(-1)**j
    geo_corr = 0.25/np.sqrt(1+lamb**2) - 5*lamb**2/(12*(1+lamb**2)**1.5)
    a_corr = geo_corr + cos(theta)**2/cos(2*theta)*lamb**2/(1+lamb**2)**1.5
    b_corr = geo_corr - sin(theta)**2/cos(2*theta)*lamb**2/(1+lamb**2)**1.5
    c_corr = geo_corr - lamb**2/(1+lamb**2)**1.5
    an = pre*np.cos(2*theta)*(1+a_corr/x)
    bn = -pre*np.cos(2*theta)*(1+b_corr/x)
    cn = -pre*np.sin(2*theta)*(1+c_corr/x)
    dn = -pre*np.sin(2*theta)*(1+c_corr/x)
    return an, bn, cn, dn


def pemc_n0(theta, x):
    """
    calculates the zero-th order of the mie coefficients for a perfect electro-
    magnetic conductor sphere.

    See Also
    --------
    mie_pemc

    """
    a0 = (cos(theta)**2*cosh(x) - sin(theta)**2*sinh(x))*exp(x)
    b0 = (-cos(theta)**2*sinh(x) + sin(theta)**2*cosh(x))*exp(x)
    c0 = -sin(theta)*cos(theta)*(cosh(x)+sinh(x))*exp(x)
    d0 = -sin(theta)*cos(theta)*(cosh(x)+sinh(x))*exp(x)
    return a0, b0, c0, d0


def mie_biisotropic(mL, mm, x, nmax, scaling=False):
    """
    calculates the reflection coefficients of a bi-isotropic sphere for order
    `0` to `nmax`.

    Parameters
    ----------
    mL : float
        relative refractive index
    mm : float
        relative impedance
    x : float
        size parameter
    nmax : int
        maximal order of the Mie coefficients
    scaling : bool, default=False
        if `scaling=True` the scaled Mie coefficients are returned

    Returns
    -------
    an : array_like
        reflection coefficients corresponding to electric polarized incomming
        wave and an electric reflected wave
    bn : array_like
        reflection coefficients corresponding to magnetic polarized incomming
        wave and a magnetic reflected wave
    cn : array_like
        reflection coefficients corresponding to magnetic polarized incomming
        wave and an electric reflected wave
    dn : array_like
        reflection coefficients corresponding to electric polarized incomming
        wave and a magnetic reflected wave

    References
    ----------
    See Ref. [Bo74]_ and Eq. (27)-(30) in Ref. [github]_.

    .. [Bo74] C. F. Bohren, Chem. Phys. Lett. 29, 3 (1974)
    .. [github] https://github.com/gertingold/plane-wave-basis/blob/master
        /chiral-spheres/WKB.tex

    """
    mR = np.conjugate(mL)
    mp = np.conjugate(mm)
    j = np.arange(0, nmax+1)
    ratio_xR = np.array([bsl.ratio(n+1/2, mR*x) for n in range(0, nmax+1)])
    ratio_xL = np.array([bsl.ratio(n+1/2, mL*x) for n in range(0, nmax+1)])
    ratio_x = np.array([bsl.ratio(n+1/2, x) for n in range(0, nmax+1)])
    invratioK = 1/bsl.ratioK(nmax, x)
    ratioIK = ratiobessel(x, nmax, scaling)

    xR = mR*x
    xL = mL*x

    VR = mp*(ratio_xR - j/xR) + invratioK + j/x
    VL = mm*(ratio_xL - j/xL) + invratioK + j/x
    WR = ratio_xR - j/xR + mp*(invratioK + j/x)
    WL = ratio_xL - j/xL + mm*(invratioK + j/x)
    delta = VR*WL + VL*WR
    AR = mp*(ratio_x - j/x) - (ratio_xR - j/xR)
    AL = mm*(ratio_x - j/x) - (ratio_xL - j/xL)
    BR = ratio_x - j/x - mp*(ratio_xR - j/xR)
    BL = ratio_x - j/x - mm*(ratio_xL - j/xL)
    pre_fac = (-1)**j * 0.5*pi*ratioIK

    an = pre_fac*(VR*AL + VL*AR)/delta
    bn = pre_fac*(WR*BL + WL*BR)/delta
    cn = 1j*pre_fac*((ratio_x - j/x + invratioK + j/x) *
                     (mm*(ratio_xR - j/xR) - mp*(ratio_xL - j/xL)))/delta
    dn = 1j*pre_fac*((ratio_x - j/x + invratioK + j/x) *
                     (mm*(ratio_xL - j/xL) - mp*(ratio_xR - j/xR)))/delta
    return an, bn, cn, dn


def mie_biisotropic_x0(mL, mm, x, nmax, scaling=False):
    """
    calculates the Mie coefficients for a bi-isotropic sphere in the limit of
    low frequencies.

    """
    mR = np.conjugate(mL)
    mp = np.conjugate(mm)
    j = np.arange(1, nmax+1)
    if not scaling:
        pre = 0.5*(-1)**j * np.exp(np.log(j+1)+2*gammaln(j+1)-np.log(j)
                                   - np.log(2*j+1)
                                   - 2*gammaln(2*j+1))*(2*x)**(2*j+1)
    else:
        pre = 1
    aux = np.exp(np.log(j)-np.log(j+1))
    delta = (1/mL+mm*aux)*(mp/mR+aux)+(1/mR+mp*aux)*(mm/mL+aux)
    model_an = aux*((mm-1/mL)*(mp/mR+aux)+(mp-1/mR)*(mm/mL+aux))/delta
    model_bn = aux*((1-mp/mR)*(1/mL + mm*aux)+(1-mm/mL)*(1/mR+mp*aux))/delta
    aux2 = np.exp(np.log(j)+np.log(2*j+1)-2*np.log(j+1))
    model_cn = 1j*aux2*(mm/mR-mp/mL)/delta
    model_dn = 1j*aux2*(mm/mL-mp/mR)/delta
    an = pre*model_an
    bn = pre*model_bn
    cn = pre*model_cn
    dn = pre*model_dn
    return an, bn, cn, dn


def wkb_biisotropic(mL, mm, x, nmax, scaling=False):
    """
    calculates the Mie coefficients for a bi-isotropic sphere in the limit of
    large aspect ratios (see Eq. (37)-(40) in Ref. [github]_).

    """
    mR = np.conjugate(mL)
    mp = np.conjugate(mm)
    j = np.arange(0, nmax+1)
    lamb = (j+1/2)/x

    cos_i = np.sqrt(1+lamb**2)
    cos_tL = np.sqrt(1+(lamb/mL)**2)
    cos_tR = np.sqrt(1+(lamb/mR)**2)
    tan2_i = 0.5*lamb**2/(1+lamb**2)
    tan2_tR = 0.5*lamb**2/(mR*(mR**2 + lamb**2))
    tan2_tL = 0.5*lamb**2/(mL*(mL**2 + lamb**2))

    if not scaling:
        epsi = 0.5*(-1)**j*np.exp(2*x*(cos_i - lamb*np.arcsinh(lamb)))
    else:
        epsi = 0.5*(-1)**j
    geo_corr = 0.25/cos_i - 5*lamb**2/(12*cos_i**3)

    DL = cos_i + mm*cos_tL
    DR = cos_i + mp*cos_tR
    CL = mm*cos_i + cos_tL
    CR = mp*cos_i + cos_tR
    f = cos_i*(mp*DL + mm*DR)
    g = cos_tL*DR + cos_tR*DL
    F = cos_i*(CL + CR)
    G = mm*cos_tL*CR + mp*cos_tR*CL

    r_tmtm = (f-g)/(f+g)
    r_tete = (F-G)/(F+G)
    r_tmte = 2j*cos_i*(DL-DR)/(f+g)
    r_tetm = 2j*cos_i*(mm*CR-mp*CL)/(f+g)

    a_mat = 2*(- tan2_tL*DR*(f + cos_i*mm*(mm*CR - mp*CL))
               - tan2_tR*DL*(f - cos_i*mp*(mm*CR - mp*CL))
               + tan2_i*(f**2/cos_i - cos_i*(DL-DR)*(mm*CR - mp*CL))
               )/((f+g)**2)
    b_mat = 2*(- tan2_tL*CR*(mm*F - cos_i*(DL-DR))
               - tan2_tR*CL*(mp*F + cos_i*(DL-DR))
               + tan2_i*(F**2/cos_i - cos_i*(DL-DR)*(mm*CR - mp*CL))
               )/((F+G)**2)
    c_mat = 2j*(- tan2_tL*CR*(f + cos_i*mm*(mm*CR - mp*CL))
                + tan2_tR*CL*(f - cos_i*mp*(mm*CR - mp*CL))
                + tan2_i*cos_i*(mm*CR - mp*CL)*(CL+CR + mm*DR + mp*DL)
                )/((f+g)**2)
    d_mat = 2j*(tan2_tL*DR*(mm*F - cos_i*(DL-DR))
                - tan2_tR*DL*(mp*F + cos_i*(DL-DR))
                + tan2_i*cos_i*(DL-DR)*(CL+CR + mm*DR + mp*DL)
                )/((F+G)**2)

    a_corr = r_tmtm*geo_corr + a_mat
    b_corr = r_tete*geo_corr + b_mat
    c_corr = r_tetm*geo_corr + c_mat
    d_corr = r_tmte*geo_corr + d_mat

    an = epsi*(r_tmtm + a_corr/x)
    bn = epsi*(r_tete + b_corr/x)
    cn = epsi*(r_tetm + c_corr/x)
    dn = epsi*(r_tmte + d_corr/x)
    return an, bn, cn, dn


def biisotropic_n0(mL, mm, x):
    """
    calculates the zeroth order of the Mie coefficients for a bi-isotropic
    sphere.

    See Also
    --------
    mie_biisotropic

    """
    mR = np.conjugate(mL)
    mp = np.conjugate(mm)
    xR = mR*x
    xL = mL*x
    VR0 = mp/tanh(xR) + 1
    VL0 = mm/tanh(xL) + 1
    WR0 = 1/tanh(xR) + mp
    WL0 = 1/tanh(xL) + mm
    delta0 = VR0*WL0 + VL0*WR0
    AR0 = mp/tanh(x) - 1/tanh(xR)
    AL0 = mm/tanh(x) - 1/tanh(xL)
    BR0 = 1/tanh(x) - mp/tanh(xR)
    BL0 = 1/tanh(x) - mm/tanh(xL)
    a0 = (VR0*AL0 + VL0*AR0)*sinh(x)*exp(x)/delta0
    b0 = (WR0*BL0 + WL0*BR0)*sinh(x)*exp(x)/delta0
    c0 = 1j*(1/tanh(x) + 1)*(mm/tanh(xR) - mp/tanh(xL))*sinh(x)*exp(x)/delta0
    d0 = 1j*(1/tanh(x) + 1)*(mm/tanh(xL) - mp/tanh(xR))*sinh(x)*exp(x)/delta0
    return a0, b0, c0, d0
