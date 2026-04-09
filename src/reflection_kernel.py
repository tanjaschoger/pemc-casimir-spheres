r"""
reflection_kernel.py calculates the kernel function of the reflection operator
of a spherical particle

.. math::

    K_\mathcal{R}(\mathbf{k}_1, p_1; \mathbf{k}_2, p_2) =
    \frac{A S_{p_1, p_2} + (-1)^{p_1+p_2} B S_{\bar{p_1},\bar{p_2}}
           -(-1)^{p_1} C S_{\bar{p_1}, p_2}
           + (-1)^{p_2} D S_{p_1, \bar{p_2}}}
    {\sqrt{(1+ (\mathcal{K}/k_1)^2)(1+ (\mathcal{K}/k_2)^2)}}

:math:`S_{p_1, p_2}` defines the Mie scattering amplitudes which are calculated
in :py:mod:`src.scattering_amplitudes`.

"""
from math import pi, ceil, floor, lgamma
import numpy as np
from numba import njit

import src.mie_coeff as mie
import src.scattering_amplitudes as scam


@njit
def polarization_transform(Y, vec1, vec2, sgn):
    """
    calculates the polarization transformation coefficients given in Eq. (A8)
    in Ref. [Sp18]_.

    Parameters
    ----------
    sgn : int
        defines the direction of the incident wave
        (sgn = +1 for positive z-directon, sgn = -1 for negative z-direction)

    Returns
    -------
    A, B, C, D : tuple of floats

    """
    y1, Phi1 = vec1
    y2, Phi2 = vec2
    pre_fac = ((1+(Y/y1)**2)*(1+(Y/y2)**2))**0.25
    if Y == 0:
        return 1/pre_fac, 0, 0, 0
    if y1 == y2 and Phi1 == Phi2:
        return 1/pre_fac, 0, 0, 0
    if y1 == y2 and (Phi1-Phi2) % 0.5 == .0:
        return -1/pre_fac, 0, 0, 0

    dPhi = 2*pi*(Phi1 - Phi2)
    cosdPhi = np.cos(dPhi)
    sqrt_y1 = np.sqrt(1+(Y/y1)**2)
    sqrt_y2 = np.sqrt(1+(Y/y2)**2)

    denom = pre_fac*(Y**4 - (y1*y2)**2 * (cosdPhi + sqrt_y1*sqrt_y2)**2)

    A = (Y**4*cosdPhi
         - (y1*y2)**2*((cosdPhi+sqrt_y1*sqrt_y2)*(1+cosdPhi*sqrt_y1*sqrt_y2)
                       ))/denom
    if (Phi1-Phi2) % 0.5 == .0:
        B, C, D = 0, 0, 0
    else:
        sindPhi = np.sin(dPhi)
        B = -Y**2*y1*y2*sindPhi**2/denom
        C = sgn*Y*sindPhi*y1**2*y2*(sqrt_y1*cosdPhi + sqrt_y2)/denom
        D = -sgn*Y*sindPhi*y2**2*y1*(sqrt_y2*cosdPhi + sqrt_y1)/denom
    return A, B, C, D


@njit
def refl_kernel(Y, y1, y2, scatampl, poltrans):
    """
    calculates the kernel functions of the reflection operator for a spheres
    described by an arbitrary model.

    Returns
    -------
    tuple of floats
        values of the kernel functions for TM-TM, TE-TE, TE-TM and TM-TE
        polarization

    """
    stmtm, stete, stetm, stmte = scatampl
    A, B, C, D = poltrans

    kRtmtm = A*stmtm + B*stete - C*stetm + D*stmte
    kRtete = A*stete + B*stmtm + C*stmte - D*stetm
    kRtmte = A*stmte - B*stetm - C*stete - D*stmtm
    kRtetm = A*stetm - B*stmte + C*stmtm + D*stete
    return kRtmtm, kRtete, kRtetm, kRtmte


@njit
def kernelR(RbyL, Y, vec1, vec2, sgn, model):
    """
    calculates the kernel functions of the reflection operator for a spheres
    described by an arbitrary model.

    Parameters
    ----------
    model : tuple of 1Darrays
        Mie reflection coefficients

    Returns
    -------
    tuple of floats
        values of the kernel functions for TM-TM, TE-TE, TE-TM and TM-TE
        polarization

    """
    an, bn, cn, dn = model

    stete = scam.ScatteringAmplitude(RbyL, Y, an, bn).evalf(vec1, vec2)
    stmtm = scam.ScatteringAmplitude(RbyL, Y, bn, an).evalf(vec1, vec2)
    stmte = scam.ScatteringAmplitude(RbyL, Y, -cn, dn).evalf(vec1, vec2)
    stetm = scam.ScatteringAmplitude(RbyL, Y, -dn, cn).evalf(vec1, vec2)
    A, B, C, D = polarization_transform(Y, vec1, vec2, sgn)

    kRtmtm = A*stmtm + B*stete - C*stetm + D*stmte
    kRtete = A*stete + B*stmtm + C*stmte - D*stetm
    kRtmte = A*stmte - B*stetm - C*stete - D*stmtm
    kRtetm = A*stetm - B*stmte + C*stmtm + D*stete
    return kRtmtm, kRtete, kRtetm, kRtmte


def kernelR_die(RbyL, Y, vec1, vec2, sgn, N):
    """
    calculates the kernel function for a dielectric sphere.

    """
    nmax = 12*ceil(RbyL)
    x = RbyL*Y
    an, bn = mie.mie_die(N=N, x=x, nmax=nmax, scaling=True)
    model = (an, bn,
             np.zeros(nmax+1, dtype=np.complex128),
             np.zeros(nmax+1, dtype=np.complex128))
    model = mie.add_prefac(model, nmax)
    return kernelR(RbyL, Y, vec1, vec2, sgn, model)


def kernelR_pec(RbyL, Y, vec1, vec2, sgn):
    """
    calculates the kernel function for a dielectric sphere.

    """
    nmax = 12*ceil(RbyL)
    x = RbyL*Y
    an, bn = mie.mie_pec(x=x, nmax=nmax, scaling=True)
    model = (an, bn,
            np.zeros(nmax+1, dtype=np.complex128),
            np.zeros(nmax+1, dtype=np.complex128))
    model = mie.add_prefac(model, nmax)
    return kernelR(RbyL, Y, vec1, vec2, sgn, model)


def kernelR_biisotropic(RbyL, Y, vec1, vec2, sgn, mm, mL):
    """
    calculates the kernel function for a bi-isotropic sphere.

    """
    nmax = 12*ceil(RbyL)
    x = RbyL*Y
    model = mie.mie_biisotropic(mL, mm, x, nmax, scaling=True)
    model = mie.add_prefac(model, nmax)
    return kernelR(RbyL, Y, vec1, vec2, sgn, model)


def kernelR_pemc(RbyL, Y, vec1, vec2, sgn, theta):
    """
    calculates the kernel function for a pemc sphere.

    """
    nmax = 12*ceil(RbyL)
    x = RbyL*Y
    model = mie.mie_pemc(theta=theta, x=x, nmax=nmax, scaling=True)
    model = mie.add_prefac(model, nmax)
    return kernelR(RbyL, Y, vec1, vec2, sgn, model)


@njit
def kernelRlowfreq(RbyL, vec1, vec2, an):
    """
    calculates the kernel function in the low-frequency limit for an arbitrary
    model.

    References
    ----------
    See also Eq. (5.23) in Ref. [Sp20]_ .

    """
    if not np.any(an):
        return 0

    y1, Phi1 = vec1
    y2, Phi2 = vec2

    x2cos_theta = -RbyL**2*y1*y2*(1+np.cos(2*pi*(Phi2-Phi1)))
    if x2cos_theta == 0:
        return 0

    y_sp = RbyL*np.sqrt(2*y1*y2*(1+np.cos(2*pi*(Phi2-Phi1))))
    arg_trans = RbyL*(y1+y2)
    asymp = np.exp(y_sp - arg_trans)
    if asymp < 1e-20:
        return 0

    nmax = len(an)-1
    nsp = min(max(1, floor(np.sqrt(-x2cos_theta))), nmax)
    log_2x2cos_theta = np.log(-2*x2cos_theta)

    arg = an[nsp-1]*np.exp(nsp*log_2x2cos_theta - lgamma(2*nsp+1) - arg_trans)
    scat_ampl = arg
    # upward summation
    for n in range(nsp+1, nmax+1):
        arg = an[n-1]*np.exp(n*log_2x2cos_theta - lgamma(2*n+1) - arg_trans)
        scat_ampl += arg
        if abs(arg) < 1e-17*abs(scat_ampl):
            break
    # downward summation
    for n in range(nsp-1, 0, -1):
        arg = an[n-1]*np.exp(n*log_2x2cos_theta - lgamma(2*n+1) - arg_trans)
        scat_ampl += arg
        if abs(arg) < 1e-17*abs(scat_ampl):
            break

    return 2*pi*RbyL*scat_ampl.real


def kernelRlowfreq_die(RbyL, vec1, vec2, N):
    """
    calculates the low-frequency limit of the kernel function for an dielectric
    sphere.

    """
    nmax = 12*ceil(RbyL)
    an, bn = mie.mie_die_x0(N=N, x=0, nmax=nmax, scaling=True)
    return (kernelRlowfreq(RbyL, vec1, vec2, an),
            kernelRlowfreq(RbyL, vec1, vec2, bn))


def kernelRlowfreq_pec(RbyL, vec1, vec2):
    nmax = 12*ceil(RbyL)
    n = np.arange(1, nmax+1)
    an = np.ones(nmax)
    bn = - n/(n+1)
    return (kernelRlowfreq(RbyL, vec1, vec2, an),
            kernelRlowfreq(RbyL, vec1, vec2, bn))


@njit
def kernelRlowfreq_pemc(RbyL, vec1, vec2, theta):
    nmax = 12*ceil(RbyL)
    n = np.arange(1, nmax+1)
    model_te = - n/(n+1)
    model_tm = np.ones(nmax+1)
    kRte = kernelRlowfreq(RbyL, vec1, vec2, model_te)
    kRtm = kernelRlowfreq(RbyL, vec1, vec2, model_tm)
    kRtmtm = np.cos(theta)**2 * kRtm + np.sin(theta)**2 * kRte
    kRtete = np.cos(theta)**2 * kRte + np.sin(theta)**2 * kRtm
    kRtetm = 0.5*np.sin(2*theta)*(-kRtm + kRte)
    kRtmte = kRtetm
    return kRtmtm, kRtete, kRtetm, kRtmte


def kernelRlowfreq_biisotropic(RbyL, vec1, vec2, mm, mL):
    nmax = 12*ceil(RbyL)
    an, bn, cn, dn = mie.mie_biisotropic_x0(mL=mL, mm=mm, x=0, nmax=nmax,
                                            scaling=True)
    kRtmtm = kernelRlowfreq(RbyL, vec1, vec2, an)
    kRtete = kernelRlowfreq(RbyL, vec1, vec2, bn)
    kRtetm = kernelRlowfreq(RbyL, vec1, vec2, cn)
    kRtmte = kernelRlowfreq(RbyL, vec1, vec2, dn)
    return kRtmtm, kRtete, kRtetm, kRtmte


def wkb_kernelR(Y, vec1, vec2, sgn, amplitudes):
    """
    calculates the wkb approximation of the kernel functions.

    """
    A, B, C, D = polarization_transform(Y, vec1, vec2, sgn)
    stmtm, stete, stetm, stmte = amplitudes

    kRtmtm = A*stmtm + B*stete - C*stetm + D*stmte
    kRtete = A*stete + B*stmtm + C*stmte - D*stetm
    kRtmte = A*stmte - B*stetm - C*stete - D*stmtm
    kRtetm = A*stetm - B*stmte + C*stmtm + D*stete
    return kRtmtm, kRtete, kRtetm, kRtmte


def wkb_kernelR_pec(RbyL, Y, vec1, vec2, sgn):
    amplitudes = scam.amplitude_pec(RbyL, Y, vec1, vec2)
    return wkb_kernelR(Y, vec1, vec2, sgn, amplitudes)


def wkb_kernelR_die(RbyL, Y, vec1, vec2, sgn, N):
    amplitudes = scam.amplitude_die(RbyL, Y, vec1, vec2, N)
    return wkb_kernelR(Y, vec1, vec2, sgn, amplitudes)


def wkb_kernelR_pemc(RbyL, Y, vec1, vec2, sgn, theta):
    amplitudes = scam.amplitude_pemc(RbyL, Y, vec1, vec2, theta)
    return wkb_kernelR(Y, vec1, vec2, sgn, amplitudes)


def wkb_kernelR_biisotropic(RbyL, Y, vec1, vec2, sgn, mm, mL):
    amplitudes = scam.amplitude_biisotropic(RbyL, Y, vec1, vec2, mL, mm)
    return wkb_kernelR(Y, vec1, vec2, sgn, amplitudes)
