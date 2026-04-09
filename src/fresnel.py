"""
fresnel.py determines the Fresnel reflection coefficients for a PEMC plate,
a dielectric plate and a bi-isotropic plate for imaginary frequencies.

"""
import numpy as np

from src.scattering_amplitudes import cosTheta


def fresnel_pec():
    """
    returns the Fresnel coefficients for a perfect electric conductor.

    Returns
    -------
    tuple of floats

    """
    return 1, -1, 0, 0


def fresnel_pemc(theta):
    r"""
    calculates the Fresnel coefficients for a perfect electromagnetic conductor.

    Parameters
    ----------
    theta : float
        takes values between :math:`0` and :math:`\pi/2`, interpolating
        between a perfect electic and perfect magnetic conductor

    Returns
    -------
    tuple of floats
        TM-TM, TE-TE, TE-TM and TM-TE reflection coefficients

    """
    arg = 2*theta
    return np.cos(arg), -np.cos(arg), -np.sin(arg), -np.sin(arg)


def fresnel_die(Y, vec1, vec2, N):
    """
    calculates the Fresnel coefficients for a dielectric plane.

    Parameters
    ----------
    N : float
        reltive refractive index

    Returns
    -------
    tuple of floats
        TM-TM, TE-TE, TE-TM and TM-TE reflection coefficients

    References
    ----------
    The Fresnel reflection coefficients for real frequencies are e. g. defined
    in Eq. (2.67), (2.69) of Ref. [BH04]_ .

    .. [BH04] C. F. Bohren, D. R. Huffman, Absorption and Scattering of Light
        by Small Particles, WILEY-VCH, 2004

    """
    if Y == 0:
        r_tmtm = (N**2 - 1)/(N**2 + 1)
        r_tete = 0
        return r_tmtm, r_tete, 0, 0

    cos_theta = cosTheta(Y, vec1, vec2)
    c = -1j*np.sqrt(0.5*(-cos_theta-1))
    s = np.sqrt(0.5*(-cos_theta+1))
    # direct reflection coefficients
    r_tmtm = (N**2*s - np.sqrt(N**2-c**2))/(N**2*s + np.sqrt(N**2-c**2))
    r_tete = -(np.sqrt(N**2-c**2) - s)/(np.sqrt(N**2-c**2) + s)
    return r_tmtm, r_tete, 0, 0


def fresnel_biisotropic(Y, vec1, vec2, mL, mm):
    """
    calculates the Fresnel coefficients for a bi-isotropic plane.

    Parameters
    ----------
    mL : float
        relative refractive index
    mm : float
        relative impedance

    Returns
    -------
    tuple of floats
         TM-TM, TE-TE, TE-TM and TM-TE reflection coefficients


    References
    ----------
    Definitions of the reflection coefficient can e. g. be found in [Lin94]_ .

    .. [Lin94] I. V. Lindell et al., Electromagnetic Waves in Chiral and Bi-
        isotropic Media, Artech House, 1994

    """
    mR = np.conjugate(mL)
    mp = np.conjugate(mm)

    if Y == 0:
        DL = 1 + mm/mL
        DR = 1 + mp/mR
        CL = mm + 1/mL
        CR = mp + 1/mR
        f = mp*DL + mm*DR
        g = DR/mL + DL/mR
        F = CL + CR
        G = mm*CR/mL + mp*CL/mR
        cos_i = 1

    else:
        cos_theta = cosTheta(Y, vec1, vec2)
        c = -1j*np.sqrt(0.5*(-cos_theta-1))
        s = np.sqrt(0.5*(-cos_theta+1))

        cos_i = s
        cos_tL = np.sqrt(1-(c/mL)**2)
        cos_tR = np.sqrt(1-(c/mR)**2)
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
    return r_tmtm, r_tete, r_tetm, r_tmte
