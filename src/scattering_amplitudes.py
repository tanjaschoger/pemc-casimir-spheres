r"""
scattering_amplitudes.py calculates the Mie scattering amplitudes

.. math::
    S_{p_1, p_2}(\mathrm{cos}(\Theta)) = \frac{2\pi}{\mathcal{K}}
    \sum_{\ell = 1}^\infty
    \left[ r_\ell^{p_1, p_2} \tau_\ell(\mathrm{cos}(\Theta))
           + r_\ell^{\bar{p_1}, \bar{p_2}}
           \pi_\ell(\mathrm{cos}(\Theta)) \right]
    e^{-R\left(\sqrt{k_1^2 + \mathcal{K}^2}
                + \sqrt{k_2^2 + \mathcal{K}^2}\right)}

The reflection coefficients :math:`r_\ell^{p_1, p_2}` are calculated in
:py:mod:`src.mie_coeff`.

The angular functions :math:`\tau_\ell, \pi_\ell` are computed in
:py:mod:`src.angular_func`.

"""
from math import pi, floor
import numpy as np
from numba import njit, float64, complex128, int64, objmode, boolean
from numba.experimental import jitclass
from scipy.special import loggamma


from src.angular_func import angularj, angularj_large_arg


@njit(cache=True)
def cosTheta(Y, vec1, vec2):
    """
    calculates the cosine of the scattering angle for imaginary frequencies.

    References
    ----------
    See Eq. (10) in Ref. [Sp18]_

    .. [Sp18] B. Spreng et al., Phys. Rev. A 97, 062504 (2018)

    """
    y1, Phi1 = vec1
    y2, Phi2 = vec2
    if abs(y1-y2) < 1e-15:
        return -1 - 2*(y1*np.cos(pi*(Phi1-Phi2))/Y)**2
    return -(1/Y**2)*(np.sqrt((y1**2+(Y)**2)*(y2**2 + (Y)**2))
                      + y1*y2*np.cos(2*pi*(Phi1-Phi2)))


spec = [('RbyL', float64), ('Y', float64), ('an', complex128[:]),
        ('bn', complex128[:]), ('x', float64), ('nmax', int64),
        ('arg_mie', float64[:]), ('all_zeros', boolean)
        ]


@jitclass(spec)
class ScatteringAmplitude:

    def __init__(self, RbyL, Y, an, bn):
        self.RbyL = RbyL
        self.Y = Y
        self.an = an
        self.bn = bn
        if not np.any(self.an) and not np.any(self.bn):
            self.all_zeros = True
            return

        self.all_zeros = False
        self.x = RbyL*Y
        self.nmax = len(self.an)-1
        m = np.arange(1, self.nmax + 1)
        if self.x < 10:
            with objmode(arg_mie='float64[:]'):
                if self.Y < 1e-10:
                    arg_mie = np.log(m+1) + np.log(m) - loggamma(2*m+3)
                else:
                    arg_mie = ((2*m+1)*np.log(0.5*self.x) - np.log(m+1/2)
                               + np.log(pi) - 2*loggamma(m+0.5))
        else:
            Lamb = (m+0.5)/self.x
            arg_mie = 2*self.x*(np.sqrt(1+Lamb**2) - Lamb*np.arcsinh(Lamb))

        self.arg_mie = arg_mie

    def scat_ampl_lowfreq(self, y1, Phi1, y2, Phi2, cos_theta):
        """
        calculates the scattering amplitude for low-frequencies.

        """
        arg_trans = -self.RbyL*(y1*np.sqrt(1+(self.Y/y1)**2)
                                + y2*np.sqrt(1+(self.Y/y2)**2))
        log_2x2cos_theta = np.log(2*self.RbyL**2
                                  * (np.sqrt((y1**2 + self.Y**2)
                                             * (y2**2 + self.Y**2))
                                     + y1*y2*np.cos(2*pi*(Phi1-Phi2))))
        u = np.arccosh(abs(cos_theta))
        nsp = min(max(1, floor(self.x*np.sqrt(-0.5*(cos_theta+1)))),
                  self.nmax-1)

        pi_nsp, tau_nsp = angularj_large_arg(nsp, u)
        exp_scal = np.exp(arg_trans+nsp*log_2x2cos_theta+self.arg_mie[nsp-1])
        arg_nsp = exp_scal*(self.an[nsp]*pi_nsp/cos_theta
                            + nsp*self.bn[nsp]*tau_nsp)

        n = nsp + 1
        pi_n, tau_n = angularj_large_arg(n, u)
        exp_scal = np.exp(arg_trans + n*log_2x2cos_theta + self.arg_mie[n-1])
        arg_n = exp_scal*(self.an[n]*pi_n/cos_theta + n*self.bn[n]*tau_n)

        scat_ampl = arg_nsp + arg_n
        # upward summation
        for n in range(nsp+2, self.nmax):
            if abs(arg_n) < 1e-17*abs(scat_ampl):
                break
            pi_n, tau_n = angularj_large_arg(n, u)
            exp_scal = np.exp(arg_trans+n*log_2x2cos_theta+self.arg_mie[n-1])
            arg_n = exp_scal*(self.an[n]*pi_n/cos_theta + n*self.bn[n]*tau_n)
            scat_ampl += arg_n

        # downward summation
        for n in range(nsp-1, 0, -1):
            pi_n, tau_n = angularj_large_arg(n, u)
            exp_scal = np.exp(arg_trans+n*log_2x2cos_theta+self.arg_mie[n-1])
            arg_n = exp_scal*(self.an[n]*pi_n/cos_theta + n*self.bn[n]*tau_n)
            scat_ampl += arg_n
            if abs(arg_n) < 1e-17*abs(scat_ampl):
                break

        return 4*pi*self.RbyL*scat_ampl.real

    def scat_largeradii(self, arg_trans, cos_theta):
        """
        calculates the scattering amplitude for large aspect ratios.

        """
        u = np.arccosh(-cos_theta)
        nsp = min(max(1, floor(self.x*np.sqrt(-0.5*(cos_theta+1)))),
                  self.nmax-1)

        pi_nm1, tau_nm1 = angularj(nsp, u, scaling=True)
        exp_scal = np.exp(arg_trans + self.arg_mie[nsp-1] + (nsp+0.5)*u)
        arg_nsp = exp_scal * (-self.an[nsp]*pi_nm1 + self.bn[nsp]*tau_nm1)

        n = nsp + 1
        pi_n, tau_n = angularj(n, u, scaling=True)
        exp_scal = np.exp(arg_trans + self.arg_mie[n-1] + (n+0.5)*u)
        arg = exp_scal * (-self.an[n]*pi_n + self.bn[n]*tau_n)

        scat_ampl = arg_nsp + arg
        # upward summation
        exp_u = np.exp(-u)
        exp_2u = np.exp(-2*u)
        for n in range(nsp+1, self.nmax):
            if abs(arg) < 1e-17*abs(scat_ampl):
                break
            if n <= 1000:
                pi_np1 = (-(2*n+1)*cos_theta*pi_n/n * exp_u
                          - (n+1)*pi_nm1/n * exp_2u)
                tau_np1 = -(n+1)*cos_theta*pi_np1 - (n+2)*pi_n*exp_u
            else:
                pi_np1, tau_np1 = angularj(n+1, u, scaling=True)
            exp_scal = np.exp(arg_trans + self.arg_mie[n] + (n+1.5)*u)
            arg = exp_scal*(-self.an[n+1]*pi_np1 + self.bn[n+1]*tau_np1)

            scat_ampl += arg
            pi_nm1 = pi_n
            pi_n = pi_np1

        # downward summation
        for n in range(nsp-1, 0, -1):
            pi_n, tau_n = angularj(n, u, scaling=True)
            exp_scal = np.exp(arg_trans + self.arg_mie[n-1] + (n+0.5)*u)
            arg = exp_scal*(-self.an[n]*pi_n + self.bn[n]*tau_n)
            scat_ampl += arg
            if abs(arg) < 1e-17*abs(scat_ampl):
                break

        return 2*pi*scat_ampl.real/self.Y

    def evalf(self, vec1, vec2):
        """
        returns the scattering amplitude.

        """
        if self.all_zeros:
            return 0

        y1, Phi1 = vec1
        y2, Phi2 = vec2
        cos_theta = cosTheta(self.Y, vec1, vec2)
        sin_theta2 = np.sqrt(0.5*(-cos_theta+1))
        arg_trans = - self.RbyL*(y1*np.sqrt(1+(self.Y/y1)**2)
                                 + y2*np.sqrt(1+(self.Y/y2)**2))
        asym = self.RbyL*np.exp(2*self.x*sin_theta2 + arg_trans)

        if asym < 1e-20:
            return 0
        if self.Y < 1e-10:
            return self.scat_ampl_lowfreq(y1, Phi1, y2, Phi2, cos_theta)
        return self.scat_largeradii(arg_trans, cos_theta)


def amplitude_pec(RbyL, Y, vec1, vec2):
    """
    calculates the wkb approximation of the scattering amplitudes for
    perfect electric conductor spheres (see Eq. (C.23) in Ref. [Sp20]_).

    """
    x = RbyL*Y
    y1, Phi1 = vec1
    y2, Phi2 = vec2
    cos_theta = cosTheta(Y, vec1, vec2)
    sin_theta2 = np.sqrt(0.5*(-cos_theta+1))

    # leading-order
    pre_fac = pi*RbyL*np.exp(2*x*sin_theta2 - RbyL*(y1*np.sqrt(1+(Y/y1)**2)
                                                    + y2*np.sqrt(1+(Y/y2)**2)))
    s0_tmtm = 1*pre_fac
    s0_tete = -1*pre_fac
    # leading correction
    s1_tmtm = -0.5/sin_theta2**3
    s1_tete = 0.5*cos_theta/sin_theta2**3
    # wkb approximation of the scattering amplitudes
    stmtm = s0_tmtm*(1 + s1_tmtm/x)
    stete = s0_tete*(1 + s1_tete/x)
    return stmtm, stete, 0, 0


def wkb_amplitude_x0_pec(RbyL, vec1, vec2):
    """
    calculates the wkb approximation of the scattering amplitudes for
    perfect electric conductor spheres in the zero-frequency limit
    (see Eq. (5.23) in  Ref. [Sp20]_).

    """
    y1, Phi1 = vec1
    y2, Phi2 = vec2
    _2xsin_theta2 = RbyL*np.sqrt(2*y1*y2*(1+np.cos(2*pi*(Phi1-Phi2))))

    pre_fac = pi*RbyL*np.exp(_2xsin_theta2 - RbyL*(y1+y2))

    stmtm = 1*pre_fac
    stete = -1*pre_fac
    return stmtm, stete, 0, 0


def amplitude_die(RbyL, Y, vec1, vec2, N):
    """
    calculates the wkb approximation of the scattering amplitudes for
    dielectric spheres (see Eq. (C.25) in Ref. [Sp20]_).

    Parameters
    ----------
    N : float
        relative refractive index

    Returns
    -------
    stmtm, stete : tuple of floats
        leading-oder and correction of the wkb approximation of the scattering
        amplitudes

    """
    x = RbyL*Y
    y1, Phi1 = vec1
    y2, Phi2 = vec2
    cos_theta = cosTheta(Y, vec1, vec2)
    c = -1j*np.sqrt(0.5*(-cos_theta-1))
    s = np.sqrt(0.5*(-cos_theta+1))
    # direct reflection coefficients
    r_tmtm = (N**2*s - np.sqrt(N**2-c**2))/(N**2*s + np.sqrt(N**2-c**2))
    r_tete = -(np.sqrt(N**2-c**2) - s)/(np.sqrt(N**2-c**2) + s)
    # leading-order see Eq. C.25 in [1]
    pre_fac = pi*RbyL*np.exp(2*x*s - RbyL*(y1*np.sqrt(1+(Y/y1)**2)
                                           + y2*np.sqrt(1+(Y/y2)**2)))
    s0_tmtm = pre_fac*r_tmtm
    s0_tete = pre_fac*r_tete
    # leading corrections see Eq. C.26 in [1]
    s1_tmtm = (-0.5/s**3 + 1/(s*c**2 - s**2*np.sqrt(N**2 - c**2))
               - ((2*(s*c*N**2)**2 - (N*c*c)**2*(1+s**2-s**4) + c**8)
                  / (s**3 * (N**2 - c**2)*((N*s)**2 - c**2)**2))
               + 0.5*((2*N**6 - (N**2*c)**2*(1+c**2) - N**2*c**4)
                      / ((N**2 - c**2)**(1.5) * ((N*s)**2 - c**2)**2)))
    s1_tete = (0.5*cos_theta/s**3 + 1/(s*c**2 + s**2*np.sqrt(N**2 - c**2))
               - (N**2 - 0.5*c**2)/(N**2 - c**2)**(1.5))
    # wkb approximation of the scattering amplitudes
    stmtm = s0_tmtm*(1 + s1_tmtm/x)
    stete = s0_tete*(1 + s1_tete/x)
    return stmtm, stete, 0, 0


def wkb_amplitude_x0_die(RbyL, vec1, vec2, N):
    """
    calculates the wkb approximation of the scattering amplitudes for
    dielectric spheres in the zero-frequency limit (see Eq. (5.23) in Ref.
    [Sp20]_).

    Parameters
    ----------
    N : float
        relative refractive index

    Returns
    -------
    stmtm, stete : tuple of floats
        leading-oder and correction of the wkb approximation of the scattering
        amplitudes

    """
    # direct reflection coefficients
    r_tmtm = (N**2 - 1)/(N**2 + 1)
    r_tete = 0

    y1, Phi1 = vec1
    y2, Phi2 = vec2
    _2xsin_theta2 = RbyL*np.sqrt(2*y1*y2*(1+np.cos(2*pi*(Phi1-Phi2))))
    pre_fac = pi*RbyL*np.exp(_2xsin_theta2 - RbyL*(y1+y2))
    stmtm = pre_fac*r_tmtm
    stete = pre_fac*r_tete
    return stmtm, stete, 0, 0


def amplitude_pemc(RbyL, Y, vec1, vec2, alpha):
    x = RbyL*Y
    y1, Phi1 = vec1
    y2, Phi2 = vec2
    cos_theta = cosTheta(Y, vec1, vec2)
    c = -1j*np.sqrt(0.5*(-cos_theta-1))
    s = np.sqrt(0.5*(-cos_theta+1))
    # leading order
    pre_fac = pi*RbyL*np.exp(2*x*s - RbyL*(y1*np.sqrt(1+(Y/y1)**2)
                                           + y2*np.sqrt(1+(Y/y2)**2)))
    s0_tmtm = pre_fac*np.cos(2*alpha)
    s0_tete = -s0_tmtm
    s0_tmte = -pre_fac*np.sin(2*alpha)
    # leading correction
    s1_geo = 0.5*cos_theta/s**3
    s1_mat = -c**2/s**3
    s1_tmtm = pre_fac*(np.cos(2*alpha)*s1_geo + np.cos(alpha)**2 * s1_mat)
    s1_tete = pre_fac*(-np.cos(2*alpha)*s1_geo + np.sin(alpha)**2 * s1_mat)
    s1_tmte = -np.sin(2*alpha)*pre_fac*(s1_geo + 0.5*s1_mat)
    # wkb approximation of the scattering amplitudes
    stmtm = s0_tmtm + s1_tmtm/x
    stete = s0_tete + s1_tete/x
    stmte = s0_tmte + s1_tmte/x
    stetm = stmte
    return stmtm.real, stete.real, stetm.real, stmte.real


def wkb_amplitude_x0_pemc(RbyL, vec1, vec2, alpha):
    # leading order
    y1, Phi1 = vec1
    y2, Phi2 = vec2
    _2xsin_theta2 = RbyL*np.sqrt(2*y1*y2*(1+np.cos(2*pi*(Phi1-Phi2))))
    pre_fac = pi*RbyL*np.exp(_2xsin_theta2 - RbyL*(y1+y2))
    stmtm = pre_fac*np.cos(2*alpha)
    stete = -stmtm
    stmte = -pre_fac*np.sin(2*alpha)
    stetm = stmte
    return stmtm, stete, stetm, stmte


def amplitude_biisotropic(RbyL, Y, vec1, vec2, mL, mm):
    """
    calculates the wkb approximation of the scattering amplitudes for
    dielectric spheres (see Ref. [github]_).

    Parameters
    ----------
    x : float
        size parameter
    theta : float
        (pi-theta)/2 is the angle of incidence
    mL : float
        relative refractive index
    mm : float
        relative impedance

    Returns
    -------
    stmtm, stete, stetm, stmte : tuple of floats
        leading-oder and correction of the wkb approximation of the scattering
        amplitudes

    """
    mR = np.conjugate(mL)
    mp = np.conjugate(mm)

    x = RbyL*Y
    y1, Phi1 = vec1
    y2, Phi2 = vec2
    cos_theta = cosTheta(Y, vec1, vec2)
    c = -1j*np.sqrt(0.5*(-cos_theta-1))
    s = np.sqrt(0.5*(-cos_theta+1))
    pre_fac = pi*RbyL*np.exp(2*x*s - RbyL*(y1*np.sqrt(1+(Y/y1)**2)
                                           + y2*np.sqrt(1+(Y/y2)**2)))
    # definition of several auxiliary variables and their derivatives
    cos_i = s
    cos_tL = np.sqrt(1-(c/mL)**2)
    cos_tR = np.sqrt(1-(c/mR)**2)
    # cos_tR = np.conjugate(cos_tL)
    tan2_i = -0.5*(c/s)**2
    tan2_tR = -0.5*c**2/(mR*(mR**2 - c**2))
    tan2_tL = -0.5*c**2/(mL*(mL**2 - c**2))
    deriv1_cos_i = 1j*c/cos_i
    deriv1_cos_tL = 1j*c/(mL**2*cos_tL)
    deriv1_cos_tR = 1j*c/(mR**2*cos_tR)
    deriv2_cos_i = 1/cos_i**3
    deriv2_cos_tL = 1/(mL**2*cos_tL**3)
    deriv2_cos_tR = 1/(mR**2*cos_tR**3)
    DL = cos_i + mm*cos_tL
    deriv1_DL = deriv1_cos_i + mm*deriv1_cos_tL
    deriv2_DL = deriv2_cos_i + mm*deriv2_cos_tL
    DR = cos_i + mp*cos_tR
    deriv1_DR = deriv1_cos_i + mp*deriv1_cos_tR
    deriv2_DR = deriv2_cos_i + mp*deriv2_cos_tR
    CL = mm*cos_i + cos_tL
    deriv1_CL = mm*deriv1_cos_i + deriv1_cos_tL
    deriv2_CL = mm*deriv2_cos_i + deriv2_cos_tL
    CR = mp*cos_i + cos_tR
    deriv1_CR = mp*deriv1_cos_i + deriv1_cos_tR
    deriv2_CR = mp*deriv2_cos_i + deriv2_cos_tR
    f = cos_i*(mp*DL + mm*DR)
    deriv1_f = deriv1_cos_i*(mp*DL+mm*DR)+cos_i*(mp*deriv1_DL+mm*deriv1_DR)
    deriv2_f = (deriv2_cos_i*(mp*DL+mm*DR)+cos_i*(mp*deriv2_DL+mm*deriv2_DR)
                + 2*deriv1_cos_i*(mp*deriv1_DL + mm*deriv1_DR))
    g = cos_tL*DR + cos_tR*DL
    deriv1_g = (deriv1_cos_tL*DR + cos_tL*deriv1_DR
                + deriv1_cos_tR*DL + cos_tR*deriv1_DL)
    deriv2_g = (deriv2_cos_tL*DR + cos_tL*deriv2_DR
                + deriv2_cos_tR*DL + cos_tR*deriv2_DL
                + 2*deriv1_cos_tL*deriv1_DR + 2*deriv1_cos_tR*deriv1_DL)
    F = cos_i*(CL + CR)
    deriv1_F = deriv1_cos_i*(CL+CR) + cos_i*(deriv1_CL + deriv1_CR)
    deriv2_F = (deriv2_cos_i*(CL+CR) + cos_i*(deriv2_CL + deriv2_CR)
                + 2*deriv1_cos_i*(deriv1_CL + deriv1_CR))
    G = mm*cos_tL*CR + mp*cos_tR*CL
    deriv1_G = (mm*deriv1_cos_tL*CR + mp*deriv1_cos_tR*CL
                + mm*cos_tL*deriv1_CR + mp*cos_tR*deriv1_CL)
    deriv2_G = (mm*deriv2_cos_tL*CR + mp*deriv2_cos_tR*CL
                + mm*cos_tL*deriv2_CR + mp*cos_tR*deriv2_CL
                + 2*mm*deriv1_cos_tL*deriv1_CR + 2*mp*deriv1_cos_tR*deriv1_CL)
    # direct reflection coefficients
    r_tmtm = (f-g)/(f+g)
    r_tete = (F-G)/(F+G)
    r_tmte = 2j*cos_i*(DL-DR)/(f+g)
    r_tetm = 2j*cos_i*(mm*CR-mp*CL)/(f+g)
    # leading order of the scattering amplitudes
    s0_tmtm = pre_fac*r_tmtm
    s0_tete = pre_fac*r_tete
    s0_tetm = pre_fac*r_tetm
    s0_tmte = pre_fac*r_tmte
    # geometrical correction
    s_geo = 0.5*(1-2*s**2)/s**3
    # leading correction of the wkb approximaton of the mie coefficients
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
    # derivatives of the direct reflection coefficients
    deriv1_rtmtm = (g*deriv1_f - f*deriv1_g)/(f+g)**2
    deriv2_rtmtm = (g*deriv2_f - f*deriv2_g
                    - 2*(g*deriv1_f - f*deriv1_g)*(deriv1_f+deriv1_g)/(f+g)
                    )/(f+g)**2
    deriv1_rtete = (G*deriv1_F - F*deriv1_G)/(F+G)**2
    deriv2_rtete = (G*deriv2_F - F*deriv2_G
                    - 2*(G*deriv1_F - F*deriv1_G)*(deriv1_F+deriv1_G)/(F+G)
                    )/(F+G)**2
    deriv1_rtmte = 1j*((DL-DR)*(1j*c/s - s*(deriv1_f + deriv1_g)/(f+g))
                       + s*(deriv1_DL - deriv1_DR))/(f+g)
    deriv2_rtmte = 1j*((DL-DR)*(1/s**3 - 2j*c/s * (deriv1_f + deriv1_g)/(f+g)
                                + 2*s*((deriv1_f + deriv1_g)/(f+g))**2
                                - s*(deriv2_f + deriv2_g)/(f+g))
                       + 2*((deriv1_DL - deriv1_DR)
                            * (1j*c/s - s*(deriv1_f + deriv1_g)/(f+g)))
                       + s*(deriv2_DL - deriv2_DR)
                       )/(f+g)
    deriv1_rtetm = 1j*((mm*CR-mp*CL)*(1j*c/s - s*(deriv1_F + deriv1_G)/(F+G))
                       + s*(mm*deriv1_CR - mp*deriv1_CL))/(F+G)
    deriv2_rtetm = 1j*((mm*CR-mp*CL)*(1/s**3 - 2j*c/s*(deriv1_f+deriv1_g)/(f+g)
                                      + 2*s*((deriv1_f + deriv1_g)/(f+g))**2
                                      - s*(deriv2_f + deriv2_g)/(f+g))
                       + 2*((mm*deriv1_CR - mp*deriv1_CL)
                            * (1j*c/s - s*(deriv1_f + deriv1_g)/(f+g)))
                       + s*(mm*deriv2_CR - mp*deriv2_CL)
                       )/(f+g)
    deriv_tmtm = 0.5*(1j*(1-2*s**2)/(s*c) * deriv1_rtmtm + s*deriv2_rtmtm)
    deriv_tete = 0.5*(1j*(1-2*s**2)/(s*c) * deriv1_rtete + s*deriv2_rtete)
    deriv_tmte = 0.5*(1j*(1-2*s**2)/(s*c) * deriv1_rtmte + s*deriv2_rtmte)
    deriv_tetm = 0.5*(1j*(1-2*s**2)/(s*c) * deriv1_rtetm + s*deriv2_rtetm)
    # leading-order correction of the scattering amplidues
    s1_tmtm = pre_fac*(r_tmtm*s_geo + a_mat + 0.5*(r_tmtm + r_tete)/(s*c**2)
                       + deriv_tmtm).real
    s1_tete = pre_fac*(r_tete*s_geo + b_mat + 0.5*(r_tmtm + r_tete)/(s*c**2)
                       + deriv_tete).real
    s1_tmte = pre_fac*(r_tmte*s_geo + d_mat + 0.5*(r_tmte - r_tetm)/(s*c**2)
                       + deriv_tmte).real
    s1_tetm = pre_fac*(r_tetm*s_geo + c_mat + 0.5*(r_tetm - r_tmte)/(s*c**2)
                       + deriv_tetm).real
    # wkb approximation of the scattering amplitudes and the leading-order
    # correction
    stmtm = s0_tmtm + s1_tmtm/x
    stete = s0_tete + s1_tete/x
    stetm = s0_tetm + s1_tetm/x
    stmte = s0_tmte + s1_tmte/x
    return stmtm, stete, stetm, stmte


def wkb_amplitude_x0_biisotropic(RbyL, vec1, vec2, mL, mm):
    """
    calculates the wkb approximation of the scattering amplitudes for
    dielectric spheres at zero-frequency (see Ref. [github]_).

    Parameters
    ----------
    x : float
        size parameter
    theta : float
        (pi-theta)/2 is the angle of incidence
    mL : float
        relative refractive index
    mm : float
        relative impedance

    Returns
    -------
    stmtm, stete, stetm, stmte : tuple of floats
        leading-oder and correction of the wkb approximation of the scattering
        amplitudes

    """
    mR = np.conjugate(mL)
    mp = np.conjugate(mm)

    y1, Phi1 = vec1
    y2, Phi2 = vec2
    _2xsin_theta2 = RbyL*np.sqrt(2*y1*y2*(1+np.cos(2*pi*(Phi1-Phi2))))
    pre_fac = pi*RbyL*np.exp(_2xsin_theta2 - RbyL*(y1+y2))

    DL = 1 + mm/mL
    DR = 1 + mp/mR
    CL = mm + 1/mL
    CR = mp + 1/mR
    f = mp*DL + mm*DR
    g = DR/mL + DL/mR
    F = CL + CR
    G = mm*CR/mL + mp*CL/mR

    # direct reflection coefficients
    r_tmtm = (f-g)/(f+g)
    r_tete = (F-G)/(F+G)
    r_tmte = 2j*(DL-DR)/(f+g)
    r_tetm = 2j*(mm*CR-mp*CL)/(f+g)
    # leading order of the scattering amplitudes
    stmtm = pre_fac*r_tmtm
    stete = pre_fac*r_tete
    stetm = pre_fac*r_tetm
    stmte = pre_fac*r_tmte
    return stmtm, stete, stetm, stmte
