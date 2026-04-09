r"""
determines the logarithm of the Fredholm determinant for finite frequencies

.. math::

    \det[1 - \mathcal{M}(i\xi)]
    = \det[\delta(\mathbf{k}_1 - \mathbf{k}_2)\delta_{p_1, p_2}
                  - \langle \mathbf{k}_1, p_1 | \mathcal{M}(i\xi)|
                      \mathbf{k}_2, p_2 \rangle
                  ]

"""
from math import pi, sqrt, ceil
from itertools import product
from numba import njit
import numpy as np

from src.reflection_kernel import kernelR, refl_kernel, polarization_transform
from src.reflection_kernel import kernelRlowfreq
from src.quad import quad_chebychev, quad_trapezoidal
from src.mie_coeff import add_prefac
import src.scattering_amplitudes as scam


class Fredholm:
    r"""
    calculates the logarithm of the Fredholm determinant
    :math:`\det(1-\mathcal{M})`.

    """

    def __init__(self, sphere1, eta):
        self.LbyR1, self.nmax1, self.mie_coeff1 = sphere1
        self.LbyReff = self.LbyR1

        if self.LbyReff > 1:
            self.dim_angular = 50
            self.dim_radial = 50
        else:
            self.dim_angular = ceil(eta)*ceil(sqrt(1/self.LbyReff))
            self.dim_radial = ceil(eta)*ceil(sqrt(1/self.LbyReff))

        self.angular = quad_trapezoidal(self.dim_angular)
        self.radial = quad_chebychev(self.dim_radial)


    def roundtrip_kernel(self, vec1, vec2):
        """
        Returns the kernel function of the roundtrip operator for a given in-
        and outgoing wave-vector and should be overridden in the inheriting
        class.

        Returns
        -------
        None.

        """
        pass

    def evalf(self):
        """
        computes the components of the roundtrip matrix and evaluates the
        logarithm of the Fredholm determinant for a fixed frequency.

        Returns
        -------
        float
            value of the logarithm of the Fredholm determinant

        """
        Phi1 = self.angular[0][0]
        w_i = self.angular[1][0]
        w_m = self.radial[1]

        fMtmtm = np.zeros((self.dim_angular, self.dim_radial, self.dim_radial),
                          dtype=np.complex128)
        fMtmte = np.zeros((self.dim_angular, self.dim_radial, self.dim_radial),
                          dtype=np.complex128)
        fMtetm = np.zeros((self.dim_angular, self.dim_radial, self.dim_radial),
                          dtype=np.complex128)
        fMtete = np.zeros((self.dim_angular, self.dim_radial, self.dim_radial),
                          dtype=np.complex128)

        dfMtmtm = np.zeros((self.dim_angular, self.dim_radial, self.dim_radial),
                          dtype=np.complex128)
        dfMtmte = np.zeros((self.dim_angular, self.dim_radial, self.dim_radial),
                          dtype=np.complex128)
        dfMtetm = np.zeros((self.dim_angular, self.dim_radial, self.dim_radial),
                          dtype=np.complex128)
        dfMtete = np.zeros((self.dim_angular, self.dim_radial, self.dim_radial),
                          dtype=np.complex128)

        for (k, y2), (m, y1) in product(enumerate(self.radial[0]),
                                        enumerate(self.radial[0])):
            Mtmtm_y1y2 = []
            Mtmte_y1y2 = []
            Mtetm_y1y2 = []
            Mtete_y1y2 = []

            dMtmtm_y1y2 = []
            dMtmte_y1y2 = []
            dMtetm_y1y2 = []
            dMtete_y1y2 = []

            for Phi2 in self.angular[0]:
                vec1 = y1, Phi1
                vec2 = y2, Phi2
                Mtmtm, Mtete, Mtmte, Mtetm, dMtmtm, dMtete, dMtmte, dMtetm = self.roundtrip_kernel(vec1, vec2)

                Mtmtm_y1y2.append(Mtmtm)
                Mtete_y1y2.append(Mtete)
                Mtmte_y1y2.append(Mtmte)
                Mtetm_y1y2.append(Mtetm)

                dMtmtm_y1y2.append(dMtmtm)
                dMtete_y1y2.append(dMtete)
                dMtmte_y1y2.append(dMtmte)
                dMtetm_y1y2.append(dMtetm)

            fMtmtm[:, m, k] = np.fft.fft(Mtmtm_y1y2)
            fMtmte[:, m, k] = np.fft.fft(Mtmte_y1y2)
            fMtetm[:, m, k] = np.fft.fft(Mtetm_y1y2)
            fMtete[:, m, k] = np.fft.fft(Mtete_y1y2)

            dfMtmtm[:, m, k] = np.fft.fft(dMtmtm_y1y2)
            dfMtmte[:, m, k] = np.fft.fft(dMtmte_y1y2)
            dfMtetm[:, m, k] = np.fft.fft(dMtetm_y1y2)
            dfMtete[:, m, k] = np.fft.fft(dMtete_y1y2)

        fMtmtm *= w_i*w_m/(2*pi)
        fMtmte *= w_i*w_m/(2*pi)
        fMtetm *= w_i*w_m/(2*pi)
        fMtete *= w_i*w_m/(2*pi)

        dfMtmtm *= w_i*w_m/(2*pi)
        dfMtmte *= w_i*w_m/(2*pi)
        dfMtetm *= w_i*w_m/(2*pi)
        dfMtete *= w_i*w_m/(2*pi)

        logdet = 0
        trdM = 0
        sign = 1
        unit_mat = np.eye(self.dim_radial)
        for (fMtmtm_n, fMtete_n,
             fMtmte_n, fMtetm_n,
             dfMtmtm_n, dfMtete_n,
             dfMtmte_n, dfMtetm_n) in zip(fMtmtm, fMtete, fMtmte, fMtetm,
                                          dfMtmtm, dfMtete, dfMtmte, dfMtetm):
            fD = np.block([[unit_mat - fMtmtm_n, - fMtmte_n],
                           [- fMtetm_n, unit_mat - fMtete_n]])
            sign_n, logdet_n = np.linalg.slogdet(fD)
            invfD = np.linalg.inv(fD)

            dM = np.block([[dfMtmtm_n, dfMtmte_n],
                           [dfMtetm_n, dfMtete_n]])
            trdM += np.trace(dM @ invfD)
            sign *= sign_n
            logdet += logdet_n
        return logdet/2, trdM.real/2


@njit
def kernel_M(Y, sphere1, sphere2, vec1, vecm, vec2):
    r"""
    calculates the kernel function of the roundtrip operator

    .. math::

        K_\mathcal{M} (\mathbf{k}_1, p_1; \mathbf{k}_1, p_1 ) =
        \sum_{p} \sum_{\gamma}
        \frac{w_\gamma}{(2\pi)^2}
        K_{\mathcal{R}_1} (\mathbf{k}_1, p_1; \mathbf{k}_\gamma, p)
        K_{\mathcal{R}_2} (\mathbf{k}_\gamma, p; \mathbf{k}_2, p_2)
        e^{-L(\kappa_1 + 2\kappa + \kappa_2)/2}

    where :math:`\kappa = \sqrt{\mathcal{K}^2 + k^2}`

    The kernel function of the reflection operator
    :math:`K_\mathcal{R} (\mathbf{k}_1, p_1; \mathbf{k}_2, p_2)`
    is calculated in :py:mod:`src.reflection_kernel`.

    """
    LbyR1, (a1n, b1n, c1n, d1n) = sphere1
    LbyR2, (a2n, b2n, c2n, d2n) = sphere2
    y1, _ = vec1
    y2, _ = vec2
    km_vals, Phin_vals, wm_vals, vn = vecm

    kMtmtm, kMtete, kMtmte, kMtetm = 0.0, 0.0, 0.0, 0.0

    S1tete = scam.ScatteringAmplitude(1/LbyR1, Y, a1n, b1n)
    S1tmtm = scam.ScatteringAmplitude(1/LbyR1, Y, b1n, a1n)
    S1tmte = scam.ScatteringAmplitude(1/LbyR1, Y, -c1n, d1n)
    S1tetm = scam.ScatteringAmplitude(1/LbyR1, Y, -d1n, c1n)

    S2tete = scam.ScatteringAmplitude(1/LbyR2, Y, a2n, b2n)
    S2tmtm = scam.ScatteringAmplitude(1/LbyR2, Y, b2n, a2n)
    S2tmte = scam.ScatteringAmplitude(1/LbyR2, Y, -c2n, d2n)
    S2tetm = scam.ScatteringAmplitude(1/LbyR2, Y, -d2n, c2n)

    for Phin in Phin_vals:
        for km, wm in zip(km_vals, wm_vals):
            vec = km, Phin

            scatampl1 = (S1tmtm.evalf(vec1, vec),
                         S1tete.evalf(vec1, vec),
                         S1tetm.evalf(vec1, vec),
                         S1tmte.evalf(vec1, vec))

            scatampl2 = (S2tmtm.evalf(vec, vec2),
                         S2tete.evalf(vec, vec2),
                         S2tetm.evalf(vec, vec2),
                         S2tmte.evalf(vec, vec2))

            poltrans1 = polarization_transform(Y, vec1, vec, 1)
            poltrans2 = polarization_transform(Y, vec, vec2, -1)

            k1tmtm, k1tete, k1tetm, k1tmte = refl_kernel(Y, y1, km,
                                                         scatampl1, poltrans1)
            k2tmtm, k2tete, k2tetm, k2tmte = refl_kernel(Y, km, y2,
                                                         scatampl2, poltrans2)

            exp_trans = np.exp(-0.5*(y1*np.sqrt(1+(Y/y1)**2)
                                     + y2*np.sqrt(1+(Y/y2)**2)
                                     + 2*km*np.sqrt(1+(Y/km)**2)))

            kMtmtm += wm*(k1tmtm*k2tmtm + k1tmte*k2tetm)*exp_trans
            kMtete += wm*(k1tete*k2tete + k1tetm*k2tmte)*exp_trans
            kMtmte += wm*(k1tmtm*k2tmte + k1tmte*k2tete)*exp_trans
            kMtetm += wm*(k1tetm*k2tmtm + k1tete*k2tetm)*exp_trans

    kMtmtm *= vn/(2*pi)
    kMtete *= vn/(2*pi)
    kMtmte *= vn/(2*pi)
    kMtetm *= vn/(2*pi)

    arg_trans = y1*np.sqrt(1+(Y/y1)**2) +  y2*np.sqrt(1+(Y/y2)**2)
    return (kMtmtm, kMtete, kMtmte, kMtetm,
            arg_trans*kMtmtm, arg_trans*kMtete, arg_trans*kMtmte, arg_trans*kMtetm)



class FredholmSphereSphere(Fredholm):
    r"""
    calculates the logarithm of the Fredholm determinant
    :math:`\det(1-\mathcal{M})` for the sphere-sphere geometry at finite
    frequency.

    """

    def __init__(self, Y, sphere1, sphere2, eta):
        super().__init__(sphere1, eta)
        self.modelR1 = add_prefac(self.mie_coeff1, self.nmax1)
        self.Y = Y
        self.LbyR2, nmax2, mie_coeff2 = sphere2
        self.modelR2 = add_prefac(mie_coeff2, nmax2)

        self.LbyReff = self.LbyR1 + self.LbyR2

        if self.LbyReff > 1:
            self.dim_angular = 50
            self.dim_radial = 50
            deg_in = 50
        else:
            self.dim_angular = ceil(eta)*ceil(sqrt(1/self.LbyReff))
            self.dim_radial = ceil(eta)*ceil(sqrt(1/self.LbyReff))
            deg_in = ceil(eta+1)*ceil(sqrt(1/self.LbyR1 + 1/self.LbyR2))

        self.angular = quad_trapezoidal(self.dim_angular)
        self.radial = quad_chebychev(self.dim_radial)

        self.radial2 = quad_chebychev(deg_in)


    def roundtrip_kernel(self, vec1, vec2):
        """
        See Also
        --------
        src.fredholm.Fredholm.roundtrip_kernel

        """
        sphere1 = self.LbyR1, self.modelR1
        sphere2 = self.LbyR2, self.modelR2

        vecm = (self.radial2[0], self.angular[0],
                self.radial2[1], self.angular[1][0])

        return kernel_M(self.Y, sphere1, sphere2, vec1, vecm, vec2)


class FredholmSpherePlane(Fredholm):
    r"""
    calculates the logarithm of the Fredholm determinant
    :math:`\det(1-\mathcal{M})` for the sphere-plane geometry at finite
    frequency.

    """

    def __init__(self, Y, sphere1, plane, eta):
        super().__init__(sphere1, eta)
        self.modelR1 = add_prefac(self.mie_coeff1, self.nmax1)
        self.Y = Y
        self.fresnel_coeff = plane

    def roundtrip_kernel(self, vec1, vec2):
        """
        See Also
        --------
        src.fredholm.Fredholm.roundtrip_kernel

        """
        y1, _ = vec1
        y2, _ = vec2
        sgn = 1

        arg_trans = (np.sqrt(y1**2 + self.Y**2) + np.sqrt(y2**2 + self.Y**2))
        exp_trans = np.exp(-np.sqrt(y1**2 + self.Y**2)
                           - np.sqrt(y2**2 + self.Y**2))
        kernelR1 = kernelR(1/self.LbyReff, self.Y, vec1, vec2, sgn, self.modelR1)

        k1tmtm, k1tete, k1tetm, k1tmte = kernelR1
        k2tmtm, k2tete, k2tetm, k2tmte = self.fresnel_coeff

        kMtmtm = (k1tmtm*k2tmtm + k1tmte*k2tetm)*exp_trans
        kMtete = (k1tete*k2tete + k1tetm*k2tmte)*exp_trans
        kMtmte = (k1tmtm*k2tmte + k1tmte*k2tete)*exp_trans
        kMtetm = (k1tetm*k2tmtm + k1tete*k2tetm)*exp_trans
        return (kMtmtm, kMtete, kMtmte, kMtetm,
                arg_trans*kMtmtm, arg_trans*kMtete, arg_trans*kMtmte, arg_trans*kMtetm)

@njit
def kernel_M_ht(sphere1, sphere2, vec1, vecm, vec2):
    """
    See Also
    --------
    src.pemc.kernel_M

    """
    LbyR1, (a1n, b1n, c1n, d1n) = sphere1
    LbyR2, (a2n, b2n, c2n, d2n) = sphere2
    y1, _ = vec1
    y2, _ = vec2
    km_vals, Phin_vals, wm_vals, vn = vecm

    kMtmtm, kMtete, kMtmte, kMtetm = 0.0, 0.0, 0.0, 0.0
    dkMtmtm, dkMtete, dkMtmte, dkMtetm = 0.0, 0.0, 0.0, 0.0

    for Phin in Phin_vals:
        for km, wm in zip(km_vals, wm_vals):
            vec = km, Phin

            k1tmtm = kernelRlowfreq(1/LbyR1, vec1, vec, a1n)
            k1tete = kernelRlowfreq(1/LbyR1, vec1, vec, b1n)
            k1tetm = kernelRlowfreq(1/LbyR1, vec1, vec, c1n)
            k1tmte = kernelRlowfreq(1/LbyR1, vec1, vec, d1n)

            k2tmtm = kernelRlowfreq(1/LbyR2, vec, vec2, a2n)
            k2tete = kernelRlowfreq(1/LbyR2, vec, vec2, b2n)
            k2tetm = kernelRlowfreq(1/LbyR2, vec, vec2, c2n)
            k2tmte = kernelRlowfreq(1/LbyR2, vec, vec2, d2n)

            exp_trans = np.exp(-km-(y1+y2)/2)
            kMtmtm += wm*(k1tmtm*k2tmtm + k1tmte*k2tetm)*exp_trans
            kMtete += wm*(k1tete*k2tete + k1tetm*k2tmte)*exp_trans
            kMtmte += wm*(k1tmtm*k2tmte + k1tmte*k2tete)*exp_trans
            kMtetm += wm*(k1tetm*k2tmtm + k1tete*k2tetm)*exp_trans

    kMtmtm *= vn/(2*pi)
    kMtete *= vn/(2*pi)
    kMtmte *= vn/(2*pi)
    kMtetm *= vn/(2*pi)

    return (kMtmtm, kMtete, kMtmte, kMtetm,
            (y1+y2)*kMtmtm, (y1+y2)*kMtete, (y1+y2)*kMtmte, (y1+y2)*kMtetm)


class FredholmSphereSphereHT(Fredholm):
    r"""
    calculates the logarithm of the Fredholm determinant
    :math:`\det(1-\mathcal{M})` for the sphere-sphere geometry at zero
    frequency.

    """

    def __init__(self, sphere1, sphere2, eta):
        super().__init__(sphere1, eta)
        self.modelR1 = self.mie_coeff1
        self.LbyR2, _, self.modelR2 = sphere2

        self.LbyReff = self.LbyR1 + self.LbyR2

        if self.LbyReff > 1:
            self.dim_angular = 50
            self.dim_radial = 50
            deg_in = 50
        else:
            self.dim_angular = ceil(eta)*ceil(sqrt(1/self.LbyReff))
            self.dim_radial = ceil(eta)*ceil(sqrt(1/self.LbyReff))
            deg_in = ceil(eta+1)*ceil(sqrt(1/self.LbyR1 + 1/self.LbyR2))

        self.angular = quad_trapezoidal(self.dim_angular)
        self.radial = quad_chebychev(self.dim_radial)

        self.radial2 = quad_chebychev(deg_in)

    def roundtrip_kernel(self, vec1, vec2):
        """
        See Also
        --------
        src.fredholm.Fredholm.roundtrip_kernel

        """
        sphere1 = self.LbyR1, self.modelR1
        sphere2 = self.LbyR2, self.modelR2
        vecm = (self.radial2[0], self.angular[0],
                self.radial2[1], self.angular[1][0])

        return kernel_M_ht(sphere1, sphere2, vec1, vecm, vec2)


class FredholmSpherePlaneHT(Fredholm):
    r"""
    calculates the logarithm of the Fredholm determinant
    :math:`\det(1-\mathcal{M})` for the sphere-plane geometry at zero
    frequency.

    """

    def __init__(self, sphere1, plane, eta):
        super().__init__(sphere1, eta)
        self.modelR1 = self.mie_coeff1
        self.fresnel_coeff = plane

    def roundtrip_kernel(self, vec1, vec2):
        """
        See Also
        --------
        src.fredholm.Fredholm.roundtrip_kernel

        """
        y1, _ = vec1
        y2, _ = vec2

        a1n, b1n, c1n, d1n = self.modelR1
        k1tmtm = kernelRlowfreq(1/self.LbyReff, vec1, vec2, a1n)
        k1tete = kernelRlowfreq(1/self.LbyReff, vec1, vec2, b1n)
        k1tetm = kernelRlowfreq(1/self.LbyReff, vec1, vec2, c1n)
        k1tmte = kernelRlowfreq(1/self.LbyReff, vec1, vec2, d1n)

        k2tmtm, k2tete, k2tetm, k2tmte = self.fresnel_coeff

        exp_trans = np.exp(-(y1+y2))
        kMtmtm = (k1tmtm*k2tmtm + k1tmte*k2tetm)*exp_trans
        kMtete = (k1tete*k2tete + k1tetm*k2tmte)*exp_trans
        kMtmte = (k1tmtm*k2tmte + k1tmte*k2tete)*exp_trans
        kMtetm = (k1tetm*k2tmtm + k1tete*k2tetm)*exp_trans
        return (kMtmtm, kMtete, kMtmte, kMtetm,
                (y1+y2)*kMtmtm, (y1+y2)*kMtete, (y1+y2)*kMtmte, (y1+y2)*kMtetm)

