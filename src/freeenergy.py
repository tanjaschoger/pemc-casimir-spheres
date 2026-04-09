r"""
calculates the Casimir free energy in units of :math:`k_B T` in terms of the
Pade spectrum decomposition (:py:mod:`src.psd`).
"""
from math import ceil, pi
import numpy as np
from numpy.polynomial.laguerre import laggauss

from src.psd import psd
from src.pemc import fredholm_pemc, fredholm_pemc_ht
from src.quad import quad_chebychev


def freeenergy_pemc(LbyReff, LbyLambT, theta1, theta2, u, etas):
    r"""
    calculates the Casimir free energy and force between two PEMC spheres at
    finite temperature. The free energy is given in units of :math:`k_B T`.

    Parameters
    ----------
    LbyReff : float
        ratio of the surface-to-surface distance and the effective radius
    LbyLambT : float
        ratio of the surface-to-surface distance and the thermal wavelength
        :math:`\lambda_T = \hbar c/k_B T`
    theta1, theta2 : floats
        parameters in the range [0, pi/2] specifying the material of the
        spheres
    u : float
        parameter between 0 and 1/4 specifying the ratio of the sphere radii
    etas : tuple
        parameters determining the accuracy of the result

    Returns
    -------
    float
        value of the Casimir free energy in units of :math:`k_B T`
    float
        value of the Casimir force in units of :math:`k_B T/L`

    """
    eta1, eta2 = etas
    N = ceil(eta2*np.sqrt(1/LbyLambT))
    xi_vals, eta_vals = psd(N)

    energy0, force0 = fredholm_pemc_ht(LbyReff, theta1=theta1, theta2=theta2,
                                       u=u, eta=eta1)

    casimir_energy = energy0
    casimir_force = force0
    for xi, eta in zip(xi_vals, eta_vals):
        energyn, forcen  = fredholm_pemc(LbyReff, LbyLambT,
                                         theta1=theta1, theta2=theta2,
                                         u=u, freq=xi, eta=eta1)
        casimir_energy += 2*eta*energyn
        casimir_force += 2*eta*forcen

    return casimir_energy, casimir_force


def energy_pemc(LbyReff, theta1, theta2, u, etas):
    """
    calculates the Casimir energy and force between two PEMC spheres at zero
    temperature.

    Returns
    -------
    float
        Casimir energy in units of :math:`\hbar c /L`
    float
        Casimir force in units of :math:`\hbar c /L^2`

    """
    eta1, eta2 = etas
    N = max(10, ceil(eta2/np.sqrt(LbyReff)))
    xi_vals, eta_vals = laggauss(N)
    LbyLambT = 1

    casimir_energy = 0
    casimir_force = 0
    for xi, eta in zip(xi_vals, eta_vals):
        energyn, forcen  = fredholm_pemc(LbyReff, LbyLambT,
                                         theta1=theta1, theta2=theta2,
                                         u=u, freq=xi, eta=eta1)
        casimir_energy += eta*energyn*np.exp(xi)
        casimir_force += eta*forcen*np.exp(xi)

    return casimir_energy/pi, casimir_force/pi
