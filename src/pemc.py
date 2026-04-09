"""
Returns the logarithm of the Fredholm determinant for a system of two PEMC
spheres or a PEMC sphere and a plate.

"""
from math import ceil
import numpy as np

import src.fredholm as fh
import src.mie_coeff as mie
import src.fresnel as fres


def multipole_order(eta, LbyR):
    """
    defines the multipole order for a given accuracy

    Parameters
    ----------
    eta : float
        controls the numerical error of the multipole expansion
    LbyR : float
        aspect ratio

    Returns
    -------
    int
        maximal multipole order

    """
    return max(50, ceil(eta/LbyR))


def fredholm_pemc(LbyReff, LbyLambT, theta1, theta2, u, freq, eta):
    """
    returns the logarithm of the Fredholm determinant in the high-temperature
    limit (zero-frequency limit)


    Parameters
    ----------
    Parameters
    ----------
    LbyReff : float
        ratio surface-to-surface distance and the effective radius
    LbyLambT : float
        ratio surface-to-surface distance and thermal wavelength
    theta1, theta2 : floats
        parameter in the range [0, pi/2] specifying the material
        of the sphere, with theta=0 for a perfect electric reflector and
        theta=pi/2 for a perfect magnetic reflector
    u : float
        parameter between 0 and 1/4 specifying the ratio of the sphere radii
    freq : float
        frequency
    eta : float
        parameter specifying the accuracy of the result

    Returns
    -------
    float
        logarithm of the Fredholm determinant

    """
    eta_multipole = 12
    Y = LbyLambT*freq

    # sphere-plane result
    if u == 0:
        nmax = multipole_order(eta_multipole, LbyReff)
        x = Y/LbyReff
        mie_coeff = mie.mie_pemc(theta=theta1, x=x, nmax=nmax, scaling=True)
        sphere = LbyReff, nmax, mie_coeff
        plane = fres.fresnel_pemc(theta2)
        return fh.FredholmSpherePlane(Y, sphere, plane, eta).evalf()

    # sphere-sphere result
    R1byR2 = u/(0.5-u + np.sqrt((0.5-u)**2-u**2))
    LbyR1 = LbyReff/(1 + R1byR2)
    LbyR2 = LbyReff/(1 + 1/R1byR2)

    nmax1 = multipole_order(eta_multipole, LbyR1)
    nmax2 = multipole_order(eta_multipole, LbyR2)

    x1 = Y/LbyR1
    x2 = Y/LbyR2

    mie_coeff1 = mie.mie_pemc(theta=theta1, x=x1, nmax=nmax1, scaling=True)
    mie_coeff2 = mie.mie_pemc(theta=theta2, x=x2, nmax=nmax2, scaling=True)

    sphere1 = LbyR1, nmax1, mie_coeff1
    sphere2 = LbyR2, nmax2, mie_coeff2

    return fh.FredholmSphereSphere(Y, sphere1, sphere2, eta).evalf()


def fredholm_pemc_ht(LbyReff, theta1, theta2, u, eta):
    """
    returns the logarithm of the Fredholm determinant in the high-temperature
    limit (zero-frequency limit)

    """
    eta_multipole = 12

    # sphere-plane result
    if u == 0:
        nmax1 = multipole_order(eta_multipole, LbyReff)
        modelR1 = mie.mie_pemc_x0(theta=theta1, x=0, nmax=nmax1, scaling=True)
        sphere1 = LbyReff, nmax1, modelR1
        plane = fres.fresnel_pemc(theta2)
        return fh.FredholmSpherePlaneHT(sphere1, plane, eta).evalf()

    # sphere-sphere result
    R1byR2 = u/(0.5-u + np.sqrt((0.5-u)**2-u**2))
    LbyR1 = LbyReff/(1 + R1byR2)
    LbyR2 = LbyReff/(1 + 1/R1byR2)

    nmax1 = multipole_order(eta_multipole, LbyR1)
    nmax2 = multipole_order(eta_multipole, LbyR2)

    modelR1 = mie.mie_pemc_x0(theta=theta1, x=0, nmax=nmax1, scaling=True)
    modelR2 = mie.mie_pemc_x0(theta=theta2, x=0, nmax=nmax2, scaling=True)

    sphere1 = LbyR1, nmax1, modelR1
    sphere2 = LbyR2, nmax2, modelR2

    return fh.FredholmSphereSphereHT(sphere1, sphere2, eta).evalf()
