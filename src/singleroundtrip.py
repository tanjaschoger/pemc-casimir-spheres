"""
singleroundtrip.py calculates the single-roundrip result of the Casimir free
energy in the high-temperature limit.

"""
import numpy as np


def singleroundtrip_pemc(delta, u, l_reff_vals):
    """
    determines the singleroundtrip result of the Casimir free energy for two
    PEMC spheres.

    Parameters
    ----------
    delta : array_like
        specifies the materials of the spheres
    u : float
        in the range [0, 1/4], specifies the ratio of the sphere radii
    l_reff_vals : array_like
        specifies the effective distance between the two spheres

    Returns
    -------
    array_like
        singleroundtrip expression for two spheres or a sphere and plane

    """
    if u == 0:
        y = 1 + l_reff_vals
        srt_te = 0.5*(y/(2*(y**2-1)) + y*np.log1p(-1/y**2)/2)
        srt_tm = 0.5*(y/(2*(y**2-1)) - 1/(2*y))
        srt_corr = 0.5*(y/((y**2-1)) - 1/(2*y) + y*np.log1p(-1/y**2)/2)
        return 0.5*((1 + np.cos(2*delta))*(srt_te + srt_tm)
                    - (1 - np.cos(2*delta))*srt_corr)
    else:
        r1r2_val = u/(0.5-u + np.sqrt((0.5-u)**2-u**2))
        alpha = 1/r1r2_val
        beta = r1r2_val
        y = 1 + l_reff_vals + l_reff_vals**2/(2*(1+alpha)*(1+beta))

        srt_te = 0.5*(y/(2*(y**2 - 1))
                      - (2*((2*y + alpha + beta)
                            * np.log1p(1/(2*y*(2*y + alpha + beta))))
                      - (2*y + alpha + beta)*np.log1p(-1/y**2)
                      + 2*beta*(np.arctanh(np.sqrt(alpha*(2*y + alpha + beta))
                                           / (1 - alpha*y - 2*y**2))
                                / np.sqrt(alpha*(2*y + alpha + beta)))
                      + 2*alpha*(np.arctanh(np.sqrt(beta*(2*y + alpha + beta))
                                            / (1 - beta*y - 2*y**2))
                                 / np.sqrt(beta*(2*y + alpha + beta))))/6
                      )

        srt_tm = 0.5*(0.5*y/(y**2-1)
                      - 1/(2*y+alpha) - 1/(2*y+beta) + 1/(2*y+alpha+beta))

        srt_corr = 0.5*(y/(y**2-1)
                        - 1/(2*y+alpha) - 1/(2*y+beta)
                        - (alpha + beta)*np.log1p(1/(2*y*(2*y + alpha + beta)))
                        + 0.5*(alpha+beta)*np.log1p(-1/y**2)
                        - beta*(np.arctanh(np.sqrt(alpha*(2*y + alpha + beta))
                                           / (1 - alpha*y - 2*y**2))
                                / np.sqrt(alpha*(2*y + alpha + beta)))
                        - alpha*(np.arctanh(np.sqrt(beta*(2*y + alpha + beta))
                                            / (1 - beta*y - 2*y**2))
                                 / np.sqrt(beta*(2*y + alpha + beta)))
                        )

        return 0.5*((1 + np.cos(2*delta))*(srt_te + srt_tm)
                    - (1 - np.cos(2*delta))*srt_corr)
