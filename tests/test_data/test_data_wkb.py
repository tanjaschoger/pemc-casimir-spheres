import pytest
import numpy as np
from sympy import sqrt

from numerics.mie_coeff import wkb_die, wkb_biisotropic


def scaled_wkb_die(N, x, j):
    lamb = (j+1/2)/x
    cos_i = sqrt(1+lamb**2)
    cos_t = sqrt(N**2 + lamb**2)
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
    return complex(an.evalf()), complex(bn.evalf())


def scaled_wkb_biisotropic(mL, mm, x, j):
    mR = np.conjugate(mL)
    mp = np.conjugate(mm)
    lamb = (j+1/2)/x

    cos_i = sqrt(1+lamb**2)
    cos_tL = sqrt(1+(lamb/mL)**2)
    cos_tR = sqrt(1+(lamb/mR)**2)
    tan2_i = 0.5*lamb**2/(1+lamb**2)
    tan2_tR = 0.5*lamb**2/(mR*(mR**2 + lamb**2))
    tan2_tL = 0.5*lamb**2/(mL*(mL**2 + lamb**2))

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
    return (complex(an.evalf()), complex(bn.evalf()),
            complex(cn.evalf()), complex(dn.evalf()))


data_wkbBiisotropic = open("wkb_biisotropic.dat", "w")

for mL in [0.2+0.1j, 1+0.1j, 5+0.1j]:
    for mm in [1+0.1j, 10j]:
        for n in [1, 10, 1000, 5000]:
            for x in np.logspace(1, 3, 5):
                an, bn, cn, dn, = scaled_wkb_biisotropic(mL, mm, x, n)
                data_wkbBiisotropic.write("{}, {}, {}, {}, {}, {}, {}, {}\n"
                                          .format(mL, mm, x, n, an, bn, cn, dn))

data_wkbBiisotropic.close()

data_wkbMieDie = open("wkb_die.dat", "w")

for N in [0.2, 2, 1+0.1j, 5]:
    for n in [1, 10, 500, 1000, 5000]:
        for x in np.logspace(1, 3, 10):
            an, bn = scaled_wkb_die(N, x, n)
            data_wkbMieDie.write("{}, {}, {}, {}, {}\n".format(N, x, n, an, bn))


data_wkbMieDie.close()
