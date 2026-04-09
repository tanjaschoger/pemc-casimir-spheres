import numpy as np
from sympy.functions.special.bessel import besseli, besselk
from sympy import exp, sqrt, asinh, log, loggamma, sin, cos
from math import pi


def scaled_mie_die(N, x, j):
    ratioI_Nx = (besseli(j-0.5, N*x)/besseli(j+0.5, N*x)).evalf()
    ratioI_x = (besseli(j-0.5, x)/besseli(j+0.5, x)).evalf()
    invratioK_x = (besselk(j-0.5, x)/besselk(j+0.5, x)).evalf()
    Lamb = (j+0.5)/x
    if x < 10:
        scaling = exp(-(2*j+1)*log(2*x) - loggamma(j+1) - loggamma(j+2)
                      + loggamma(2*j+3) + loggamma(2*j+1))
    else:
        scaling = exp(-2*x*(sqrt(1+Lamb**2) - Lamb*asinh(Lamb)))
    ratioKI = besseli(j+0.5, x)/besselk(j+0.5, x)*scaling

    sj_a = (x*ratioI_x - j)
    sj_b = (N*x*ratioI_Nx - j)
    sj_c = x*invratioK_x + j
    sj_d = N*x*ratioI_Nx - j
    pre_fac = (-1)**j * 0.5*pi*ratioKI

    an = pre_fac*(N**2*sj_a - sj_b)/(N**2*sj_c + sj_d)
    bn = pre_fac*(sj_a - sj_b)/(sj_c + sj_d)
    return complex(an.evalf()), complex(bn.evalf())


def scaled_mie_pemc(theta, x, j):
    ratioI_x = (besseli(j-0.5, x)/besseli(j+0.5, x)).evalf()
    invratioK_x = (besselk(j-0.5, x)/besselk(j+0.5, x)).evalf()

    Lamb = (j+0.5)/x
    if x < 10:
        scaling = exp(-(2*j+1)*log(2*x) - loggamma(j+1) - loggamma(j+2)
                      + loggamma(2*j+3) + loggamma(2*j+1))
    else:
        scaling = exp(-2*x*(sqrt(1+Lamb**2) - Lamb*asinh(Lamb)))
    ratioKI = besseli(j+0.5, x)/besselk(j+0.5, x)*scaling

    termI = x*ratioI_x - j
    termK = x*invratioK_x + j
    pre_fac = (-1)**j * 0.5*pi*ratioKI
    an = pre_fac*(cos(theta)**2 * termI
                  - sin(theta)**2 * termK)/(x*invratioK_x + j)
    bn = pre_fac*(-cos(theta)**2 * termK
                  + sin(theta)**2 * termI)/(x*invratioK_x + j)
    cn = -0.5*pre_fac*sin(2*theta)*(termI + termK)/(x*invratioK_x + j)
    dn = cn
    return (complex(an.evalf()), complex(bn.evalf()),
            complex(cn.evalf()), complex(dn.evalf()))


def scaled_mie_biisotropic(mL, mm, x, j):
    mR = np.conjugate(mL)
    mp = np.conjugate(mm)
    xR = mR*x
    xL = mL*x
    ratioI_xR = (besseli(j-0.5, xR)/besseli(j+0.5, xR))
    ratioI_xL = (besseli(j-0.5, xL)/besseli(j+0.5, xL))
    ratioI_x = (besseli(j-0.5, x)/besseli(j+0.5, x))
    invratioK_x = (besselk(j-0.5, x)/besselk(j+0.5, x))

    Lamb = (j+0.5)/x
    if x < 10:
        scaling = exp(-(2*j+1)*log(2*x) - loggamma(j+1) - loggamma(j+2)
                      + loggamma(2*j+3) + loggamma(2*j+1))
    else:
        scaling = exp(-2*x*(sqrt(1+Lamb**2) - Lamb*asinh(Lamb)))
    ratioKI = besseli(j+0.5, x)/besselk(j+0.5, x)*scaling

    VR = mp*(ratioI_xR - j/xR) + invratioK_x + j/x
    VL = mm*(ratioI_xL - j/xL) + invratioK_x + j/x
    WR = ratioI_xR - j/xR + mp*(invratioK_x + j/x)
    WL = ratioI_xL - j/xL + mm*(invratioK_x + j/x)
    delta = VR*WL + VL*WR
    AR = mp*(ratioI_x - j/x) - (ratioI_xR - j/xR)
    AL = mm*(ratioI_x - j/x) - (ratioI_xL - j/xL)
    BR = ratioI_x - j/x - mp*(ratioI_xR - j/xR)
    BL = ratioI_x - j/x - mm*(ratioI_xL - j/xL)

    pre_fac = (-1)**j * 0.5*pi*ratioKI
    an = pre_fac*(VR*AL + VL*AR)/delta
    bn = pre_fac*(WR*BL + WL*BR)/delta
    cn = 1j*pre_fac*((ratioI_x - j/x + invratioK_x + j/x) *
                     (mm*(ratioI_xR - j/xR) - mp*(ratioI_xL - j/xL)))/delta
    dn = 1j*pre_fac*((ratioI_x - j/x + invratioK_x + j/x) *
                     (mm*(ratioI_xL - j/xL) - mp*(ratioI_xR - j/xR)))/delta
    return (complex(an.evalf()), complex(bn.evalf()),
            complex(cn.evalf()), complex(dn.evalf()))


data_mieBiisotropic = open("mie_biisotropic.dat", "w")

for mL in [0.2+0.1j, 1+0.1j, 5+0.1j]:
    for mm in [1+0.1j, 10j]:
        for n in [1, 10, 1000, 5000]:
            for x in np.logspace(-2, 3, 5):
                an, bn, cn, dn, = scaled_mie_biisotropic(mL, mm, x, n)
                data_mieBiisotropic.write("{}, {}, {}, {}, {}, {}, {}, {}\n"
                                          .format(mL, mm, x, n, an, bn, cn, dn))
data_mieBiisotropic.close()


data_miePemc = open("mie_pemc.dat", "w")

for theta in [pi/16, pi/4, pi/3, pi/2]:
    for n in [1, 10, 500, 1000, 5000]:
        for x in np.logspace(-2, 3, 10):
            an, bn, cn, dn = scaled_mie_pemc(theta, x, n)
            data_miePemc.write("{}, {}, {}, {}, {}, {}, {}\n"
                               .format(theta, x, n, an, bn, cn, dn))

data_miePemc.close()


data_mieDie = open("mie_die.dat", "w")

for N in [0.2, 1, 1+0.1j, 5]:
    for n in [1, 10, 500, 1000, 5000]:
        for x in np.logspace(-2, 3, 10):
            an, bn = scaled_mie_die(N, x, n)
            data_mieDie.write("{}, {}, {}, {}, {}\n".format(N, x, n, an, bn))

data_mieDie.close()
