from sympy.functions.special.bessel import besseli, besselk
from sympy import exp, sqrt, asinh
import numpy as np


data_ratioBesseli = open("ratioBesseli.dat", "w")
data_ratioBesselk = open("ratioBesselk.dat", "w")
data_scaledBessel = open("scaledBessel.dat", "w")


for N in [0.2, 1, 1+0.1j, 5]:
    for n in [1, 10, 500, 1000, 5000]:
        for x in np.logspace(-2, 2, 10):
            arg = N*x
            ratioi = besseli(n-0.5, arg)/besseli(n+0.5, arg)
            data_ratioBesseli.write("{}, {}, {}\n".format(n, arg,
                                                          ratioi.evalf()))
            ratiok = besselk(n+0.5, arg)/besselk(n-0.5, arg)
            data_ratioBesselk.write("{}, {}, {}\n".format(n, arg,
                                                          ratiok.evalf()))
            Lamb = (n+0.5)/arg
            scaledBesseli = besseli(n+0.5, arg)*exp(-arg*(sqrt(1+Lamb**2)
                                                          - Lamb*asinh(Lamb)))
            scaledBesselk = besselk(n+0.5, arg)*exp(arg*(sqrt(1+Lamb**2)
                                                         - Lamb*asinh(Lamb)))
            data_scaledBessel.write("{}, {}, {}, {}\n".
                                    format(n, arg, scaledBesseli.evalf(),
                                           scaledBesselk.evalf()))

data_ratioBesseli.close()
data_ratioBesselk.close()
data_scaledBessel.close()
