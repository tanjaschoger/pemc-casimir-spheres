import numpy as np

from sympy.functions.special.polynomials import legendre, assoc_legendre
from sympy import cosh, exp


data_legendre = open("scaledlegendre.dat", "w")
data_assoclegendre = open("scaledassoclegendre.dat", "w")


for x in np.linspace(10, 100, 5):
    for n in range(1000, 2000, 200):
        legendrePl = legendre(n, cosh(x))*exp(-x*(n+0.5))
        data_legendre.write("{}, {}, {}\n".format(n, x, legendrePl.evalf()))

        assoclegendrePl1 = 1j*assoc_legendre(n, 1, cosh(x))*exp(-x*(n+0.5))
        data_assoclegendre.write("{}, {}, {}\n".
                                 format(n, x, assoclegendrePl1.evalf()))


data_legendre.close()
data_assoclegendre.close()
