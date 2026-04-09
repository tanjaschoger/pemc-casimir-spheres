import numpy as np

from sympy.functions.special.polynomials import legendre, assoc_legendre
from sympy import cosh, exp, sqrt


data_angular = open("scaledangular.dat", "w")


for x in np.linspace(10, 100, 5):
    for n in range(1, 2000, 200):
        pi = 1j*assoc_legendre(n, 1, cosh(x))*exp(-x*(n+0.5))/sqrt(cosh(x)**2-1)
        tau = -cosh(x)*pi + n*(n+1)*legendre(n, cosh(x))*exp(-x*(n+0.5))
        data_angular.write("{}, {}, {}, {}\n".format(n, x, pi.evalf(),
                                                     tau.evalf()))

data_angular.close()
