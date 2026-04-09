import os
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, os.path.abspath('../src/'))

from pemc import fredholm_pemc_ht

DATADIR = Path('data')

filename_u0 = DATADIR / 'u0p0.csv'
filename_u0p10 = DATADIR / 'u0p10.csv'
filename_u0p25 = DATADIR / 'u0p25.csv'


def extract_u(s):
    return float('0.' + str(s.stem).split('p')[1])


datafiles = [filename_u0, filename_u0p10, filename_u0p25]

error = {'rel': 1e-4, 'abs': 1e-4}
eta = 8
theta1 = 0
theta2 = 0


def test_pec():
    """
    compares the zero-frequency result for two perfect reflector spheres with
    the results obtained by B. Spreng

    """
    for filename in datafiles:
        l_reff_vals, tm_vals, te_vals = np.loadtxt(filename, delimiter=',',
                                                   unpack=True)
        u = extract_u(filename)

        for idx in [80]:
            LbyReff = l_reff_vals[idx]

            spreng = tm_vals[idx] + te_vals[idx]
            logdet_ht = fredholm_pemc_ht(LbyReff, theta1=theta1, theta2=theta2,
                                         u=u, eta=eta)[0]

            assert spreng == pytest.approx(logdet_ht, **error)
