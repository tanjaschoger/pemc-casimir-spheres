import os
import sys
from sympy import N
import pytest
from pathlib import Path

sys.path.insert(0, os.path.abspath('../src/'))
os.environ["NUMBA_DISABLE_JIT"] = "1"

import bessel as bsl


DATADIR = Path('test_data')


def test_ratioBesseli():
    data = open(DATADIR / "ratioBesseli.dat", encoding="latin1")
    for line in data.readlines():
        line = line.strip()
        n, x, besselratio = line.split(", ")
        besselratio = complex(N(besselratio))
        n = int(n)
        x = complex(x)
        try:
            assert (bsl.ratio(n+0.5, x)
                    == pytest.approx(besselratio, rel=1e-11, abs=1e-11))
        except:
            data.close()
            raise AssertionError
    data.close()


def test_ratioBesselk():
    data = open(DATADIR / "ratioBesselk.dat", encoding="latin1")
    for line in data.readlines():
        line = line.strip()
        n, x, besselratio = line.split(", ")
        besselratio = complex(N(besselratio))
        n = int(n)
        x = complex(x)
        try:
            assert (bsl.ratioK(n, x)[-1]
                    == pytest.approx(besselratio, rel=1e-11, abs=1e-11))
        except:
            data.close()
            raise AssertionError
    data.close()


def test_scaledBesseli():
    data = open(DATADIR / "scaledBessel.dat", encoding="latin1")

    error = {'rel': 1e-11,
             'abs': 1e-11}

    for line in data.readlines():
        line = line.strip()
        k, x, besseli, besselk = line.split(", ")
        besseli = complex(N(besseli))
        besselk = complex(N(besselk))
        k = int(k)
        x = complex(x)
        try:
            assert (bsl.scaled2_modbesselI(k, x)[-1]
                    == pytest.approx(besseli, **error) and
                    bsl.scaled2_modbesselK(k, x)[-1]
                    == pytest.approx(besselk, **error)
                    )
        except:
            data.close()
            raise AssertionError
    data.close()
