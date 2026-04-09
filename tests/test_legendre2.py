import os
import sys
import pytest
from pathlib import Path

sys.path.insert(0, os.path.abspath('../src/'))
os.environ["NUMBA_DISABLE_JIT"] = "1"

from legendre import legendre_pl, legendre_pl1

DATADIR = Path('test_data')


def test_legendre():
    data = open(DATADIR / "scaledlegendre.dat", encoding="latin1")
    for line in data.readlines():
        line = line.strip()
        n, x, legendre = line.split(", ")
        n = int(n)
        x = float(x)
        legendre = float(legendre)
        try:
            assert legendre_pl(n, x) == pytest.approx(legendre, rel=1e-11,
                                                      abs=1e-11)
        except:
            data.close()
            raise AssertionError
    data.close()


def test_assoclegendre():
    data = open(DATADIR / "scaledassoclegendre.dat", encoding="latin1")
    for line in data.readlines():
        line = line.strip()
        n, x, legendre = line.split(", ")
        n = int(n)
        x = float(x)
        legendre = float(legendre)
        try:
            assert legendre_pl1(n, x) == pytest.approx(legendre, rel=1e-11,
                                                       abs=1e-11)
        except:
            data.close()
            raise AssertionError
    data.close()
