import os
import sys
import pytest
from pathlib import Path

sys.path.insert(0, os.path.abspath('../src/'))
os.environ["NUMBA_DISABLE_JIT"] = "1"
from angular_func import angularj


DATADIR = Path('test_data')


def test_angular():
    data = open(DATADIR / "scaledangular.dat", encoding="latin1")

    error = {'rel': 1e-11,
             'abs': 1e-11}

    for line in data.readlines():
        line = line.strip()
        n, x, pi, tau = line.split(", ")
        n = int(n)
        x = float(x)
        pi = float(pi)
        tau = float(tau)
        Pi, Tau = angularj(n, x, scaling=True)
        try:
            assert (pi == pytest.approx(Pi, **error) and
                    tau == pytest.approx(Tau, **error))
        except:
            data.close()
            raise AssertionError
    data.close()
