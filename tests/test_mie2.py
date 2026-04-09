
import os
import sys
import pytest
from pathlib import Path

sys.path.insert(0, os.path.abspath('../src/'))
os.environ["NUMBA_DISABLE_JIT"] = "1"

import mie_coeff as mie

DATADIR = Path('test_data')


def test_scaledMieDie():
    data = open(DATADIR / "mie_die.dat", encoding="latin1")

    error = {'rel': 1e-11,
             'abs': 1e-11}

    for line in data.readlines():
        line = line.strip()
        N, x, n, an, bn = line.split(", ")
        an = complex(an)
        bn = complex(bn)
        n = int(n)
        x = float(x)
        N = complex(N)
        An, Bn = mie.mie_die(N, x, n, scaling=True)
        try:
            assert (An[-1] == pytest.approx(an, **error) and
                    Bn[-1] == pytest.approx(bn, **error)
                    )
        except:
            data.close()
            raise AssertionError
    data.close()


def test_scaledMiePemc():
    data = open(DATADIR / "mie_pemc.dat", encoding="latin1")

    error = {'rel': 1e-11,
             'abs': 1e-11}

    for line in data.readlines():
        line = line.strip()
        theta, x, n, an, bn, cn, dn = line.split(", ")
        an = complex(an)
        bn = complex(bn)
        cn = complex(cn)
        dn = complex(dn)
        n = int(n)
        x = float(x)
        theta = complex(theta)
        An, Bn, Cn, Dn = mie.mie_pemc(theta, x, n, scaling=True)
        try:
            assert (An[-1] == pytest.approx(an, **error) and
                    Bn[-1] == pytest.approx(bn, **error) and
                    Cn[-1] == pytest.approx(cn, **error) and
                    Dn[-1] == pytest.approx(dn, **error)
                    )
        except:
            data.close()
            raise AssertionError
    data.close()


def test_scaledMieBiisotropic():
    data = open(DATADIR / "mie_biisotropic.dat", encoding="latin1")

    error = {'rel': 1e-11,
             'abs': 1e-11}

    for line in data.readlines():
        line = line.strip()
        mL, mm, x, n, an, bn, cn, dn = line.split(", ")
        an = complex(an)
        bn = complex(bn)
        cn = complex(cn)
        dn = complex(dn)
        n = int(n)
        x = float(x)
        mL = complex(mL)
        mm = complex(mm)
        An, Bn, Cn, Dn = mie.mie_biisotropic(mL, mm, x, n, scaling=True)
        try:
            assert (An[-1] == pytest.approx(an, **error) and
                    Bn[-1] == pytest.approx(bn, **error) and
                    Cn[-1] == pytest.approx(cn, **error) and
                    Dn[-1] == pytest.approx(dn, **error)
                    )
        except:
            data.close()
            raise AssertionError
    data.close()
