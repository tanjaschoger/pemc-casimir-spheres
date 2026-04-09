import os
import sys
import numpy as np
from hypothesis import given, settings
from hypothesis.strategies import floats, builds
import pytest

sys.path.insert(0, os.path.abspath('../src/'))
os.environ["NUMBA_DISABLE_JIT"] = "1"

import fresnel


def target(Y, y1, Phi1, y2, Phi2):
    vec1 = y1, Phi1
    vec2 = y2, Phi2
    return Y, vec1, vec2


Strategy = builds(target,
                  Y=floats(min_value=1e-5, max_value=1e5),
                  y1=floats(min_value=1e-5, max_value=1e5),
                  y2=floats(min_value=1e-5, max_value=1e5),
                  Phi1=floats(min_value=0, max_value=1),
                  Phi2=floats(min_value=0, max_value=1)
                  )


@settings(max_examples=20, deadline=2000)
@given(param=Strategy)
def test_pec_die(param):
    """
    compares the Fresnel coefficients for a dielectric material with the
    perfect reflector limit

    """
    error = {'rel': 1e-10,
             'abs': 1e-10}

    Y, vec1, vec2 = param
    N = 1e15
    rtmtm, rtete, rtetm, rtmte = fresnel.fresnel_die(Y, vec1, vec2, N)
    assert (rtmtm == pytest.approx(1, **error) and
            rtete == pytest.approx(-1, **error))


@settings(max_examples=20, deadline=2000)
@given(param=Strategy)
def test_pec_biisotropic(param):
    """
    compares the Fresnel coefficients for a bi-isotropic material with the
    perfect reflector limit

    """
    error = {'rel': 1e-10,
             'abs': 1e-10}

    Y, vec1, vec2 = param
    mL = 1e15
    mm = mL
    rtmtm, rtete, rtetm, rtmte = fresnel.fresnel_biisotropic(Y, vec1, vec2,
                                                             mL, mm)
    assert (rtmtm == pytest.approx(1, **error) and
            rtete == pytest.approx(-1, **error) and
            rtetm == pytest.approx(0, **error) and
            rtmte == pytest.approx(0, **error)
            )


@settings(max_examples=20, deadline=2000)
@given(param=Strategy,
       theta=floats(min_value=0.1, max_value=3.14/2)
       )
def test_pemc_biisotropic(param, theta):
    """
    compares the Fresnel coefficients for a bi-isotropic material with the
    perfect electromagnetic conductor limit

    """
    error = {'rel': 1e-15,
             'abs': 1e-15}

    Y, vec1, vec2 = param
    mm = 1j*np.cos(theta)/np.sin(theta)
    mL = 1
    rtmtm, rtete, rtetm, rtmte = fresnel.fresnel_biisotropic(Y, vec1, vec2,
                                                             mL, mm)
    ptmtm, ptete, ptetm, ptmte = fresnel.fresnel_pemc(theta)
    assert (rtmtm == pytest.approx(ptmtm, **error) and
            rtete == pytest.approx(ptete, **error) and
            rtetm == pytest.approx(ptetm, **error) and
            rtmte == pytest.approx(ptmte, **error)
            )


@settings(max_examples=20, deadline=2000)
@given(param=Strategy,
       N=floats(min_value=1, max_value=5)
       )
def test_die_biisotropic(param, N):
    """
    compares the Fresnel coefficients for a bi-isotropic material with the
    perfect electromagnetic conductor limit

    """
    error = {'rel': 1e-15,
             'abs': 1e-15}

    Y, vec1, vec2 = param
    mm = N
    mL = N
    rtmtm, rtete, rtetm, rtmte = fresnel.fresnel_biisotropic(Y, vec1, vec2,
                                                             mL, mm)
    ptmtm, ptete, ptetm, ptmte = fresnel.fresnel_die(Y, vec1, vec2, N)
    assert (rtmtm == pytest.approx(ptmtm, **error) and
            rtete == pytest.approx(ptete, **error) and
            rtetm == pytest.approx(ptetm, **error) and
            rtmte == pytest.approx(ptmte, **error)
            )
