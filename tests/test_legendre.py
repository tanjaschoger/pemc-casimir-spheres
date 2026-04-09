import os
import sys
import numpy as np
from hypothesis import given, settings
from hypothesis.strategies import floats, integers, builds
from scipy.special import eval_legendre, lpmn, i0e, i1e
import pytest

sys.path.insert(0, os.path.abspath('../src/'))
os.environ["NUMBA_DISABLE_JIT"] = "1"
from legendre import legendre_pl, legendre_pl1, ratio_10


def target(j, x):
    return j, x


Strategy = builds(target,
                  j=integers(min_value=1001, max_value=5000),
                  x=floats(min_value=1e-4, max_value=1e-1)
                  )


@settings(max_examples=10, deadline=None)
@given(param=Strategy)
def test_legendre(param):
    j, x = param
    assert (eval_legendre(j, np.cosh(x))
            == pytest.approx(legendre_pl(j, x)*np.exp((j+0.5)*x),
                             rel=1e-07, abs=1e-07))


@settings(max_examples=10, deadline=None)
@given(param=Strategy)
def test_ratio(param):
    """
    compares the ratio of the Legendre poylnomials with the asymptotic
    expansion for large orders j
    """
    j, x = param
    lamb = j+0.5
    asym = j*(j+1)*i1e(lamb*x)/(lamb*i0e(lamb*x))
    assert ratio_10(j, x) == pytest.approx(asym, rel=1e-03)


@settings(max_examples=10, deadline=None)
@given(param=Strategy)
def test_recursion_pl(param):
    """
    tests whether the recursion in j is fulfilled (see Eq. 14.10.3 in Ref. [1])

    Reference
    ---------
    ..[1] https://dlmf.nist.gov/14.10
    """
    j, x = param
    lpj = legendre_pl(j, x)
    lpjp1 = legendre_pl(j+1, x)*np.exp(x)
    lpjp2 = legendre_pl(j+2, x)*np.exp(2*x)
    eps = (j+2)*lpjp2 - (2*j+3)*np.cosh(x)*lpjp1 + (j+1)*lpj
    assert (eps == pytest.approx(0.0,  rel=1e-04, abs=1e-04))


@settings(max_examples=10, deadline=None)
@given(param=Strategy)
def test_legendre1(param):
    j, x = param
    lp1, dlp1 = lpmn(1, j, np.cosh(x))
    assert (lp1[1][-1] == pytest.approx(legendre_pl1(j, x)*np.exp((j+0.5)*x),
                                        rel=1e-07, abs=1e-07))


@settings(max_examples=10, deadline=None)
@given(param=Strategy)
def test_recursion_pl1(param):
    """
    tests whether the recursion in j is fulfilled (see Eq. 14.10.3 in Ref. [1])

    Reference
    ---------
    ..[1] https://dlmf.nist.gov/14.10
    """
    j, x = param
    lpj = legendre_pl1(j, x)
    lpjp1 = legendre_pl1(j+1, x)*np.exp(x)
    lpjp2 = legendre_pl1(j+2, x)*np.exp(2*x)
    eps = (j+1)*lpjp2 - (2*j+3)*np.cosh(x)*lpjp1 + (j+2)*lpj
    assert (eps == pytest.approx(0.0,  rel=1e-04, abs=1e-04))
