import os
import sys
import numpy as np
from hypothesis import given, settings
from hypothesis.strategies import floats, composite, integers
from scipy.special import iv, i0e, kv, gammaln
import pytest

sys.path.insert(0, os.path.abspath('../src/'))
os.environ["NUMBA_DISABLE_JIT"] = "1"

import bessel as bsl


@composite
def complex_arg(draw):
    x = draw(floats(min_value=1e-2, max_value=1e2))
    y = draw(floats(min_value=1e-4, max_value=1e-3))
    return x+1j*y


def scipy_bessel(nmax, z):
    knu = []
    inu = []
    for n in range(nmax+1):
        knu.append(kv(n+1/2, z))
        inu.append(iv(n+1/2, z))
    return inu, knu


def normalization(z, eps0):
    """
    calculates the sum over modified Bessel functions of even order
    (see Eq. (10.35.4) in Ref. [1])

    Reference
    ---------
    ..[1] https://dlmf.nist.gov/10.35
    """
    bessel_sum = bsl.In(0, z, scaling=True)
    eps = 1
    n = 0
    while eps > eps0:
        n += 1
        term = (-1)**n * 2*bsl.In(2*n, z, scaling=True)
        bessel_sum += term
        eps = abs(term/bessel_sum)
    return bessel_sum


def besselSeries(z, eps0):
    """
    calculates the sums over modified Bessel functions of even and odd order
    (see Eq. (10.35.6) in Ref. [1])

    Reference
    ---------
    ..[1] https://dlmf.nist.gov/10.35
    """
    besselSum_even = bsl.In(0, z, scaling=True)
    besselSum_odd = 2*bsl.In(1, z, scaling=True)
    eps = 1
    n = 0
    while eps > eps0:
        n += 1
        evenTerm = 2*bsl.In(2*n, z, scaling=True)
        besselSum_even += evenTerm
        eps = abs(evenTerm/besselSum_even)
    eps = 1
    n = 0
    while eps > eps0:
        n += 1
        oddTerm = 2*bsl.In(2*n+1, z, scaling=True)
        besselSum_odd += oddTerm
        eps = abs(oddTerm/besselSum_odd)
    return besselSum_even, besselSum_odd


@settings(max_examples=50, deadline=2000)
@given(z=complex_arg())
def test_bessel(z):
    error = {'rel': 1e-07,
             'abs': 1e-07}

    nu_max = bsl.n_max(z)
    knu = bsl.modbesselK(nu_max, z)
    inu = bsl.modbesselI(nu_max, z)
    scipy_inu, scipy_knu = scipy_bessel(nu_max, z)
    assert (scipy_knu == pytest.approx(knu, **error) and
            scipy_inu == pytest.approx(inu, **error))


@settings(max_examples=50, deadline=2000)
@given(z=complex_arg())
def test_wronskian(z):
    """
    tests if the Wronskian (Eq. (10.28.2) in Ref. [1]) is fulfilled

    Reference
    ---------
    ..[1] https://dlmf.nist.gov/10.28
    """
    error = {'rel': 1e-07,
             'abs': 1e-07}

    nmax = bsl.n_max(z)
    n = np.arange(nmax)

    knu = bsl.modbesselK(nmax, z)
    inu = bsl.modbesselI(nmax, z)
    wron = inu[0:nmax]*knu[1:nmax+1] + knu[0:nmax]*inu[1:nmax+1]

    sc_knu = bsl.scaled_modbesselK(nmax, z)
    sc_inu = bsl.scaled_modbesselI(nmax, z)
    sc_wron = (sc_inu[0:nmax]*sc_knu[1:nmax+1]/z
               + sc_inu[1:nmax+1]*sc_knu[0:nmax]*z/((2*n+3)*(2*n+1)))

    sc2_knu = bsl.scaled2_modbesselK(nmax, z)
    sc2_inu = bsl.scaled2_modbesselI(nmax, z)
    scaling = np.exp(z*(bsl.arg(n, z)-bsl.arg(n+1, z)))
    sc2_wron = (sc2_inu[0:nmax]*sc2_knu[1:nmax+1]*scaling
                + sc2_inu[1:nmax+1]*sc2_knu[0:nmax]/scaling)
    assert (np.ones(nmax)/z == pytest.approx(wron, **error) and
            np.ones(nmax)/z == pytest.approx(sc_wron, **error) and
            np.ones(nmax)/z == pytest.approx(sc2_wron, **error)
            )


@settings(max_examples=50, deadline=2000)
@given(z=complex_arg(),
       n=integers(min_value=500, max_value=5000))
def test_ratio(n, z):
    asym = np.exp(z*(bsl.arg(n-1, z)-bsl.arg(n, z)))
    assert (bsl.ratio(n, z) == pytest.approx(asym, rel=1e-03, abs=1e-03))


@settings(max_examples=50, deadline=2000)
@given(n=integers(min_value=1, max_value=1000),
       z=floats(min_value=1e-1, max_value=1e2))
def test_bessel_in(n, z):
    assert bsl.In(n, z) == pytest.approx(iv(n, z), rel=1e-06, abs=1e-06)


@settings(max_examples=50, deadline=2000)
@given(z=floats(min_value=1e-4, max_value=1e4))
def test_i0(z):
    assert bsl.I0(z, scaling=True) == pytest.approx(i0e(z),
                                                    rel=1e-15, abs=1e-15)


@settings(max_examples=50, deadline=2000)
@given(n=integers(min_value=1, max_value=1000),
       z=floats(min_value=1e-5, max_value=1e-1))
def test_small_arg(n, z):
    """
    compares the modified Bessel functions for small arguments with the
    limiting form given in Eq. (10.30.1) in Ref. [1]

    Reference
    ---------
    ..[1] https://dlmf.nist.gov/10.30
    """
    approx_in = (0.5*z)**n * np.exp(-gammaln(n+1))
    assert bsl.In(n, z) == pytest.approx(approx_in, rel=1e-03, abs=1e-03)


@settings(max_examples=50, deadline=2000)
@given(z=floats(min_value=1e-4, max_value=1e3))
def test_normalization(z):
    """
    See Also
    --------
    normalization
    """
    eps0 = 1e-10
    assert normalization(z, eps0) == pytest.approx(np.exp(-z),
                                                   rel=1e-08, abs=1e-08)


@settings(max_examples=50, deadline=2000)
@given(z=floats(min_value=1e-4, max_value=1e3))
def test_series(z):
    """
    See Also
    --------
    besselSeries
    """
    error = {'rel': 1e-08,
             'abs': 1e-08}

    eps0 = 1e-10
    besselSum_even, besselSum_odd = besselSeries(z, eps0)
    assert (besselSum_even == pytest.approx(0.5*(1+np.exp(-2*z)), **error) and
            besselSum_odd == pytest.approx(0.5*(1-np.exp(-2*z)), **error)
            )
