import os
import sys
import numpy as np
from hypothesis import given, settings
from hypothesis.strategies import floats, booleans, composite
import pytest

sys.path.insert(0, os.path.abspath('../src/'))
os.environ["NUMBA_DISABLE_JIT"] = "1"

import mie_coeff as mie
import bessel as bsl


@composite
def complex_arg(draw):
    x = draw(floats(min_value=1e-2, max_value=1e2))
    y = draw(floats(min_value=1e-4, max_value=1e-3))
    return x + 1j*y


@settings(max_examples=20, deadline=2000)
@given(N=floats(min_value=1e-3, max_value=2),
       x=floats(min_value=1e-2, max_value=1e2))
def test_die(N, x):
    """
    compares the zero-order Mie coefficients for a dielectric sphere
    """
    error = {'rel': 1e-07,
             'abs': 1e-07}

    a0, b0 = mie.die_n0(N, x)
    an, bn = mie.mie_die(N, x, 1)
    assert (a0 == pytest.approx(an[0], **error) and
            b0 == pytest.approx(bn[0], **error))


@settings(max_examples=20, deadline=2000)
@given(x=floats(min_value=1e-2, max_value=1e2))
def test_pec(x):
    """
    compares the zero-order Mie coefficients for a pec sphere
    """
    error = {'rel': 1e-07,
             'abs': 1e-07}

    a0, b0 = mie.pec_n0(x)
    an, bn = mie.mie_pec(x, 1)
    assert (a0 == pytest.approx(an[0], **error) and
            b0 == pytest.approx(bn[0], **error))


@settings(max_examples=20, deadline=2000)
@given(theta=floats(min_value=0, max_value=3.14/2),
       x=floats(min_value=1e-2, max_value=1e2))
def test_pemc(theta, x):
    """
    compares the zero-order Mie coefficients for a pec sphere
    """
    error = {'rel': 1e-07,
             'abs': 1e-07}

    a0, b0, c0, d0 = mie.pemc_n0(theta, x)
    an, bn, cn, dn = mie.mie_pemc(theta, x, 1)
    assert (a0 == pytest.approx(an[0], **error) and
            b0 == pytest.approx(bn[0], **error) and
            c0 == pytest.approx(cn[0], **error) and
            d0 == pytest.approx(dn[0], **error)
            )


@settings(max_examples=20, deadline=2000)
@given(mL=complex_arg(),
       mm=complex_arg(),
       x=floats(min_value=1e-2, max_value=1e2))
def test_biisotropic(mL, mm, x):
    """
    compares the zero-order Mie coefficients for a biisotropic sphere
    """
    error = {'rel': 1e-07,
             'abs': 1e-07}

    a0, b0, c0, d0 = mie.biisotropic_n0(mL, mm, x)
    an, bn, cn, dn = mie.mie_biisotropic(mL, mm, x, 1)
    assert (a0 == pytest.approx(an[0], **error) and
            b0 == pytest.approx(bn[0], **error) and
            c0 == pytest.approx(cn[0], **error) and
            d0 == pytest.approx(dn[0], **error)
            )


@settings(max_examples=20, deadline=2000)
@given(N=floats(min_value=1e-3, max_value=2),
       x=floats(min_value=1e-2, max_value=1e2),
       scaling=booleans())
def test_die_biisotropic(N, x, scaling):
    """
    compares the Mie coefficients of a dielectric and biisotropic sphere up to
    order nmax
    """
    error = {'rel': 1e-07,
             'abs': 1e-07}

    mL = N
    mm = N
    nmax = bsl.n_max(x)
    an_die, bn_die = mie.mie_die(N, x, nmax, scaling)
    an, bn, cn, dn = mie.mie_biisotropic(mL, mm, x, nmax, scaling)
    assert (an_die == pytest.approx(an, **error) and
            bn_die == pytest.approx(bn, **error) and
            np.zeros(len(cn)) == pytest.approx(cn, **error) and
            np.zeros(len(dn)) == pytest.approx(dn, **error)
            )


@settings(max_examples=20, deadline=2000)
@given(theta=floats(min_value=0.1, max_value=3.14/2),
       x=floats(min_value=1e-2, max_value=1e2),
       scaling=booleans())
def test_pemc_biisotropic(theta, x, scaling):
    """
    compares the Mie coefficients of a pemc and biisotropic sphere up to
    order nmax
    """
    error = {'rel': 1e-07,
             'abs': 1e-07}

    mm = 1j*np.cos(theta)/np.sin(theta)
    mL = 1
    nmax = bsl.n_max(x)
    an_pemc, bn_pemc, cn_pemc, dn_pemc = mie.mie_pemc(theta, x, nmax, scaling)
    an, bn, cn, dn = mie.mie_biisotropic(mL, mm, x, nmax, scaling)
    assert (an_pemc == pytest.approx(an, **error) and
            bn_pemc == pytest.approx(bn, **error) and
            cn_pemc == pytest.approx(cn, **error) and
            dn_pemc == pytest.approx(dn, **error)
            )


@settings(max_examples=20, deadline=2000)
@given(N=floats(min_value=1e-3, max_value=2),
       x=floats(min_value=1e-5, max_value=1e-1))
def test_die_lowfrequency(N, x):
    """
    compares the low-frequency Mie coefficients of a dielectric sphere
    """
    error = {'abs': 1e-03}

    nmax = bsl.n_max(x)
    an, bn = mie.mie_die(N, x, nmax)
    an_x0, bn_x0 = mie.mie_die_x0(N, x, nmax)
    assert (an[1:] == pytest.approx(an_x0, **error) and
            bn[1:] == pytest.approx(bn_x0, **error))


@settings(max_examples=20, deadline=2000)
@given(theta=floats(min_value=0.1, max_value=3.14/2),
       x=floats(min_value=1e-5, max_value=1e-1))
def test_pemc_lowfrequency(theta, x):
    """
    compares the low-frequency Mie coefficients of a pemc sphere
    """
    error = {'abs': 1e-03}

    nmax = bsl.n_max(x)
    an, bn, cn, dn = mie.mie_pemc(theta, x, nmax)
    an_x0, bn_x0, cn_x0, dn_x0 = mie.mie_pemc_x0(theta, x, nmax)
    assert (an[1:] == pytest.approx(an_x0, **error) and
            bn[1:] == pytest.approx(bn_x0, **error) and
            cn[1:] == pytest.approx(cn_x0, **error) and
            dn[1:] == pytest.approx(dn_x0, **error)
            )


@settings(max_examples=20, deadline=2000)
@given(mL=complex_arg(),
       mm=complex_arg(),
       x=floats(min_value=1e-5, max_value=1e-1))
def test_biisotropic_lowfrequency(mL, mm, x):
    """
    compares the low-frequency Mie coefficients of a biisotropic sphere
    """
    error = {'abs': 1e-03}

    nmax = bsl.n_max(x)
    an, bn, cn, dn = mie.mie_biisotropic(mL, mm, x, nmax)
    an_x0, bn_x0, cn_x0, dn_x0 = mie.mie_biisotropic_x0(mL, mm, x, nmax)
    assert (an[1:] == pytest.approx(an_x0,  **error) and
            bn[1:] == pytest.approx(bn_x0,  **error) and
            cn[1:] == pytest.approx(cn_x0,  **error) and
            dn[1:] == pytest.approx(dn_x0,  **error)
            )


@settings(max_examples=20, deadline=2000)
@given(N=floats(min_value=1e-3, max_value=2),
       x=floats(min_value=500, max_value=2000))
def test_die_wkb(N, x):
    """
    compares the WKB-approximation of the Mie coefficients of a dielectric
    sphere
    """
    error = {'rel': 1e-01}

    nmax = bsl.n_max(x)
    an, bn = mie.mie_die(N, x, nmax, scaling=True)
    an_wkb, bn_wkb = mie.wkb_die(N, x, nmax, scaling=True)
    assert (an == pytest.approx(an_wkb, **error) and
            bn == pytest.approx(bn_wkb, **error))


@settings(max_examples=20, deadline=2000)
@given(theta=floats(min_value=0.1, max_value=3.14/2),
       x=floats(min_value=500, max_value=2000))
def test_pemc_wkb(theta, x):
    """
    compares the WKB-approximation of the Mie coefficients of a pemc sphere
    """
    error = {'rel': 1e-01}

    nmax = bsl.n_max(x)
    an, bn, cn, dn = mie.mie_pemc(theta, x, nmax, scaling=True)
    an_wkb, bn_wkb, cn_wkb, dn_wkb = mie.wkb_pemc(theta, x, nmax, scaling=True)
    assert (an == pytest.approx(an_wkb, **error) and
            bn == pytest.approx(bn_wkb, **error) and
            cn == pytest.approx(cn_wkb, **error) and
            dn == pytest.approx(dn_wkb, **error)
            )
