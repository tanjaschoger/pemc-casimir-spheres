import os
import sys
import numpy as np
from hypothesis import given, settings
from hypothesis.strategies import floats, integers
from scipy.special import lpmn
import pytest

sys.path.insert(0, os.path.abspath('../src/'))
os.environ["NUMBA_DISABLE_JIT"] = "1"
from angular_func import angularj
from bessel import I0, In


def scipy_angular(jmax, z):
    lp1, dlp1 = lpmn(1, jmax, np.cosh(z))
    pi_j = lp1[1][-1]/np.sqrt(np.cosh(z)**2 - 1)
    tau_j = - np.cosh(z)*pi_j + jmax*(jmax+1)*lp1[0][-1]
    return pi_j, tau_j


def asym_angularj(j, z):
    lamb = j+0.5
    pi_j = (j*(j+1)*np.sqrt(z/np.sinh(z)**3)/lamb
            * (In(1, lamb*z, scaling=True)
               + 3*(1-z/np.tanh(z))*In(2, lamb*z, scaling=True)/(8*lamb*z)))

    tau_j = (j*(j+1)*np.sqrt(z/np.sinh(z))
             * (I0(lamb*z, scaling=True)
                - (1+7*z/np.tanh(z))*In(1, lamb*z, scaling=True)/(8*lamb*z)))
    return pi_j, tau_j


@settings(max_examples=50, deadline=2000)
@given(
    jmax=integers(min_value=1, max_value=1000),
    x=floats(min_value=1e-3, max_value=1e-1)
)
def test_angular(jmax, x):
    error = {'rel': 1e-10,
             'abs': 1e-10}

    pi_j, tau_j = angularj(jmax, x)
    scipy_pi_j, scipy_tau_j = scipy_angular(jmax, x)
    assert (pi_j == pytest.approx(scipy_pi_j, **error) and
            tau_j == pytest.approx(scipy_tau_j, **error))


@settings(max_examples=50, deadline=2000)
@given(
    jmax=integers(min_value=1000, max_value=5000),
    x=floats(min_value=1e-2, max_value=1e2)
)
def test_asym_angular(jmax, x):
    error = {'rel': 1e-05,
             'abs': 1e-05}

    pi_j, tau_j = angularj(jmax, x, scaling=True)
    asym_pi_j, asym_tau_j = asym_angularj(jmax, x)
    assert (pi_j == pytest.approx(asym_pi_j, **error) and
            tau_j == pytest.approx(asym_tau_j, **error))
