import os
import sys
from hypothesis import given, settings
from hypothesis.strategies import floats, builds
import pytest
from math import ceil, pi
import numpy as np

sys.path.insert(0, os.path.abspath('../src/'))

import scattering_amplitudes as scam
import mie_coeff as mie


def target(RbyL, Y, y1, Phi1, y2, Phi2):
    vec1 = y1, Phi1
    vec2 = y2, Phi2
    return RbyL, Y, vec1, vec2


Strategy = builds(target,
                  RbyL=floats(min_value=5e2, max_value=1e4),
                  Y=floats(min_value=1e-1, max_value=1e1),
                  y1=floats(min_value=1e-4, max_value=1e4),
                  Phi1=floats(min_value=0, max_value=1),
                  y2=floats(min_value=1e-4, max_value=1e4),
                  Phi2=floats(min_value=0, max_value=1)
                  )

error = {'rel': 1e-3,
         'abs': 1e-6}

@settings(max_examples=20, deadline=None)
@given(param=Strategy)
def test_scatampl_pec(param):

    RbyL, Y, vec1, vec2 = param
    nmax = 12*ceil(RbyL)
    x = RbyL*Y
    refl_coeff = mie.mie_pec(x=x, nmax=nmax, scaling=True)
    an, bn = mie.add_prefac(refl_coeff, nmax)

    stete = scam.ScatteringAmplitude(RbyL, Y, an, bn).evalf(vec1, vec2)
    stmtm = scam.ScatteringAmplitude(RbyL, Y, bn, an).evalf(vec1, vec2)
    wkb_ampl = scam.amplitude_pec(RbyL, Y, vec1, vec2)

    assert (stmtm == pytest.approx(wkb_ampl[0], **error) and
            stete == pytest.approx(wkb_ampl[1], **error))


@settings(max_examples=20, deadline=None)
@given(param=Strategy,
       theta=floats(min_value=0, max_value=pi/2)
       )
def test_scatampl_pemc(param, theta):

    RbyL, Y, vec1, vec2 = param
    nmax = 12*ceil(RbyL)
    x = RbyL*Y
    refl_coeff = mie.mie_pemc(theta=theta, x=x, nmax=nmax, scaling=True)
    an, bn, cn, dn = mie.add_prefac(refl_coeff, nmax)

    stete = scam.ScatteringAmplitude(RbyL, Y, an, bn).evalf(vec1, vec2)
    stmtm = scam.ScatteringAmplitude(RbyL, Y, bn, an).evalf(vec1, vec2)
    stmte = scam.ScatteringAmplitude(RbyL, Y, -cn, dn).evalf(vec1, vec2)
    stetm = scam.ScatteringAmplitude(RbyL, Y, -dn, cn).evalf(vec1, vec2)
    wkb_ampl = scam.amplitude_pemc(RbyL, Y, vec1, vec2, theta)

    assert (stmtm == pytest.approx(wkb_ampl[0], **error) and
            stete == pytest.approx(wkb_ampl[1], **error) and
            stetm == pytest.approx(wkb_ampl[2], **error) and
            stmte == pytest.approx(wkb_ampl[3], **error)
            )


@settings(max_examples=20, deadline=None)
@given(param=Strategy,
       mL=floats(min_value=1, max_value=2),
       mm=floats(min_value=1, max_value=2)
       )
def test_scatampl_biisotropic(param, mL, mm):

    RbyL, Y, vec1, vec2 = param
    nmax = 12*ceil(RbyL)
    x = RbyL*Y
    refl_coeff = mie.mie_biisotropic(mL=mL, mm=mm, x=x, nmax=nmax, scaling=True)
    an, bn, cn, dn = mie.add_prefac(refl_coeff, nmax)

    stete = scam.ScatteringAmplitude(RbyL, Y, an, bn).evalf(vec1, vec2)
    stmtm = scam.ScatteringAmplitude(RbyL, Y, bn, an).evalf(vec1, vec2)
    stmte = scam.ScatteringAmplitude(RbyL, Y, -cn, dn).evalf(vec1, vec2)
    stetm = scam.ScatteringAmplitude(RbyL, Y, -dn, cn).evalf(vec1, vec2)
    wkb_ampl = scam.amplitude_biisotropic(RbyL, Y, vec1, vec2, mL, mm)

    assert (stmtm == pytest.approx(wkb_ampl[0], **error) and
            stete == pytest.approx(wkb_ampl[1], **error) and
            stetm == pytest.approx(wkb_ampl[2], **error) and
            stmte == pytest.approx(wkb_ampl[3], **error)
            )
