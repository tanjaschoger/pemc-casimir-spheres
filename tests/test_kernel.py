from math import pi
import os
import sys
from hypothesis import given, settings
from hypothesis.strategies import floats, builds
import pytest

sys.path.insert(0, os.path.abspath('../src/'))

import reflection_kernel as rflk


def target(RbyL, Y, y1, Phi1, y2, Phi2):
    vec1 = y1, Phi1
    vec2 = y2, Phi2
    return RbyL, Y, vec1, vec2


Strategy_lowfreq = builds(target,
                          RbyL=floats(min_value=1e-2, max_value=1e4),
                          Y=floats(min_value=1e-11, max_value=1e-05),
                          y1=floats(min_value=1e-4, max_value=1e4),
                          Phi1=floats(min_value=0, max_value=1),
                          y2=floats(min_value=1e-4, max_value=1e4),
                          Phi2=floats(min_value=0, max_value=1)
                          )

Strategy_wkb = builds(target,
                      RbyL=floats(min_value=5e2, max_value=1e4),
                      Y=floats(min_value=1e-1, max_value=1e1),
                      y1=floats(min_value=1e-4, max_value=1e4),
                      Phi1=floats(min_value=0, max_value=1),
                      y2=floats(min_value=1e-4, max_value=1e4),
                      Phi2=floats(min_value=0, max_value=1)
                      )


error = {'rel': 1e-2,
         'abs': 1e-6}


@settings(max_examples=10, deadline=None)
@given(param=Strategy_lowfreq,
       N=floats(min_value=1, max_value=10))
def test_lowfreq_die(param, N):
    """
    compares the reflection kernel functions of a dielectric sphere with the
    low-frequency results

    """
    RbyL, Y, vec1, vec2 = param
    kernel = rflk.kernelR_die(*param, sgn=1, N=N)
    lowfreq_kernel = rflk.kernelRlowfreq_die(RbyL, vec1, vec2, N=N)
    assert (kernel[0] == pytest.approx(lowfreq_kernel[0], **error) and
            kernel[1] == pytest.approx(lowfreq_kernel[1], **error))


@settings(max_examples=10, deadline=None)
@given(param=Strategy_lowfreq,
       theta=floats(min_value=0, max_value=pi/2))
def test_lowfreq_pemc(param, theta):
    """
    compares the reflection kernel functions of a PEMC sphere with the
    low-frequency results

    """
    RbyL, Y, vec1, vec2 = param
    kernel = rflk.kernelR_pemc(*param, sgn=1, theta=theta)
    lowfreq_kernel = rflk.kernelRlowfreq_pemc(RbyL, vec1, vec2, theta=theta)
    assert (kernel[0] == pytest.approx(lowfreq_kernel[0], **error) and
            kernel[1] == pytest.approx(lowfreq_kernel[1], **error) and
            kernel[2] == pytest.approx(lowfreq_kernel[2], **error) and
            kernel[3] == pytest.approx(lowfreq_kernel[3], **error)
            )


@settings(max_examples=10, deadline=None)
@given(param=Strategy_lowfreq,
       mL=floats(min_value=1, max_value=2),
       mm=floats(min_value=1, max_value=2))
def test_lowfreq_biisotropic(param, mL, mm):
    """
    compares the reflection kernel functions of a bi-isotropic sphere with the
    low-frequency results

    """
    RbyL, Y, vec1, vec2 = param
    kernel = rflk.kernelR_biisotropic(*param, sgn=1, mm=mm, mL=mL)
    lowfreq_kernel = rflk.kernelRlowfreq_biisotropic(RbyL, vec1, vec2,
                                                     mm=mm, mL=mL)
    assert (kernel[0] == pytest.approx(lowfreq_kernel[0], **error) and
            kernel[1] == pytest.approx(lowfreq_kernel[1], **error) and
            kernel[2] == pytest.approx(lowfreq_kernel[2], **error) and
            kernel[3] == pytest.approx(lowfreq_kernel[3], **error)
            )


@settings(max_examples=10, deadline=None)
@given(param=Strategy_wkb,
       N=floats(min_value=1, max_value=10))
def test_wkb_die(param, N):
    """
    compares the reflection kernel functions of a dielectic sphere with the
    WKB results

    """
    kernel = rflk.kernelR_die(*param, sgn=1, N=N)
    wkb_kernel = rflk.wkb_kernelR_die(*param, sgn=1, N=N)
    assert (kernel[0] == pytest.approx(wkb_kernel[0], **error) and
            kernel[1] == pytest.approx(wkb_kernel[1], **error))


@settings(max_examples=10, deadline=None)
@given(param=Strategy_wkb,
       theta=floats(min_value=0, max_value=pi/2))
def test_wkb_pemc(param, theta):
    """
    compares the reflection kernel functions of a PEMC sphere with the
    WKB results

    """
    kernel = rflk.kernelR_pemc(*param, sgn=1, theta=theta)
    wkb_kernel = rflk.wkb_kernelR_pemc(*param, sgn=1, theta=theta)
    assert (kernel[0] == pytest.approx(wkb_kernel[0], **error) and
            kernel[1] == pytest.approx(wkb_kernel[1], **error) and
            kernel[2] == pytest.approx(wkb_kernel[2], **error) and
            kernel[3] == pytest.approx(wkb_kernel[3], **error)
            )


@settings(max_examples=10, deadline=None)
@given(param=Strategy_wkb,
       mL=floats(min_value=1, max_value=2),
       mm=floats(min_value=1, max_value=2))
def test_wkb_biisotropic(param, mL, mm):
    """
    compares the reflection kernel functions of a bi-isotropic sphere with the
    WKB results

    """
    kernel = rflk.kernelR_biisotropic(*param, sgn=1, mm=mm, mL=mL)
    wkb_kernel = rflk.wkb_kernelR_biisotropic(*param, sgn=1, mm=mm, mL=mL)
    assert (kernel[0] == pytest.approx(wkb_kernel[0], **error) and
            kernel[1] == pytest.approx(wkb_kernel[1], **error) and
            kernel[2] == pytest.approx(wkb_kernel[2], **error) and
            kernel[3] == pytest.approx(wkb_kernel[3], **error)
            )
