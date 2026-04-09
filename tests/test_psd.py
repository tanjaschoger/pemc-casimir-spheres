import os
import sys
from hypothesis import given, settings
from hypothesis.strategies import floats, integers
import pytest

sys.path.insert(0, os.path.abspath('../src/'))
os.environ["NUMBA_DISABLE_JIT"] = "1"

import psd


@settings(max_examples=20, deadline=2000)
@given(n=integers(min_value=1, max_value=500),
       z=floats(min_value=1e-1, max_value=1e2))
def test_psd(n, z):
    assert psd.bose_func1(n, z) == pytest.approx(psd.bose_func2(n, z),
                                                 rel=1e-10, abs=1e-10)
