"""
tests/test_core.py - Unit tests for Prototype 1 (pure T2 decay).
=================================================================

Run with:  pytest tests/ -v
"""

import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.core import time_axis, simple_T2_decay, simulate_simple_coherence


# ---------------------------------------------------------------------------
# time_axis
# ---------------------------------------------------------------------------

class TestTimeAxis:
    def test_starts_at_zero(self):
        t = time_axis(10.0, 1.0)
        assert t[0] == 0.0

    def test_ends_at_t_max(self):
        t = time_axis(10.0, 1.0)
        assert np.isclose(t[-1], 10.0)

    def test_step_size(self):
        t = time_axis(5.0, 0.5)
        diffs = np.diff(t)
        assert np.allclose(diffs, 0.5)

    def test_fine_grid(self):
        t = time_axis(1.0, 0.01)
        assert len(t) == 101

    def test_invalid_t_max(self):
        with pytest.raises(ValueError):
            time_axis(-1.0, 0.1)

    def test_invalid_dt(self):
        with pytest.raises(ValueError):
            time_axis(10.0, -0.1)


# ---------------------------------------------------------------------------
# simple_T2_decay
# ---------------------------------------------------------------------------

class TestSimpleT2Decay:
    def test_initial_coherence_is_one(self):
        """L(0) must equal exactly 1."""
        t = np.array([0.0, 1.0, 2.0])
        L = simple_T2_decay(t, T2=5.0)
        assert L[0] == 1.0

    def test_at_T2_is_one_over_e(self):
        """L(T2) = exp(-1) = 1/e."""
        T2 = 7.3
        t = np.array([0.0, T2])
        L = simple_T2_decay(t, T2=T2)
        assert np.isclose(L[1], 1.0 / np.e, rtol=1e-10)

    def test_monotonically_decreasing(self):
        t = time_axis(20.0, 0.1)
        L = simple_T2_decay(t, T2=5.0)
        assert np.all(np.diff(L) < 0)

    def test_approaches_zero(self):
        """After many T2 periods the coherence is negligibly small."""
        t = np.array([50.0])
        L = simple_T2_decay(t, T2=5.0)  # t = 10 x T2
        assert L[0] < 1e-4

    def test_output_range(self):
        t = time_axis(30.0, 0.1)
        L = simple_T2_decay(t, T2=5.0)
        assert np.all(L >= 0) and np.all(L <= 1)

    def test_invalid_T2(self):
        with pytest.raises(ValueError):
            simple_T2_decay(np.array([0.0, 1.0]), T2=0.0)

    def test_array_and_scalar_consistency(self):
        """Scalar input should give the same value as array input."""
        T2 = 4.0
        L_arr = simple_T2_decay(np.array([2.0]), T2=T2)
        L_scalar = np.exp(-2.0 / T2)
        assert np.isclose(L_arr[0], L_scalar)


# ---------------------------------------------------------------------------
# simulate_simple_coherence
# ---------------------------------------------------------------------------

class TestSimulateSimpleCoherence:
    def test_returns_two_arrays(self):
        result = simulate_simple_coherence(T2=5.0, t_max=20.0, dt=0.1)
        assert len(result) == 2

    def test_same_length(self):
        t, L = simulate_simple_coherence(T2=5.0, t_max=20.0, dt=0.1)
        assert len(t) == len(L)

    def test_L0_equals_one(self):
        _, L = simulate_simple_coherence(T2=5.0, t_max=20.0, dt=0.1)
        assert L[0] == 1.0

    def test_L_at_T2(self):
        T2 = 5.0
        t, L = simulate_simple_coherence(T2=T2, t_max=20.0, dt=0.5)
        idx = np.argmin(np.abs(t - T2))
        assert np.isclose(L[idx], 1.0 / np.e, rtol=1e-6)

    def test_time_starts_at_zero(self):
        t, _ = simulate_simple_coherence(T2=5.0, t_max=10.0, dt=0.1)
        assert t[0] == 0.0

    def test_time_ends_near_t_max(self):
        t, _ = simulate_simple_coherence(T2=5.0, t_max=10.0, dt=0.1)
        assert np.isclose(t[-1], 10.0)