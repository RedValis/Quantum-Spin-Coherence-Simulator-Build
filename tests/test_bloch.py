"""
tests/test_bloch_full.py - Unit tests for Prototype 3 (full Bloch equations).
==============================================================================

Physics invariants under test:

  A. bloch_rhs correctness  - cross product, decay terms, analytic spot checks
  B. Equilibrium            - M_init = [0, 0, M0] never moves
  C. T2-only decay          - |M_perp| matches exp(-t/T2)
  D. T1 recovery            - Mz(t) matches M0*(1 - exp(-t/T1)) from Mz0=0
  E. Limiting cases         - T1>>T2, T2>>T1 (capped at T1), T1=T2
  F. Long-time behaviour    - Mz → M0, |M_perp| → 0
  G. Consistency with P2    - simulate_bloch matches bloch_precession when T1→large
  H. Input validation       - bad T1, T2, T2>T1, bad t_max/dt all raise

Run with:  pytest tests/test_bloch_full.py -v
           OR:  python tests/test_bloch_full.py
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core import (
    time_axis, bloch_rhs, simulate_bloch, bloch_precession
)

results = []

def check(name, condition):
    ok = bool(condition)
    results.append(ok)
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}  {name}")
    return ok


# -- shared parameters --------------------------------------------------------
gamma  = 1.0               # normalised gyromagnetic ratio
B0     = 2 * np.pi * 0.5  # rad/µs  →  omega0 = 2pi*0.5 MHz
B      = np.array([0.0, 0.0, B0])
M0     = 1.0
T1     = 20.0              # µs
T2     = 5.0               # µs  (T2 <= T1 always)
t_max  = 4 * T1            # long enough to see full recovery
dt     = 0.05


# ===========================================================================
# Group A - bloch_rhs spot checks
# ===========================================================================
print("\n-- Group A: bloch_rhs correctness -----------------------------------")

# At equilibrium [0, 0, M0]: all derivatives must be zero
dM_eq = bloch_rhs(0, np.array([0.0, 0.0, M0]), gamma, B, T1, T2, M0)
check("A1: dM/dt = 0 at equilibrium [0,0,M0]",       np.allclose(dM_eq, 0.0, atol=1e-14))

# At M = [M0, 0, 0], B = [0,0,B0]:
#   cross = [0*B0 - 0*0, 0*0 - M0*B0, M0*0 - 0*0] = [0, -M0*B0, 0]
#   dMx = gamma*0    - M0/T2 = -M0/T2
#   dMy = gamma*(-M0*B0) - 0/T2 = -gamma*M0*B0
#   dMz = gamma*0    - (0 - M0)/T1 = M0/T1
M_test = np.array([M0, 0.0, 0.0])
dM     = bloch_rhs(0, M_test, gamma, B, T1, T2, M0)
check("A2: dMx = -M0/T2 when M=[M0,0,0]",             np.isclose(dM[0], -M0/T2))
check("A3: dMy = -gamma*M0*B0 when M=[M0,0,0]",       np.isclose(dM[1], -gamma*M0*B0))
check("A4: dMz = +M0/T1 when M=[M0,0,0], Mz0=0",      np.isclose(dM[2],  M0/T1))

# With Mz already at M0: dMz = 0
M_test2 = np.array([0.0, 0.0, M0])
dM2 = bloch_rhs(0, M_test2, gamma, B, T1, T2, M0)
check("A5: dMz = 0 when Mz = M0",                     np.isclose(dM2[2], 0.0, atol=1e-14))

# RHS is time-independent (autonomous system) - same result at t=0 and t=99
dM_t0  = bloch_rhs(0,  M_test, gamma, B, T1, T2, M0)
dM_t99 = bloch_rhs(99, M_test, gamma, B, T1, T2, M0)
check("A6: RHS is autonomous (time-independent)",      np.allclose(dM_t0, dM_t99))


# ===========================================================================
# Group B - equilibrium stability
# ===========================================================================
print("\n-- Group B: Equilibrium invariant -----------------------------------")

t_b, Mx_b, My_b, Mz_b = simulate_bloch(
    M_init=[0.0, 0.0, M0], gamma=gamma, B=B,
    T1=T1, T2=T2, M0=M0, t_max=t_max, dt=dt)

check("B1: Mx stays 0 from equilibrium",   np.allclose(Mx_b, 0.0, atol=1e-8))
check("B2: My stays 0 from equilibrium",   np.allclose(My_b, 0.0, atol=1e-8))
check("B3: Mz stays M0 from equilibrium",  np.allclose(Mz_b, M0,  rtol=1e-6))


# ===========================================================================
# Group C - transverse decay matches exp(-t/T2)
# ===========================================================================
print("\n-- Group C: Transverse T2 decay --------------------------------------")

t_c, Mx_c, My_c, Mz_c = simulate_bloch(
    M_init=[M0, 0.0, 0.0], gamma=gamma, B=B,
    T1=T1, T2=T2, M0=M0, t_max=4*T2, dt=0.02)

M_perp_c = np.sqrt(Mx_c**2 + My_c**2)
expected_c = M0 * np.exp(-t_c / T2)
check("C1: |M_perp| matches M0*exp(-t/T2)",  np.allclose(M_perp_c, expected_c, rtol=1e-5))

idx_T2 = np.argmin(np.abs(t_c - T2))
check("C2: |M_perp|(T2) ≈ 1/e",             np.isclose(M_perp_c[idx_T2], 1/np.e, rtol=1e-4))
check("C3: |M_perp| monotonically decreasing", np.all(np.diff(M_perp_c) <= 0))
check("C4: |M_perp|(0) = M0",               np.isclose(M_perp_c[0], M0, rtol=1e-6))


# ===========================================================================
# Group D - longitudinal T1 recovery
# ===========================================================================
print("\n-- Group D: Longitudinal T1 recovery --------------------------------")

# Start at Mz0 = 0 (fully inverted or just tipped): expect Mz(t) = M0*(1-exp(-t/T1))
t_d, Mx_d, My_d, Mz_d = simulate_bloch(
    M_init=[0.0, 0.0, 0.0], gamma=gamma, B=np.array([0.0,0.0,0.0]),
    T1=T1, T2=T2, M0=M0, t_max=4*T1, dt=0.1)

expected_mz = M0 * (1 - np.exp(-t_d / T1))
check("D1: Mz(t) = M0*(1-exp(-t/T1)) from Mz0=0",  np.allclose(Mz_d, expected_mz, rtol=1e-5))
check("D2: Mz(0) = 0",                               np.isclose(Mz_d[0], 0.0, atol=1e-8))
check("D3: Mz monotonically increases from 0→M0",    np.all(np.diff(Mz_d) >= -1e-10))
check("D4: Mz(4·T1) > 0.98*M0",                     Mz_d[-1] > 0.98 * M0)

idx_T1 = np.argmin(np.abs(t_d - T1))
check("D5: Mz(T1) ≈ M0*(1-1/e) ≈ 0.632*M0",        np.isclose(Mz_d[idx_T1], M0*(1-1/np.e), rtol=1e-4))


# ===========================================================================
# Group E - limiting cases
# ===========================================================================
print("\n-- Group E: Limiting cases -------------------------------------------")

# E1: T1 = T2  → isotropic decay
T_iso = 5.0
t_e1, Mx_e1, My_e1, Mz_e1 = simulate_bloch(
    M_init=[M0, 0.0, 0.0], gamma=gamma, B=B,
    T1=T_iso, T2=T_iso, M0=M0, t_max=4*T_iso, dt=0.02)
M_perp_e1 = np.sqrt(Mx_e1**2 + My_e1**2)
check("E1: T1=T2 → |M_perp| decays as exp(-t/T)",
      np.allclose(M_perp_e1, M0*np.exp(-t_e1/T_iso), rtol=1e-4))

# E2: T1 >> T2  → transverse dies, Mz barely moves
T1_long = 1e6     # effectively infinite
T2_short = 2.0
t_e2, Mx_e2, My_e2, Mz_e2 = simulate_bloch(
    M_init=[M0, 0.0, 0.0], gamma=gamma, B=B,
    T1=T1_long, T2=T2_short, M0=M0, t_max=4*T2_short, dt=0.02)
M_perp_e2 = np.sqrt(Mx_e2**2 + My_e2**2)
check("E2a: T1>>T2 → transverse decays on T2 scale",
      np.allclose(M_perp_e2, M0*np.exp(-t_e2/T2_short), rtol=1e-4))
check("E2b: T1>>T2 → Mz barely changes (< 0.001)",
      np.max(np.abs(Mz_e2)) < 0.001)

# E3: very long T2 (= T1 for physical validity) → |M_perp| almost constant
T1_med = 50.0
T2_long = 50.0   # T2 = T1  (maximum allowed)
t_e3, Mx_e3, My_e3, Mz_e3 = simulate_bloch(
    M_init=[M0, 0.0, 0.0], gamma=gamma, B=B,
    T1=T1_med, T2=T2_long, M0=M0, t_max=2.0, dt=0.01)
M_perp_e3 = np.sqrt(Mx_e3**2 + My_e3**2)
check("E3: T2=T1 (max) → |M_perp| decays slowly over short window",
      M_perp_e3[-1] > 0.9 * M0)

# E4: T2 > T1 must raise
try:
    simulate_bloch([M0,0,0], gamma, B, T1=5.0, T2=10.0, M0=M0, t_max=10.0, dt=0.1)
    check("E4: raises when T2 > T1", False)
except ValueError:
    check("E4: raises when T2 > T1", True)


# ===========================================================================
# Group F - long-time (steady-state) limits
# ===========================================================================
print("\n-- Group F: Long-time convergence -----------------------------------")

t_f, Mx_f, My_f, Mz_f = simulate_bloch(
    M_init=[M0, 0.0, 0.0], gamma=gamma, B=B,
    T1=T1, T2=T2, M0=M0, t_max=10*T1, dt=0.2)

check("F1: Mz → M0 at 10·T1",    np.isclose(Mz_f[-1], M0, atol=1e-3))
check("F2: Mx → 0  at 10·T1",    np.isclose(Mx_f[-1], 0.0, atol=1e-3))
check("F3: My → 0  at 10·T1",    np.isclose(My_f[-1], 0.0, atol=1e-3))
M_perp_f = np.sqrt(Mx_f**2 + My_f**2)
check("F4: |M_perp| → 0 at long t", M_perp_f[-1] < 1e-3)


# ===========================================================================
# Group G - consistency with Prototype 2 analytic solution
# ===========================================================================
print("\n-- Group G: Consistency with P2 analytic solution -------------------")

# With T1 → large and same omega0, simulate_bloch should match bloch_precession
T1_large = 1e8
t_g = time_axis(4*T2, 0.02)
_, Mx_g, My_g, Mz_g = simulate_bloch(
    M_init=[M0, 0.0, 0.0], gamma=gamma, B=B,
    T1=T1_large, T2=T2, M0=M0, t_max=4*T2, dt=0.02)

omega0 = gamma * B0
Mx_p2, My_p2, Mz_p2 = bloch_precession(t_g, M0=M0, omega0=omega0, T2=T2, Mz0=0.0)

check("G1: Mx matches P2 analytic (rtol=1e-4)",  np.allclose(Mx_g, Mx_p2, rtol=1e-4))
check("G2: My matches P2 analytic (rtol=1e-4)",  np.allclose(My_g, My_p2, rtol=1e-4))
check("G3: Mz matches P2 analytic (Mz=0)",       np.allclose(Mz_g, 0.0,   atol=1e-4))


# ===========================================================================
# Group H - input validation
# ===========================================================================
print("\n-- Group H: Input validation ----------------------------------------")

for label, kwargs, exc in [
    ("H1: T1 <= 0",    dict(T1=0.0,  T2=T2,  t_max=10, dt=0.1), ValueError),
    ("H2: T2 <= 0",    dict(T1=T1,   T2=0.0, t_max=10, dt=0.1), ValueError),
    ("H3: T2 > T1",    dict(T1=5.0,  T2=10., t_max=10, dt=0.1), ValueError),
    ("H4: t_max <= 0", dict(T1=T1,   T2=T2,  t_max=-1, dt=0.1), ValueError),
    ("H5: dt <= 0",    dict(T1=T1,   T2=T2,  t_max=10, dt=-1.), ValueError),
]:
    base = dict(M_init=[M0,0,0], gamma=gamma, B=B, M0=M0)
    base.update(kwargs)
    try:
        simulate_bloch(**base)
        check(label, False)
    except exc:
        check(label, True)


# ===========================================================================
# Summary
# ===========================================================================
print(f"\n{'='*58}")
print(f"  {sum(results)}/{len(results)} tests passed")
print(f"{'='*58}\n")