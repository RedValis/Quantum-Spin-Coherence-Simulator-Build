"""
tests/test_ensemble.py - Unit tests for Prototype 5 (inhomogeneous ensemble).
==============================================================================

Physics invariants under test:

  A. sample_frequencies    - shape, mean, std, sigma=0 edge case
  B. sigma=0 FID           - ensemble matches single spin (T2 only)
  C. FID envelope          - |<M_perp>| ≈ M0·exp(-t/T2)·exp(-σ²t²/2)
  D. Faster decay          - larger sigma → faster FID
  E. Echo refocusing       - amplitude at 2τ matches exp(-2τ/T2) regardless of σ
  F. T2* < T2              - FID decays faster than echo with σ > 0
  G. Echo sweep            - fitted T2 matches input, independent of σ
  H. Input validation      - bad N, sigma, T2>T1, tau errors

Run:  python tests/test_ensemble.py
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ensemble import (
    sample_frequencies,
    simulate_ensemble_FID,
    simulate_ensemble_hahn_echo,
    sweep_ensemble_echo,
)
from src.sequences import measure_echo_amplitude

results = []

def check(name, condition):
    ok = bool(condition)
    results.append(ok)
    print(f"  {'✅ PASS' if ok else '❌ FAIL'}  {name}")
    return ok


# -- shared physics ---------------------------------------------------------
gamma  = 1.0
B0     = 2 * np.pi * 0.5        # rad/µs
omega0 = gamma * B0             # = B0 since gamma=1
sigma  = 0.3                    # rad/µs  (~10% spread)
N      = 300                    # spins (balance speed vs statistics)
M0     = 1.0
T1     = 50.0                   # µs
T2     = 10.0                   # µs
dt     = 0.05                   # µs
SEED   = 42                     # fixed seed for reproducibility


# ===========================================================================
# Group A - sample_frequencies
# ===========================================================================
print("\n-- Group A: sample_frequencies --------------------------------------")

freqs = sample_frequencies(omega0, sigma, N, seed=SEED)
check("A1: returns (N,) array",            freqs.shape == (N,))
check("A2: mean ≈ omega0 (rtol 5%)",
      np.isclose(np.mean(freqs), omega0, rtol=0.05))
check("A3: std ≈ sigma (rtol 10%)",
      np.isclose(np.std(freqs), sigma, rtol=0.10))

# sigma=0 → all identical
freqs_zero = sample_frequencies(omega0, 0.0, N, seed=SEED)
check("A4: sigma=0 → all frequencies = omega0",
      np.allclose(freqs_zero, omega0))

# reproducibility: same seed → same values
freqs2 = sample_frequencies(omega0, sigma, N, seed=SEED)
check("A5: fixed seed → reproducible",     np.allclose(freqs, freqs2))

# different seeds → different values
freqs3 = sample_frequencies(omega0, sigma, N, seed=SEED + 1)
check("A6: different seeds → different",   not np.allclose(freqs, freqs3))

try:
    sample_frequencies(omega0, sigma, N=0)
    check("A7: raises on N=0", False)
except ValueError:
    check("A7: raises on N=0", True)

try:
    sample_frequencies(omega0, sigma=-0.1, N=10)
    check("A8: raises on sigma<0", False)
except ValueError:
    check("A8: raises on sigma<0", True)


# ===========================================================================
# Group B - sigma=0 FID matches single spin
# ===========================================================================
print("\n-- Group B: sigma=0 FID matches single-spin Bloch -------------------")

# With sigma=0, ensemble FID = single spin (no dephasing)
t_fid0, Mx0, My0, Mz0 = simulate_ensemble_FID(
    omega0=omega0, sigma=0.0, N=10,   # small N fine since all identical
    T1=T1, T2=T2, M0=M0,
    t_max=2*T2, dt=dt,
    gamma=gamma, seed=SEED)

M_perp0 = np.sqrt(Mx0**2 + My0**2)
expected_single = M0 * np.exp(-t_fid0 / T2)

check("B1: sigma=0 |<M_perp>| matches exp(-t/T2)",
      np.allclose(M_perp0, expected_single, rtol=1e-4))
check("B2: sigma=0 |<M_perp>|(0) = M0",
      np.isclose(M_perp0[0], M0, rtol=1e-6))
check("B3: sigma=0 |<M_perp>|(T2) ≈ 1/e",
      np.isclose(M_perp0[np.argmin(np.abs(t_fid0 - T2))], 1/np.e, rtol=1e-3))


# ===========================================================================
# Group C - FID envelope matches Gaussian formula
# ===========================================================================
print("\n-- Group C: FID envelope matches M0·exp(-t/T2)·exp(-σ²t²/2) ---------")

# Use large N for accurate statistics; fixed seed for reproducibility
N_large = 1500
t_fid, Mx_fid, My_fid, _ = simulate_ensemble_FID(
    omega0=omega0, sigma=sigma, N=N_large,
    T1=T1, T2=T2, M0=M0,
    t_max=2*T2, dt=dt,
    gamma=gamma, seed=SEED)

M_perp_fid = np.sqrt(Mx_fid**2 + My_fid**2)
# Analytic: M0 * exp(-t/T2) * exp(-sigma^2*t^2/2)
analytic_fid = M0 * np.exp(-t_fid / T2) * np.exp(-0.5 * sigma**2 * t_fid**2)

mask = analytic_fid > 0.15
check("C1: FID envelope matches analytic where signal > 0.15 (rtol 10%)",
      np.allclose(M_perp_fid[mask], analytic_fid[mask], rtol=0.10))
check("C2: FID |<M_perp>|(0) = M0",
      np.isclose(M_perp_fid[0], M0, rtol=0.02))
check("C3: FID decays faster than exp(-t/T2) with sigma>0",
      np.all(M_perp_fid[1:] <= M0 * np.exp(-t_fid[1:] / T2) + 0.02))


# ===========================================================================
# Group D - larger sigma → faster FID decay
# ===========================================================================
print("\n-- Group D: Larger sigma → faster FID decay --------------------------")

results_sigma = {}
for sig in [0.0, 0.2, 0.5, 1.0]:
    t_s, Mx_s, My_s, _ = simulate_ensemble_FID(
        omega0=omega0, sigma=sig, N=N,
        T1=T1, T2=T2, M0=M0,
        t_max=3*T2, dt=dt,
        gamma=gamma, seed=SEED)
    # measure |<M_perp>| at t = T2/2
    idx = np.argmin(np.abs(t_s - T2/2))
    results_sigma[sig] = np.sqrt(Mx_s[idx]**2 + My_s[idx]**2)

check("D1: sigma=0.2 → smaller |M_perp| than sigma=0",
      results_sigma[0.2] < results_sigma[0.0] + 0.01)
check("D2: sigma=0.5 → smaller than sigma=0.2",
      results_sigma[0.5] < results_sigma[0.2] + 0.01)
check("D3: sigma=1.0 → smallest (fastest decay)",
      results_sigma[1.0] < results_sigma[0.5] + 0.01)


# ===========================================================================
# Group E - Hahn echo refocuses inhomogeneity
# ===========================================================================
print("\n-- Group E: Echo amplitude matches exp(-2τ/T2) regardless of σ ------")

tau = 3.0
for sig_e in [0.0, 0.3, 0.8]:
    t_e, Mx_e, My_e, _ = simulate_ensemble_hahn_echo(
        omega0=omega0, sigma=sig_e, N=N,
        T1=T1, T2=T2, M0=M0,
        tau=tau, dt=dt,
        gamma=gamma, seed=SEED)
    amp  = measure_echo_amplitude(Mx_e, My_e, t_e, echo_time=2*tau)
    exp_amp = M0 * np.exp(-2*tau / T2)
    check(f"E: sigma={sig_e} echo amp={amp:.4f} ≈ exp(-2τ/T2)={exp_amp:.4f}",
          np.isclose(amp, exp_amp, rtol=0.05))


# ===========================================================================
# Group F - T2* < T2: FID decays faster than echo
# ===========================================================================
print("\n-- Group F: FID decays faster than echo (T2* < T2) ------------------")

sigma_f = 0.25
tau_f   = 2.0

t_ff, Mx_ff, My_ff, _ = simulate_ensemble_FID(
    omega0=omega0, sigma=sigma_f, N=N,
    T1=T1, T2=T2, M0=M0,
    t_max=2*tau_f, dt=dt,
    gamma=gamma, seed=SEED)

t_fe, Mx_fe, My_fe, _ = simulate_ensemble_hahn_echo(
    omega0=omega0, sigma=sigma_f, N=N,
    T1=T1, T2=T2, M0=M0,
    tau=tau_f, dt=dt,
    gamma=gamma, seed=SEED)

# FID amplitude at t=2*tau_f
idx_fid  = np.argmin(np.abs(t_ff - 2*tau_f))
amp_fid  = np.sqrt(Mx_ff[idx_fid]**2 + My_ff[idx_fid]**2)

# Echo amplitude at t=2*tau_f
amp_echo = measure_echo_amplitude(Mx_fe, My_fe, t_fe, echo_time=2*tau_f)

check("F1: FID amplitude < echo amplitude at 2τ (T2* effect)",
      amp_fid < amp_echo - 0.01)
check("F2: Echo amplitude ≈ exp(-2τ/T2)",
      np.isclose(amp_echo, M0*np.exp(-2*tau_f/T2), rtol=0.05))
check("F3: FID amplitude ≈ exp(-2τ/T2)·exp(-σ²(2τ)²/2) (rtol 15%)",
      np.isclose(amp_fid,
                 M0 * np.exp(-2*tau_f/T2) * np.exp(-0.5*sigma_f**2*(2*tau_f)**2),
                 rtol=0.15))


# ===========================================================================
# Group G - Echo sweep gives correct T2 regardless of sigma
# ===========================================================================
print("\n-- Group G: Echo sweep → correct T2 independent of σ ----------------")

tau_arr  = np.array([1.0, 2.0, 3.0, 5.0, 7.0])

for sig_g in [0.0, 0.5]:
    two_tau_g, amps_g = sweep_ensemble_echo(
        omega0=omega0, sigma=sig_g, N=N,
        T1=T1, T2=T2, M0=M0,
        tau_values=tau_arr, dt=dt,
        gamma=gamma, seed=SEED)
    log_A   = np.log(np.clip(amps_g, 1e-10, None))
    coeffs  = np.polyfit(two_tau_g, log_A, 1)
    T2_fit  = -1.0 / coeffs[0]
    check(f"G: sigma={sig_g} → fitted T2={T2_fit:.2f} ≈ {T2} µs (rtol 10%)",
          np.isclose(T2_fit, T2, rtol=0.10))


# ===========================================================================
# Group H - Input validation
# ===========================================================================
print("\n-- Group H: Input validation -----------------------------------------")

try:
    sample_frequencies(omega0, sigma, N=-1)
    check("H1: raises on N<1", False)
except ValueError:
    check("H1: raises on N<1", True)

try:
    simulate_ensemble_FID(omega0, sigma, N, T1=5.0, T2=10.0,
                          M0=M0, t_max=20.0, dt=dt, gamma=gamma)
    check("H2: FID raises on T2>T1", False)
except ValueError:
    check("H2: FID raises on T2>T1", True)

try:
    simulate_ensemble_hahn_echo(omega0, sigma, N, T1=T1, T2=T2,
                                M0=M0, tau=-1.0, dt=dt, gamma=gamma)
    check("H3: echo raises on tau<=0", False)
except ValueError:
    check("H3: echo raises on tau<=0", True)

try:
    simulate_ensemble_hahn_echo(omega0, sigma, N, T1=T1, T2=T2,
                                M0=M0, tau=1.0, dt=5.0, gamma=gamma)
    check("H4: echo raises on dt>=tau", False)
except ValueError:
    check("H4: echo raises on dt>=tau", True)


# ===========================================================================
# Summary
# ===========================================================================
print(f"\n{'='*58}")
print(f"  {sum(results)}/{len(results)} tests passed")
print(f"{'='*58}\n")