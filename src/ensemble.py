"""
ensemble.py - Inhomogeneous spin ensemble simulation.
=====================================================

Prototype 5 models a realistic sample where N spins have slightly
different Larmor frequencies drawn from a Gaussian distribution
(e.g. due to local field variations).

Key physics results:
  - FID decays faster than single-spin T2 (apparent T2* < T2)
  - Hahn echo refocuses static inhomogeneity → echo amplitude still
    decays as exp(-2τ/T2), independent of sigma
  - Ensemble-averaged FID envelope:
      |<M_perp>(t)| = M0 · exp(-t/T2) · exp(-σ²t²/2)
    (Gaussian inhomogeneity gives Gaussian FID envelope on top of T2)
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Optional

from .core import simulate_bloch
from .sequences import hahn_echo_sequence, measure_echo_amplitude


# ===========================================================================
# Frequency sampling
# ===========================================================================

def sample_frequencies(
    omega0: float,
    sigma: float,
    N: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Draw N Larmor frequencies from a Gaussian distribution.

    Parameters
    ----------
    omega0 : float   centre frequency (rad / time_unit)
    sigma  : float   standard deviation of the distribution (rad / time_unit)
                     sigma = 0  →  all spins identical (no inhomogeneity)
    N      : int     number of spins in the ensemble
    seed   : int     optional RNG seed for reproducibility

    Returns
    -------
    frequencies : (N,) np.ndarray of angular frequencies

    Notes
    -----
    Uses numpy's modern Generator API (numpy.random.default_rng) so results
    are reproducible and independent of global random state.
    """
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    rng = np.random.default_rng(seed)
    return rng.normal(omega0, sigma, N)


# ===========================================================================
# Ensemble FID
# ===========================================================================

def simulate_ensemble_FID(
    omega0: float,
    sigma: float,
    N: int,
    T1: float,
    T2: float,
    M0: float,
    t_max: float,
    dt: float,
    gamma: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate ensemble free induction decay with inhomogeneous broadening.

    Each spin starts tipped into the transverse plane (π/2 pulse along y
    → M = [M0, 0, 0]).  Each spin precesses at its own frequency omega_i
    and decays at the common T1, T2.  The ensemble average is:

        <Mx(t)> + i<My(t)>
          = M0 · exp(iω₀t) · exp(-t/T2) · <exp(i·δω·t)>

    For Gaussian δω ~ N(0, σ²), the characteristic function gives:

        |<M_perp>(t)| = M0 · exp(-t/T2) · exp(-σ²t²/2)

    So the apparent decay is faster than T2 alone (T2* effect).

    Parameters
    ----------
    omega0 : float   central Larmor frequency (rad / time_unit)
    sigma  : float   frequency spread (rad / time_unit)
    N      : int     number of spins
    T1, T2 : float   relaxation times (T2 ≤ T1)
    M0     : float   equilibrium magnetisation
    t_max  : float   FID duration
    dt     : float   time step
    gamma  : float   gyromagnetic ratio (default 1.0)
    seed   : int     RNG seed for reproducibility

    Returns
    -------
    t             : (N_t,) np.ndarray  common time axis
    Mx_avg        : (N_t,) averaged transverse x component
    My_avg        : (N_t,) averaged transverse y component
    Mz_avg        : (N_t,) averaged longitudinal component

    Note: ensemble-averaged signal is |<M_perp>| = sqrt(Mx_avg²+My_avg²),
    NOT <|M_perp|>.  The former captures dephasing; the latter does not.
    """
    if T2 > T1:
        raise ValueError(f"T2 ({T2}) cannot exceed T1 ({T1})")

    frequencies = sample_frequencies(omega0, sigma, N, seed=seed)

    # π/2 pulse along y tips [0, 0, M0] → [M0, 0, 0]
    M_init = np.array([M0, 0.0, 0.0])

    Mx_sum = None
    My_sum = None
    Mz_sum = None
    t_out  = None

    for omega_i in frequencies:
        # Each spin sees a slightly different field along z
        B_i = np.array([0.0, 0.0, omega_i / gamma])
        t_i, Mx_i, My_i, Mz_i = simulate_bloch(
            M_init=M_init,
            gamma=gamma, B=B_i,
            T1=T1, T2=T2, M0=M0,
            t_max=t_max, dt=dt,
        )
        if t_out is None:
            t_out  = t_i
            Mx_sum = Mx_i.copy()
            My_sum = My_i.copy()
            Mz_sum = Mz_i.copy()
        else:
            # All calls use same t_max and dt → same length array
            Mx_sum += Mx_i
            My_sum += My_i
            Mz_sum += Mz_i

    return t_out, Mx_sum / N, My_sum / N, Mz_sum / N


# ===========================================================================
# Ensemble Hahn echo
# ===========================================================================

def simulate_ensemble_hahn_echo(
    omega0: float,
    sigma: float,
    N: int,
    T1: float,
    T2: float,
    M0: float,
    tau: float,
    dt: float,
    gamma: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate ensemble Hahn echo — static inhomogeneity is refocused.

    Each spin runs through the full π/2 → τ → π → τ sequence at its own
    frequency.  After the π pulse, each spin's accumulated phase is reversed,
    so at t = 2τ all spins realign regardless of their offset frequency.

    The echo amplitude is therefore:

        |<M_perp>(2τ)| = M0 · exp(-2τ/T2)

    independent of sigma.  This is the key distinction between T2 and T2*.

    Parameters
    ----------
    omega0, sigma, N, T1, T2, M0, tau, dt, gamma, seed : as above

    Returns
    -------
    t, Mx_avg, My_avg, Mz_avg : ensemble-averaged Bloch components
        Echo peak is visible near t = 2τ in |<M_perp>|.
    """
    if T2 > T1:
        raise ValueError(f"T2 ({T2}) cannot exceed T1 ({T1})")
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    if dt >= tau:
        raise ValueError(f"dt ({dt}) must be smaller than tau ({tau})")

    frequencies = sample_frequencies(omega0, sigma, N, seed=seed)

    Mx_sum = None
    My_sum = None
    Mz_sum = None
    t_out  = None

    for omega_i in frequencies:
        B_i = np.array([0.0, 0.0, omega_i / gamma])
        t_i, Mx_i, My_i, Mz_i = hahn_echo_sequence(
            gamma=gamma, B=B_i,
            T1=T1, T2=T2, M0=M0,
            tau=tau, dt=dt,
        )
        if t_out is None:
            t_out  = t_i
            Mx_sum = Mx_i.copy()
            My_sum = My_i.copy()
            Mz_sum = Mz_i.copy()
        else:
            Mx_sum += Mx_i
            My_sum += My_i
            Mz_sum += Mz_i

    return t_out, Mx_sum / N, My_sum / N, Mz_sum / N


# ===========================================================================
# Convenience: sweep tau for ensemble echo amplitude
# ===========================================================================

def sweep_ensemble_echo(
    omega0: float,
    sigma: float,
    N: int,
    T1: float,
    T2: float,
    M0: float,
    tau_values: np.ndarray,
    dt: float,
    gamma: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sweep τ and record ensemble echo amplitude at t = 2τ.

    Even with large sigma, the echo amplitude should follow:
        A_echo(2τ) = M0 · exp(-2τ / T2)

    Parameters
    ----------
    tau_values : (K,) array of τ values

    Returns
    -------
    two_tau    : (K,) echo times = 2·tau_values
    amplitudes : (K,) ensemble echo amplitudes at each 2τ
    """
    amplitudes = []
    for tau in tau_values:
        t, Mx, My, _ = simulate_ensemble_hahn_echo(
            omega0=omega0, sigma=sigma, N=N,
            T1=T1, T2=T2, M0=M0,
            tau=float(tau), dt=dt,
            gamma=gamma, seed=seed,
        )
        amp = measure_echo_amplitude(Mx, My, t, echo_time=2.0 * float(tau))
        amplitudes.append(amp)
    return 2.0 * np.asarray(tau_values), np.array(amplitudes)