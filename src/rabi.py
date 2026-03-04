"""
rabi.py - Rabi oscillation: driven qubit dynamics.
====================================================

Simulates a spin-½ system under a resonant or off-resonant drive field:

    H = (ℏω₀/2)σz + ℏΩcos(ωt)σx

In the rotating frame (rotating wave approximation, RWA), the
time-dependent drive reduces to a static transverse field and the
Hamiltonian becomes:

    H_rot = (ℏΔ/2)σz + (ℏΩ/2)σx

where Δ = ω - ω₀ is the detuning.  This maps exactly onto the Bloch
equations with an effective field (Ω, 0, Δ) — no explicit time
dependence remains.

Rotating-frame Bloch equations (with T₁, T₂ relaxation):

    dMx/dt =  Δ·My  - Mx/T₂
    dMy/dt = -Δ·Mx  + Ω·Mz - My/T₂
    dMz/dt = -Ω·My  - (Mz - 1)/T₁

Key physics results:
  - On resonance (Δ=0):  full oscillations between |↑⟩ and |↓⟩ at rate Ω
  - Off resonance:        reduced amplitude (Ω/Ω_eff)², faster oscillations Ω_eff
  - Ω_eff = √(Ω² + Δ²)  generalised Rabi frequency
  - t_π  = π/Ω_eff       π-pulse time (population inversion)
  - T₁, T₂ damp the oscillations exponentially over time
"""

from __future__ import annotations

import numpy as np
from typing import Tuple
from scipy.integrate import solve_ivp


# ============================================================================
# ODE solver — full Rabi dynamics with relaxation
# ============================================================================

def run_rabi(
    omega_rabi: float,
    delta: float,
    T1: float,
    T2: float,
    t_max: float,
    n: int = 2000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve the rotating-frame Bloch equations for a driven spin.

    The spin starts in the ground state M = (0, 0, 1) — fully aligned
    with +z (Mz = +1 corresponds to |↑⟩, the low-energy eigenstate).
    The drive field Ω tips the spin, producing Rabi oscillations.

    Parameters
    ----------
    omega_rabi : float
        Rabi frequency Ω (rad / time_unit).  Controls how fast the spin
        rotates between |↑⟩ and |↓⟩.  Proportional to drive amplitude.
    delta : float
        Detuning Δ = ω_drive - ω₀ (rad / time_unit).
        Δ = 0  →  on resonance (maximum population transfer).
        Δ ≠ 0  →  off resonance (reduced amplitude, faster oscillations).
    T1 : float
        Longitudinal relaxation time.  Damps Mz back to +1 (ground state).
    T2 : float
        Transverse relaxation time.  Damps Mx, My (coherence).  T2 ≤ T1.
    t_max : float
        Total simulation time (same units as T1, T2).
    n : int
        Number of output time points (default 2000).

    Returns
    -------
    t  : (n,) np.ndarray   time axis
    Mx : (n,) np.ndarray   rotating-frame x component
    My : (n,) np.ndarray   rotating-frame y component
    Mz : (n,) np.ndarray   longitudinal component (population contrast)

    Notes
    -----
    Excited-state population:  P↑(t) = (1 - Mz) / 2
      Mz = +1  →  P↑ = 0   (ground state, fully down)
      Mz =  0  →  P↑ = 0.5 (equal superposition)
      Mz = -1  →  P↑ = 1   (excited state, fully up / inverted)
    """
    if T2 > T1:
        raise ValueError(f"T2 ({T2}) cannot exceed T1 ({T1}) — unphysical")
    if omega_rabi < 0:
        raise ValueError(f"omega_rabi must be non-negative, got {omega_rabi}")
    if t_max <= 0:
        raise ValueError(f"t_max must be positive, got {t_max}")

    def _bloch_rotating(t, M):
        Mx, My, Mz = M
        dMx =  delta * My  - Mx / T2
        dMy = -delta * Mx  + omega_rabi * Mz - My / T2
        dMz = -omega_rabi * My - (Mz - 1.0) / T1
        return [dMx, dMy, dMz]

    t_eval = np.linspace(0, t_max, n)
    sol = solve_ivp(
        _bloch_rotating,
        t_span=(0.0, t_max),
        y0=[0.0, 0.0, 1.0],      # ground state: M = (0, 0, +1)
        t_eval=t_eval,
        method='RK45',
        rtol=1e-8,
        atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(f"Rabi ODE solver failed: {sol.message}")

    return sol.t, sol.y[0], sol.y[1], sol.y[2]


# ============================================================================
# Analytic solutions (no relaxation) — exact closed form
# ============================================================================

def analytic_rabi_population(
    t: np.ndarray,
    omega_rabi: float,
    delta: float,
) -> np.ndarray:
    """Analytic excited-state population without relaxation.

    Exact solution for T₁, T₂ → ∞:

        P↑(t) = (Ω / Ω_eff)² · sin²(Ω_eff · t / 2)

    where Ω_eff = √(Ω² + Δ²) is the generalised Rabi frequency.

    Parameters
    ----------
    t          : time array
    omega_rabi : Rabi frequency Ω
    delta      : detuning Δ

    Returns
    -------
    P_up : (N,) array of excited-state populations in [0, 1]
    """
    omega_eff = np.sqrt(omega_rabi**2 + delta**2)
    if omega_eff == 0:
        return np.zeros_like(np.asarray(t, dtype=float))
    return (omega_rabi / omega_eff)**2 * np.sin(omega_eff * np.asarray(t) / 2)**2


def analytic_chevron(
    omega_rabi: float,
    delta_max: float,
    t_max: float,
    n_delta: int = 200,
    n_t: int = 300,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analytic 2-D chevron: P↑(t, Δ) over a grid of time and detuning.

    The chevron (bowtie) pattern is the canonical signature of Rabi
    oscillations.  It arises because:
      - On resonance (Δ=0):   full amplitude (P↑ reaches 1), slowest at Ω
      - Off resonance (|Δ|>0): reduced amplitude, faster oscillations Ω_eff

    The characteristic 'V' shape in the time direction and the narrowing
    along detuning are direct measures of the drive strength Ω.

    Parameters
    ----------
    omega_rabi : Rabi frequency Ω (sets the on-resonance oscillation rate)
    delta_max  : detuning range, scanned from -delta_max to +delta_max
    t_max      : maximum time
    n_delta    : number of detuning points (rows of the heatmap)
    n_t        : number of time points (columns of the heatmap)

    Returns
    -------
    t      : (n_t,)           time axis
    deltas : (n_delta,)       detuning axis
    P      : (n_delta, n_t)   excited-state population grid
    """
    deltas    = np.linspace(-delta_max, delta_max, n_delta)
    t         = np.linspace(0, t_max, n_t)
    T, D      = np.meshgrid(t, deltas)
    omega_eff = np.sqrt(omega_rabi**2 + D**2)
    P         = (omega_rabi / omega_eff)**2 * np.sin(omega_eff * T / 2)**2
    return t, deltas, P


def pi_pulse_time(omega_rabi: float, delta: float) -> float:
    """Return the π-pulse time: t_π = π / Ω_eff.

    At t = t_π the population is fully inverted (P↑ = (Ω/Ω_eff)²).
    On resonance this equals 1.0.

    Parameters
    ----------
    omega_rabi : Rabi frequency Ω
    delta      : detuning Δ

    Returns
    -------
    t_pi : float
    """
    omega_eff = np.sqrt(omega_rabi**2 + delta**2)
    if omega_eff == 0:
        raise ValueError("omega_eff is zero — drive and detuning are both zero")
    return np.pi / omega_eff


def max_population_inversion(omega_rabi: float, delta: float) -> float:
    """Maximum achievable P↑ for given Ω and Δ.

    P↑_max = (Ω / Ω_eff)²

    On resonance (Δ=0): P↑_max = 1.0 (complete inversion).
    Off resonance: P↑_max < 1.0 (incomplete inversion).

    Parameters
    ----------
    omega_rabi : Rabi frequency Ω
    delta      : detuning Δ

    Returns
    -------
    float in [0, 1]
    """
    omega_eff = np.sqrt(omega_rabi**2 + delta**2)
    if omega_eff == 0:
        return 0.0
    return float((omega_rabi / omega_eff)**2)