"""
ramsey.py - Ramsey interference: π/2 → free evolution → π/2
=============================================================

The Ramsey sequence is the foundation of atomic clocks and quantum sensors.
It is more sensitive to frequency detuning than a single Rabi pulse because
the spin accumulates phase freely between two brief π/2 pulses, rather than
being driven continuously.

Sequence
--------
1. π/2 pulse  - tips spin from |↑⟩ to equator (+x axis in rotating frame)
2. Free evolution for time T  - spin precesses at detuning Δ, dephases at T₂*
3. π/2 pulse  - maps accumulated phase onto population (Mz)

Signal
------
    P↑(T) = ½ · (1 - cos(Δ·T) · exp(-T/T₂*))

The cosine oscillation vs free evolution time T gives "Ramsey fringes".
Fringe frequency = Δ/(2π), so measuring fringe frequency gives Δ precisely.
The envelope decay gives T₂* directly.

Key physics
-----------
- Fringe contrast → coherence (T₂*)
- Fringe frequency → detuning Δ (used in atomic clocks to lock to resonance)
- Inhomogeneous ensemble broadening enters as T₂* (Gaussian decay of envelope)
- Echo version (Hahn echo between the two π/2 pulses) recovers T₂ from T₂*
"""

from __future__ import annotations

import numpy as np
from typing import Tuple
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit


# ============================================================================
# Analytic solution (rotating frame, no relaxation)
# ============================================================================

def analytic_ramsey_population(
    T_free: np.ndarray,
    delta: float,
    T2_star: float,
    contrast: float = 1.0,
    offset: float = 0.5,
) -> np.ndarray:
    """Analytic Ramsey fringe signal with Gaussian T₂* envelope.

    P↑(T) = offset + (contrast/2) · cos(Δ·T) · exp(-T/T₂*)

    For a pure state starting in |↑⟩ with ideal π/2 pulses:
        offset   = 0.5
        contrast = 1.0  →  P↑ oscillates between 0 and 1

    Parameters
    ----------
    T_free   : free evolution time array
    delta    : detuning Δ = ω_drive - ω₀  (rad / time_unit)
    T2_star  : effective coherence time (includes inhomogeneous broadening)
    contrast : fringe visibility in [0, 1]  (default 1.0 - ideal pulses)
    offset   : signal baseline  (default 0.5 - equal superposition)

    Returns
    -------
    P_up : array of excited-state populations
    """
    T_free = np.asarray(T_free, dtype=float)
    return offset + (contrast / 2) * np.cos(delta * T_free) * np.exp(-T_free / T2_star)


# ============================================================================
# ODE solver - full Ramsey with T₁, T₂, arbitrary pulse imperfections
# ============================================================================

def _bloch_free(t, M, delta, T1, T2):
    """Rotating-frame Bloch RHS during free evolution (no drive, Ω=0)."""
    Mx, My, Mz = M
    return [
         delta * My  - Mx / T2,
        -delta * Mx  - My / T2,
        -(Mz - 1.0) / T1,
    ]


def _apply_pi2_pulse(M: np.ndarray, axis: str = 'y') -> np.ndarray:
    """Instantaneous π/2 rotation in the rotating frame.

    Default axis='y' tips +z → +x  (standard Ramsey preparation).
    Second pulse axis='y' maps +x → -z, -y → -z (converts phase → population).
    """
    Mx, My, Mz = M
    if axis == 'y':
        # Ry(π/2): x→x, y→y ... no wait, Ry(π/2): z→x, x→-z
        return np.array([ Mz,  My, -Mx])
    elif axis == '-y':
        return np.array([-Mz,  My,  Mx])
    elif axis == 'x':
        return np.array([ Mx, -Mz,  My])
    elif axis == '-x':
        return np.array([ Mx,  Mz, -My])
    else:
        raise ValueError(f"axis must be 'x', '-x', 'y', or '-y', got '{axis}'")


def run_ramsey(
    delta: float,
    T1: float,
    T2: float,
    T_free: float,
    n_points: int = 800,
    pulse1_axis: str = 'y',
    pulse2_axis: str = 'y',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate a single Ramsey sequence for one free evolution time T_free.

    Sequence: ground state → π/2 → free evolution T_free → π/2 → readout

    Parameters
    ----------
    delta      : detuning Δ = ω_drive - ω₀  (rad / time_unit)
    T1         : longitudinal relaxation time
    T2         : transverse relaxation time (T2 ≤ T1)
    T_free     : free evolution duration
    n_points   : time points during free evolution
    pulse1_axis: rotation axis for first  π/2 pulse (default 'y')
    pulse2_axis: rotation axis for second π/2 pulse (default 'y')

    Returns
    -------
    t, Mx, My, Mz : time axis and Bloch components across full sequence
        t = 0 is after the first π/2 pulse (start of free evolution).
        The final values are post-second-π/2.
    """
    if T2 > T1:
        raise ValueError(f"T2 ({T2}) cannot exceed T1 ({T1})")
    if T_free <= 0:
        raise ValueError(f"T_free must be positive, got {T_free}")

    # Ground state |↑⟩ = (0, 0, 1)
    M0 = np.array([0.0, 0.0, 1.0])

    # First π/2 pulse - tips to equator
    M1 = _apply_pi2_pulse(M0, axis=pulse1_axis)

    # Free evolution
    t_eval = np.linspace(0, T_free, n_points)
    sol = solve_ivp(
        _bloch_free,
        t_span=(0.0, T_free),
        y0=M1,
        t_eval=t_eval,
        args=(delta, T1, T2),
        method='RK45',
        rtol=1e-8,
        atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(f"Ramsey free evolution ODE failed: {sol.message}")

    t_free = sol.t
    Mx_free, My_free, Mz_free = sol.y

    # Second π/2 pulse - convert phase to population
    M_end = _apply_pi2_pulse(
        np.array([Mx_free[-1], My_free[-1], Mz_free[-1]]),
        axis=pulse2_axis)

    # Append final state as a single point
    t_out  = np.append(t_free,  t_free[-1])  # duplicate last t for the readout point
    Mx_out = np.append(Mx_free, M_end[0])
    My_out = np.append(My_free, M_end[1])
    Mz_out = np.append(Mz_free, M_end[2])

    return t_out, Mx_out, My_out, Mz_out


def sweep_ramsey(
    delta: float,
    T1: float,
    T2: float,
    T_free_max: float,
    n_sweep: int = 300,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sweep free evolution time and record P↑ after second π/2 pulse.

    This generates the Ramsey fringe pattern - the observable that atomic
    clocks and quantum sensors actually measure.

    Parameters
    ----------
    delta      : detuning (rad / time_unit)
    T1, T2     : relaxation times
    T_free_max : maximum free evolution time
    n_sweep    : number of T_free values to evaluate

    Returns
    -------
    T_free_arr : (n_sweep,) free evolution times
    P_up       : (n_sweep,) excited-state population after readout pulse
    """
    T_free_arr = np.linspace(1e-4, T_free_max, n_sweep)
    P_up       = np.empty(n_sweep)

    for i, T_f in enumerate(T_free_arr):
        _, _, _, Mz_out = run_ramsey(delta, T1, T2, T_f, n_points=200)
        # P↑ = (1 - Mz_final) / 2  - last point is post-second-pulse readout
        P_up[i] = (1.0 - Mz_out[-1]) / 2.0

    return T_free_arr, P_up


# ============================================================================
# Fringe fitting - extract Δ and T₂* from measured fringes
# ============================================================================

def _ramsey_model(T_free, delta_fit, T2_star, contrast, offset):
    """Curve-fit model: P↑ = offset + (contrast/2)·cos(Δ·T)·exp(-T/T₂*)."""
    return offset + (contrast / 2) * np.cos(delta_fit * T_free) * np.exp(-T_free / T2_star)


def fit_ramsey_fringes(
    T_free: np.ndarray,
    P_up: np.ndarray,
    delta_guess: float = 1.0,
    T2_star_guess: float = 5.0,
) -> Tuple[dict, dict, np.ndarray]:
    """Fit Ramsey fringes to extract detuning Δ and coherence time T₂*.

    Parameters
    ----------
    T_free        : free evolution time array
    P_up          : measured excited-state population
    delta_guess   : initial guess for Δ
    T2_star_guess : initial guess for T₂*

    Returns
    -------
    params : dict  {'delta': Δ_fit, 'T2_star': T₂*_fit,
                    'contrast': c_fit, 'offset': o_fit}
    errors : dict  1-sigma uncertainties for each parameter
    P_fit  : fitted fringe curve at T_free points
    """
    p0     = [delta_guess, T2_star_guess, 1.0, 0.5]
    bounds = ([0,    0.01, 0,  0  ],
              [1e4,  1e6,  1,  1  ])
    try:
        popt, pcov = curve_fit(
            _ramsey_model, T_free, P_up,
            p0=p0, bounds=bounds, maxfev=20000)
    except RuntimeError:
        # Fall back to unconstrained fit if bounded fails
        popt, pcov = curve_fit(_ramsey_model, T_free, P_up, p0=p0, maxfev=20000)

    stds   = np.sqrt(np.diag(pcov))
    names  = ['delta', 'T2_star', 'contrast', 'offset']
    params = {n: float(v) for n, v in zip(names, popt)}
    errors = {n: float(s) for n, s in zip(names, stds)}
    P_fit  = _ramsey_model(T_free, *popt)
    return params, errors, P_fit


# ============================================================================
# Detuning sensitivity - the key figure of merit for sensing
# ============================================================================

def ramsey_sensitivity(
    T2_star: float,
    n_shots: int = 1000,
) -> float:
    """Minimum detectable frequency shift δf for a Ramsey experiment.

    At optimal free evolution time T_opt = T₂*/2, the phase sensitivity
    per shot is 1/√N_shots, giving frequency sensitivity:

        δf_min = 1 / (2π · T₂* · √N_shots)   [Hz, if T₂* in seconds]

    Parameters
    ----------
    T2_star : coherence time (same units as 1/frequency)
    n_shots : number of repeated measurements

    Returns
    -------
    delta_f_min : minimum detectable frequency shift
    """
    return 1.0 / (2 * np.pi * T2_star * np.sqrt(n_shots))