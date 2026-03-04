"""
density_matrix.py - Open quantum system evolution via density matrix.
======================================================================

Moves from the Bloch vector (a classical 3-vector) to the density matrix
(a 2×2 complex Hermitian operator), which is the proper quantum mechanical
description of a spin-½ system.

State representation
--------------------
A pure state |ψ⟩ = α|↑⟩ + β|↓⟩ has density matrix:

    ρ = |ψ⟩⟨ψ| = [[|α|², αβ*],
                   [α*β,  |β|²]]

A mixed state is a classical probability mixture of pure states:

    ρ = Σ_i p_i |ψ_i⟩⟨ψ_i|

The Bloch vector r⃗ is recovered from ρ via the Pauli matrices:

    rx = 2·Re(ρ₀₁) = Tr(ρ·σx)
    ry = 2·Im(ρ₀₁) = Tr(ρ·σy)   [note sign convention]
    rz = ρ₀₀ − ρ₁₁ = Tr(ρ·σz)

|r⃗| = 1 ↔ pure state;  |r⃗| < 1 ↔ mixed state.

Lindblad master equation
------------------------
The full open-system evolution including T₁ and T₂:

    dρ/dt = −(i/ℏ)[H, ρ] + L₁[ρ] + L₂[ρ]

where the Lindblad dissipators are:

    L₁[ρ] = γ₁ (σ₋ ρ σ₊ − ½{σ₊σ₋, ρ})   T₁ decay (amplitude damping)
    L₂[ρ] = γ_φ(σz ρ σz − ρ)              pure dephasing

with  γ₁ = 1/T₁  and  γ_φ = 1/(2T₂) − 1/(4T₁).

This reduces exactly to the Bloch equations for a two-level system,
providing a cross-validation target.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple
from scipy.integrate import solve_ivp


# ============================================================================
# Pauli matrices and basis states
# ============================================================================

I2  = np.eye(2, dtype=complex)
sx  = np.array([[0, 1], [1, 0]], dtype=complex)
sy  = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz  = np.array([[1, 0], [0, -1]], dtype=complex)
s_plus  = np.array([[0, 1], [0, 0]], dtype=complex)   # |↑⟩⟨↓| — raises
s_minus = np.array([[0, 0], [1, 0]], dtype=complex)   # |↓⟩⟨↑| — lowers

# Basis states (column vectors)
KET_UP   = np.array([[1], [0]], dtype=complex)   # |↑⟩ — ground state
KET_DOWN = np.array([[0], [1]], dtype=complex)   # |↓⟩ — excited state


# ============================================================================
# State constructors
# ============================================================================

def ground_state_dm() -> np.ndarray:
    """Density matrix for |↑⟩ (ground state, Mz = +1)."""
    return np.array([[1, 0], [0, 0]], dtype=complex)


def excited_state_dm() -> np.ndarray:
    """Density matrix for |↓⟩ (excited state, Mz = −1)."""
    return np.array([[0, 0], [0, 1]], dtype=complex)


def superposition_dm(theta: float = np.pi/2, phi: float = 0.0) -> np.ndarray:
    """Density matrix for a pure state on the Bloch sphere.

    |ψ⟩ = cos(θ/2)|↑⟩ + e^(iφ)sin(θ/2)|↓⟩

    Parameters
    ----------
    theta : polar angle from north pole (0 = |↑⟩, π = |↓⟩, π/2 = equator)
    phi   : azimuthal angle (0 = +x, π/2 = +y)

    Returns
    -------
    rho : (2,2) complex density matrix
    """
    alpha = np.cos(theta / 2)
    beta  = np.exp(1j * phi) * np.sin(theta / 2)
    psi   = np.array([[alpha], [beta]], dtype=complex)
    return psi @ psi.conj().T


def mixed_state_dm(purity: float) -> np.ndarray:
    """Maximally mixed state scaled by purity p: ρ = p·|+x⟩⟨+x| + (1-p)·I/2.

    purity = 1.0 → pure |+x⟩ state  (Bloch vector on equator)
    purity = 0.0 → completely mixed  (Bloch vector at origin)

    Parameters
    ----------
    purity : float in [0, 1]

    Returns
    -------
    rho : (2,2) density matrix with Tr(ρ) = 1
    """
    if not (0 <= purity <= 1):
        raise ValueError(f"purity must be in [0, 1], got {purity}")
    rho_pure = superposition_dm(theta=np.pi/2, phi=0.0)  # |+x⟩
    return purity * rho_pure + (1 - purity) * I2 / 2


# ============================================================================
# Bloch vector ↔ density matrix conversions
# ============================================================================

def dm_to_bloch(rho: np.ndarray) -> np.ndarray:
    """Extract Bloch vector r⃗ = (rx, ry, rz) from density matrix ρ.

    r = (Tr(ρ·σx), Tr(ρ·σy), Tr(ρ·σz))

    |r⃗| = 1 → pure state;  |r⃗| < 1 → mixed state.

    Parameters
    ----------
    rho : (2,2) complex density matrix

    Returns
    -------
    r : (3,) real array  [rx, ry, rz]
    """
    rx = float(np.real(np.trace(rho @ sx)))
    ry = float(np.real(np.trace(rho @ sy)))
    rz = float(np.real(np.trace(rho @ sz)))
    return np.array([rx, ry, rz])


def bloch_to_dm(r: np.ndarray) -> np.ndarray:
    """Construct density matrix from Bloch vector r⃗.

    ρ = (I + r⃗·σ⃗) / 2 = ½(I + rx·σx + ry·σy + rz·σz)

    Parameters
    ----------
    r : (3,) array  [rx, ry, rz]  with |r| ≤ 1

    Returns
    -------
    rho : (2,2) complex density matrix
    """
    rx, ry, rz = r
    return (I2 + rx*sx + ry*sy + rz*sz) / 2


def purity(rho: np.ndarray) -> float:
    """Return Tr(ρ²) — equals 1 for pure states, 1/2 for maximally mixed.

    Parameters
    ----------
    rho : (2,2) density matrix

    Returns
    -------
    float in [0.5, 1.0] for a qubit
    """
    return float(np.real(np.trace(rho @ rho)))


def population(rho: np.ndarray) -> Tuple[float, float]:
    """Return (P_ground, P_excited) = (ρ₀₀, ρ₁₁).

    Parameters
    ----------
    rho : (2,2) density matrix

    Returns
    -------
    (P_up, P_down) : ground and excited state populations
    """
    return float(np.real(rho[0, 0])), float(np.real(rho[1, 1]))


# ============================================================================
# Hamiltonian builders
# ============================================================================

def hamiltonian_free(omega0: float) -> np.ndarray:
    """Free precession Hamiltonian: H = (ℏω₀/2)·σz.

    In natural units (ℏ = 1): H = (ω₀/2)·σz.

    Parameters
    ----------
    omega0 : Larmor frequency ω₀ (rad / time_unit)

    Returns
    -------
    H : (2,2) complex Hermitian matrix
    """
    return (omega0 / 2) * sz


def hamiltonian_rotating(delta: float, omega_rabi: float) -> np.ndarray:
    """Rotating-frame Hamiltonian (RWA): H = (Δ/2)·σz + (Ω/2)·σx.

    This is H_rot after the rotating wave approximation removes the
    fast-oscillating terms from H = (ω₀/2)σz + Ωcos(ωt)σx.

    Parameters
    ----------
    delta      : detuning Δ = ω_drive − ω₀
    omega_rabi : drive strength Ω

    Returns
    -------
    H : (2,2) complex Hermitian matrix
    """
    return (delta / 2) * sz + (omega_rabi / 2) * sx


def hamiltonian_free_evolution(delta: float) -> np.ndarray:
    """Rotating-frame free evolution: H = (Δ/2)·σz  (no drive)."""
    return (delta / 2) * sz


# ============================================================================
# Lindblad dissipators
# ============================================================================

def _dissipator(L: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Standard Lindblad dissipator: D[L]ρ = LρL† − ½{L†L, ρ}."""
    Ld = L.conj().T
    return L @ rho @ Ld - 0.5 * (Ld @ L @ rho + rho @ Ld @ L)


def lindblad_rhs(
    rho: np.ndarray,
    H: np.ndarray,
    T1: float,
    T2: float,
) -> np.ndarray:
    """Full Lindblad master equation RHS for a driven qubit.

    dρ/dt = −i[H, ρ] + γ₁·D[σ₋]ρ + γ_φ·D[σz]ρ

    Decay rates:
        γ₁  = 1/T₁          amplitude damping  (energy relaxation)
        γ_φ = 1/(2T₂) − 1/(4T₁)  pure dephasing

    This gives exactly the Bloch equations:
        dMx/dt = −Mx/T₂  +  (ω-dependent terms from H)
        dMy/dt = −My/T₂  +  (ω-dependent terms from H)
        dMz/dt = −(Mz−1)/T₁

    Parameters
    ----------
    rho : (2,2) density matrix (current state)
    H   : (2,2) Hamiltonian (ℏ=1 units)
    T1  : longitudinal relaxation time
    T2  : transverse relaxation time (T2 ≤ T1)

    Returns
    -------
    drho_dt : (2,2) complex array — time derivative of ρ
    """
    gamma1  = 1.0 / T1
    # Pure dephasing rate: γ_φ = 1/(2T₂) − 1/(4T₁)
    # Clamped to zero to avoid unphysical negative dephasing when T₂ = T₁/2
    gamma_phi = max(0.0, 1.0 / (2 * T2) - 1.0 / (4 * T1))

    commutator = H @ rho - rho @ H
    drho = -1j * commutator
    drho += gamma1    * _dissipator(s_minus, rho)   # amplitude damping
    drho += gamma_phi * _dissipator(sz,      rho)   # pure dephasing
    return drho


# ============================================================================
# ODE solver
# ============================================================================

def simulate_dm(
    rho0: np.ndarray,
    H: np.ndarray,
    T1: float,
    T2: float,
    t_max: float,
    n: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the Lindblad master equation for a qubit.

    Parameters
    ----------
    rho0  : (2,2) initial density matrix
    H     : (2,2) Hamiltonian (time-independent; ℏ=1)
    T1    : longitudinal relaxation time
    T2    : transverse relaxation time (T2 ≤ T1)
    t_max : simulation duration
    n     : number of output time points

    Returns
    -------
    t      : (n,)  time axis
    rx     : (n,)  Bloch x component
    ry     : (n,)  Bloch y component
    rz     : (n,)  Bloch z component
    pur    : (n,)  purity Tr(ρ²) — tracks pure→mixed transition
    """
    if T2 > T1:
        raise ValueError(f"T2 ({T2}) cannot exceed T1 ({T1})")
    if t_max <= 0:
        raise ValueError(f"t_max must be positive, got {t_max}")

    # Flatten 2×2 complex matrix → 8 real numbers for the ODE solver
    def _pack(rho):
        return np.concatenate([rho.real.ravel(), rho.imag.ravel()])

    def _unpack(y):
        half = len(y) // 2
        return (y[:half].reshape(2, 2) + 1j * y[half:].reshape(2, 2))

    def _rhs(t, y):
        rho    = _unpack(y)
        drho   = lindblad_rhs(rho, H, T1, T2)
        return _pack(drho)

    y0     = _pack(rho0)
    t_eval = np.linspace(0, t_max, n)

    sol = solve_ivp(
        _rhs,
        t_span=(0.0, t_max),
        y0=y0,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-8,
        atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(f"Density matrix ODE failed: {sol.message}")

    rx_arr  = np.empty(n)
    ry_arr  = np.empty(n)
    rz_arr  = np.empty(n)
    pur_arr = np.empty(n)

    for i in range(n):
        rho_i      = _unpack(sol.y[:, i])
        r          = dm_to_bloch(rho_i)
        rx_arr[i]  = r[0]
        ry_arr[i]  = r[1]
        rz_arr[i]  = r[2]
        pur_arr[i] = purity(rho_i)

    return sol.t, rx_arr, ry_arr, rz_arr, pur_arr


# ============================================================================
# Cross-validation: density matrix vs Bloch ODE
# ============================================================================

def bloch_vs_dm_error(
    omega_rabi: float,
    delta: float,
    T1: float,
    T2: float,
    t_max: float,
    n: int = 500,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compare Bloch ODE (from rabi.py) with density matrix evolution.

    Runs both methods with identical parameters and returns the maximum
    absolute error in the Bloch vector over the full trajectory.
    This is the primary correctness check for the density matrix module.

    Parameters
    ----------
    omega_rabi : Rabi drive strength Ω
    delta      : detuning Δ
    T1, T2     : relaxation times
    t_max      : simulation duration
    n          : number of time points

    Returns
    -------
    t            : time axis
    bloch_dm     : (n, 3) Bloch vectors from density matrix
    max_error    : max |r_bloch − r_dm| across all times and components
    """
    from .rabi import run_rabi

    # Density matrix path
    H    = hamiltonian_rotating(delta, omega_rabi)
    rho0 = ground_state_dm()
    t_dm, rx, ry, rz, _ = simulate_dm(rho0, H, T1, T2, t_max, n=n)
    bloch_dm = np.column_stack([rx, ry, rz])

    # Bloch ODE path
    t_bl, Mx, My, Mz = run_rabi(omega_rabi, delta, T1, T2, t_max, n=n)
    bloch_ode = np.column_stack([Mx, My, Mz])

    max_error = float(np.max(np.abs(bloch_dm - bloch_ode)))
    return t_dm, bloch_dm, max_error