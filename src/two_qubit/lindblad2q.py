"""
lindblad2q.py - Two-qubit open-system Lindblad master equation.
================================================================

Extends the single-qubit Lindblad formalism (density_matrix.py) to the
four-dimensional two-qubit Hilbert space.

Noise channels
--------------

Local noise (independent on each qubit):
    Each qubit A and B independently couples to its own bath.
    Single-qubit Lindblad operators are embedded via tensor product:
        L_A = L ⊗ I₂        (acts on qubit A only)
        L_B = I₂ ⊗ L        (acts on qubit B only)

    Implemented local channels (per-qubit):
        Amplitude damping:   L = √γ₁ · σ₊
        Phase damping:       L = √γ_φ · σz

Correlated noise:
    Two-qubit dephasing where both qubits see the same bath fluctuation.
    L_ZZ = √γ_ZZ · (σz ⊗ σz)
    This creates correlated phase noise - relevant for NV centres in
    diamond or ions in the same trap.

Depolarizing (local or global):
    Local depolarizing on each qubit (3 Lindblad operators per qubit).
    Global depolarizing contracts the full 4-qubit Bloch vector.

Entanglement sudden death (ESD)
--------------------------------
Under local amplitude damping, entanglement can reach zero in *finite*
time even though the state never fully decoheres - "sudden death" of
entanglement.  This module makes ESD easy to observe by tracking
concurrence alongside the density matrix trajectory.

State representation
--------------------
A two-qubit density matrix ρ is (4,4) complex.  It is flattened to
32 real numbers for the ODE solver (real and imaginary parts of all 16
elements).

Usage
-----
    from src.two_qubit.lindblad2q import (
        TwoQubitNoiseModel, simulate_2q, entanglement_sudden_death
    )

    nm = TwoQubitNoiseModel(T1_A=10, T2_A=8, T1_B=15, T2_B=10)
    t, rho_t, ent = simulate_2q(bell_state_dm('phi_plus'), H, nm, t_max=30)
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass, field
from scipy.integrate import solve_ivp

from .states import (
    I2, I4, sx, sy, sz,
    tensor, ket_to_dm, bell_state_dm,
)
from .entanglement import (
    track_entanglement, concurrence, partial_trace, von_neumann_entropy,
)

# Single-qubit operators
s_plus  = np.array([[0, 1], [0, 0]], dtype=complex)   # |↑⟩⟨↓| raises
s_minus = np.array([[0, 0], [1, 0]], dtype=complex)   # |↓⟩⟨↑| lowers


# ============================================================================
# Noise model
# ============================================================================

@dataclass
class TwoQubitNoiseModel:
    """
    Noise parameters for a two-qubit open system.

    Each qubit has independent T1 and T2.  Optional correlated ZZ dephasing
    and per-qubit depolarizing channels can be added.

    Parameters
    ----------
    T1_A, T2_A : relaxation/coherence times for qubit A (np.inf = no noise)
    T1_B, T2_B : relaxation/coherence times for qubit B
    T_dep_A    : depolarizing time for qubit A (np.inf = off)
    T_dep_B    : depolarizing time for qubit B (np.inf = off)
    T_ZZ       : correlated ZZ dephasing time  (np.inf = off)

    Examples
    --------
    # Identical qubits, T1/T2 only:
    nm = TwoQubitNoiseModel(T1_A=10, T2_A=8, T1_B=10, T2_B=8)

    # Asymmetric qubits:
    nm = TwoQubitNoiseModel(T1_A=20, T2_A=15, T1_B=8, T2_B=6)

    # Add correlated ZZ noise:
    nm = TwoQubitNoiseModel(T1_A=10, T2_A=8, T1_B=10, T2_B=8, T_ZZ=5)

    # Pure depolarizing:
    nm = TwoQubitNoiseModel(T_dep_A=10, T_dep_B=10)
    """
    T1_A:    float = np.inf
    T2_A:    float = np.inf
    T1_B:    float = np.inf
    T2_B:    float = np.inf
    T_dep_A: float = np.inf
    T_dep_B: float = np.inf
    T_ZZ:    float = np.inf
    _ops: list = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        # Validate T2 ≤ 2·T1 for each qubit
        for label, T1, T2 in [('A', self.T1_A, self.T2_A),
                               ('B', self.T1_B, self.T2_B)]:
            if np.isfinite(T1) and np.isfinite(T2):
                if T2 > 2.0 * T1:
                    raise ValueError(
                        f"Qubit {label}: T2 ({T2}) > 2·T1 ({2*T1}) is unphysical."
                    )
        self._ops = self._build_operators()

    def _gamma1(self, T1: float) -> float:
        return 1.0 / T1 if np.isfinite(T1) else 0.0

    def _gamma_phi(self, T1: float, T2: float) -> float:
        rate = 0.0
        if np.isfinite(T2):
            rate += 1.0 / (2.0 * T2)
        if np.isfinite(T1):
            rate -= 1.0 / (4.0 * T1)
        return max(0.0, rate)

    def _gamma_dep(self, T_dep: float) -> float:
        return 1.0 / T_dep if np.isfinite(T_dep) else 0.0

    def _build_operators(self):
        """Build list of (L_4x4, gamma) pairs for all active channels."""
        ops = []

        # ---- Qubit A (L_A = L ⊗ I) ----------------------------------------
        γ1A = self._gamma1(self.T1_A)
        if γ1A > 0:
            L = np.kron(s_plus, I2)
            ops.append((L, γ1A, 'amp_A'))

        γφA = self._gamma_phi(self.T1_A, self.T2_A)
        if γφA > 0:
            L = np.kron(sz, I2)
            ops.append((L, γφA, 'phase_A'))

        γdA = self._gamma_dep(self.T_dep_A)
        if γdA > 0:
            for pauli in (sx, sy, sz):
                L = np.kron(pauli, I2)
                ops.append((L, γdA / 4.0, 'dep_A'))

        # ---- Qubit B (L_B = I ⊗ L) ----------------------------------------
        γ1B = self._gamma1(self.T1_B)
        if γ1B > 0:
            L = np.kron(I2, s_plus)
            ops.append((L, γ1B, 'amp_B'))

        γφB = self._gamma_phi(self.T1_B, self.T2_B)
        if γφB > 0:
            L = np.kron(I2, sz)
            ops.append((L, γφB, 'phase_B'))

        γdB = self._gamma_dep(self.T_dep_B)
        if γdB > 0:
            for pauli in (sx, sy, sz):
                L = np.kron(I2, pauli)
                ops.append((L, γdB / 4.0, 'dep_B'))

        # ---- Correlated ZZ dephasing ----------------------------------------
        γZZ = self._gamma_dep(self.T_ZZ)
        if γZZ > 0:
            L = np.kron(sz, sz)
            ops.append((L, γZZ, 'ZZ'))

        return ops

    def lindblad_operators(self):
        """Return list of (L_4x4, gamma) pairs."""
        return [(L, g) for L, g, _ in self._ops]

    def active_channels(self):
        """Return list of active channel names."""
        seen = []
        for _, _, name in self._ops:
            base = name.rstrip('_AB')
            if name not in seen:
                seen.append(name)
        return seen

    def summary(self) -> str:
        lines = ["TwoQubitNoiseModel", "=" * 44]
        for label, T1, T2, Td in [
            ('A', self.T1_A, self.T2_A, self.T_dep_A),
            ('B', self.T1_B, self.T2_B, self.T_dep_B),
        ]:
            γ1   = self._gamma1(T1)
            γφ   = self._gamma_phi(T1, T2)
            γd   = self._gamma_dep(Td)
            lines.append(f"  Qubit {label}: T1={T1:.3g}, T2={T2:.3g}, T_dep={Td:.3g}")
            lines.append(f"    γ₁={γ1:.4g}, γ_φ={γφ:.4g}, γ_dep={γd:.4g}")
        γZZ = self._gamma_dep(self.T_ZZ)
        lines.append(f"  ZZ dephasing: T_ZZ={self.T_ZZ:.3g}, γ_ZZ={γZZ:.4g}")
        lines.append(f"  Active channels: {self.active_channels()}")
        return "\n".join(lines)


# ============================================================================
# Convenience constructors
# ============================================================================

def noise_identical_qubits(T1: float, T2: float,
                            T_dep: float = np.inf,
                            T_ZZ: float  = np.inf) -> TwoQubitNoiseModel:
    """Both qubits with identical noise parameters."""
    return TwoQubitNoiseModel(
        T1_A=T1, T2_A=T2, T1_B=T1, T2_B=T2,
        T_dep_A=T_dep, T_dep_B=T_dep,
        T_ZZ=T_ZZ,
    )


def noise_amplitude_only(T1_A: float, T1_B: float) -> TwoQubitNoiseModel:
    """Pure amplitude damping on each qubit (T2 = 2*T1 → zero pure dephasing)."""
    return TwoQubitNoiseModel(
        T1_A=T1_A, T2_A=2*T1_A,
        T1_B=T1_B, T2_B=2*T1_B,
    )


def noise_dephasing_only(T2_A: float, T2_B: float) -> TwoQubitNoiseModel:
    """Pure dephasing only (T1 → ∞)."""
    return TwoQubitNoiseModel(T2_A=T2_A, T2_B=T2_B)


# ============================================================================
# Lindblad RHS for two-qubit system
# ============================================================================

def _dissipator(L: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Lindblad dissipator: D[L]ρ = LρL† - ½{L†L, ρ}."""
    Ld  = L.conj().T
    LdL = Ld @ L
    return L @ rho @ Ld - 0.5 * (LdL @ rho + rho @ LdL)


def lindblad_rhs_2q(
    rho: np.ndarray,
    H: np.ndarray,
    noise: TwoQubitNoiseModel,
) -> np.ndarray:
    """Lindblad master equation RHS for a two-qubit system.

    dρ/dt = -i[H, ρ] + Σ_k γ_k · D[L_k]ρ

    Parameters
    ----------
    rho   : (4,4) density matrix
    H     : (4,4) two-qubit Hamiltonian (ℏ=1)
    noise : TwoQubitNoiseModel

    Returns
    -------
    drho_dt : (4,4) complex
    """
    drho = -1j * (H @ rho - rho @ H)
    for L, gamma_k in noise.lindblad_operators():
        drho += gamma_k * _dissipator(L, rho)
    return drho


# ============================================================================
# ODE solver
# ============================================================================

def simulate_2q(
    rho0: np.ndarray,
    H: np.ndarray,
    noise: TwoQubitNoiseModel,
    t_max: float,
    n: int = 500,
    track_ent: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Optional[Dict]]:
    """Integrate the two-qubit Lindblad master equation.

    Parameters
    ----------
    rho0      : (4,4) initial density matrix
    H         : (4,4) Hamiltonian (time-independent, ℏ=1)
    noise     : TwoQubitNoiseModel
    t_max     : simulation duration
    n         : number of output time steps
    track_ent : if True, compute entanglement measures at each step

    Returns
    -------
    t      : (n,)      time axis
    rho_t  : (n,4,4)   density matrix trajectory
    ent    : dict of (n,) arrays - only if track_ent=True, else None
        'concurrence', 'negativity', 'entropy_A', 'entanglement_of_formation',
        'purity_A', 'purity_B'
    """
    if t_max <= 0:
        raise ValueError(f"t_max must be positive, got {t_max}")
    if rho0.shape != (4, 4):
        raise ValueError(f"rho0 must be (4,4), got {rho0.shape}")

    # Pack/unpack 4×4 complex → 32 real numbers
    def _pack(rho):
        return np.concatenate([rho.real.ravel(), rho.imag.ravel()])

    def _unpack(y):
        return y[:16].reshape(4, 4) + 1j * y[16:].reshape(4, 4)

    def _rhs(t, y):
        rho  = _unpack(y)
        drho = lindblad_rhs_2q(rho, H, noise)
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
        raise RuntimeError(f"Two-qubit ODE solver failed: {sol.message}")

    rho_t = np.array([_unpack(sol.y[:, i]) for i in range(n)])

    ent = track_entanglement(rho_t) if track_ent else None
    return sol.t, rho_t, ent


# ============================================================================
# Entanglement sudden death
# ============================================================================

def entanglement_sudden_death(
    bell_state: str = "phi_plus",
    T1_A: float = 10.0,
    T1_B: float = 10.0,
    T2_A: Optional[float] = None,
    T2_B: Optional[float] = None,
    t_max: float = 30.0,
    n: int = 400,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Demonstrate entanglement sudden death under local amplitude damping.

    Starting from a Bell state, local (independent) amplitude damping
    causes the concurrence to drop to zero in *finite* time - even though
    the individual qubit coherences decay asymptotically as exp(-t/T1).

    This is a hallmark of quantum correlations: they can be more fragile
    than classical ones, and more fragile than single-qubit coherence.

    Physical picture
    ----------------
    For the Werner state ρ = p|Φ+⟩⟨Φ+| + (1-p)I/4, under amplitude
    damping the concurrence is:
        C(t) = max(0, √(ρ₀₃² + ρ₁₂²) - √(ρ₀₀ρ₁₁) )
    The first term decreases as exp(-t/T1) while the denominator grows,
    causing C to reach zero at a finite t_ESD < ∞.

    Parameters
    ----------
    bell_state : starting Bell state name
    T1_A, T1_B : amplitude damping times
    T2_A, T2_B : dephasing times (default: 2*T1 = amplitude damping only)
    t_max      : simulation duration
    n          : time steps

    Returns
    -------
    t           : (n,) time axis
    concurrence : (n,) concurrence trajectory
    purity_A    : (n,) single-qubit purity of qubit A
    """
    T2_A = T2_A or 2.0 * T1_A
    T2_B = T2_B or 2.0 * T1_B

    rho0  = bell_state_dm(bell_state)
    H     = np.zeros((4, 4), dtype=complex)
    noise = TwoQubitNoiseModel(T1_A=T1_A, T2_A=T2_A, T1_B=T1_B, T2_B=T2_B)

    t, rho_t, ent = simulate_2q(rho0, H, noise, t_max, n, track_ent=True)

    return t, ent['concurrence'], ent['purity_A']


# ============================================================================
# Correlated ZZ noise: protects or destroys entanglement?
# ============================================================================

def correlated_vs_local_dephasing(
    T2: float = 8.0,
    T_ZZ: float = 8.0,
    t_max: float = 25.0,
    n: int = 400,
) -> Dict[str, np.ndarray]:
    """Compare local dephasing vs correlated ZZ dephasing on a Bell state.

    Local dephasing (each qubit to an independent bath) destroys the
    phase coherence and kills entanglement.

    Correlated ZZ dephasing (both qubits see the same phase noise)
    preserves |Φ+⟩ = (|00⟩+|11⟩)/√2 exactly - the ZZ operator
    commutes with |Φ+⟩ and |Φ-⟩ (both are eigenstates of σz⊗σz).

    This contrast reveals that the *structure* of noise, not just its
    strength, determines which states are decoherence-free.

    Parameters
    ----------
    T2    : local dephasing time for each qubit
    T_ZZ  : correlated ZZ dephasing time
    t_max : simulation duration

    Returns
    -------
    dict with keys 't', 'C_local', 'C_correlated', 'C_both'
    """
    rho0  = bell_state_dm('phi_plus')
    H     = np.zeros((4, 4), dtype=complex)

    # 1. Local dephasing only
    nm_local = noise_dephasing_only(T2, T2)
    t, _, ent_local = simulate_2q(rho0, H, nm_local, t_max, n)

    # 2. Correlated ZZ only
    nm_corr = TwoQubitNoiseModel(T_ZZ=T_ZZ)
    _, _, ent_corr = simulate_2q(rho0, H, nm_corr, t_max, n)

    # 3. Both simultaneously
    nm_both = TwoQubitNoiseModel(T2_A=T2, T2_B=T2, T_ZZ=T_ZZ)
    _, _, ent_both = simulate_2q(rho0, H, nm_both, t_max, n)

    return {
        't':            t,
        'C_local':      ent_local['concurrence'],
        'C_correlated': ent_corr['concurrence'],
        'C_both':       ent_both['concurrence'],
    }
