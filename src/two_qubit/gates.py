"""
gates.py — Single- and two-qubit quantum gates.
================================================

All gates are exact unitary matrices in the computational basis
|00⟩, |01⟩, |10⟩, |11⟩.

Single-qubit gates (2×2 unitaries)
------------------------------------
    I   : identity
    X   : Pauli-X (NOT, bit flip)         [[0,1],[1,0]]
    Y   : Pauli-Y                          [[0,-i],[i,0]]
    Z   : Pauli-Z (phase flip)             [[1,0],[0,-1]]
    H   : Hadamard  (|0⟩↔|+⟩)            [[1,1],[1,-1]]/√2
    S   : phase gate (Z^½)                 [[1,0],[0,i]]
    T   : π/8 gate  (Z^¼)                 [[1,0],[0,e^(iπ/4)]]
    Rx  : rotation about x-axis  e^(-iθσx/2)
    Ry  : rotation about y-axis  e^(-iθσy/2)
    Rz  : rotation about z-axis  e^(-iθσz/2)
    U3  : general single-qubit unitary (3 Euler angles)

Two-qubit gates (4×4 unitaries)
---------------------------------
    CNOT(ctrl, tgt) : controlled-NOT, flips target iff control=|1⟩
    CZ              : controlled-Z,   applies Z to target iff control=|1⟩
    SWAP            : swaps the two qubits
    iSWAP           : SWAP with an extra i phase on |01⟩ and |10⟩
    XX(θ)           : Ising XX coupling  e^(-iθ XX/2)
    CP(φ)           : controlled phase

Gate application
-----------------
    apply_gate_state(U, psi)  : U|ψ⟩
    apply_gate_dm(U, rho)     : U ρ U†
    apply_1q_gate(gate, q, psi/rho): embed 1-qubit gate on qubit q
    run_circuit(psi0, ops)    : execute a list of (gate_name, *args) instructions

Unitary evolution
------------------
    evolve_unitary(rho, H, t) : e^(-iHt) ρ e^(iHt)
"""

from __future__ import annotations

import numpy as np
from typing import Union, List, Tuple

from .states import (
    I2, I4, tensor, tensor_op_on_qubit, ket_to_dm,
    KET_00, KET_01, KET_10, KET_11,
)

# Type alias
State   = np.ndarray   # (4,) complex  or  (4,4) complex
Gate1Q  = np.ndarray   # (2,2) complex
Gate2Q  = np.ndarray   # (4,4) complex


# ============================================================================
# Single-qubit gate library
# ============================================================================

# Clifford / standard gates
GATE_I = np.eye(2, dtype=complex)
GATE_X = np.array([[0, 1], [1, 0]],  dtype=complex)
GATE_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
GATE_Z = np.array([[1, 0], [0, -1]], dtype=complex)
GATE_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
GATE_S = np.array([[1, 0], [0, 1j]], dtype=complex)
GATE_T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
GATE_S_DAG = GATE_S.conj().T
GATE_T_DAG = GATE_T.conj().T

# Rotation gates
def Rx(theta: float) -> Gate1Q:
    """Rotation by θ around x-axis: e^(-iθσx/2)."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j*s], [-1j*s, c]], dtype=complex)

def Ry(theta: float) -> Gate1Q:
    """Rotation by θ around y-axis: e^(-iθσy/2)."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)

def Rz(theta: float) -> Gate1Q:
    """Rotation by θ around z-axis: e^(-iθσz/2)."""
    return np.array([[np.exp(-1j*theta/2), 0],
                     [0, np.exp(1j*theta/2)]], dtype=complex)

def U3(theta: float, phi: float, lam: float) -> Gate1Q:
    """General single-qubit unitary (IBM Qiskit U3 convention).

    U3(θ,φ,λ) = [[cos(θ/2),         -e^(iλ)sin(θ/2)      ],
                  [e^(iφ)sin(θ/2),   e^(i(φ+λ))cos(θ/2)  ]]

    Special cases:
        U3(π, 0, π)      = X
        U3(π/2, 0, π)    = H
        U3(0, 0, θ)      = Rz(θ)  (up to global phase)
    """
    c, s = np.cos(theta/2), np.sin(theta/2)
    return np.array([
        [c,                      -np.exp(1j*lam)*s],
        [np.exp(1j*phi)*s,        np.exp(1j*(phi+lam))*c],
    ], dtype=complex)

def phase_gate(phi: float) -> Gate1Q:
    """Phase gate P(φ): |0⟩→|0⟩, |1⟩→e^(iφ)|1⟩."""
    return np.array([[1, 0], [0, np.exp(1j*phi)]], dtype=complex)

# Named gate registry for the circuit runner
_1Q_GATES = {
    'I':  GATE_I, 'X': GATE_X, 'Y': GATE_Y, 'Z': GATE_Z,
    'H':  GATE_H, 'S': GATE_S, 'T': GATE_T,
    'Sd': GATE_S_DAG, 'Td': GATE_T_DAG,
}


# ============================================================================
# Two-qubit gate library
# ============================================================================

def CNOT(control: int = 0, target: int = 1) -> Gate2Q:
    """Controlled-NOT gate.

    Flips the target qubit if and only if the control qubit is |1⟩.

    Matrix in |00⟩,|01⟩,|10⟩,|11⟩ basis (control=0, target=1):
        [[1,0,0,0],
         [0,1,0,0],
         [0,0,0,1],
         [0,0,1,0]]

    The controlled-NOT is universal for quantum computing when combined
    with arbitrary single-qubit rotations.  It creates entanglement:
        H⊗I → CNOT transforms |00⟩ → |Φ+⟩ (Bell state preparation).

    Parameters
    ----------
    control : qubit index of the control (0 or 1)
    target  : qubit index of the target  (0 or 1)

    Returns
    -------
    (4,4) unitary
    """
    if control == target:
        raise ValueError("control and target must be different qubits")
    if control == 0 and target == 1:
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ], dtype=complex)
    else:   # control=1, target=0 (reverse CNOT)
        return np.array([
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
        ], dtype=complex)


def CZ() -> Gate2Q:
    """Controlled-Z gate.

    Applies a Z phase to the target qubit iff control=|1⟩.
    Symmetric: CZ = CZ† (same with qubits swapped).

        [[1,0,0, 0],
         [0,1,0, 0],
         [0,0,1, 0],
         [0,0,0,-1]]
    """
    return np.diag([1, 1, 1, -1]).astype(complex)


def SWAP_gate() -> Gate2Q:
    """SWAP gate: exchanges the two qubits.

    |01⟩ ↔ |10⟩,  others unchanged.

    Can be decomposed as 3 CNOTs: CNOT(0,1)·CNOT(1,0)·CNOT(0,1).
    """
    return np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ], dtype=complex)


def iSWAP_gate() -> Gate2Q:
    """iSWAP gate: SWAP with an extra i phase on the swapped components.

    Used as a native gate on superconducting hardware.

        [[1, 0,  0,  0],
         [0, 0,  i,  0],
         [0, i,  0,  0],
         [0, 0,  0,  1]]
    """
    return np.array([
        [1, 0,  0,  0],
        [0, 0,  1j, 0],
        [0, 1j, 0,  0],
        [0, 0,  0,  1],
    ], dtype=complex)


def XX_gate(theta: float) -> Gate2Q:
    """Ising XX coupling: e^(-iθ·σx⊗σx / 2).

    Used in trapped-ion quantum computers (Mølmer-Sørensen gate).
    At θ=π/2 creates maximal entanglement from |00⟩.

    Parameters
    ----------
    theta : coupling angle

    Returns
    -------
    (4,4) unitary
    """
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([
        [c,    0,    0,  -1j*s],
        [0,    c, -1j*s,    0],
        [0, -1j*s,  c,      0],
        [-1j*s, 0,   0,     c],
    ], dtype=complex)


def CP_gate(phi: float) -> Gate2Q:
    """Controlled-Phase (CP) gate.

    Applies phase e^(iφ) to |11⟩ only.  Reduces to CZ at φ=π.
    """
    return np.diag([1, 1, 1, np.exp(1j*phi)]).astype(complex)


# Named two-qubit gate registry
_2Q_GATES = {
    'CNOT':  CNOT(),
    'CX':    CNOT(),
    'rCNOT': CNOT(control=1, target=0),
    'CZ':    CZ(),
    'SWAP':  SWAP_gate(),
    'iSWAP': iSWAP_gate(),
}


# ============================================================================
# Gate application
# ============================================================================

def apply_gate_state(U: Gate2Q, psi: np.ndarray) -> np.ndarray:
    """Apply unitary U to state vector: |ψ'⟩ = U|ψ⟩.

    Parameters
    ----------
    U   : (4,4) unitary
    psi : (4,) complex state vector

    Returns
    -------
    (4,) complex state vector (normalised)
    """
    return U @ np.asarray(psi, dtype=complex)


def apply_gate_dm(U: Gate2Q, rho: np.ndarray) -> np.ndarray:
    """Apply unitary U to density matrix: ρ' = U ρ U†.

    Parameters
    ----------
    U   : (4,4) unitary
    rho : (4,4) density matrix

    Returns
    -------
    (4,4) density matrix
    """
    return U @ rho @ U.conj().T


def apply_1q_gate(
    gate: Union[str, Gate1Q],
    qubit: int,
    state: State,
) -> State:
    """Apply a single-qubit gate to qubit *qubit* in a two-qubit system.

    Automatically embeds the gate as  G⊗I  (qubit=0) or  I⊗G  (qubit=1).
    Accepts both state vectors and density matrices.

    Parameters
    ----------
    gate  : (2,2) unitary  OR  string key from the gate registry
              ('I','X','Y','Z','H','S','T','Sd','Td')
    qubit : 0 (qubit A) or 1 (qubit B)
    state : (4,) state vector  OR  (4,4) density matrix

    Returns
    -------
    same type and shape as *state*
    """
    if isinstance(gate, str):
        if gate not in _1Q_GATES:
            raise ValueError(f"Unknown gate '{gate}'. "
                             f"Available: {list(_1Q_GATES)}")
        G = _1Q_GATES[gate]
    else:
        G = np.asarray(gate, dtype=complex)

    U2 = tensor_op_on_qubit(G, qubit, n_qubits=2)
    state = np.asarray(state, dtype=complex)

    if state.ndim == 1:   # state vector
        return apply_gate_state(U2, state)
    else:                  # density matrix
        return apply_gate_dm(U2, state)


# ============================================================================
# Bell state preparation circuits
# ============================================================================

def prepare_bell_state(which: str = "phi_plus") -> Tuple[np.ndarray, np.ndarray]:
    """Prepare a Bell state from |00⟩ using H + CNOT.

    The canonical Bell state circuit:
        |00⟩ → H⊗I → CNOT → |Φ+⟩

    Other Bell states are reached by additional single-qubit rotations:
        |Φ-⟩: Z on qubit A after the circuit
        |Ψ+⟩: X on qubit B after the circuit
        |Ψ-⟩: Z on qubit A, X on qubit B

    Parameters
    ----------
    which : 'phi_plus', 'phi_minus', 'psi_plus', 'psi_minus'

    Returns
    -------
    psi : (4,) Bell state vector
    rho : (4,4) Bell state density matrix
    """
    # Start in |00⟩
    psi = KET_00.copy().astype(complex)

    # H on qubit A
    psi = apply_1q_gate(GATE_H, qubit=0, state=psi)

    # CNOT(0→1)
    psi = apply_gate_state(CNOT(0, 1), psi)

    # Corrections
    corrections = {
        'phi_plus':  [],
        'phi_minus': [('Z', 0)],
        'psi_plus':  [('X', 1)],
        'psi_minus': [('Z', 0), ('X', 1)],
    }
    if which not in corrections:
        raise ValueError(f"Unknown Bell state '{which}'")

    for gname, q in corrections[which]:
        psi = apply_1q_gate(gname, q, psi)

    rho = ket_to_dm(psi)
    return psi, rho


# ============================================================================
# Circuit runner
# ============================================================================

def run_circuit(
    psi0: np.ndarray,
    ops: List,
    return_all: bool = False,
) -> Union[np.ndarray, List[np.ndarray]]:
    """Execute a quantum circuit as an ordered list of gate operations.

    Each operation is a tuple:
        ('1q',  gate, qubit)               single-qubit gate on qubit 0 or 1
        ('2q',  gate)                       two-qubit gate (4×4 unitary)
        ('2q',  gate_name)                  named gate from registry
        ('cnot', control, target)           CNOT shorthand
        ('rx',  theta, qubit)              Rx rotation
        ('ry',  theta, qubit)              Ry rotation
        ('rz',  theta, qubit)              Rz rotation
        ('u3',  theta, phi, lam, qubit)    U3 rotation

    Parameters
    ----------
    psi0       : (4,) initial state vector
    ops        : list of operation tuples (see above)
    return_all : if True, return list of states after each operation

    Returns
    -------
    psi_final : (4,) final state vector
    OR
    states    : list of (4,) state vectors [psi0, psi_after_op1, ...]
    """
    psi    = np.asarray(psi0, dtype=complex).copy()
    states = [psi.copy()] if return_all else None

    for op in ops:
        kind = op[0].lower()

        if kind == '1q':
            _, gate, qubit = op
            psi = apply_1q_gate(gate, qubit, psi)

        elif kind == '2q':
            gate = op[1]
            if isinstance(gate, str):
                if gate not in _2Q_GATES:
                    raise ValueError(f"Unknown 2Q gate '{gate}'")
                U = _2Q_GATES[gate]
            else:
                U = np.asarray(gate, dtype=complex)
            psi = apply_gate_state(U, psi)

        elif kind == 'cnot' or kind == 'cx':
            _, ctrl, tgt = op
            psi = apply_gate_state(CNOT(ctrl, tgt), psi)

        elif kind == 'rx':
            _, theta, qubit = op
            psi = apply_1q_gate(Rx(theta), qubit, psi)

        elif kind == 'ry':
            _, theta, qubit = op
            psi = apply_1q_gate(Ry(theta), qubit, psi)

        elif kind == 'rz':
            _, theta, qubit = op
            psi = apply_1q_gate(Rz(theta), qubit, psi)

        elif kind == 'u3':
            _, theta, phi, lam, qubit = op
            psi = apply_1q_gate(U3(theta, phi, lam), qubit, psi)

        elif kind == 'cz':
            psi = apply_gate_state(CZ(), psi)

        elif kind == 'swap':
            psi = apply_gate_state(SWAP_gate(), psi)

        elif kind == 'h':
            _, qubit = op
            psi = apply_1q_gate(GATE_H, qubit, psi)

        else:
            raise ValueError(f"Unknown operation type '{kind}'")

        if return_all:
            states.append(psi.copy())

    return states if return_all else psi


# ============================================================================
# Unitary (Hamiltonian) evolution
# ============================================================================

def evolve_unitary(
    rho: np.ndarray,
    H: np.ndarray,
    t: float,
) -> np.ndarray:
    """Unitary evolution: ρ(t) = e^(-iHt) ρ e^(+iHt).

    Uses matrix exponentiation via eigendecomposition (exact for
    time-independent H, ℏ=1 units).

    Parameters
    ----------
    rho : (N,N) density matrix
    H   : (N,N) Hermitian Hamiltonian
    t   : evolution time

    Returns
    -------
    (N,N) evolved density matrix
    """
    evals, evecs = np.linalg.eigh(H)
    U    = evecs @ np.diag(np.exp(-1j * evals * t)) @ evecs.conj().T
    return U @ rho @ U.conj().T


def two_qubit_hamiltonian(
    Jx: float = 0.0,
    Jy: float = 0.0,
    Jz: float = 0.0,
    hA: float = 0.0,
    hB: float = 0.0,
) -> np.ndarray:
    """Heisenberg-type two-qubit Hamiltonian.

    H = Jx·(σx⊗σx) + Jy·(σy⊗σy) + Jz·(σz⊗σz)
        + hA·(σz⊗I) + hB·(I⊗σz)

    Parameters
    ----------
    Jx, Jy, Jz : exchange coupling strengths
    hA, hB     : local Zeeman fields on qubits A and B

    Returns
    -------
    (4,4) Hermitian matrix
    """
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)

    H = (Jx * np.kron(sx, sx) +
         Jy * np.kron(sy, sy) +
         Jz * np.kron(sz, sz) +
         hA * np.kron(sz, np.eye(2, dtype=complex)) +
         hB * np.kron(np.eye(2, dtype=complex), sz))
    return H


def gate_fidelity(U_ideal: np.ndarray, U_actual: np.ndarray) -> float:
    """Average gate fidelity between two unitaries.

    F = |Tr(U_ideal† U_actual)|² / N²

    where N is the dimension.  F=1 means identical gates.

    Parameters
    ----------
    U_ideal, U_actual : (N,N) unitary matrices

    Returns
    -------
    float in [0, 1]
    """
    N = U_ideal.shape[0]
    overlap = np.trace(U_ideal.conj().T @ U_actual)
    return float(abs(overlap)**2 / N**2)
