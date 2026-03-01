"""
examples/run_prototype1.py
==========================
Demonstrates Prototype 1: single-spin pure T2 coherence decay.

Usage:
    python examples/run_prototype1.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.core import simulate_simple_coherence, plot_coherence_decay

# -- Parameters --------------------------------------------------------------
T2    = 5   # µs   - coherence time
t_max = 20.0  # µs   - simulate 4 x T2
dt    = 0.05  # µs   - time step

# -- Simulate -----------------------------------------------------------------
t, L = simulate_simple_coherence(T2=T2, t_max=t_max, dt=dt)

# -- Quick verification prints ------------------------------------------------
print("=== Prototype 1 - Pure T₂ Decay ===")
print(f"  T2          = {T2} µs")
print(f"  Time points = {len(t)}")
print(f"  L(t=0)      = {L[0]:.6f}   (expected 1.000000)")

idx_T2 = np.argmin(np.abs(t - T2))
print(f"  L(t=T₂)     = {L[idx_T2]:.6f}   (expected {1/np.e:.6f}  = 1/e)")
print(f"  L(t=4·T₂)   = {L[np.argmin(np.abs(t - 4*T2))]:.6f}   (expected {np.exp(-4):.6f} = e⁻⁴)")

# -- Plot ---------------------------------------------------------------------
fig = plot_coherence_decay(
    t, L,
    T2=T2,
    title=rf"Single-Spin Coherence Decay  ($T_2 = {T2}$ µs)",
    time_unit="µs",
    save_path="examples/prototype1_decay.png",
)
print("\n  Plot saved → examples/prototype1_decay.png")
fig.show()