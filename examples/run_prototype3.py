"""
examples/run_prototype3.py
==========================
Demonstrates Prototype 3: full Bloch equations with T1 and T2 relaxation.

Generates four output files:
  1. prototype3_relaxation.png    - Mx/My/Mz/|M_perp| for a canonical case
  2. prototype3_limiting_cases.png - overlay of T1>>T2, T1=T2, T1≈T2 regimes
  3. prototype3_bloch_sphere.png  - 3-D Bloch sphere with decaying spiral
  4. prototype3_recovery.png      - Mz recovery from Mz0=0 (inversion recovery)

Usage:
    python examples/run_prototype3.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib; matplotlib.use("Agg")
import numpy as np

from src.core import (
    time_axis, simulate_bloch,
    plot_bloch_relaxation, plot_T1_T2_comparison,
)
from src.visualization import plot_bloch_sphere_trajectory

OUT = os.path.dirname(__file__)

# -- Shared parameters ---------------------------------------------------------
gamma  = 1.0
f0     = 0.5                    # MHz
omega0 = 2 * np.pi * f0         # rad/µs
B      = np.array([0.0, 0.0, omega0])   # gamma=1 so B0 = omega0
M0     = 1.0

print("=== Prototype 3 - Full Bloch Equations (T1 + T2) ===\n")

# -----------------------------------------------------------------------------
# Plot 1: canonical relaxation  (T1=20, T2=5 µs)
# -----------------------------------------------------------------------------
T1, T2 = 20.0, 5.0
t, Mx, My, Mz = simulate_bloch(
    M_init=[M0, 0.0, 0.0], gamma=gamma, B=B,
    T1=T1, T2=T2, M0=M0, t_max=4*T1, dt=0.05)

M_perp = np.sqrt(Mx**2 + My**2)
print(f"  Canonical run: T1={T1} µs, T2={T2} µs")
print(f"  |M_perp|(T2)  = {M_perp[np.argmin(np.abs(t-T2))]:.6f}  (expected {1/np.e:.6f} = 1/e)")
print(f"  Mz(T1)        = {Mz[np.argmin(np.abs(t-T1))]:.6f}  (expected {M0*(1-1/np.e):.6f} = M0*(1-1/e))")
print(f"  Mz(10·T1)     = {Mz[-1]:.6f}  (expected ~{M0:.6f})")

p1 = os.path.join(OUT, "prototype3_relaxation.png")
plot_bloch_relaxation(t, Mx, My, Mz, T1=T1, T2=T2, M0=M0, time_unit="µs",
    title=rf"Full Bloch Relaxation  ($T_1$={T1} µs, $T_2$={T2} µs)",
    save_path=p1)
print(f"\n  Saved → {p1}")

# -----------------------------------------------------------------------------
# Plot 2: limiting cases comparison
# -----------------------------------------------------------------------------
t_cmp = time_axis(5*T2, 0.02)

scenarios = []
configs = [
    # (label, T1, T2, color, ls)
    (r"$T_1=T_2=5$ µs  (isotropic)",         5.0,  5.0,  "#E63946", "-"),
    (r"$T_1=20$, $T_2=5$ µs  (typical NMR)",  20.0, 5.0,  "#2C7BB6", "-"),
    (r"$T_1=1\times10^6$, $T_2=5$ µs  ($T_1{\gg}T_2$)", 1e6, 5.0,  "#6A994E", "--"),
    (r"$T_1=T_2=50$ µs  (slow dephasing)",    50.0, 50.0, "#9B2226", ":"),
]
for label, T1c, T2c, col, ls in configs:
    _, Mx_c, My_c, Mz_c = simulate_bloch(
        M_init=[M0, 0.0, 0.0], gamma=gamma, B=B,
        T1=T1c, T2=T2c, M0=M0, t_max=5*T2, dt=0.02)
    scenarios.append(dict(Mx=Mx_c, My=My_c, Mz=Mz_c,
                          label=label, color=col, ls=ls, lw=1.9))

p2 = os.path.join(OUT, "prototype3_limiting_cases.png")
plot_T1_T2_comparison(t_cmp, scenarios, time_unit="µs", save_path=p2)
print(f"  Saved → {p2}")

# -----------------------------------------------------------------------------
# Plot 3: Bloch sphere - shows the inward spiral with T2 decay + Mz recovery
# -----------------------------------------------------------------------------
T1_s, T2_s = 20.0, 5.0
_, Mx_s, My_s, Mz_s = simulate_bloch(
    M_init=[M0, 0.0, 0.0], gamma=gamma, B=B,
    T1=T1_s, T2=T2_s, M0=M0, t_max=3*T1_s, dt=0.03)

p3 = os.path.join(OUT, "prototype3_bloch_sphere.png")
plot_bloch_sphere_trajectory(Mx_s, My_s, Mz_s,
    title=rf"Bloch Sphere — $T_1$={T1_s}, $T_2$={T2_s} µs",
    save_path=p3)
print(f"  Saved → {p3}")

# -----------------------------------------------------------------------------
# Plot 4: Inversion recovery  (Mz0 = -M0, no transverse component)
# -----------------------------------------------------------------------------
import matplotlib.pyplot as plt

T1_r = 20.0
t_r, _, _, Mz_r = simulate_bloch(
    M_init=[0.0, 0.0, -M0], gamma=gamma, B=np.array([0.,0.,0.]),
    T1=T1_r, T2=T1_r, M0=M0, t_max=5*T1_r, dt=0.1)

theory = M0 * (1 - 2*np.exp(-t_r / T1_r))   # inversion recovery formula

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(t_r, Mz_r,   color="#6A994E", lw=2.2, label=r"$M_z(t)$ (ODE)")
ax.plot(t_r, theory, "--", color="#E63946", lw=1.5, alpha=0.9,
        label=r"$M_0(1-2e^{-t/T_1})$  (analytic)")
ax.axhline(0,   color="black",  lw=0.7, alpha=0.4)
ax.axhline(M0,  color="#888888", lw=1.0, ls="--", alpha=0.6, label=rf"$M_0={M0}$")
ax.axhline(-M0, color="#888888", lw=1.0, ls="--", alpha=0.4)
ax.axvline(T1_r * np.log(2), color="#457B9D", lw=1.0, ls=":",
           label=rf"$t_{{null}} = T_1\ln 2 = {T1_r*np.log(2):.2f}$ µs")
ax.set_xlabel("Time (µs)", fontsize=12)
ax.set_ylabel(r"$M_z(t)$", fontsize=12)
ax.set_title(rf"Inversion Recovery — $M_z(0)=-M_0$, $T_1$={T1_r} µs", fontsize=13, fontweight="bold")
ax.legend(fontsize=9, framealpha=0.85)
ax.grid(True, ls="--", alpha=0.35)
ax.set_facecolor("#F9F9F9")
fig.patch.set_facecolor("white")
fig.tight_layout()

p4 = os.path.join(OUT, "prototype3_recovery.png")
fig.savefig(p4, dpi=150, bbox_inches="tight")
print(f"  Saved → {p4}")

max_err = np.max(np.abs(Mz_r - theory))
print(f"\n  Inversion recovery max error vs analytic: {max_err:.2e}")
print("\nDone.")