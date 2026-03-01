"""
examples/run_prototype2.py
==========================
Demonstrates Prototype 2: Bloch vector + Larmor precession + T2 decay.

Generates three output files:
  1. prototype2_components.png   - Mx, My, Mz, |M_perp| vs time (4-panel)
  2. prototype2_bloch_sphere.png - 3-D Bloch sphere trajectory
  3. prototype2_bloch.gif        - animated Bloch vector (optional)

Usage:
    python examples/run_prototype2.py
    python examples/run_prototype2.py --animate   (slower, also renders GIF)
"""

import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import matplotlib
matplotlib.use("Agg")
import numpy as np

from src.core import time_axis, bloch_precession, plot_bloch_components
from src.visualization import plot_bloch_sphere_trajectory, animate_bloch_trajectory

# -- Parameters ---------------------------------------------------------------
T2     = 5.0                   # µs
f0     = 0.5                   # MHz  →  0.5 cycles / µs
omega0 = 2 * np.pi * f0        # rad / µs
M0     = 1.0
t_max  = 4 * T2                # simulate 4 x T2
dt     = 0.01                  # µs

# -- Build time axis & compute Bloch vectors -----------------------------------
t = time_axis(t_max, dt)

Mx_nodecay, My_nodecay, Mz_nodecay = bloch_precession(
    t, M0=M0, omega0=omega0, T2=None)

Mx, My, Mz = bloch_precession(
    t, M0=M0, omega0=omega0, T2=T2)

# -- Verification printout -----------------------------------------------------
print("=== Prototype 2 - Bloch Vector & Larmor Precession ===\n")
print(f"  T2     = {T2} µs       omega0 = {omega0:.4f} rad/µs  ({f0} MHz)")
print(f"  t_max  = {t_max} µs    dt     = {dt} µs   ({len(t)} points)\n")

print("  -- Without T2 (pure precession) --")
M_perp_nd = np.sqrt(Mx_nodecay**2 + My_nodecay**2)
print(f"  |M_perp| at t=0   : {M_perp_nd[0]:.10f}   (expected {M0:.10f})")
print(f"  |M_perp| at t=T2  : {M_perp_nd[np.argmin(np.abs(t-T2))]:.10f}   (expected {M0:.10f})")
print(f"  |M_perp| at t=end : {M_perp_nd[-1]:.10f}   (expected {M0:.10f})")
drift = np.max(np.abs(M_perp_nd - M0))
print(f"  Max drift from M0 : {drift:.2e}   (should be < 1e-12)\n")

print("  -- With T2 decay --")
M_perp = np.sqrt(Mx**2 + My**2)
idx_T2 = np.argmin(np.abs(t - T2))
print(f"  |M_perp|(0)       : {M_perp[0]:.8f}   (expected {M0:.8f})")
print(f"  |M_perp|(T2)      : {M_perp[idx_T2]:.8f}   (expected {1/np.e:.8f} = 1/e)")
print(f"  |M_perp|(4·T2)    : {M_perp[-1]:.8f}   (expected {np.exp(-4):.8f} = e^-4)")
max_err = np.max(np.abs(M_perp - M0 * np.exp(-t/T2)))
print(f"  Max |M_perp| error: {max_err:.2e}   (should be < 1e-10)\n")

# -- Plot 1: four-panel component figure ---------------------------------------
out_dir = os.path.join(os.path.dirname(__file__))
p_comp  = os.path.join(out_dir, "prototype2_components.png")

fig_comp = plot_bloch_components(
    t, Mx, My, Mz,
    T2=T2, omega0=omega0,
    time_unit="µs",
    title=rf"Bloch Components — $\omega_0$ = 2πx{f0} MHz,  $T_2$ = {T2} µs",
    save_path=p_comp,
)
print(f"  Saved → {p_comp}")

# -- Plot 2: Bloch sphere trajectory ------------------------------------------
p_sphere = os.path.join(out_dir, "prototype2_bloch_sphere.png")

fig_sphere = plot_bloch_sphere_trajectory(
    Mx, My, Mz,
    title=rf"Bloch Sphere — $\omega_0$=2πx{f0} MHz,  $T_2$={T2} µs",
    save_path=p_sphere,
)
print(f"  Saved → {p_sphere}")

# -- Plot 3 (optional): animation ---------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--animate", action="store_true")
args, _ = parser.parse_known_args()

if args.animate:
    p_gif = os.path.join(out_dir, "prototype2_bloch.gif")
    print("\n  Rendering animation (this takes ~10 s) …")
    anim = animate_bloch_trajectory(
        Mx, My, Mz,
        stride=6, interval=30,
        title=rf"Bloch Vector — $T_2$={T2} µs",
        save_path=p_gif,
        writer="pillow",
    )
    print(f"  Saved → {p_gif}")

print("\nDone.")