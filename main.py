"""
app.py – Spin Coherence Simulator (Interactive Edition)
========================================================
UI only. All physics/simulation logic lives in src/.

src/core.py           — Bloch equations, ODE solver
src/ensemble.py       — inhomogeneous spin ensemble
src/sequences.py      — Hahn echo, CPMG
src/rabi.py           — Rabi oscillation, chevron, analytic helpers
src/ramsey.py         — Ramsey interference, fringe fitting, sensitivity
src/density_matrix.py — Lindblad master equation, open quantum systems
src/fitting.py        — curve fitting, model registry
src/two_qubit/        — Bell states, CNOT, entanglement, two-qubit Lindblad

Run:  streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from src.sequences import cpmg_sequence
from src.rabi import (
    run_rabi, analytic_chevron,
    analytic_rabi_population, pi_pulse_time, max_population_inversion,
)
from src.ramsey import (
    sweep_ramsey, analytic_ramsey_population,
    fit_ramsey_fringes, ramsey_sensitivity,
)
from src.density_matrix import (
    simulate_dm, ground_state_dm, superposition_dm, mixed_state_dm,
    hamiltonian_rotating, hamiltonian_free_evolution,
    dm_to_bloch, bloch_to_dm, purity, population,
    bloch_vs_dm_error,
)
from src.fitting import (
    fit_T2_to_data, fit_multi_to_data,
    generate_synthetic_data, MODEL_REGISTRY,
)

# ── Two-qubit imports ─────────────────────────────────────────────────────────
from src.two_qubit.states import (
    bell_state, all_bell_states, ket_to_dm, product_state,
    random_pure_state, state_summary, KET_00,
)
from src.two_qubit.gates import (
    CNOT, CZ, SWAP_gate, iSWAP_gate, XX_gate, prepare_bell_state,
    run_circuit, evolve_unitary, two_qubit_hamiltonian,
    GATE_H, GATE_X, GATE_Y, GATE_Z,
)
from src.two_qubit.entanglement import (
    concurrence, entanglement_entropy, negativity, logarithmic_negativity,
    von_neumann_entropy, partial_trace, entanglement_summary,
    track_entanglement, chsh_value, entanglement_of_formation,
    schmidt_decomposition,
)
from src.two_qubit.lindblad2q import (
    TwoQubitNoiseModel, simulate_2q,
    entanglement_sudden_death, correlated_vs_local_dephasing,
    noise_identical_qubits, noise_amplitude_only, noise_dephasing_only,
)


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spin Coherence Simulator",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    IS_DARK = (st.get_option("theme.base") == "dark")
except Exception:
    IS_DARK = False

TMPL   = "plotly_dark"  if IS_DARK else "plotly_white"
C_BLUE = "#4a9eda"      if IS_DARK else "#2C7BB6"
C_RED  = "#ff6b7a"      if IS_DARK else "#E63946"
C_GREEN= "#7fd77f"      if IS_DARK else "#6A994E"
C_ORNG = "#ffb347"      if IS_DARK else "#F4A261"
C_PURP = "#c792ea"      if IS_DARK else "#7B2D8B"
C_TEAL = "#4dd0e1"      if IS_DARK else "#00838F"
C_GREY = "#aaaaaa"

st.markdown("""
<style>
    .block-container { padding-top: 1.2rem; padding-bottom: 0.5rem; }
    h1 { font-size: 1.5rem; font-weight: 700; }
    .section-divider { border-top: 1px solid rgba(128,128,128,0.2);
                       margin: 0.5rem 0 0.7rem 0; }
</style>
""", unsafe_allow_html=True)

MODEL_LABELS = {
    "simple_T2":    "L(t) = exp(−t/T₂)",
    "gaussian_fid": "L(t) = exp(−t/T₂) · exp(−σ²t²/2)",
    "hahn_echo":    "A(τ) = exp(−2τ/T₂)",
    "T1_recovery":  "Mz(t) = M0·(1 − exp(−t/T₁))",
    "stretched_T2": "L(t) = exp(−(t/T₂)^β)",
}

EXPERIMENTS = [
    "Single Spin (Bloch)",
    "Ensemble FID",
    "Hahn Echo",
    "CPMG Train",
    "Echo Sweep",
    "FID vs Echo (T₂* comparison)",
    "Rabi Oscillation",
    "Ramsey Interference",
    "Density Matrix",
    "Two-Qubit: Bell States & Gates",
    "Two-Qubit: Noise & Entanglement",
]

BELL_LABELS = {
    "phi_plus":  "|Φ+⟩ = (|00⟩ + |11⟩)/√2",
    "phi_minus": "|Φ-⟩ = (|00⟩ − |11⟩)/√2",
    "psi_plus":  "|Ψ+⟩ = (|01⟩ + |10⟩)/√2",
    "psi_minus": "|Ψ-⟩ = (|01⟩ − |10⟩)/√2",
}

BASIS_LABELS_2Q = ["|00⟩", "|01⟩", "|10⟩", "|11⟩"]


# ============================================================================
# Analytic solutions — no ODE, run in microseconds
# ============================================================================

def _analytic_single_spin(omega0, T1, T2, t_max, n=1200):
    t  = np.linspace(0, t_max, n)
    Mx =  np.cos(omega0 * t) * np.exp(-t / T2)
    My = -np.sin(omega0 * t) * np.exp(-t / T2)
    Mz =  1 - np.exp(-t / T1)
    return t, Mx, My, Mz


def _analytic_ensemble_fid(omega0, sigma, T1, T2, t_max, n=1200):
    t        = np.linspace(0, t_max, n)
    envelope = np.exp(-t / T2) * np.exp(-0.5 * sigma**2 * t**2)
    Mx       =  np.cos(omega0 * t) * envelope
    My       = -np.sin(omega0 * t) * envelope
    Mz       =  1 - np.exp(-t / T1)
    return t, Mx, My, Mz


def _analytic_hahn_echo(omega0, T1, T2, tau, n=1200):
    half = n // 2
    t1   = np.linspace(0,   tau,   half,     endpoint=False)
    t2   = np.linspace(tau, 2*tau, half + 1, endpoint=True)
    Mx1  =  np.cos(omega0 * t1) * np.exp(-t1 / T2)
    My1  = -np.sin(omega0 * t1) * np.exp(-t1 / T2)
    Mz1  =  1 - np.exp(-t1 / T1)
    Mx2  =  np.cos(omega0 * (2*tau - t2)) * np.exp(-t2 / T2)
    My2  =  np.sin(omega0 * (2*tau - t2)) * np.exp(-t2 / T2)
    Mz2  =  1 - np.exp(-t2 / T1)
    return (np.concatenate([t1, t2]),
            np.concatenate([Mx1, Mx2]),
            np.concatenate([My1, My2]),
            np.concatenate([Mz1, Mz2]))


def _analytic_echo_sweep(T2, tau_min, tau_max, n=500):
    tau  = np.linspace(tau_min, tau_max, n)
    amps = np.exp(-2 * tau / T2)
    return 2 * tau, amps


def _analytic_fid_vs_echo(sigma, T2, tau, t_max, n=1200):
    t       = np.linspace(0, t_max, n)
    fid_env = np.exp(-t / T2) * np.exp(-0.5 * sigma**2 * t**2)
    t2_only = np.exp(-t / T2)
    return t, fid_env, t2_only


# ============================================================================
# Cached ODE wrappers
# ============================================================================

@st.cache_data(show_spinner=False)
def _run_cpmg(gamma, omega0, T1, T2, tau, n_echoes, dt):
    B = np.array([0.0, 0.0, omega0 / gamma])
    return cpmg_sequence(
        gamma=gamma, B=B, T1=T1, T2=T2, M0=1.0,
        tau=tau, n_echoes=n_echoes, dt=dt)


@st.cache_data(show_spinner=False)
def _run_rabi(omega_rabi, delta, T1, T2, t_max):
    return run_rabi(omega_rabi=omega_rabi, delta=delta,
                    T1=T1, T2=T2, t_max=t_max)


@st.cache_data(show_spinner=False)
def _run_ramsey_sweep(delta, T1, T2, T_free_max, n_sweep):
    return sweep_ramsey(delta=delta, T1=T1, T2=T2,
                        T_free_max=T_free_max, n_sweep=n_sweep)


@st.cache_data(show_spinner=False)
def _run_dm(rho0_tag, H_flat, T1, T2, t_max, n):
    H = np.array(H_flat, dtype=complex).reshape(2, 2)
    rho0_map = {
        "ground":        ground_state_dm(),
        "superposition": superposition_dm(theta=np.pi/2, phi=0.0),
        "mixed_0.5":     mixed_state_dm(0.5),
        "mixed_0.0":     mixed_state_dm(0.0),
    }
    rho0 = rho0_map.get(rho0_tag, ground_state_dm())
    return simulate_dm(rho0, H, T1, T2, t_max, n=n)


@st.cache_data(show_spinner=False)
def _run_dm_validation(omega_rabi, delta, T1, T2, t_max):
    return bloch_vs_dm_error(omega_rabi, delta, T1, T2, t_max, n=400)


# ── Two-qubit cached runners ─────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _run_2q_noise(T1_A, T2_A, T1_B, T2_B, T_dep_A, T_dep_B, T_ZZ,
                  bell_name, t_max, n):
    """Integrate two-qubit Lindblad from a Bell state with given noise."""
    rho0  = ket_to_dm(bell_state(bell_name))
    H     = np.zeros((4, 4), dtype=complex)
    nm    = TwoQubitNoiseModel(
        T1_A=T1_A, T2_A=T2_A,
        T1_B=T1_B, T2_B=T2_B,
        T_dep_A=T_dep_A, T_dep_B=T_dep_B,
        T_ZZ=T_ZZ,
    )
    t, rho_t, ent = simulate_2q(rho0, H, nm, t_max=t_max, n=n)
    # Serialise rho_t as flat float array for cache hashing
    return t, rho_t.real, rho_t.imag, ent


@st.cache_data(show_spinner=False)
def _run_esd(bell_name, T1_A, T1_B, T2_A, T2_B, t_max, n):
    rho0  = ket_to_dm(bell_state(bell_name))
    H     = np.zeros((4, 4), dtype=complex)
    noise = TwoQubitNoiseModel(T1_A=T1_A, T2_A=T2_A, T1_B=T1_B, T2_B=T2_B)
    t, rho_t, ent = simulate_2q(rho0, H, noise, t_max=t_max, n=n)
    return t, ent['concurrence'], ent['purity_A']


@st.cache_data(show_spinner=False)
def _run_correlated_comparison(T2, T_ZZ, t_max, n):
    return correlated_vs_local_dephasing(T2=T2, T_ZZ=T_ZZ, t_max=t_max, n=n)


@st.cache_data(show_spinner=False)
def _run_xx_sweep(n_angles):
    """Concurrence of XX(θ)|00⟩ as θ sweeps 0→π."""
    angles = np.linspace(0, np.pi, n_angles)
    concs  = np.empty(n_angles)
    for i, theta in enumerate(angles):
        psi  = XX_gate(theta) @ KET_00.astype(complex)
        rho  = ket_to_dm(psi)
        concs[i] = concurrence(rho)
    return angles, concs


# ============================================================================
# Shared Plotly helpers
# ============================================================================

def _base_layout(title="", height=420):
    return dict(
        title=title,
        template=TMPL,
        hovermode="x unified",
        height=height,
        margin=dict(l=60, r=20, t=55, b=50),
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.02, xanchor="right", x=1),
    )


def _fig_bloch_components(t, Mx, My, Mz, T2, title):
    fig = make_subplots(rows=1, cols=3, shared_yaxes=False,
                        subplot_titles=("Mx(t)", "My(t)", "Mz(t)"))
    env = np.exp(-t / T2)
    for col, data, color, name in [
        (1, Mx, C_BLUE,  "Mx"),
        (2, My, C_RED,   "My"),
        (3, Mz, C_GREEN, "Mz"),
    ]:
        fig.add_trace(go.Scatter(x=t, y=data, name=name,
                                 line=dict(color=color, width=2)),
                      row=1, col=col)
    for col in (1, 2):
        for sign in (1, -1):
            fig.add_trace(go.Scatter(x=t, y=sign*env, showlegend=False,
                                     line=dict(color=C_GREY, dash="dash", width=1),
                                     hoverinfo="skip"),
                          row=1, col=col)
    fig.update_xaxes(title_text="Time (µs)")
    fig.update_yaxes(title_text="Magnetisation", row=1, col=1)
    fig.update_layout(title=title, template=TMPL, height=360,
                      hovermode="x unified",
                      margin=dict(l=60, r=20, t=65, b=50))
    return fig


def _fig_echo_detail(t, Mx, My, tau, T2):
    M_perp   = np.sqrt(Mx**2 + My**2)
    echo_amp = float(np.interp(2*tau, t, M_perp))
    expected = float(np.exp(-2*tau / T2))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=M_perp, name="|M⊥|(t)",
                             line=dict(color=C_BLUE, width=2.5)))
    fig.add_trace(go.Scatter(x=t, y=np.exp(-t/T2),
                             name=f"exp(−t/T₂)  T₂={T2} µs",
                             line=dict(color=C_GREY, dash="dash", width=1.3)))
    fig.add_vline(x=tau,   line_dash="dot",  line_color=C_RED,
                  annotation_text="π pulse", annotation_position="top right")
    fig.add_vline(x=2*tau, line_dash="dash", line_color=C_GREEN,
                  annotation_text=f"echo  {echo_amp:.4f}",
                  annotation_position="top left")
    fig.add_trace(go.Scatter(
        x=[2*tau], y=[echo_amp], mode="markers",
        marker=dict(color=C_GREEN, size=12, symbol="star"),
        name=f"Echo = {echo_amp:.4f}  (expected {expected:.4f})"))
    fig.update_layout(**_base_layout(f"Hahn Echo  τ={tau} µs  T₂={T2} µs"))
    fig.update_xaxes(title_text="Time (µs)")
    fig.update_yaxes(title_text="|M⊥|", range=[-0.05, 1.1])
    return fig


def _fig_echo_sweep(two_tau, amps, T2):
    log_A  = np.log(np.clip(amps, 1e-10, None))
    coeffs = np.polyfit(two_tau, log_A, 1)
    T2_fit = -1.0 / coeffs[0]
    tt = np.linspace(two_tau[0], two_tau[-1], 600)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tt, y=np.exp(-tt / T2),
                             name=f"exp(−2τ/T₂)  T₂={T2} µs",
                             line=dict(color=C_BLUE, dash="dash", width=1.8)))
    fig.add_trace(go.Scatter(x=tt, y=np.exp(coeffs[0]*tt + coeffs[1]),
                             name=f"Fitted  T₂ = {T2_fit:.2f} µs",
                             line=dict(color=C_GREEN, dash="dot", width=1.8)))
    fig.add_trace(go.Scatter(x=two_tau, y=amps, name="Echo amplitudes",
                             mode="markers+lines",
                             marker=dict(color=C_RED, size=7),
                             line=dict(color=C_RED, width=1)))
    fig.add_hline(y=1/np.e, line_dash="dot", line_color=C_GREY,
                  annotation_text="1/e", annotation_position="right")
    fig.update_layout(**_base_layout("Echo Amplitude vs 2τ"))
    fig.update_xaxes(title_text="Echo time 2τ (µs)")
    fig.update_yaxes(title_text="Echo amplitude", range=[-0.02, 1.1])
    return fig, T2_fit


def _fig_fid_vs_echo(t, fid_env, t2_only, sigma, T2, tau):
    echo_amp = float(np.exp(-2*tau / T2))
    fid_at   = float(np.exp(-2*tau/T2) * np.exp(-0.5*sigma**2*(2*tau)**2))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=fid_env,
                             name=f"FID envelope  σ={sigma:.2f} rad/µs",
                             line=dict(color=C_RED, width=2.5)))
    fig.add_trace(go.Scatter(x=t, y=t2_only,
                             name=f"exp(−t/T₂)  T₂={T2} µs",
                             line=dict(color=C_BLUE, dash="dash", width=1.8)))
    fig.add_trace(go.Scatter(x=[2*tau], y=[echo_amp], mode="markers",
                             marker=dict(color=C_GREEN, size=14, symbol="star"),
                             name=f"Echo amp at 2τ = {echo_amp:.4f}"))
    fig.add_trace(go.Scatter(x=[2*tau], y=[fid_at], mode="markers",
                             marker=dict(color=C_RED, size=12, symbol="x"),
                             name=f"FID at 2τ = {fid_at:.4f}"))
    fig.add_vline(x=2*tau, line_dash="dot", line_color=C_GREY,
                  annotation_text=f"2τ = {2*tau} µs",
                  annotation_position="top right")
    fig.add_hline(y=1/np.e, line_dash="dot", line_color=C_GREY,
                  annotation_text="1/e", annotation_position="right")
    fig.update_layout(**_base_layout(
        f"FID vs Echo  —  T₂* vs T₂   σ={sigma:.2f}  τ={tau} µs"))
    fig.update_xaxes(title_text="Time (µs)")
    fig.update_yaxes(title_text="|⟨M⊥⟩|", range=[-0.02, 1.1])
    return fig


def _fig_fid_ensemble(t, Mx, My, sigma, T2):
    M_perp = np.sqrt(Mx**2 + My**2)
    t_s    = np.linspace(0, t[-1], 600)
    env    = np.exp(-t_s/T2) * np.exp(-0.5*sigma**2*t_s**2)
    fig    = make_subplots(rows=1, cols=2,
                           subplot_titles=("Transverse components", "FID envelope"))
    fig.add_trace(go.Scatter(x=t, y=Mx, name="⟨Mx⟩",
                             line=dict(color=C_BLUE, width=1.8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=My, name="⟨My⟩",
                             line=dict(color=C_RED, width=1.8), opacity=0.85),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=M_perp, name="|⟨M⊥⟩|",
                             line=dict(color=C_BLUE, width=2.5)), row=1, col=2)
    fig.add_trace(go.Scatter(x=t_s, y=env, name="Analytic envelope",
                             line=dict(color=C_RED, dash="dash", width=1.5)),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=t_s, y=np.exp(-t_s/T2), name="exp(−t/T₂)",
                             line=dict(color=C_GREY, dash="dot", width=1.2)),
                  row=1, col=2)
    fig.update_xaxes(title_text="Time (µs)")
    fig.update_layout(
        title=f"Ensemble FID  σ={sigma:.2f} rad/µs  T₂={T2} µs",
        template=TMPL, height=370, hovermode="x unified",
        margin=dict(l=60, r=20, t=65, b=50))
    return fig


def _fig_cpmg(t, Mx, My, Mz, echo_times, tau, T2, n_echoes):
    M_perp    = np.sqrt(Mx**2 + My**2)
    echo_amps = [float(np.interp(et, t, M_perp)) for et in echo_times]
    fig = make_subplots(rows=2, cols=1, row_heights=[0.65, 0.35],
                        shared_xaxes=True,
                        subplot_titles=("Echo train", "Echo amplitude decay"))
    fig.add_trace(go.Scatter(x=t, y=M_perp, name="|M⊥|(t)",
                             line=dict(color=C_BLUE, width=1.8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=Mz, name="Mz(t)",
                             line=dict(color=C_GREEN, width=1.2, dash="dash"),
                             opacity=0.7), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=np.exp(-t/T2), name="exp(−t/T₂)",
                             line=dict(color=C_GREY, dash="dot", width=1)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=echo_times, y=echo_amps, mode="markers",
                             marker=dict(color=C_ORNG, size=9, symbol="diamond"),
                             name="Echo peaks", showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=echo_times, y=echo_amps, name="Measured",
                             mode="markers+lines",
                             marker=dict(color=C_BLUE, size=9),
                             line=dict(color=C_BLUE, width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=echo_times,
                             y=np.exp(-np.array(echo_times)/T2),
                             name="exp(−t/T₂)",
                             line=dict(color=C_RED, dash="dash", width=1.8)),
                  row=2, col=1)
    fig.update_xaxes(title_text="Time (µs)", row=2, col=1)
    fig.update_yaxes(title_text="Magnetisation", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude",     row=2, col=1)
    fig.update_layout(
        title=f"CPMG  τ={tau} µs  N={n_echoes}  T₂={T2} µs",
        template=TMPL, height=520, hovermode="x unified",
        margin=dict(l=60, r=20, t=60, b=50))
    return fig


def _fig_bloch_sphere_3d(Mx, My, Mz, title="Bloch Sphere"):
    theta  = np.linspace(0, 2*np.pi, 80)
    traces = []
    for zval in np.linspace(-0.8, 0.8, 5):
        r = np.sqrt(max(0, 1 - zval**2))
        traces.append(go.Scatter3d(
            x=r*np.cos(theta), y=r*np.sin(theta), z=np.full_like(theta, zval),
            mode='lines', line=dict(color='rgba(160,160,160,0.2)', width=1),
            showlegend=False, hoverinfo='skip'))
    for angle in np.linspace(0, np.pi, 7):
        traces.append(go.Scatter3d(
            x=np.cos(angle)*np.cos(theta),
            y=np.sin(angle)*np.cos(theta),
            z=np.sin(theta),
            mode='lines', line=dict(color='rgba(160,160,160,0.2)', width=1),
            showlegend=False, hoverinfo='skip'))
    for xs, ys, zs, lbl in [
        ([0, 1.25], [0, 0],    [0, 0],    "x"),
        ([0, 0],    [0, 1.25], [0, 0],    "y"),
        ([0, 0],    [0, 0],    [0, 1.25], "z"),
    ]:
        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs, mode='lines+text',
            line=dict(color='rgba(200,200,200,0.6)', width=2),
            text=['', lbl], textposition='top center',
            showlegend=False, hoverinfo='skip'))
    n = len(Mx)
    traces.append(go.Scatter3d(
        x=Mx, y=My, z=Mz, mode='lines',
        line=dict(color=np.linspace(0, 1, n), colorscale='Plasma', width=5,
                  colorbar=dict(title="t / t_max", thickness=12, len=0.5, x=1.02)),
        name='Trajectory'))
    traces.append(go.Scatter3d(
        x=[Mx[0]], y=[My[0]], z=[Mz[0]], mode='markers',
        marker=dict(color='lime', size=7), name='Start'))
    traces.append(go.Scatter3d(
        x=[Mx[-1]], y=[My[-1]], z=[Mz[-1]], mode='markers',
        marker=dict(color=C_RED, size=7), name='End'))
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title, template=TMPL, height=520,
        margin=dict(l=0, r=40, t=50, b=0),
        scene=dict(
            xaxis=dict(title='Mx', range=[-1.4, 1.4]),
            yaxis=dict(title='My', range=[-1.4, 1.4]),
            zaxis=dict(title='Mz', range=[-1.4, 1.4]),
            aspectmode='cube'))
    return fig


def _fig_fit(t_data, L_data, L_fit, params, errors, model_label):
    resid = L_data - L_fit
    fig   = make_subplots(rows=2, cols=1, row_heights=[0.75, 0.25],
                          shared_xaxes=True,
                          subplot_titles=("Fit", "Residuals"))
    fig.add_trace(go.Scatter(x=t_data, y=L_data, mode='markers', name='Data',
                             marker=dict(color=C_BLUE, size=6, opacity=0.75)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=t_data, y=L_fit, name='Best fit',
                             line=dict(color=C_RED, width=2.2)), row=1, col=1)
    fig.add_trace(go.Bar(x=t_data, y=resid, name='Residual',
                         marker_color=C_GREEN, opacity=0.7), row=2, col=1)
    fig.add_hline(y=0, line_color=C_GREY, line_width=0.8, row=2, col=1)
    param_str = "    ".join(
        f"{k} = {v:.4f} ± {errors.get(k,0):.4f}" for k, v in params.items())
    fig.update_layout(
        title=f"Model: {model_label}<br><sup>{param_str}</sup>",
        template=TMPL, height=480, hovermode="x unified",
        margin=dict(l=60, r=20, t=85, b=50))
    fig.update_xaxes(title_text="Time (µs)", row=2, col=1)
    fig.update_yaxes(title_text="Coherence", row=1, col=1)
    fig.update_yaxes(title_text="Residual",  row=2, col=1)
    return fig


# ── Rabi figure builders ──────────────────────────────────────────────────────

def _fig_rabi_population(t, Mz, omega_rabi, delta, T1, T2):
    omega_eff  = np.sqrt(omega_rabi**2 + delta**2)
    t_pi       = np.pi / omega_eff
    P_ode      = (1.0 - Mz) / 2.0
    P_analytic = analytic_rabi_population(t, omega_rabi, delta)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=P_ode,
                             name="P↑(t)  [with T₁, T₂]",
                             line=dict(color=C_BLUE, width=2.5)))
    fig.add_trace(go.Scatter(x=t, y=P_analytic,
                             name="P↑(t)  [no relaxation]",
                             line=dict(color=C_GREY, dash="dash", width=1.5)))
    n_marks = min(int(t[-1] / t_pi), 8)
    for k in range(1, n_marks + 1):
        tp    = k * t_pi
        is_pi = (k % 2 == 1)
        fig.add_vline(x=tp, line_dash="dot",
                      line_color=C_RED if is_pi else C_GREEN,
                      annotation_text="π" if is_pi else "2π",
                      annotation_position="top right")
    fig.add_hline(y=0.5, line_dash="dot", line_color=C_GREY,
                  annotation_text="P↑ = 0.5", annotation_position="right")
    fig.update_layout(**_base_layout(
        f"Rabi Oscillations  Ω={omega_rabi:.2f}  Δ={delta:.2f}"
        f"  Ω_eff={omega_eff:.2f} rad/µs"))
    fig.update_xaxes(title_text="Time (µs)")
    fig.update_yaxes(title_text="P↑  (excited state population)",
                     range=[-0.05, 1.1])
    return fig


def _fig_rabi_components(t, Mx, My, Mz):
    fig = make_subplots(rows=1, cols=3, shared_yaxes=False,
                        subplot_titles=("Mx  (rot. frame)",
                                        "My  (rot. frame)",
                                        "Mz  (population)"))
    for col, data, color, name in [
        (1, Mx, C_BLUE,  "Mx"),
        (2, My, C_PURP,  "My"),
        (3, Mz, C_GREEN, "Mz"),
    ]:
        fig.add_trace(go.Scatter(x=t, y=data, name=name,
                                 line=dict(color=color, width=2)),
                      row=1, col=col)
    fig.add_hline(y= 1, line_dash="dot", line_color=C_GREY,
                  annotation_text="|↑⟩", annotation_position="right",
                  row=1, col=3)
    fig.add_hline(y=-1, line_dash="dot", line_color=C_GREY,
                  annotation_text="|↓⟩", annotation_position="right",
                  row=1, col=3)
    fig.update_xaxes(title_text="Time (µs)")
    fig.update_yaxes(title_text="Component", row=1, col=1)
    fig.update_layout(title="Bloch Vector — Rotating Frame",
                      template=TMPL, height=340, hovermode="x unified",
                      margin=dict(l=60, r=20, t=65, b=50))
    return fig


def _fig_rabi_chevron(omega_rabi, delta_max, t_max, current_delta=None):
    t, deltas, P = analytic_chevron(omega_rabi, delta_max, t_max)
    fig = go.Figure(go.Heatmap(
        x=t, y=deltas, z=P,
        colorscale="Viridis",
        colorbar=dict(title="P↑", thickness=14),
        zmin=0, zmax=1))
    fig.add_hline(y=0, line_color="white", line_dash="dash", line_width=1.2,
                  annotation_text="Δ = 0  (resonance)",
                  annotation_font_color="white",
                  annotation_position="top right")
    if current_delta is not None:
        fig.add_hline(y=current_delta, line_color=C_ORNG, line_dash="dot",
                      line_width=1.5,
                      annotation_text=f"Δ = {current_delta:.2f}",
                      annotation_font_color=C_ORNG,
                      annotation_position="bottom right")
    fig.update_layout(**_base_layout(
        f"Chevron  P↑(t, Δ)   Ω = {omega_rabi:.2f} rad/µs", height=420))
    fig.update_xaxes(title_text="Time (µs)")
    fig.update_yaxes(title_text="Detuning Δ (rad/µs)")
    return fig


def _fig_rabi_resonance_scan(omega_rabi, delta_max, t_pi):
    deltas    = np.linspace(-delta_max, delta_max, 500)
    omega_eff = np.sqrt(omega_rabi**2 + deltas**2)
    P         = (omega_rabi / omega_eff)**2 * np.sin(omega_eff * t_pi / 2)**2
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=deltas, y=P,
                             name=f"P↑ at t_π = {t_pi:.3f} µs",
                             line=dict(color=C_PURP, width=2.5)))
    fig.add_vline(x=0, line_dash="dot", line_color=C_GREY,
                  annotation_text="Resonance", annotation_position="top right")
    fig.add_hline(y=1.0, line_dash="dot", line_color=C_GREY,
                  annotation_text="P↑ = 1", annotation_position="right")
    fig.update_layout(**_base_layout(
        f"Resonance Scan  P↑ vs Δ  (at t = t_π)   Ω = {omega_rabi:.2f}"))
    fig.update_xaxes(title_text="Detuning Δ (rad/µs)")
    fig.update_yaxes(title_text="P↑", range=[-0.05, 1.1])
    return fig


# ── Ramsey figure builders ────────────────────────────────────────────────────

def _fig_ramsey_fringes(T_free, P_up, params, errors, P_fit, delta, T2, T2_star_fit):
    T_analytic = np.linspace(T_free[0], T_free[-1], 600)
    P_analytic = analytic_ramsey_population(T_analytic, delta, T2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=T_free, y=P_up,
                             name="Simulated fringes",
                             mode="markers+lines",
                             marker=dict(color=C_BLUE, size=4),
                             line=dict(color=C_BLUE, width=1.2)))
    fig.add_trace(go.Scatter(x=T_analytic, y=P_analytic,
                             name=f"Analytic  T₂={T2:.1f} µs",
                             line=dict(color=C_GREY, dash="dash", width=1.4)))
    if P_fit is not None:
        fig.add_trace(go.Scatter(x=T_free, y=P_fit,
                                 name=f"Fit  T₂*={T2_star_fit:.2f} µs  "
                                      f"Δ_fit={params['delta']:.3f}",
                                 line=dict(color=C_RED, width=2.0)))
    fig.add_hline(y=0.5, line_dash="dot", line_color=C_GREY,
                  annotation_text="P↑ = 0.5", annotation_position="right")
    fig.update_layout(**_base_layout(
        f"Ramsey Fringes  Δ={delta:.3f} rad/µs  T₂={T2:.1f} µs"))
    fig.update_xaxes(title_text="Free evolution time T (µs)")
    fig.update_yaxes(title_text="P↑  (excited state population)",
                     range=[-0.05, 1.1])
    return fig


def _fig_ramsey_detuning_scan(T2, T_free_fixed, delta_range):
    deltas = np.linspace(-delta_range, delta_range, 500)
    P      = 0.5 + 0.5 * np.cos(deltas * T_free_fixed) * np.exp(-T_free_fixed / T2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=deltas / (2*np.pi), y=P,
                             name=f"T = {T_free_fixed:.1f} µs",
                             line=dict(color=C_TEAL, width=2.5)))
    fig.add_vline(x=0, line_dash="dot", line_color=C_GREY,
                  annotation_text="ω = ω₀", annotation_position="top right")
    fig.add_hline(y=0.5, line_dash="dot", line_color=C_GREY,
                  annotation_text="P↑ = 0.5", annotation_position="right")
    fig.update_layout(**_base_layout(
        f"Ramsey Spectroscopy Scan  T = {T_free_fixed:.1f} µs"))
    fig.update_xaxes(title_text="Detuning Δ/2π  (MHz)")
    fig.update_yaxes(title_text="P↑", range=[-0.05, 1.1])
    return fig


# ── Density matrix figure builders ───────────────────────────────────────────

def _fig_dm_bloch(t, rx, ry, rz, pur, T2, title):
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=("rx(t) = Mx", "ry(t) = My",
                                        "rz(t) = Mz", "Purity Tr(ρ²)"))
    for (row, col), data, color, name in [
        ((1,1), rx,  C_BLUE,  "rx"),
        ((1,2), ry,  C_PURP,  "ry"),
        ((2,1), rz,  C_GREEN, "rz"),
        ((2,2), pur, C_ORNG,  "Tr(ρ²)"),
    ]:
        fig.add_trace(go.Scatter(x=t, y=data, name=name,
                                 line=dict(color=color, width=2)),
                      row=row, col=col)
    env = np.exp(-t / T2)
    for (row, col) in [(1,1), (1,2)]:
        for sign in (1, -1):
            fig.add_trace(go.Scatter(x=t, y=sign*env, showlegend=False,
                                     line=dict(color=C_GREY, dash="dash", width=1),
                                     hoverinfo="skip"),
                          row=row, col=col)
    fig.add_hline(y=1.0, line_dash="dot", line_color=C_GREY,
                  annotation_text="pure", row=2, col=2)
    fig.add_hline(y=0.5, line_dash="dot", line_color=C_GREY,
                  annotation_text="max mixed", row=2, col=2)
    fig.update_xaxes(title_text="Time (µs)")
    fig.update_layout(title=title, template=TMPL, height=500,
                      hovermode="x unified",
                      margin=dict(l=60, r=20, t=65, b=50))
    return fig


def _fig_dm_validation(t, bloch_dm, max_err, omega_rabi, delta):
    fig = go.Figure()
    labels = ["rx  (Mx)", "ry  (My)", "rz  (Mz)"]
    colors = [C_BLUE, C_PURP, C_GREEN]
    for i, (lbl, col) in enumerate(zip(labels, colors)):
        fig.add_trace(go.Scatter(x=t, y=bloch_dm[:, i],
                                 name=lbl,
                                 line=dict(color=col, width=2)))
    fig.update_layout(**_base_layout(
        f"DM vs Bloch ODE  —  max |error| = {max_err:.2e}  "
        f"Ω={omega_rabi:.2f}  Δ={delta:.2f}"))
    fig.update_xaxes(title_text="Time (µs)")
    fig.update_yaxes(title_text="Bloch component", range=[-1.1, 1.1])
    return fig


# ============================================================================
# Two-qubit figure builders
# ============================================================================

def _fig_dm_heatmap_2q(rho, title="Density Matrix  ρ", show_imag=True):
    """Re(ρ) and Im(ρ) as annotated Plotly heatmaps."""
    n_cols = 2 if show_imag else 1
    subtitles = ["Re(ρ)", "Im(ρ)"] if show_imag else ["Re(ρ)"]
    fig = make_subplots(rows=1, cols=n_cols, subplot_titles=subtitles,
                        horizontal_spacing=0.12)

    for col, (label, data, cscale) in enumerate([
        ("Re", np.real(rho), "RdBu_r"),
        ("Im", np.imag(rho), "RdBu_r"),
    ][:n_cols], start=1):
        annotations = []
        for i in range(4):
            for j in range(4):
                annotations.append(dict(
                    x=j, y=i, xref=f"x{col}", yref=f"y{col}",
                    text=f"{data[i,j]:.3f}",
                    showarrow=False,
                    font=dict(size=9, color="black" if abs(data[i,j]) < 0.4 else "white"),
                ))
        fig.add_trace(go.Heatmap(
            z=data,
            x=BASIS_LABELS_2Q,
            y=BASIS_LABELS_2Q,
            colorscale=cscale,
            zmid=0, zmin=-1, zmax=1,
            showscale=True,
            colorbar=dict(x=0.46 if col == 1 else 1.0, thickness=14, len=0.8),
        ), row=1, col=col)
        for ann in annotations:
            fig.add_annotation(**ann)

    fig.update_layout(
        title=title, template=TMPL, height=380,
        margin=dict(l=60, r=60, t=65, b=50),
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def _fig_bloch_pair_2q(rho, title="Reduced Qubit States"):
    """Two Bloch spheres side-by-side for the reduced single-qubit states."""
    sx2 = np.array([[0, 1], [1, 0]], dtype=complex)
    sy2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz2 = np.array([[1, 0], [0, -1]], dtype=complex)

    def _bloch_vec(rho2):
        return np.array([
            float(np.real(np.trace(rho2 @ sx2))),
            float(np.real(np.trace(rho2 @ sy2))),
            float(np.real(np.trace(rho2 @ sz2))),
        ])

    rho_A = partial_trace(rho, keep=0)
    rho_B = partial_trace(rho, keep=1)
    r_A   = _bloch_vec(rho_A)
    r_B   = _bloch_vec(rho_B)

    def _sphere_traces(r_vec, color, label):
        theta = np.linspace(0, 2*np.pi, 60)
        traces = []
        for zval in np.linspace(-0.8, 0.8, 5):
            rv = np.sqrt(max(0, 1 - zval**2))
            traces.append(go.Scatter3d(
                x=rv*np.cos(theta), y=rv*np.sin(theta),
                z=np.full_like(theta, zval),
                mode='lines', line=dict(color='rgba(150,150,150,0.15)', width=1),
                showlegend=False, hoverinfo='skip'))
        for ang in np.linspace(0, np.pi, 7):
            traces.append(go.Scatter3d(
                x=np.cos(ang)*np.cos(theta),
                y=np.sin(ang)*np.cos(theta),
                z=np.sin(theta),
                mode='lines', line=dict(color='rgba(150,150,150,0.15)', width=1),
                showlegend=False, hoverinfo='skip'))
        for xs, ys, zs, lbl, c in [
            ([0,1.3],[0,0],[0,0], 'x', '#E63946'),
            ([0,0],[0,1.3],[0,0], 'y', '#2C7BB6'),
            ([0,0],[0,0],[0,1.3], 'z', '#6A994E'),
        ]:
            traces.append(go.Scatter3d(
                x=xs, y=ys, z=zs, mode='lines+text',
                line=dict(color=c, width=2),
                text=['', lbl], textposition='top center',
                showlegend=False, hoverinfo='skip'))
        rx, ry, rz = r_vec
        rnorm = float(np.linalg.norm(r_vec))
        traces.append(go.Scatter3d(
            x=[0, rx], y=[0, ry], z=[0, rz], mode='lines+markers',
            line=dict(color=color, width=8),
            marker=dict(size=[0, 8], color=color),
            name=f'{label}  |r|={rnorm:.3f}'))
        return traces

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
        subplot_titles=[
            f"Qubit A — ρ_A = Tr_B[ρ]  |r|={np.linalg.norm(r_A):.3f}",
            f"Qubit B — ρ_B = Tr_A[ρ]  |r|={np.linalg.norm(r_B):.3f}",
        ],
    )
    for trace in _sphere_traces(r_A, C_RED,  "A"):
        fig.add_trace(trace, row=1, col=1)
    for trace in _sphere_traces(r_B, C_BLUE, "B"):
        fig.add_trace(trace, row=1, col=2)

    scene_cfg = dict(
        xaxis=dict(range=[-1.4, 1.4], title='x'),
        yaxis=dict(range=[-1.4, 1.4], title='y'),
        zaxis=dict(range=[-1.4, 1.4], title='z'),
        aspectmode='cube',
        camera=dict(eye=dict(x=1.4, y=1.2, z=0.8)),
    )
    fig.update_layout(
        title=title, template=TMPL, height=480,
        scene=scene_cfg, scene2=scene_cfg,
        margin=dict(l=0, r=0, t=65, b=0),
    )
    return fig, float(np.linalg.norm(r_A)), float(np.linalg.norm(r_B))


def _fig_bell_fidelities(rho):
    """Bar chart: fidelity of ρ with each Bell state."""
    names  = list(BELL_LABELS.keys())
    labels = [BELL_LABELS[n] for n in names]
    fids   = []
    for nm in names:
        bket = bell_state(nm)
        brho = np.outer(bket, bket.conj())
        fids.append(float(np.real(np.trace(brho @ rho))))

    colors = [C_RED, C_BLUE, C_GREEN, C_ORNG]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=fids,
        marker_color=colors,
        text=[f"{f:.3f}" for f in fids],
        textposition="outside",
        width=0.5,
    ))
    fig.add_hline(y=0.25, line_dash="dot", line_color=C_GREY,
                  annotation_text="Max mixed (0.25)", annotation_position="right")
    fig.update_layout(**_base_layout("Bell State Fidelities", height=340))
    fig.update_xaxes(title_text="Bell state")
    fig.update_yaxes(title_text="Fidelity  F(ρ, |Bell⟩)", range=[0, 1.2])
    return fig


def _fig_xx_sweep(angles, concs):
    """Concurrence vs XX gate angle θ."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=angles, y=concs,
        mode='lines',
        fill='tozeroy', fillcolor=f'rgba(230,57,70,0.10)',
        line=dict(color=C_RED, width=2.5),
        name='Concurrence  C(θ)',
    ))
    # Mark maximum
    idx_max = int(np.argmax(concs))
    fig.add_trace(go.Scatter(
        x=[angles[idx_max]], y=[concs[idx_max]], mode='markers',
        marker=dict(color=C_RED, size=12, symbol='star'),
        name=f'Max C={concs[idx_max]:.3f} at θ={angles[idx_max]:.3f}',
    ))
    fig.add_vline(x=np.pi/2, line_dash="dot", line_color=C_GREY,
                  annotation_text="θ = π/2", annotation_position="top right")
    fig.update_layout(**_base_layout(
        "Entanglement vs XX Gate Angle  —  XX(θ)|00⟩", height=340))
    fig.update_xaxes(title_text="Gate angle θ (rad)",
                     tickvals=[0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
                     ticktext=["0", "π/4", "π/2", "3π/4", "π"])
    fig.update_yaxes(title_text="Concurrence  C", range=[-0.05, 1.15])
    return fig


def _fig_entanglement_evolution(t, ent, title):
    """Three-panel: concurrence, entanglement entropy, purity."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.38, 0.32, 0.30],
        subplot_titles=(
            "Concurrence  C(ρ)",
            "Entanglement Entropy  S(ρ_A)",
            "Qubit Purity  Tr(ρ_A²)",
        ),
        vertical_spacing=0.08,
    )

    # Concurrence + EoF
    fig.add_trace(go.Scatter(x=t, y=ent['concurrence'],
                             name='Concurrence', fill='tozeroy',
                             fillcolor='rgba(230,57,70,0.10)',
                             line=dict(color=C_RED, width=2.5)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=ent['entanglement_of_formation'],
                             name='Entanglement of Formation',
                             line=dict(color=C_PURP, dash='dash', width=1.5)),
                  row=1, col=1)
    fig.add_hline(y=0, line_color=C_GREY, line_width=0.6, row=1, col=1)

    # Entropy
    fig.add_trace(go.Scatter(x=t, y=ent['entropy_A'],
                             name='S(ρ_A)', fill='tozeroy',
                             fillcolor='rgba(44,123,182,0.10)',
                             line=dict(color=C_BLUE, width=2.2)),
                  row=2, col=1)
    fig.add_hline(y=1.0, line_dash='dot', line_color=C_GREY,
                  annotation_text='1 ebit', annotation_position='right',
                  row=2, col=1)

    # Purity A and B
    fig.add_trace(go.Scatter(x=t, y=ent['purity_A'],
                             name='Purity A', line=dict(color=C_ORNG, width=2)),
                  row=3, col=1)
    fig.add_trace(go.Scatter(x=t, y=ent['purity_B'],
                             name='Purity B',
                             line=dict(color=C_GREY, dash='dash', width=1.5)),
                  row=3, col=1)
    fig.add_hline(y=0.5, line_dash='dot', line_color=C_GREY,
                  annotation_text='max mixed', annotation_position='right',
                  row=3, col=1)
    fig.add_hline(y=1.0, line_dash='dot', line_color=C_GREY,
                  annotation_text='pure', annotation_position='right',
                  row=3, col=1)

    fig.update_yaxes(range=[-0.05, 1.15], row=1, col=1)
    fig.update_yaxes(range=[-0.05, 1.15], row=2, col=1)
    fig.update_yaxes(range=[0.30, 1.08],  row=3, col=1)
    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_layout(
        title=title, template=TMPL, height=580,
        hovermode='x unified',
        margin=dict(l=60, r=20, t=65, b=50),
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='right', x=1),
    )
    return fig


def _fig_esd(t, conc, purity_A, T1_A, T1_B):
    """Entanglement sudden death plot: concurrence vs purity on same axes."""
    T1_mean = (T1_A + T1_B) / 2.0
    purity_theory = 0.5 + 0.5 * np.exp(-t / T1_mean)

    esd_idx = int(np.argmax(conc < 1e-3))
    t_esd   = t[esd_idx] if conc[esd_idx] < 1e-3 else None

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45],
                        subplot_titles=(
                            "Concurrence  C(t)  — Sudden Death",
                            "Single-Qubit Purity  Tr(ρ_A²)",
                        ),
                        vertical_spacing=0.10)

    fig.add_trace(go.Scatter(x=t, y=conc, name='Concurrence',
                             fill='tozeroy', fillcolor='rgba(230,57,70,0.12)',
                             line=dict(color=C_RED, width=2.5)),
                  row=1, col=1)
    if t_esd is not None:
        fig.add_vline(x=t_esd, line_dash='dash', line_color=C_RED,
                      annotation_text=f't_ESD ≈ {t_esd:.2f}',
                      annotation_font_color=C_RED,
                      annotation_position='top right', row=1, col=1)

    fig.add_trace(go.Scatter(x=t, y=purity_A, name='Purity A (ODE)',
                             line=dict(color=C_ORNG, width=2.2)),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=purity_theory,
                             name=f'½ + ½·e^(-t/T₁)  T₁={T1_mean:.1f}',
                             line=dict(color=C_GREY, dash='dash', width=1.4)),
                  row=2, col=1)
    fig.add_hline(y=0.5, line_dash='dot', line_color=C_GREY,
                  annotation_text='max mixed', annotation_position='right',
                  row=2, col=1)

    fig.update_yaxes(range=[-0.05, 1.12], row=1, col=1)
    fig.update_yaxes(range=[0.35, 1.08], row=2, col=1)
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_layout(
        title=(f"Entanglement Sudden Death  —  "
               f"T₁_A={T1_A:.1f}  T₁_B={T1_B:.1f}"),
        template=TMPL, height=480,
        hovermode='x unified',
        margin=dict(l=60, r=20, t=65, b=50),
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='right', x=1),
    )
    return fig, t_esd


def _fig_corr_vs_local(res):
    """Concurrence under local dephasing vs correlated ZZ noise."""
    t = res['t']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=res['C_local'],
                             name='Local dephasing only',
                             line=dict(color=C_RED, width=2.2)))
    fig.add_trace(go.Scatter(x=t, y=res['C_correlated'],
                             name='Correlated ZZ only',
                             line=dict(color=C_GREEN, width=2.2)))
    fig.add_trace(go.Scatter(x=t, y=res['C_both'],
                             name='Both simultaneously',
                             line=dict(color=C_ORNG, dash='dash', width=1.8)))
    fig.add_hline(y=1.0, line_dash='dot', line_color=C_GREY,
                  annotation_text='C = 1 (max entangled)',
                  annotation_position='right')
    fig.add_hline(y=0, line_dash='dot', line_color=C_GREY)
    fig.update_layout(**_base_layout(
        "Correlated vs Local Dephasing on |Φ+⟩", height=360))
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Concurrence  C(ρ)", range=[-0.05, 1.15])
    return fig


# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    st.markdown("## Spin Coherence Simulator")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    experiment = st.selectbox("Experiment type", EXPERIMENTS)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    is_two_qubit = experiment.startswith("Two-Qubit")

    if not is_two_qubit:
        # ── Single-qubit parameters ───────────────────────────────────────────
        st.markdown("**Physical parameters**")
        gamma  = 1.0
        f0     = st.slider("B₀ frequency f₀ (MHz)", 0.1, 2.0, 0.5, 0.05)
        omega0 = 2 * np.pi * f0
        T2     = st.slider("T₂ (µs)", 0.5, 50.0, 10.0, 0.5)
        T1     = st.slider("T₁ (µs)", T2, 200.0, float(max(T2*3, T2+1.0)), 1.0)

        needs_sigma = experiment in ["Ensemble FID", "FID vs Echo (T₂* comparison)"]
        sigma = (st.slider("Frequency spread σ (rad/µs)", 0.0, 2.0, 0.3, 0.05)
                 if needs_sigma else 0.0)

        needs_tau = experiment in [
            "Hahn Echo", "CPMG Train", "FID vs Echo (T₂* comparison)"]
        tau = (st.slider("Half-echo time τ (µs)", 0.1, 30.0, 3.0, 0.1)
               if needs_tau else 3.0)

        n_echoes = (st.slider("Number of echoes", 1, 20, 8, 1)
                    if experiment == "CPMG Train" else 8)

        if experiment == "Echo Sweep":
            tau_min = st.slider("τ min (µs)", 0.1, 5.0,  0.5,  0.1)
            tau_max = st.slider("τ max (µs)", 1.0, 50.0, 20.0, 1.0)
            n_tau   = st.slider("Sweep points", 10, 200, 60, 10)
        else:
            tau_min, tau_max, n_tau = 0.5, 20.0, 60

        needs_rabi = experiment in ["Rabi Oscillation", "Density Matrix"]
        if needs_rabi:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown("**Drive parameters**")
            omega_rabi     = st.slider("Rabi frequency Ω (rad/µs)", 0.05, 5.0, 1.0, 0.05)
            delta_rabi     = st.slider("Detuning Δ (rad/µs)", -5.0, 5.0, 0.0, 0.05)
            delta_max_chev = st.slider("Chevron Δ range (rad/µs)", 1.0, 10.0, 3.0, 0.5)
            show_chevron   = st.checkbox("Show chevron plot", value=True)
            show_res_scan  = st.checkbox("Show resonance scan",
                                         value=(experiment == "Rabi Oscillation"))
        else:
            omega_rabi, delta_rabi = 1.0, 0.0
            delta_max_chev, show_chevron, show_res_scan = 3.0, False, False

        needs_ramsey = (experiment == "Ramsey Interference")
        if needs_ramsey:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown("**Ramsey parameters**")
            delta_ramsey   = st.slider("Detuning Δ (rad/µs) ", -3.0, 3.0, 0.5, 0.05)
            T_free_max     = st.slider("Max free evolution T (µs)", 1.0, 100.0,
                                       float(min(4*T2, 60.0)), 1.0)
            n_ramsey       = st.slider("Fringe points", 50, 500, 200, 50)
            n_shots_sens   = st.slider("Shots (sensitivity)", 100, 10000, 1000, 100)
            show_det_scan  = st.checkbox("Show detuning scan", value=True)
            run_ramsey_fit = st.checkbox("Fit fringes", value=True)
        else:
            delta_ramsey   = 0.5
            T_free_max     = 20.0
            n_ramsey       = 200
            n_shots_sens   = 1000
            show_det_scan  = False
            run_ramsey_fit = False

        needs_dm = (experiment == "Density Matrix")
        if needs_dm:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown("**Initial state**")
            rho0_choice = st.selectbox(
                "Initial state ρ₀",
                ["ground", "superposition", "mixed_0.5", "mixed_0.0"])
            show_dm_validation = st.checkbox("Show DM vs Bloch validation", value=True)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("**Display**")
        t_max_def = float(max(4*T2, 4*tau if needs_tau else 4*T2))
        if needs_rabi:
            oe_est    = np.sqrt(omega_rabi**2 + delta_rabi**2)
            t_max_def = float(max(t_max_def, 6 * np.pi / max(oe_est, 0.01)))
        t_max       = st.slider("t_max (µs)", 5.0, 200.0, min(t_max_def, 200.0), 5.0)
        show_sphere = st.checkbox("Show 3D Bloch sphere", value=False)

        if experiment == "CPMG Train":
            dt = st.select_slider("Time step dt (µs)",
                                  [0.005, 0.01, 0.02, 0.05, 0.1], value=0.05)
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            run_cpmg     = st.button("Run CPMG",   type="primary", width='stretch')
            run_rabi_btn = run_ramsey_btn = run_dm_btn = False
        elif experiment == "Rabi Oscillation":
            dt = 0.05
            run_cpmg     = False
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            run_rabi_btn = st.button("Run Rabi",   type="primary", width='stretch')
            run_ramsey_btn = run_dm_btn = False
        elif experiment == "Ramsey Interference":
            dt = 0.05
            run_cpmg = run_rabi_btn = False
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            run_ramsey_btn = st.button("Run Ramsey", type="primary", width='stretch')
            run_dm_btn = False
        elif experiment == "Density Matrix":
            dt = 0.05
            run_cpmg = run_rabi_btn = run_ramsey_btn = False
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            run_dm_btn = st.button("Run DM Simulation", type="primary", width='stretch')
        else:
            dt = 0.05
            run_cpmg = run_rabi_btn = run_ramsey_btn = run_dm_btn = False

    else:
        # ── Two-qubit parameters ──────────────────────────────────────────────
        st.markdown("**Bell state / initial state**")
        bell_choice = st.selectbox(
            "Starting Bell state",
            list(BELL_LABELS.keys()),
            format_func=lambda k: BELL_LABELS[k],
        )

        if experiment == "Two-Qubit: Bell States & Gates":
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown("**XX gate sweep**")
            n_xx_angles = st.slider("Sweep resolution", 50, 300, 120, 10)
            show_bloch_pair  = st.checkbox("Show reduced Bloch spheres", value=True)
            show_hinton      = st.checkbox("Show Im(ρ) panel", value=True)

        elif experiment == "Two-Qubit: Noise & Entanglement":
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown("**Qubit A noise**")
            T1_A   = st.slider("T₁_A", 1.0, 100.0, 15.0, 1.0)
            T2_A   = st.slider("T₂_A", 0.5, float(2*15), float(min(12.0, 2*15)), 0.5)

            st.markdown("**Qubit B noise**")
            T1_B   = st.slider("T₁_B", 1.0, 100.0, 15.0, 1.0)
            T2_B   = st.slider("T₂_B", 0.5, float(2*15), float(min(12.0, 2*15)), 0.5)

            st.markdown("**Additional channels**")
            use_dep = st.checkbox("Local depolarizing", value=False)
            T_dep   = st.slider("T_dep (both qubits)", 1.0, 100.0, 30.0, 1.0) if use_dep else np.inf
            use_zz  = st.checkbox("Correlated ZZ dephasing", value=False)
            T_ZZ    = st.slider("T_ZZ", 1.0, 100.0, 15.0, 1.0) if use_zz else np.inf

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown("**Entanglement sudden death**")
            show_esd  = st.checkbox("Show ESD demo", value=True)
            show_corr = st.checkbox("Show correlated vs local comparison", value=True)
            T2_corr   = st.slider("T₂ (comparison)", 1.0, 50.0, 8.0, 0.5) if show_corr else 8.0
            T_ZZ_corr = st.slider("T_ZZ (comparison)", 1.0, 50.0, 8.0, 0.5) if show_corr else 8.0

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            t_max_2q = st.slider("t_max (simulation units)", 5.0, 100.0, 40.0, 5.0)
            n_2q     = st.select_slider("ODE steps", [100, 200, 400, 600], value=200)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        if experiment == "Two-Qubit: Bell States & Gates":
            run_2q_btn = False   # Bell state tab is fully live (no ODE)
        else:
            run_2q_btn = st.button("Run Two-Qubit Simulation",
                                   type="primary", width='stretch')

        # Unused single-qubit vars (keep linters happy)
        T1 = T2 = omega0 = f0 = sigma = tau = 0.0
        n_echoes = 8; tau_min = tau_max = n_tau = 0
        omega_rabi = delta_rabi = delta_max_chev = 1.0
        delta_ramsey = T_free_max = 0.0; n_ramsey = n_shots_sens = 100
        show_chevron = show_res_scan = show_det_scan = run_ramsey_fit = False
        show_sphere = False; t_max = 20.0; dt = 0.05
        rho0_choice = "ground"; show_dm_validation = False
        run_cpmg = run_rabi_btn = run_ramsey_btn = run_dm_btn = False


# ============================================================================
# Main
# ============================================================================

st.markdown("# Spin Coherence Simulator")
st.caption(
    "Bloch equations  ·  T₁/T₂ relaxation  ·  Pulse sequences  ·  "
    "Rabi oscillations  ·  Ramsey interference  ·  Density matrix  ·  "
    "Two-qubit entanglement  ·  Fitting")
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

tab_sim, tab_fit = st.tabs(["Simulation", "Fitting"])

with tab_sim:

    # ── Two-Qubit: Bell States & Gates ────────────────────────────────────────
    if experiment == "Two-Qubit: Bell States & Gates":

        psi_bell, rho_bell = prepare_bell_state(bell_choice)
        summ = entanglement_summary(psi_bell)

        # Metrics row
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Concurrence  C",      f"{summ['concurrence']:.4f}")
        c2.metric("Entanglement entropy", f"{summ['entanglement_entropy']:.4f}")
        c3.metric("Negativity  N",        f"{summ['negativity']:.4f}")
        c4.metric("CHSH value  B",        f"{chsh_value(rho_bell):.3f}",
                  delta="Bell violation ✓" if chsh_value(rho_bell) > 2 else "no violation")
        c5.metric("Schmidt rank",         str(summ['schmidt_number']))

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Density matrix heatmap + Bell fidelities side by side
        col_dm, col_fid = st.columns([1.4, 1])
        with col_dm:
            st.plotly_chart(
                _fig_dm_heatmap_2q(
                    rho_bell,
                    title=f"Density Matrix  ρ  —  {BELL_LABELS[bell_choice]}",
                    show_imag=show_hinton,
                ),
                width='stretch',
            )
        with col_fid:
            st.plotly_chart(_fig_bell_fidelities(rho_bell), width='stretch')

        # Reduced Bloch spheres
        if show_bloch_pair:
            fig_bp, r_A, r_B = _fig_bloch_pair_2q(
                rho_bell,
                title=f"Reduced Qubit States  —  {BELL_LABELS[bell_choice]}",
            )
            st.plotly_chart(fig_bp, width='stretch')
            rA_col, rB_col, note_col = st.columns([1, 1, 2])
            rA_col.metric("|r_A| (Bloch radius)", f"{r_A:.4f}",
                          delta="0 = max entangled" if r_A < 0.01 else None)
            rB_col.metric("|r_B| (Bloch radius)", f"{r_B:.4f}",
                          delta="0 = max entangled" if r_B < 0.01 else None)
            note_col.info(
                "For a maximally entangled Bell state, both reduced states "
                "are I/2 — maximally mixed — and appear at the **centre** of "
                "their Bloch spheres. Tracing out one qubit erases all local "
                "information; it lives entirely in the correlations."
            )

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("#### XX Gate Entanglement Sweep")
        st.caption(
            "XX(θ)|00⟩ starts separable (θ=0) and reaches maximum entanglement "
            "at θ=π/2 (Mølmer-Sørensen gate used in trapped-ion hardware)."
        )
        angles, concs = _run_xx_sweep(n_xx_angles)
        st.plotly_chart(_fig_xx_sweep(angles, concs), width='stretch')

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Entanglement summary table
        st.markdown("#### Entanglement Measures — All Four Bell States")
        rows = []
        for bname, blabel in BELL_LABELS.items():
            psi_b = bell_state(bname)
            rho_b = ket_to_dm(psi_b)
            s = entanglement_summary(psi_b)
            rows.append({
                "State":   blabel,
                "C":       f"{s['concurrence']:.4f}",
                "E (ebit)": f"{s['entanglement_entropy']:.4f}",
                "E_F":     f"{s['entanglement_of_formation']:.4f}",
                "N":       f"{s['negativity']:.4f}",
                "E_N":     f"{s['log_negativity']:.4f}",
                "CHSH":    f"{chsh_value(rho_b):.3f}",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width=900)

        st.markdown("---")
        # CSV export of current Bell state DM
        dm_df = pd.DataFrame(
            np.abs(rho_bell),
            index=BASIS_LABELS_2Q,
            columns=BASIS_LABELS_2Q,
        )
        st.download_button(
            "Download |ρ| as CSV",
            data=dm_df.to_csv().encode(),
            file_name=f"dm_{bell_choice}.csv",
            mime="text/csv",
        )

    # ── Two-Qubit: Noise & Entanglement ───────────────────────────────────────
    elif experiment == "Two-Qubit: Noise & Entanglement":

        # Clamp T2 ≤ 2·T1 (user sliders may be out of order)
        T2_A = min(T2_A, 2.0 * T1_A)
        T2_B = min(T2_B, 2.0 * T1_B)

        if run_2q_btn:
            with st.spinner("Integrating two-qubit Lindblad master equation..."):
                try:
                    t2q, rho_r, rho_i, ent = _run_2q_noise(
                        T1_A, T2_A, T1_B, T2_B,
                        T_dep, T_dep,   # T_dep_A, T_dep_B
                        T_ZZ,
                        bell_choice, t_max_2q, n_2q,
                    )
                    st.session_state["2q"] = dict(
                        t=t2q, rho_r=rho_r, rho_i=rho_i, ent=ent,
                        T1_A=T1_A, T2_A=T2_A, T1_B=T1_B, T2_B=T2_B,
                        bell=bell_choice, t_max=t_max_2q,
                    )
                except Exception as e:
                    st.error(f"Simulation failed: {e}")

            if show_esd:
                with st.spinner("Running ESD demo..."):
                    t_esd_arr, c_esd, pA_esd = _run_esd(
                        bell_choice, T1_A, T1_B, T2_A, T2_B, t_max_2q, n_2q)
                    st.session_state["esd"] = dict(
                        t=t_esd_arr, conc=c_esd, purity_A=pA_esd,
                        T1_A=T1_A, T1_B=T1_B,
                    )

            if show_corr:
                with st.spinner("Running correlated vs local dephasing comparison..."):
                    res_corr = _run_correlated_comparison(
                        T2_corr, T_ZZ_corr, t_max_2q, n_2q)
                    st.session_state["corr"] = res_corr

        if "2q" not in st.session_state:
            st.info("Configure noise parameters and press **Run Two-Qubit Simulation**.")
        else:
            r   = st.session_state["2q"]
            ent = r["ent"]
            t   = r["t"]

            # Reconstruct rho at t=0 and t=final
            rho_0   = r["rho_r"][0]  + 1j * r["rho_i"][0]
            rho_end = r["rho_r"][-1] + 1j * r["rho_i"][-1]

            C_init  = float(ent['concurrence'][0])
            C_final = float(ent['concurrence'][-1])
            E_init  = float(ent['entropy_A'][0])
            E_final = float(ent['entropy_A'][-1])

            # Find ESD time
            esd_arr  = np.argmax(ent['concurrence'] < 1e-3)
            t_esd_v  = float(t[esd_arr]) if ent['concurrence'][esd_arr] < 1e-3 else None
            esd_label = f"{t_esd_v:.2f}" if t_esd_v else "—"

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("C (t=0)",     f"{C_init:.4f}")
            c2.metric("C (final)",   f"{C_final:.4f}",
                      delta=f"Δ = {C_final - C_init:.4f}")
            c3.metric("S(ρ_A) init", f"{E_init:.4f}")
            c4.metric("S(ρ_A) end",  f"{E_final:.4f}")
            c5.metric("t_ESD",       esd_label,
                      delta="sudden death" if t_esd_v else "no ESD detected")

            st.plotly_chart(
                _fig_entanglement_evolution(
                    t, ent,
                    title=(
                        f"Entanglement Dynamics  —  {BELL_LABELS[r['bell']]}  ·  "
                        f"T₁_A={r['T1_A']:.1f}  T₂_A={r['T2_A']:.1f}  "
                        f"T₁_B={r['T1_B']:.1f}  T₂_B={r['T2_B']:.1f}"
                    ),
                ),
                width='stretch',
            )

            # Final density matrix heatmap
            col_init, col_fin = st.columns(2)
            with col_init:
                st.plotly_chart(
                    _fig_dm_heatmap_2q(rho_0,   title="ρ at t = 0",     show_imag=True),
                    width='stretch')
            with col_fin:
                st.plotly_chart(
                    _fig_dm_heatmap_2q(rho_end, title="ρ at t = final", show_imag=True),
                    width='stretch')

            # Reduced Bloch spheres for final state
            fig_bp, r_A_f, r_B_f = _fig_bloch_pair_2q(
                rho_end, title="Reduced States at Final Time")
            st.plotly_chart(fig_bp, width='stretch')

        # ESD demo
        if "esd" in st.session_state and show_esd:
            st.markdown("---")
            st.markdown("#### Entanglement Sudden Death")
            st.caption(
                "Entanglement reaches zero in **finite** time under amplitude "
                "damping — even though the individual qubit purity is still > 0.5. "
                "This is qualitatively different from classical correlations."
            )
            e = st.session_state["esd"]
            fig_esd, t_esd_val = _fig_esd(
                e["t"], e["conc"], e["purity_A"], e["T1_A"], e["T1_B"])
            st.plotly_chart(fig_esd, width='stretch')
            if t_esd_val:
                st.success(
                    f"ESD detected at t ≈ **{t_esd_val:.3f}** — "
                    f"concurrence drops to zero while single-qubit purity "
                    f"≈ {0.5 + 0.5*np.exp(-t_esd_val/((e['T1_A']+e['T1_B'])/2)):.3f} "
                    f"(still > 0.5)."
                )

        # Correlated vs local comparison
        if "corr" in st.session_state and show_corr:
            st.markdown("---")
            st.markdown("#### Correlated vs Local Dephasing")
            st.caption(
                "|Φ+⟩ = (|00⟩+|11⟩)/√2 is an **eigenstate** of σ_z⊗σ_z, so "
                "correlated ZZ noise leaves its concurrence exactly at 1. "
                "Local (independent) dephasing destroys it. "
                "This reveals that noise *structure*, not just strength, determines "
                "which states decohere."
            )
            st.plotly_chart(_fig_corr_vs_local(st.session_state["corr"]),
                            width='stretch')

        if "2q" in st.session_state:
            st.markdown("---")
            r = st.session_state["2q"]
            st.download_button(
                "Download entanglement trajectory as CSV",
                data=pd.DataFrame({
                    "t":                        r["t"],
                    "concurrence":              r["ent"]["concurrence"],
                    "entanglement_of_formation": r["ent"]["entanglement_of_formation"],
                    "entropy_A":                r["ent"]["entropy_A"],
                    "negativity":               r["ent"]["negativity"],
                    "purity_A":                 r["ent"]["purity_A"],
                    "purity_B":                 r["ent"]["purity_B"],
                }).to_csv(index=False).encode(),
                file_name="two_qubit_entanglement.csv",
                mime="text/csv",
            )

    # ── CPMG ─────────────────────────────────────────────────────────────────
    elif experiment == "CPMG Train":
        if run_cpmg:
            with st.spinner("Running CPMG simulation..."):
                t_cp, Mx_cp, My_cp, Mz_cp, echo_times = _run_cpmg(
                    gamma, omega0, T1, T2, tau, n_echoes, dt)
                st.session_state["cpmg"] = dict(
                    t=t_cp, Mx=Mx_cp, My=My_cp, Mz=Mz_cp,
                    echo_times=echo_times, tau=tau, T2=T2, n_echoes=n_echoes)

        if "cpmg" not in st.session_state:
            st.info("Configure parameters and press **Run CPMG**.")
        else:
            r      = st.session_state["cpmg"]
            Mp     = np.sqrt(r["Mx"]**2 + r["My"]**2)
            e_amps = [float(np.interp(et, r["t"], Mp)) for et in r["echo_times"]]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("T₂ (µs)", f"{r['T2']:.1f}")
            c2.metric("τ (µs)",  f"{r['tau']:.1f}")
            c3.metric("First echo", f"{e_amps[0]:.4f}",
                      delta=f"exp = {np.exp(-2*r['tau']/r['T2']):.4f}")
            c4.metric("Last echo",  f"{e_amps[-1]:.4f}",
                      delta=f"exp = {np.exp(-r['echo_times'][-1]/r['T2']):.4f}")
            st.plotly_chart(
                _fig_cpmg(r["t"], r["Mx"], r["My"], r["Mz"],
                          r["echo_times"], r["tau"], r["T2"], r["n_echoes"]),
                width='stretch')
            if show_sphere:
                st.plotly_chart(
                    _fig_bloch_sphere_3d(r["Mx"], r["My"], r["Mz"],
                                         "Bloch Sphere — CPMG"),
                    width='stretch')

    # ── Rabi ──────────────────────────────────────────────────────────────────
    elif experiment == "Rabi Oscillation":
        if show_chevron:
            st.plotly_chart(
                _fig_rabi_chevron(omega_rabi, delta_max_chev, t_max,
                                  current_delta=delta_rabi),
                width='stretch')
        if run_rabi_btn:
            with st.spinner("Running Rabi simulation..."):
                t_rb, Mx_rb, My_rb, Mz_rb = _run_rabi(
                    omega_rabi, delta_rabi, T1, T2, t_max)
                st.session_state["rabi"] = dict(
                    t=t_rb, Mx=Mx_rb, My=My_rb, Mz=Mz_rb,
                    omega_rabi=omega_rabi, delta=delta_rabi, T1=T1, T2=T2)

        if "rabi" not in st.session_state:
            st.info("Configure parameters and press **Run Rabi**.")
        else:
            r       = st.session_state["rabi"]
            t_pi    = pi_pulse_time(r["omega_rabi"], r["delta"])
            max_Pup = max_population_inversion(r["omega_rabi"], r["delta"])
            oe      = np.sqrt(r["omega_rabi"]**2 + r["delta"]**2)
            P_exc   = (1.0 - r["Mz"]) / 2.0
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Ω (rad/µs)",     f"{r['omega_rabi']:.3f}")
            c2.metric("Δ (rad/µs)",     f"{r['delta']:.3f}")
            c3.metric("Ω_eff (rad/µs)", f"{oe:.3f}")
            c4.metric("t_π (µs)",       f"{t_pi:.3f}",
                      delta=f"Max P↑ = {max_Pup:.3f}")
            st.plotly_chart(
                _fig_rabi_population(r["t"], r["Mz"],
                                     r["omega_rabi"], r["delta"],
                                     r["T1"], r["T2"]),
                width='stretch')
            st.plotly_chart(
                _fig_rabi_components(r["t"], r["Mx"], r["My"], r["Mz"]),
                width='stretch')
            if show_res_scan:
                st.plotly_chart(
                    _fig_rabi_resonance_scan(r["omega_rabi"], delta_max_chev, t_pi),
                    width='stretch')
            if show_sphere:
                st.plotly_chart(
                    _fig_bloch_sphere_3d(r["Mx"], r["My"], r["Mz"],
                                         "Bloch Sphere — Rabi (Rotating Frame)"),
                    width='stretch')
            st.markdown("---")
            st.download_button(
                "Download Rabi simulation as CSV",
                data=pd.DataFrame({
                    "t_us": r["t"], "Mx": r["Mx"],
                    "My": r["My"], "Mz": r["Mz"], "P_excited": P_exc,
                }).to_csv(index=False).encode(),
                file_name="rabi_sim.csv", mime="text/csv")

    # ── Ramsey ────────────────────────────────────────────────────────────────
    elif experiment == "Ramsey Interference":
        if run_ramsey_btn:
            with st.spinner("Running Ramsey sweep..."):
                T_arr, P_arr = _run_ramsey_sweep(
                    delta_ramsey, T1, T2, T_free_max, n_ramsey)
                fit_params = fit_errors = P_fitted = None
                T2_star_fit = T2
                if run_ramsey_fit:
                    try:
                        fit_params, fit_errors, P_fitted = fit_ramsey_fringes(
                            T_arr, P_arr,
                            delta_guess=abs(delta_ramsey) or 0.1,
                            T2_star_guess=T2)
                        T2_star_fit = fit_params["T2_star"]
                    except Exception as e:
                        st.warning(f"Fringe fit failed: {e}")
                st.session_state["ramsey"] = dict(
                    T_arr=T_arr, P_arr=P_arr,
                    fit_params=fit_params, fit_errors=fit_errors,
                    P_fitted=P_fitted,
                    delta=delta_ramsey, T2=T2, T2_star_fit=T2_star_fit)

        if "ramsey" not in st.session_state:
            st.info("Configure parameters and press **Run Ramsey**.")
        else:
            r = st.session_state["ramsey"]
            T2sf = r["T2_star_fit"]
            sens = ramsey_sensitivity(T2sf, n_shots_sens)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Δ (rad/µs)",   f"{r['delta']:.3f}")
            c2.metric("T₂ (µs)",      f"{r['T2']:.1f}")
            c3.metric("T₂* fit (µs)", f"{T2sf:.2f}")
            c4.metric("δf_min (MHz)",
                      f"{sens*1e-6:.4f}",
                      delta=f"N={n_shots_sens} shots")
            st.plotly_chart(
                _fig_ramsey_fringes(
                    r["T_arr"], r["P_arr"],
                    r["fit_params"] or {}, r["fit_errors"] or {},
                    r["P_fitted"], r["delta"], r["T2"], T2sf),
                width='stretch')
            if show_det_scan:
                T_fixed = float(T_free_max / 4)
                st.plotly_chart(
                    _fig_ramsey_detuning_scan(T2, T_fixed, delta_range=5.0),
                    width='stretch')
            if r["fit_params"]:
                st.markdown("**Fit parameters**")
                fc = st.columns(len(r["fit_params"]))
                for col, (pname, pval) in zip(fc, r["fit_params"].items()):
                    col.metric(pname, f"{pval:.4f}",
                               delta=f"±{r['fit_errors'].get(pname,0):.4f}")
            st.markdown("---")
            st.download_button(
                "Download Ramsey fringes as CSV",
                data=pd.DataFrame({
                    "T_free_us": r["T_arr"],
                    "P_up":      r["P_arr"],
                    **({"P_fit": r["P_fitted"]} if r["P_fitted"] is not None else {}),
                }).to_csv(index=False).encode(),
                file_name="ramsey_fringes.csv", mime="text/csv")

    # ── Density Matrix ────────────────────────────────────────────────────────
    elif experiment == "Density Matrix":
        if show_chevron:
            st.plotly_chart(
                _fig_rabi_chevron(omega_rabi, delta_max_chev, t_max,
                                  current_delta=delta_rabi),
                width='stretch')
        if run_dm_btn:
            with st.spinner("Running density matrix simulation..."):
                H       = hamiltonian_rotating(delta_rabi, omega_rabi)
                H_flat  = list(H.ravel())
                t_dm, rx, ry, rz, pur = _run_dm(
                    rho0_choice, H_flat, T1, T2, t_max, n=1000)
                st.session_state["dm"] = dict(
                    t=t_dm, rx=rx, ry=ry, rz=rz, pur=pur,
                    omega_rabi=omega_rabi, delta=delta_rabi, T1=T1, T2=T2,
                    rho0_tag=rho0_choice)
                if show_dm_validation:
                    t_val, bloch_dm_val, max_err = _run_dm_validation(
                        omega_rabi, delta_rabi, T1, T2, t_max)
                    st.session_state["dm_val"] = dict(
                        t=t_val, bloch_dm=bloch_dm_val, max_err=max_err,
                        omega_rabi=omega_rabi, delta=delta_rabi)

        if "dm" not in st.session_state:
            st.info("Configure parameters and press **Run DM Simulation**.")
        else:
            r = st.session_state["dm"]
            rho_final = bloch_to_dm(np.array([r["rx"][-1], r["ry"][-1], r["rz"][-1]]))
            P_gnd, P_exc_dm = population(rho_final)
            pur_final = float(r["pur"][-1])
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Ω (rad/µs)",     f"{r['omega_rabi']:.3f}")
            c2.metric("Δ (rad/µs)",     f"{r['delta']:.3f}")
            c3.metric("Purity  Tr(ρ²)", f"{pur_final:.4f}",
                      delta="pure" if pur_final > 0.99 else "mixed")
            c4.metric("P↑ (final)",     f"{P_exc_dm:.4f}",
                      delta=f"ρ₀ = {r['rho0_tag']}")
            st.plotly_chart(
                _fig_dm_bloch(
                    r["t"], r["rx"], r["ry"], r["rz"], r["pur"],
                    r["T2"],
                    f"Density Matrix  Ω={r['omega_rabi']:.2f}  "
                    f"Δ={r['delta']:.2f}  ρ₀={r['rho0_tag']}"),
                width='stretch')
            if show_sphere:
                st.plotly_chart(
                    _fig_bloch_sphere_3d(r["rx"], r["ry"], r["rz"],
                                         "Bloch Sphere — Density Matrix"),
                    width='stretch')
            if "dm_val" in st.session_state:
                v = st.session_state["dm_val"]
                st.plotly_chart(
                    _fig_dm_validation(v["t"], v["bloch_dm"], v["max_err"],
                                       v["omega_rabi"], v["delta"]),
                    width='stretch')
                if v["max_err"] < 1e-4:
                    st.success(f"✓ DM vs Bloch ODE max error = {v['max_err']:.2e} "
                               f"— equations are consistent.")
                else:
                    st.warning(f"DM vs Bloch ODE max error = {v['max_err']:.2e} "
                               f"— check T₂ ≤ T₁ constraint.")
            st.markdown("---")
            st.download_button(
                "Download density matrix simulation as CSV",
                data=pd.DataFrame({
                    "t_us":   r["t"],
                    "rx":     r["rx"],
                    "ry":     r["ry"],
                    "rz":     r["rz"],
                    "purity": r["pur"],
                    "P_up":   (1.0 - r["rz"]) / 2.0,
                }).to_csv(index=False).encode(),
                file_name="dm_sim.csv", mime="text/csv")

    # ── All analytic experiments ──────────────────────────────────────────────
    else:
        Mx = My = Mz = None

        if experiment == "Single Spin (Bloch)":
            t, Mx, My, Mz = _analytic_single_spin(omega0, T1, T2, t_max)
            M_perp = np.sqrt(Mx**2 + My**2)
            idx    = np.argmin(np.abs(t - T2))
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("T₂ (µs)",     f"{T2:.1f}")
            c2.metric("T₁ (µs)",     f"{T1:.1f}")
            c3.metric("|M⊥| at T₂", f"{M_perp[idx]:.4f}",
                      delta=f"expected {1/np.e:.4f}")
            c4.metric("f₀ (MHz)",    f"{f0:.2f}")
            st.plotly_chart(
                _fig_bloch_components(t, Mx, My, Mz, T2,
                    f"Single Spin  T₁={T1} µs  T₂={T2} µs  f₀={f0:.2f} MHz"),
                width='stretch')

        elif experiment == "Ensemble FID":
            t, Mx, My, Mz = _analytic_ensemble_fid(omega0, sigma, T1, T2, t_max)
            M_perp = np.sqrt(Mx**2 + My**2)
            T2star = 1.0 / (1.0/T2 + sigma) if sigma > 0 else T2
            idx    = np.argmin(np.abs(t - T2))
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("T₂ (µs)",           f"{T2:.1f}")
            c2.metric("T₂* estimate (µs)", f"{T2star:.2f}")
            c3.metric("σ (rad/µs)",         f"{sigma:.2f}")
            c4.metric("|⟨M⊥⟩| at T₂",     f"{M_perp[idx]:.4f}")
            st.plotly_chart(
                _fig_fid_ensemble(t, Mx, My, sigma, T2), width='stretch')

        elif experiment == "Hahn Echo":
            t, Mx, My, Mz = _analytic_hahn_echo(omega0, T1, T2, tau)
            M_perp   = np.sqrt(Mx**2 + My**2)
            echo_amp = float(np.interp(2*tau, t, M_perp))
            expected = float(np.exp(-2*tau / T2))
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("T₂ (µs)",        f"{T2:.1f}")
            c2.metric("τ (µs)",         f"{tau:.1f}")
            c3.metric("Echo amp at 2τ", f"{echo_amp:.4f}",
                      delta=f"expected {expected:.4f}")
            c4.metric("Error", f"{abs(echo_amp-expected):.2e}")
            st.plotly_chart(
                _fig_echo_detail(t, Mx, My, tau, T2), width='stretch')
            st.plotly_chart(
                _fig_bloch_components(t, Mx, My, Mz, T2,
                    f"Hahn Echo Components  τ={tau} µs  T₂={T2} µs"),
                width='stretch')

        elif experiment == "Echo Sweep":
            two_tau, amps = _analytic_echo_sweep(T2, tau_min, tau_max, n_tau)
            log_A  = np.log(np.clip(amps, 1e-10, None))
            T2_fit = -1.0 / np.polyfit(two_tau, log_A, 1)[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("True T₂ (µs)",   f"{T2:.1f}")
            c2.metric("Fitted T₂ (µs)", f"{T2_fit:.2f}",
                      delta=f"Δ = {T2_fit-T2:.3f}")
            c3.metric("Sweep points",    str(n_tau))
            fig_sw, _ = _fig_echo_sweep(two_tau, amps, T2)
            st.plotly_chart(fig_sw, width='stretch')

        elif experiment == "FID vs Echo (T₂* comparison)":
            t, fid_env, t2_only = _analytic_fid_vs_echo(sigma, T2, tau, t_max)
            echo_amp = float(np.exp(-2*tau / T2))
            fid_at   = float(np.exp(-2*tau/T2) * np.exp(-0.5*sigma**2*(2*tau)**2))
            T2star   = 1.0/(1.0/T2 + sigma) if sigma > 0 else T2
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("T₂ (µs)",           f"{T2:.1f}")
            c2.metric("T₂* estimate (µs)", f"{T2star:.2f}")
            c3.metric("FID |M⊥| at 2τ",    f"{fid_at:.4f}")
            c4.metric("Echo amp at 2τ",     f"{echo_amp:.4f}",
                      delta=f"{echo_amp/max(fid_at,1e-9):.1f}× refocusing")
            st.plotly_chart(
                _fig_fid_vs_echo(t, fid_env, t2_only, sigma, T2, tau),
                width='stretch')

        if show_sphere and experiment not in (
                "Echo Sweep", "FID vs Echo (T₂* comparison)"):
            st.plotly_chart(
                _fig_bloch_sphere_3d(Mx, My, Mz,
                                     f"Bloch Sphere — {experiment}"),
                width='stretch')

        st.markdown("---")
        if experiment == "Echo Sweep":
            st.download_button(
                "Download echo sweep as CSV",
                data=pd.DataFrame({"two_tau_us": two_tau,
                                   "echo_amplitude": amps})
                    .to_csv(index=False).encode(),
                file_name="echo_sweep.csv", mime="text/csv")
        elif experiment == "FID vs Echo (T₂* comparison)":
            st.download_button(
                "Download FID vs Echo as CSV",
                data=pd.DataFrame({
                    "t_us": t, "FID_envelope": fid_env, "T2_decay": t2_only,
                }).to_csv(index=False).encode(),
                file_name="fid_vs_echo.csv", mime="text/csv")
        else:
            st.download_button(
                "Download simulation as CSV",
                data=pd.DataFrame({
                    "t_us":   t,
                    "Mx":     Mx,
                    "My":     My,
                    "Mz":     Mz,
                    "M_perp": np.sqrt(Mx**2 + My**2),
                }).to_csv(index=False).encode(),
                file_name="spin_sim.csv", mime="text/csv")


# ── Fitting tab ───────────────────────────────────────────────────────────────
with tab_fit:
    st.markdown("## Parameter Fitting")
    st.caption("Fit T₂, T₁, σ, or β to experimental or synthetic data.")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    col_ctrl, col_data = st.columns([1, 2])

    with col_ctrl:
        data_source = st.radio("Data source",
                               ["Synthetic (demo)", "Upload CSV"])
        fit_model   = st.selectbox("Fit model", list(MODEL_LABELS.keys()),
                                   format_func=lambda k: MODEL_LABELS[k])
        st.markdown("**Synthetic data settings**")
        true_T2_fit = st.slider("True T₂ (µs)",    0.5, 30.0, 8.0,  0.5)
        noise_fit   = st.slider("Noise level",      0.0,  0.1, 0.03, 0.005)
        n_pts_fit   = st.slider("Number of points", 10,   100, 40,   5)
        t_fit_range = st.slider("Time range (µs)",  1.0, 100.0,
                                float(min(4*true_T2_fit, 60.0)), 1.0)
        extra_params = {}
        if fit_model == "gaussian_fid":
            extra_params["sigma"] = st.slider(
                "True σ (synthetic)", 0.05, 1.0, 0.25, 0.05)
        elif fit_model == "T1_recovery":
            extra_params["T1"] = st.slider(
                "True T₁ (synthetic)", 1.0, 100.0, 25.0, 1.0)
        st.markdown('<div class="section-divider"></div>',
                    unsafe_allow_html=True)
        run_fit = st.button("Run Fit", type="primary", width='stretch')

    with col_data:
        if data_source == "Upload CSV":
            uploaded = st.file_uploader(
                "Two-column CSV: time, coherence", type=["csv", "txt"])
            if uploaded is not None:
                try:
                    df_up = pd.read_csv(uploaded, header=None)
                    t_up  = df_up.iloc[:, 0].values.astype(float)
                    L_up  = df_up.iloc[:, 1].values.astype(float)
                    st.session_state["uploaded_t"] = t_up
                    st.session_state["uploaded_L"] = L_up
                    st.success(f"Loaded {len(t_up)} points.")
                    pfig = go.Figure(go.Scatter(
                        x=t_up, y=L_up, mode='markers',
                        marker=dict(color=C_BLUE, size=5)))
                    pfig.update_layout(title="Uploaded data preview",
                                       template=TMPL, height=280,
                                       margin=dict(l=50,r=20,t=45,b=40))
                    pfig.update_xaxes(title_text="Time")
                    pfig.update_yaxes(title_text="Coherence")
                    st.plotly_chart(pfig, width='stretch')
                except Exception as e:
                    st.error(f"Could not parse: {e}")
            elif "uploaded_t" in st.session_state:
                st.info(f"Using previously uploaded data "
                        f"({len(st.session_state['uploaded_t'])} points).")
            else:
                st.info("Upload a two-column CSV.")
        else:
            t_prev = np.linspace(0, t_fit_range, n_pts_fit)
            rng    = np.random.default_rng(42)
            L_prev = (np.exp(-t_prev/true_T2_fit)
                      + rng.normal(0, noise_fit, len(t_prev)))
            pfig = go.Figure()
            pfig.add_trace(go.Scatter(
                x=t_prev, y=L_prev, mode='markers', name="Noisy preview",
                marker=dict(color=C_BLUE, size=5, opacity=0.8)))
            pfig.add_trace(go.Scatter(
                x=t_prev, y=np.exp(-t_prev/true_T2_fit),
                name=f"True  T₂={true_T2_fit} µs",
                line=dict(color=C_RED, dash="dash", width=1.5)))
            pfig.update_layout(title="Synthetic data preview",
                               template=TMPL, height=280,
                               hovermode="x unified",
                               margin=dict(l=50,r=20,t=45,b=40))
            pfig.update_xaxes(title_text="Time (µs)")
            pfig.update_yaxes(title_text="Coherence")
            st.plotly_chart(pfig, width='stretch')

    if run_fit:
        t_fd = L_fd = None
        with st.spinner("Fitting..."):
            try:
                if data_source == "Upload CSV":
                    if "uploaded_t" not in st.session_state:
                        st.error("No data loaded. Upload a CSV first.")
                    else:
                        t_fd = st.session_state["uploaded_t"]
                        L_fd = st.session_state["uploaded_L"]
                else:
                    t_fd, L_fd = generate_synthetic_data(
                        t_max=t_fit_range, n_points=n_pts_fit,
                        T2=true_T2_fit, noise_level=noise_fit,
                        model=fit_model, seed=42, **extra_params)
                if t_fd is not None:
                    _, param_names, _ = MODEL_REGISTRY[fit_model]
                    if fit_model in ("simple_T2", "hahn_echo"):
                        T2_r, T2_e, L_fitted = fit_T2_to_data(
                            t_fd, L_fd,
                            initial_guess=float(true_T2_fit),
                            model=fit_model)
                        params_r = {param_names[0]: T2_r}
                        errors_r = {param_names[0]: T2_e}
                    else:
                        params_r, errors_r, L_fitted = fit_multi_to_data(
                            t_fd, L_fd, model=fit_model)
                    st.session_state["fit_result"] = {
                        "t": t_fd, "L_data": L_fd, "L_fit": L_fitted,
                        "params": params_r, "errors": errors_r,
                        "model": fit_model,
                    }
            except Exception as e:
                st.error(f"Fit failed: {e}")
                st.session_state.pop("fit_result", None)

    if "fit_result" in st.session_state:
        fr = st.session_state["fit_result"]
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("**Fit result**")
        st.plotly_chart(
            _fig_fit(fr["t"], fr["L_data"], fr["L_fit"],
                     fr["params"], fr["errors"],
                     MODEL_LABELS[fr["model"]]),
            width='stretch')
        res_cols = st.columns(len(fr["params"]))
        for col, (pname, pval) in zip(res_cols, fr["params"].items()):
            col.metric(pname, f"{pval:.4f}",
                       delta=f"±{fr['errors'].get(pname, 0):.4f}")
        resid = fr["L_data"] - fr["L_fit"]
        st.caption(
            f"RMSE = {np.sqrt(np.mean(resid**2)):.5f}   ·   "
            f"χ² = {np.sum(resid**2):.5f}   ·   "
            f"N = {len(fr['t'])} points   ·   "
            f"Model: {MODEL_LABELS[fr['model']]}")
        st.download_button(
            "Download fit results as CSV",
            data=pd.DataFrame({
                "t": fr["t"], "data": fr["L_data"],
                "fit": fr["L_fit"], "residual": resid,
            }).to_csv(index=False).encode(),
            file_name="fit_results.csv", mime="text/csv")


st.markdown("---")
st.caption(
    "Spin Coherence Simulator  ·  "
    "Bloch equations  T₁/T₂  Pulse sequences  Ensemble  "
    "Rabi  Ramsey  Density Matrix  Two-Qubit Entanglement  Fitting"
)