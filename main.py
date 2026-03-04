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
    "simple_T2":    "L(t) = exp(-t/T₂)",
    "gaussian_fid": "L(t) = exp(-t/T₂) · exp(-σ²t²/2)",
    "hahn_echo":    "A(τ) = exp(-2τ/T₂)",
    "T1_recovery":  "Mz(t) = M0·(1 - exp(-t/T₁))",
    "stretched_T2": "L(t) = exp(-(t/T₂)^β)",
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
]


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
    """rho0_tag is a string key so @cache_data can hash it."""
    H = np.array(H_flat, dtype=complex).reshape(2, 2)
    rho0_map = {
        "ground":       ground_state_dm(),
        "superposition": superposition_dm(theta=np.pi/2, phi=0.0),
        "mixed_0.5":    mixed_state_dm(0.5),
        "mixed_0.0":    mixed_state_dm(0.0),
    }
    rho0 = rho0_map.get(rho0_tag, ground_state_dm())
    return simulate_dm(rho0, H, T1, T2, t_max, n=n)


@st.cache_data(show_spinner=False)
def _run_dm_validation(omega_rabi, delta, T1, T2, t_max):
    return bloch_vs_dm_error(omega_rabi, delta, T1, T2, t_max, n=400)


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
                             name=f"exp(-t/T₂)  T₂={T2} µs",
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
                             name=f"exp(-2τ/T₂)  T₂={T2} µs",
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
                             name=f"exp(-t/T₂)  T₂={T2} µs",
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
    fig.add_trace(go.Scatter(x=t_s, y=np.exp(-t_s/T2), name="exp(-t/T₂)",
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
    fig.add_trace(go.Scatter(x=t, y=np.exp(-t/T2), name="exp(-t/T₂)",
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
                             name="exp(-t/T₂)",
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
    """Ramsey fringe pattern with analytic overlay and fit."""
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
    """P↑ vs detuning Δ at fixed free evolution time — spectroscopy scan."""
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
    """Bloch vector components and purity from density matrix evolution."""
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
    # T₂ envelope on rx, ry
    env = np.exp(-t / T2)
    for (row, col) in [(1,1), (1,2)]:
        for sign in (1, -1):
            fig.add_trace(go.Scatter(x=t, y=sign*env, showlegend=False,
                                     line=dict(color=C_GREY, dash="dash", width=1),
                                     hoverinfo="skip"),
                          row=row, col=col)
    # Purity references
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
    """Density matrix vs Bloch ODE cross-validation."""
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
# Sidebar
# ============================================================================

with st.sidebar:
    st.markdown("## Spin Coherence Simulator")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    experiment = st.selectbox("Experiment type", EXPERIMENTS)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
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

    # ── Rabi parameters ───────────────────────────────────────────────────────
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

    # ── Ramsey parameters ─────────────────────────────────────────────────────
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

    # ── Density matrix parameters ─────────────────────────────────────────────
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


# ============================================================================
# Main
# ============================================================================

st.markdown("# Spin Coherence Simulator")
st.caption(
    "Bloch equations  ·  T₁/T₂ relaxation  ·  Pulse sequences  ·  "
    "Rabi oscillations  ·  Ramsey interference  ·  Density matrix  ·  Fitting")
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

tab_sim, tab_fit = st.tabs(["Simulation", "Fitting"])

with tab_sim:

    # ── CPMG ─────────────────────────────────────────────────────────────────
    if experiment == "CPMG Train":
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
                T2_star_fit = T2  # fallback
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
        # Chevron is always live (analytic)
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

            # Key metrics from final state
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
        Mx = My = Mz = None  # safe init

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
st.caption("Spin Coherence Simulator  ·  "
           "Bloch equations  T₁/T₂  Pulse sequences  Ensemble  "
           "Rabi  Ramsey  Density Matrix  Fitting")