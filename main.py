"""
app.py – Spin Coherence Simulator (Interactive Edition)
========================================================
- Analytic fast path for 5/6 experiments: instant slider response
- Plotly interactive charts: hover, zoom, pan, download PNG
- ODE solver only for CPMG (button-gated)

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

# Theme
try:
    IS_DARK = (st.get_option("theme.base") == "dark")
except Exception:
    IS_DARK = False

TMPL    = "plotly_dark"  if IS_DARK else "plotly_white"
C_BLUE  = "#4a9eda"      if IS_DARK else "#2C7BB6"
C_RED   = "#ff6b7a"      if IS_DARK else "#E63946"
C_GREEN = "#7fd77f"      if IS_DARK else "#6A994E"
C_ORNG  = "#ffb347"      if IS_DARK else "#F4A261"
C_GREY  = "#aaaaaa"

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


# ============================================================================
# Analytic solutions — exact, run in microseconds
# ============================================================================

def _analytic_single_spin(omega0, T1, T2, t_max, n=1200):
    """Full Bloch solution for spin tipped to +x: precession + T₁ + T₂."""
    t  = np.linspace(0, t_max, n)
    Mx =  np.cos(omega0 * t) * np.exp(-t / T2)
    My = -np.sin(omega0 * t) * np.exp(-t / T2)
    Mz =  1 - np.exp(-t / T1)           # Mz0 = 0 after π/2 tip
    return t, Mx, My, Mz


def _analytic_ensemble_fid(omega0, sigma, T1, T2, t_max, n=1200):
    """Exact ensemble average for Gaussian frequency spread σ.

    <Mx(t)> =  cos(ω₀t) · exp(-t/T₂) · exp(-σ²t²/2)
    <My(t)> = -sin(ω₀t) · exp(-t/T₂) · exp(-σ²t²/2)
    """
    t        = np.linspace(0, t_max, n)
    envelope = np.exp(-t / T2) * np.exp(-0.5 * sigma**2 * t**2)
    Mx       =  np.cos(omega0 * t) * envelope
    My       = -np.sin(omega0 * t) * envelope
    Mz       =  1 - np.exp(-t / T1)
    return t, Mx, My, Mz


def _analytic_hahn_echo(omega0, T1, T2, tau, n=1200):
    """Exact single-spin Hahn echo in the lab frame.

    Phase 1 (0 → τ):
        Mx =  cos(ω₀t) · exp(-t/T₂)
        My = -sin(ω₀t) · exp(-t/T₂)

    π_x pulse at t=τ flips My → -My, giving My_τ⁺ = +sin(ω₀τ)·exp(-τ/T₂)

    Phase 2 (τ → 2τ):   [derived via rotation matrix from τ⁺ initial conditions]
        Mx = cos(ω₀(2τ-t)) · exp(-t/T₂)
        My = sin(ω₀(2τ-t)) · exp(-t/T₂)

    At t=2τ:  Mx = exp(-2τ/T₂),  My = 0  →  echo.
    """
    half = n // 2
    t1   = np.linspace(0,   tau,   half,     endpoint=False)
    t2   = np.linspace(tau, 2*tau, half + 1, endpoint=True)

    # Phase 1
    Mx1 =  np.cos(omega0 * t1) * np.exp(-t1 / T2)
    My1 = -np.sin(omega0 * t1) * np.exp(-t1 / T2)
    Mz1 =  1 - np.exp(-t1 / T1)

    # Phase 2
    Mx2 =  np.cos(omega0 * (2*tau - t2)) * np.exp(-t2 / T2)
    My2 =  np.sin(omega0 * (2*tau - t2)) * np.exp(-t2 / T2)
    Mz2 =  1 - np.exp(-t2 / T1)

    t  = np.concatenate([t1, t2])
    Mx = np.concatenate([Mx1, Mx2])
    My = np.concatenate([My1, My2])
    Mz = np.concatenate([Mz1, Mz2])
    return t, Mx, My, Mz


def _analytic_echo_sweep(T2, tau_min, tau_max, n=500):
    """A(2τ) = exp(-2τ/T₂)."""
    tau  = np.linspace(tau_min, tau_max, n)
    amps = np.exp(-2 * tau / T2)
    return 2 * tau, amps


def _analytic_fid_vs_echo(sigma, T2, tau, t_max, n=1200):
    """FID envelope vs T₂-only decay for direct T₂*/T₂ comparison."""
    t       = np.linspace(0, t_max, n)
    fid_env = np.exp(-t / T2) * np.exp(-0.5 * sigma**2 * t**2)
    t2_only = np.exp(-t / T2)
    return t, fid_env, t2_only


# ============================================================================
# Cached ODE solver — only used for CPMG
# ============================================================================

@st.cache_data(show_spinner=False)
def _run_cpmg(gamma, omega0, T1, T2, tau, n_echoes, dt):
    B = np.array([0.0, 0.0, omega0 / gamma])
    return cpmg_sequence(
        gamma=gamma, B=B, T1=T1, T2=T2, M0=1.0,
        tau=tau, n_echoes=n_echoes, dt=dt)


# ============================================================================
# Plotly figure builders
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
    fig = make_subplots(
        rows=1, cols=3, shared_yaxes=False,
        subplot_titles=("Mx(t)", "My(t)", "Mz(t)"))
    env = np.exp(-t / T2)

    for col, data, color, name in [
        (1, Mx, C_BLUE,  "Mx"),
        (2, My, C_RED,   "My"),
        (3, Mz, C_GREEN, "Mz"),
    ]:
        fig.add_trace(
            go.Scatter(x=t, y=data, name=name,
                       line=dict(color=color, width=2)),
            row=1, col=col)

    # T₂ envelope on Mx and My panels
    for col in (1, 2):
        for sign in (1, -1):
            fig.add_trace(
                go.Scatter(x=t, y=sign*env, showlegend=False,
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
    fig.update_layout(**_base_layout(
        f"Hahn Echo  τ={tau} µs  T₂={T2} µs"))
    fig.update_xaxes(title_text="Time (µs)")
    fig.update_yaxes(title_text="|M⊥|", range=[-0.05, 1.1])
    return fig


def _fig_echo_sweep(two_tau, amps, T2):
    log_A  = np.log(np.clip(amps, 1e-10, None))
    coeffs = np.polyfit(two_tau, log_A, 1)
    T2_fit = -1.0 / coeffs[0]

    # Smooth theory curves
    tt = np.linspace(two_tau[0], two_tau[-1], 600)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tt, y=np.exp(-tt / T2),
        name=f"exp(-2τ/T₂)  T₂={T2} µs",
        line=dict(color=C_BLUE, dash="dash", width=1.8)))
    fig.add_trace(go.Scatter(
        x=tt, y=np.exp(coeffs[0]*tt + coeffs[1]),
        name=f"Fitted  T₂ = {T2_fit:.2f} µs",
        line=dict(color=C_GREEN, dash="dot", width=1.8)))
    fig.add_trace(go.Scatter(
        x=two_tau, y=amps, name="Echo amplitudes",
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
    fig.add_trace(go.Scatter(
        x=t, y=fid_env,
        name=f"FID envelope  σ={sigma:.2f} rad/µs",
        line=dict(color=C_RED, width=2.5)))
    fig.add_trace(go.Scatter(
        x=t, y=t2_only,
        name=f"exp(-t/T₂)  T₂={T2} µs",
        line=dict(color=C_BLUE, dash="dash", width=1.8)))
    fig.add_trace(go.Scatter(
        x=[2*tau], y=[echo_amp], mode="markers",
        marker=dict(color=C_GREEN, size=14, symbol="star"),
        name=f"Echo amp at 2τ = {echo_amp:.4f}"))
    fig.add_trace(go.Scatter(
        x=[2*tau], y=[fid_at], mode="markers",
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

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Transverse components", "FID envelope"))
    fig.add_trace(go.Scatter(x=t, y=Mx, name="⟨Mx⟩",
                             line=dict(color=C_BLUE, width=1.8)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=My, name="⟨My⟩",
                             line=dict(color=C_RED, width=1.8), opacity=0.85),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=M_perp, name="|⟨M⊥⟩|",
                             line=dict(color=C_BLUE, width=2.5)),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=t_s, y=env, name="Analytic envelope",
                             line=dict(color=C_RED, dash="dash", width=1.5)),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=t_s, y=np.exp(-t_s/T2),
                             name="exp(-t/T₂)",
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

    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.65, 0.35],
        shared_xaxes=True,
        subplot_titles=("Echo train", "Echo amplitude decay"))

    fig.add_trace(go.Scatter(x=t, y=M_perp, name="|M⊥|(t)",
                             line=dict(color=C_BLUE, width=1.8)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=Mz, name="Mz(t)",
                             line=dict(color=C_GREEN, width=1.2, dash="dash"),
                             opacity=0.7), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=np.exp(-t/T2),
                             name="exp(-t/T₂)",
                             line=dict(color=C_GREY, dash="dot", width=1)),
                  row=1, col=1)

    # Echo peak markers in waveform panel
    fig.add_trace(go.Scatter(
        x=echo_times, y=echo_amps, mode="markers",
        marker=dict(color=C_ORNG, size=9, symbol="diamond"),
        name="Echo peaks", showlegend=True), row=1, col=1)

    # Echo amplitude decay panel
    fig.add_trace(go.Scatter(
        x=echo_times, y=echo_amps, name="Measured",
        mode="markers+lines",
        marker=dict(color=C_BLUE, size=9),
        line=dict(color=C_BLUE, width=1.5)), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=echo_times, y=np.exp(-np.array(echo_times)/T2),
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
    theta = np.linspace(0, 2*np.pi, 80)
    traces = []

    # Latitude circles
    for zval in np.linspace(-0.8, 0.8, 5):
        r = np.sqrt(max(0, 1 - zval**2))
        traces.append(go.Scatter3d(
            x=r*np.cos(theta), y=r*np.sin(theta), z=np.full_like(theta, zval),
            mode='lines', line=dict(color='rgba(160,160,160,0.2)', width=1),
            showlegend=False, hoverinfo='skip'))

    # Longitude lines
    for angle in np.linspace(0, np.pi, 7):
        traces.append(go.Scatter3d(
            x=np.cos(angle)*np.cos(theta),
            y=np.sin(angle)*np.cos(theta),
            z=np.sin(theta),
            mode='lines', line=dict(color='rgba(160,160,160,0.2)', width=1),
            showlegend=False, hoverinfo='skip'))

    # Axis arrows
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

    # Trajectory colored by time
    n = len(Mx)
    traces.append(go.Scatter3d(
        x=Mx, y=My, z=Mz,
        mode='lines',
        line=dict(color=np.linspace(0, 1, n),
                  colorscale='Plasma', width=5,
                  colorbar=dict(title="t / t_max",
                                thickness=12, len=0.5, x=1.02)),
        name='Trajectory'))

    # Start / end
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
            aspectmode='cube',
        ))
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
                             line=dict(color=C_RED, width=2.2)),
                  row=1, col=1)
    fig.add_trace(go.Bar(x=t_data, y=resid, name='Residual',
                         marker_color=C_GREEN, opacity=0.7), row=2, col=1)
    fig.add_hline(y=0, line_color=C_GREY, line_width=0.8, row=2, col=1)

    param_str = "    ".join(
        f"{k} = {v:.4f} ± {errors.get(k,0):.4f}"
        for k, v in params.items())
    fig.update_layout(
        title=f"Model: {model_label}<br><sup>{param_str}</sup>",
        template=TMPL, height=480, hovermode="x unified",
        margin=dict(l=60, r=20, t=85, b=50))
    fig.update_xaxes(title_text="Time (µs)", row=2, col=1)
    fig.update_yaxes(title_text="Coherence", row=1, col=1)
    fig.update_yaxes(title_text="Residual",  row=2, col=1)
    return fig


# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    st.markdown("## Spin Coherence Simulator")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    experiment = st.selectbox(
        "Experiment type",
        ["Single Spin (Bloch)",
         "Ensemble FID",
         "Hahn Echo",
         "CPMG Train",
         "Echo Sweep",
         "FID vs Echo (T₂* comparison)"],
    )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("**Physical parameters**")

    gamma  = 1.0
    f0     = st.slider("B₀ frequency f₀ (MHz)", 0.1, 2.0, 0.5, 0.05)
    omega0 = 2 * np.pi * f0
    T2     = st.slider("T₂ (µs)", 0.5, 50.0, 10.0, 0.5)
    T1     = st.slider("T₁ (µs)", T2, 200.0, float(max(T2*3, T2+1.0)), 1.0)

    needs_sigma = experiment in [
        "Ensemble FID", "FID vs Echo (T₂* comparison)"]
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

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("**Display**")

    t_max_def = float(max(4*T2, 4*tau if needs_tau else 4*T2))
    t_max     = st.slider("t_max (µs)", 5.0, 200.0, t_max_def, 5.0)
    show_sphere = st.checkbox("Show 3D Bloch sphere", value=False)

    if experiment == "CPMG Train":
        dt = st.select_slider("Time step dt (µs)",
                              [0.005, 0.01, 0.02, 0.05, 0.1], value=0.05)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        run_cpmg = st.button("Run CPMG", type="primary",
                             width='stretch')
    else:
        dt, run_cpmg = 0.05, False


# ============================================================================
# Main
# ============================================================================

st.markdown("# Spin Coherence Simulator")
st.caption(
    "Bloch equations  ·  T₁/T₂ relaxation  ·  Pulse sequences  ·  "
    "Inhomogeneous ensemble  ·  Fitting  ·  Interactive Plotly charts")
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

tab_sim, tab_fit = st.tabs(["Simulation", "Fitting"])


# ── Simulation tab ────────────────────────────────────────────────────────────
with tab_sim:

    # ── CPMG (ODE, button-gated) ──────────────────────────────────────────────
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
            r = st.session_state["cpmg"]
            Mp = np.sqrt(r["Mx"]**2 + r["My"]**2)
            e_amps = [float(np.interp(et, r["t"], Mp)) for et in r["echo_times"]]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("T₂ (µs)", f"{r['T2']:.1f}")
            c2.metric("τ (µs)",  f"{r['tau']:.1f}")
            c3.metric("First echo",
                      f"{e_amps[0]:.4f}",
                      delta=f"exp = {np.exp(-2*r['tau']/r['T2']):.4f}")
            c4.metric("Last echo",
                      f"{e_amps[-1]:.4f}",
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

    # ── All other experiments: analytic, instant slider updates ───────────────
    else:

        # Initialise variables that are only set in some branches so that
        # the shared download / Bloch-sphere code below never hits a NameError.
        Mx = My = Mz = None

        # Single Spin
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

        # Ensemble FID
        elif experiment == "Ensemble FID":
            t, Mx, My, Mz = _analytic_ensemble_fid(omega0, sigma, T1, T2, t_max)
            M_perp = np.sqrt(Mx**2 + My**2)
            T2star = 1.0 / (1.0/T2 + sigma) if sigma > 0 else T2
            idx    = np.argmin(np.abs(t - T2))
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("T₂ (µs)",            f"{T2:.1f}")
            c2.metric("T₂* estimate (µs)",  f"{T2star:.2f}")
            c3.metric("σ (rad/µs)",          f"{sigma:.2f}")
            c4.metric("|⟨M⊥⟩| at T₂",      f"{M_perp[idx]:.4f}")
            st.plotly_chart(
                _fig_fid_ensemble(t, Mx, My, sigma, T2),
                width='stretch')

        # Hahn Echo
        elif experiment == "Hahn Echo":
            t, Mx, My, Mz = _analytic_hahn_echo(omega0, T1, T2, tau)
            M_perp   = np.sqrt(Mx**2 + My**2)
            echo_amp = float(np.interp(2*tau, t, M_perp))
            expected = float(np.exp(-2*tau / T2))
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("T₂ (µs)",             f"{T2:.1f}")
            c2.metric("τ (µs)",              f"{tau:.1f}")
            c3.metric("Echo amp at 2τ",      f"{echo_amp:.4f}",
                      delta=f"expected {expected:.4f}")
            c4.metric("Error",
                      f"{abs(echo_amp-expected):.2e}")
            st.plotly_chart(
                _fig_echo_detail(t, Mx, My, tau, T2),
                width='stretch')
            st.plotly_chart(
                _fig_bloch_components(t, Mx, My, Mz, T2,
                    f"Hahn Echo Components  τ={tau} µs  T₂={T2} µs"),
                width='stretch')

        # Echo Sweep
        elif experiment == "Echo Sweep":
            two_tau, amps = _analytic_echo_sweep(T2, tau_min, tau_max, n_tau)
            log_A   = np.log(np.clip(amps, 1e-10, None))
            T2_fit  = -1.0 / np.polyfit(two_tau, log_A, 1)[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("True T₂ (µs)",   f"{T2:.1f}")
            c2.metric("Fitted T₂ (µs)", f"{T2_fit:.2f}",
                      delta=f"Δ = {T2_fit-T2:.3f}")
            c3.metric("Sweep points",    str(n_tau))
            fig_sw, _ = _fig_echo_sweep(two_tau, amps, T2)
            st.plotly_chart(fig_sw, width='stretch')

        # FID vs Echo
        elif experiment == "FID vs Echo (T₂* comparison)":
            t, fid_env, t2_only = _analytic_fid_vs_echo(sigma, T2, tau, t_max)
            echo_amp = float(np.exp(-2*tau / T2))
            fid_at   = float(np.exp(-2*tau/T2) * np.exp(-0.5*sigma**2*(2*tau)**2))
            T2star   = 1.0/(1.0/T2 + sigma) if sigma > 0 else T2
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("T₂ (µs)",             f"{T2:.1f}")
            c2.metric("T₂* estimate (µs)",   f"{T2star:.2f}")
            c3.metric("FID |M⊥| at 2τ",      f"{fid_at:.4f}")
            c4.metric("Echo amp at 2τ",       f"{echo_amp:.4f}",
                      delta=f"{echo_amp/max(fid_at,1e-9):.1f}× refocusing")
            st.plotly_chart(
                _fig_fid_vs_echo(t, fid_env, t2_only, sigma, T2, tau),
                width='stretch')

        # ── 3D Bloch sphere — only for experiments that produce Mx/My/Mz ──────
        # FID vs Echo and Echo Sweep do not compute per-component trajectories,
        # so exclude them here to avoid a NameError.
        if show_sphere and experiment not in (
                "Echo Sweep", "FID vs Echo (T₂* comparison)"):
            st.plotly_chart(
                _fig_bloch_sphere_3d(Mx, My, Mz,
                    f"Bloch Sphere — {experiment}"),
                width='stretch')

        # ── CSV export ────────────────────────────────────────────────────────
        st.markdown("---")
        if experiment == "Echo Sweep":
            st.download_button(
                "Download echo sweep as CSV",
                data=pd.DataFrame({"two_tau_us": two_tau,
                                   "echo_amplitude": amps})
                    .to_csv(index=False).encode(),
                file_name="echo_sweep.csv", mime="text/csv")
        elif experiment == "FID vs Echo (T₂* comparison)":
            # Mx/My/Mz are not computed for this experiment; export
            # the envelope arrays that are actually calculated instead.
            st.download_button(
                "Download FID vs Echo as CSV",
                data=pd.DataFrame({
                    "t_us":        t,
                    "FID_envelope": fid_env,
                    "T2_decay":    t2_only,
                }).to_csv(index=False).encode(),
                file_name="fid_vs_echo.csv", mime="text/csv")
        else:
            st.download_button(
                "Download simulation as CSV",
                data=pd.DataFrame({
                    "t_us": t, "Mx": Mx, "My": My, "Mz": Mz,
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
        run_fit = st.button("Run Fit", type="primary",
                            width='stretch')

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
            # Live preview — updates as sliders move, never touches fit state
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

    # Run fit — generate data + fit together as one atomic action
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

                    # Store complete snapshot — persists across all rerenders
                    st.session_state["fit_result"] = {
                        "t": t_fd, "L_data": L_fd, "L_fit": L_fitted,
                        "params": params_r, "errors": errors_r,
                        "model": fit_model,
                    }
            except Exception as e:
                st.error(f"Fit failed: {e}")
                st.session_state.pop("fit_result", None)

    # Render persisted fit result
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
           "Bloch equations  T₁/T₂  Pulse sequences  Ensemble  Fitting")