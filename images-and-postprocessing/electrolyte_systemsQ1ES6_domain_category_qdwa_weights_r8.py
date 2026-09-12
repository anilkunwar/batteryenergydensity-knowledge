import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator
from io import BytesIO
import base64
import shutil

# Optional cubic-spline smoothing
try:
    from scipy.interpolate import make_interp_spline
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Real LaTeX almost never exists on Streamlit Cloud -> detect it
LATEX_AVAILABLE = shutil.which("latex") is not None

# ------------------- DATA -------------------
# QDWA scoring for Q1ES6 — Bayesian-smoothed weights:
#
#   W_k = (α + raw_k) / (6·α + Σ_j raw_j),   α = 0.25
#
# Weights calibrated for Q1ES6: Electrolyte Systems is highest by a wide margin —
# the query names two electrolyte-domain levers (FEC additives and solid
# electrolytes). Performance Metrics follows directly ("high rate capability" →
# rate / C-rate / power outcomes). Anode Materials (FEC-driven SEI/interphase
# site; Li plating limits fast charge) and Cathode Materials (rate kinetics,
# high-voltage interfaces) are present but secondary to the electrolyte
# formulation itself. Degradation (FEC stabilizes cycling) and Manufacturing
# receive minor weights.
#
# "–" marks a component that contributed nothing for that category.
# >>> Edit numbers in QDWA_TABLE only. <<<

ALPHA = 0.25   # Bayesian prior α (same for all six categories)

QDWA_TABLE = [
    # Category,             Keyword Hits,  Concept Mass,  Soft Term Sum
    ("Electrolyte Systems", 5.0,           1.5,           0.0),   # raw_k = 6.5 → W_k ≈ 0.3245
    ("Performance Metrics", 3.5,           1.0,           0.0),   # raw_k = 4.5 → W_k ≈ 0.2284
    ("Anode Materials",     2.5,           1.0,           0.0),   # raw_k = 3.5 → W_k ≈ 0.1803
    ("Cathode Materials",   2.0,           0.8,           0.0),   # raw_k = 2.8 → W_k ≈ 0.1466
    ("Degradation",         1.2,           0.3,           0.0),   # raw_k = 1.5 → W_k ≈ 0.0841
    ("Manufacturing",       0.3,           0.2,           0.0),   # raw_k = 0.5 → W_k ≈ 0.0361
]

_BASE_CATEGORIES = [row[0] for row in QDWA_TABLE]
_N_CATEGORIES = len(_BASE_CATEGORIES)

_DASH = {"-", "–", "−", ""}          # accept any dash/blank style you paste in

def _term(x):
    """A dash (or blank) contributes 0 to raw_k."""
    return 0.0 if isinstance(x, str) and x.strip() in _DASH else float(x)

# ---- Build raw_k from the three sub-terms ----
df_qdwa = pd.DataFrame(
    QDWA_TABLE,
    columns=["Category", "Keyword Hits", "Concept Mass", "Soft Term Sum"],
)
df_qdwa["raw_k"] = (
    df_qdwa["Keyword Hits"].map(_term)
    + df_qdwa["Concept Mass"].map(_term)
    + df_qdwa["Soft Term Sum"].map(_term)
).round(4)

# ---- Bayesian-smoothed weight  W_k = (α + raw_k) / (N·α + Σ raw_j) ----
_sum_raw = df_qdwa["raw_k"].sum()
_denominator = _N_CATEGORIES * ALPHA + _sum_raw
df_qdwa["α (prior)"] = ALPHA
df_qdwa["Numerator"] = (ALPHA + df_qdwa["raw_k"]).round(4)
df_qdwa["Denominator"] = round(_denominator, 4)
df_qdwa["W_k"] = (df_qdwa["Numerator"] / df_qdwa["Denominator"]).round(4)

def _chart_df(round_to=None):
    """Chart data: raw_k and W_k from the Bayesian QDWA table."""
    df = pd.DataFrame({"Category": df_qdwa["Category"],
                       "raw_k": df_qdwa["raw_k"],
                       "W_k": df_qdwa["W_k"]})
    if round_to is not None:
        df["raw_k"] = df["raw_k"].round(round_to)
    return df

df_exact   = _chart_df()               # exact scores
df_rounded = _chart_df(round_to=1)     # 1-decimal rounded variant

# ------------------- PLOT FUNCTION -------------------
def plot_dual_axis(df, cfg):
    # ---- Global style: fonts, math engine, spines ----
    mpl.rcParams.update({
        "font.family": cfg["font_family"],
        "font.size": cfg["font_size"],
        "mathtext.fontset": cfg["mathtext_fontset"],   # 'cm' = LaTeX Computer Modern look
        "text.usetex": cfg["use_usetex"],              # real LaTeX (only if installed)
        "axes.labelweight": cfg["label_weight"],
    })
    if cfg["use_usetex"]:
        mpl.rcParams["text.latex.preamble"] = (
            r"\usepackage{amsmath}" "\n" r"\usepackage{amssymb}"
        )

    fig, ax1 = plt.subplots(figsize=(cfg["fig_width"], cfg["fig_height"]))
    xs = np.arange(len(df))

    # ---------- Bars (left axis) ----------
    ax1.bar(xs, df["raw_k"], width=cfg["bar_width"], color=cfg["bar_color"],
            alpha=cfg["bar_alpha"], edgecolor=cfg["bar_edge_color"],
            linewidth=cfg["bar_edge_width"], label="Raw Evidence (left)")
    ax1.set_xlabel("Category", fontsize=cfg["font_size"])
    ax1.set_ylabel(cfg["ylabel_left"], fontsize=cfg["font_size"],
                   color=cfg["bar_color"], fontweight=cfg["label_weight"])

    ax1.set_xticks(xs)
    ax1.set_xticklabels(df["Category"], fontsize=cfg["font_size"],
                        rotation=cfg["x_rotation"])
    if cfg["x_rotation"] != 0:
        plt.setp(ax1.get_xticklabels(), ha="right", rotation_mode="anchor")

    # ---------- Line / spline (right axis) ----------
    ax2 = ax1.twinx()
    if cfg["smooth_line"] and SCIPY_AVAILABLE and len(xs) > 3:
        x_s = np.linspace(xs.min(), xs.max(), 300)
        w_s = make_interp_spline(xs, df["W_k"].to_numpy(), k=3)(x_s)
        ax2.plot(x_s, w_s, color=cfg["line_color"],
                 linewidth=cfg["line_width"],
                 linestyle=cfg["line_style"], label="Bayesian Weight (right)")
        ax2.plot(xs, df["W_k"], linestyle="none", marker=cfg["marker_style"],
                 markersize=cfg["marker_size"], color=cfg["line_color"],
                 markeredgewidth=cfg["marker_edge_width"],
                 markeredgecolor=cfg["marker_edge_color"])
    else:
        ax2.plot(xs, df["W_k"], color=cfg["line_color"],
                 linewidth=cfg["line_width"],
                 linestyle=cfg["line_style"], marker=cfg["marker_style"],
                 markersize=cfg["marker_size"],
                 markeredgewidth=cfg["marker_edge_width"],
                 markeredgecolor=cfg["marker_edge_color"],
                 label="Bayesian Weight (right)")

    ax2.set_ylabel(cfg["ylabel_right"], fontsize=cfg["font_size"],
                   color=cfg["line_color"], fontweight=cfg["label_weight"])

    # ---------- Ticks: length, width, direction, padding ----------
    tkw = dict(length=cfg["tick_length"], width=cfg["tick_width"],
               direction=cfg["tick_direction"], pad=cfg["tick_pad"])
    ax1.tick_params(axis="y", labelcolor=cfg["bar_color"],
                    labelsize=cfg["font_size"], **tkw)
    ax1.tick_params(axis="x", **tkw)
    ax2.tick_params(axis="y", labelcolor=cfg["line_color"],
                    labelsize=cfg["font_size"], **tkw)

    # ---------- Minor ticks ----------
    if cfg["minor_ticks"]:
        ax1.yaxis.set_minor_locator(AutoMinorLocator())
        ax2.yaxis.set_minor_locator(AutoMinorLocator())
        for ax in (ax1, ax2):
            ax.tick_params(which="minor", length=cfg["tick_length"] * 0.5,
                           width=cfg["tick_width"] * 0.75,
                           direction=cfg["tick_direction"])

    # ---------- Spines (axis frame) ----------
    for ax in (ax1, ax2):
        for spine in ax.spines.values():
            spine.set_linewidth(cfg["spine_width"])
        ax.spines["top"].set_visible(cfg["top_spines"])
    ax1.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)
    ax1.spines["left"].set_color(cfg["bar_color"])
    ax2.spines["right"].set_color(cfg["line_color"])

    # ---------- Grid ----------
    ax1.grid(cfg["show_grid"], axis="y", linestyle=cfg["grid_style"],
             linewidth=cfg["grid_width"], alpha=cfg["grid_alpha"])
    ax2.grid(False)
    ax1.set_axisbelow(True)

    # ---------- Title & legend ----------
    version = "Rounded" if cfg["use_rounded"] else "Exact"
    ax1.set_title(f"Q1ES6 Dual-Axis Chart — {version} Data",
                  fontsize=cfg["font_size"] + 2, pad=12,
                  fontweight="bold" if cfg["bold_title"] else "normal")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc=cfg["legend_loc"],
               frameon=cfg["legend_frame"], fontsize=cfg["legend_fontsize"],
               framealpha=0.9, edgecolor="black")

    fig.tight_layout()
    return fig

# ------------------- DOWNLOAD FUNCTION -------------------
MIME_TYPES = {"png": "image/png", "pdf": "application/pdf",
              "svg": "image/svg+xml", "eps": "application/postscript",
              "tiff": "image/tiff"}

def get_download_link(fig, dpi, fmt, transparent):
    buf = BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight",
                transparent=transparent)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = (f'<a href="data:{MIME_TYPES[fmt]};base64,{b64}" '
            f'download="Q1ES6_dual_axis_chart.{fmt}">Download {fmt.upper()}</a>')
    return href

# ------------------- STREAMLIT UI -------------------
st.set_page_config(page_title="Q1ES6: FEC Additives & Solid Electrolytes", layout="wide")
st.title("📊 Q1ES6: Optimizing FEC Additives and Solid Electrolytes for High Rate Capability")
st.markdown("**Query:** To optimize FEC additives and solid electrolytes for high rate capability.  "
            "**Highest Domain:** Electrolyte Systems")
st.markdown("Compare raw evidence (bars, left) and Bayesian-smoothed weights "
            "$W_k$ (line, right) with full publication-style control.")

st.sidebar.header("Chart Customization")

# ---- X-Axis Order Expander ----
with st.sidebar.expander("🔀 X-Axis Domain Order", expanded=True):
    st.caption("Select the order of categories on the x-axis:")
    ordered_categories = []
    remaining_cats = _BASE_CATEGORIES.copy()
    for i in range(len(remaining_cats)):
        selected = st.selectbox(f"Position {i+1}", remaining_cats, key=f"pos_{i}")
        ordered_categories.append(selected)
        remaining_cats = [c for c in remaining_cats if c != selected]

# ---- Data & Colors Expander ----
with st.sidebar.expander("🎨 Data & Colors", expanded=True):
    use_rounded = st.checkbox("Use Rounded Data", value=False)
    bar_color = st.color_picker("Bar Color", "#1f77b4")
    line_color = st.color_picker("Line Color", "#ff7f0e")
    bar_alpha = st.slider("Bar transparency", 0.1, 1.0, 0.7, 0.05)
    bar_width = st.slider("Bar width", 0.2, 1.0, 0.8, 0.05)
    bar_edge_color = st.color_picker("Bar edge color", "#000000")
    bar_edge_width = st.slider("Bar edge width", 0.0, 2.0, 0.5, 0.1)

# ---- Main Area: QDWA Scoring Table ----
with st.expander("🧮 QDWA scoring (Bayesian formula + table)"):
    st.markdown(r"$$W_k \;=\; \frac{\alpha + \mathrm{raw}_k}"
                r"{N\,\alpha + \sum_{j} \mathrm{raw}_j}"
                r"\qquad \alpha = 0.25,\; N = 6$$")
    st.markdown(r"$$\mathrm{raw}_k \;=\; \sum_{\mathrm{kw}} \mathbf{1}[\mathrm{kw} \in k]"
                r"\;+\;\sum_{c} w_c\,\mathbf{1}[\mathrm{cat}(c) = k]"
                r"\;+\;\sum_{t \in q} m_k(t)$$")
    st.caption("raw_k = keyword hits + concept mass + soft-term sum per category. "
               "W_k = (α + raw_k) / (N·α + Σ raw_j) — Bayesian-smoothed weight with "
               "uniform prior α = 0.25 over N = 6 categories.")
    # Display the full Bayesian table, reordered to match the chart
    cols_display = ["Category", "α (prior)", "raw_k", "Numerator", "Denominator", "W_k"]
    df_qdwa_display = df_qdwa[cols_display].set_index("Category").loc[ordered_categories].reset_index()
    st.dataframe(df_qdwa_display.style.format({
        "α (prior)": "{:.2f}",
        "raw_k": "{:.4f}",
        "Numerator": "{:.4f}",
        "Denominator": "{:.4f}",
        "W_k": "{:.4f}",
    }))

# ---- Main Area: Math Cheat Sheet ----
with st.expander("✍️ Math notation cheat sheet (works in the label boxes below)"):
    st.code(r"""
 $w_k$              subscript
 $x^2$              superscript
 $\frac{a}{b}$      fraction
 $\sqrt{x}$         square root
 $\sum_{j=1}^{K}$   summation
 $\int_0^1 f\,dx$   integral
 $\alpha\beta\Delta\Omega$   Greek letters
 $\mathrm{raw}$     upright roman text inside math
 $\mathbf{w}$       bold   /  $\mathcal{L}$  calligraphic
""", language=None)

# ---- Sidebar: Line / Spline ----
with st.sidebar.expander("✏️ Line / Spline", expanded=True):
    line_width = st.slider("Spline (line) width", 0.5, 6.0, 2.0, 0.1)
    line_style = st.selectbox("Line style", ["solid", "dashed", "dashdot", "dotted"])
    marker_style = st.selectbox("Marker", ["o", "s", "^", "D", "P", "*", "v", "X", "None"])
    marker_size = st.slider("Marker size", 3, 16, 8)
    marker_edge_width = st.slider("Marker edge width", 0.0, 3.0, 1.5, 0.1)
    marker_edge_color = st.color_picker("Marker edge color", "#ffffff")
    if SCIPY_AVAILABLE:
        smooth_line = st.checkbox("Smooth curve (cubic spline fit)", value=False)
    else:
        st.caption("scipy not installed — smoothing disabled")
        smooth_line = False

# ---- Sidebar: Fonts & Math ----
with st.sidebar.expander("🔤 Fonts & Math", expanded=True):
    font_family = st.selectbox("Font family", ["sans-serif", "serif", "monospace"])
    mathtext_fontset = st.selectbox(
        "Math font set (mathtext)",
        ["dejavusans", "dejavuserif", "cm", "stix", "stixsans"], index=2,
        help="'cm' = Computer Modern → classic LaTeX look. "
             "'stix' + serif family ≈ Times New Roman. No LaTeX install needed.")
    ylabel_left = st.text_input("Left y-label (LaTeX ok)",
                                value=r"Raw Evidence $\mathrm{raw}_k$")
    ylabel_right = st.text_input("Right y-label (LaTeX ok)",
                                 value=r"Bayesian Weight $W_k$")
    label_weight = st.selectbox("Axis label weight", ["normal", "bold"])
    bold_title = st.checkbox("Bold title", value=True)
    if LATEX_AVAILABLE:
        use_usetex = st.checkbox("Use real LaTeX (text.usetex)", value=False)
    else:
        st.info("No LaTeX installation found → using built-in mathtext. "
                "Pick 'cm' for the LaTeX look.")
        use_usetex = False

# ---- Sidebar: Ticks & Spines ----
with st.sidebar.expander("📏 Ticks & Spines"):
    tick_length = st.slider("Major tick length", 0, 15, 5)
    tick_width = st.slider("Major tick width", 0.1, 3.0, 1.0, 0.1)
    tick_direction = st.selectbox("Tick direction", ["out", "in", "inout"])
    tick_pad = st.slider("Tick label padding", 1, 15, 4)
    minor_ticks = st.checkbox("Show minor ticks (y axes)", value=False)
    spine_width = st.slider("Spine (frame) width", 0.5, 3.0, 1.0, 0.1)
    top_spines = st.checkbox("Show top spine", value=False)
    x_rotation = st.select_slider("X-label rotation", options=[0, 15, 30, 45, 60, 90], value=45)

# ---- Sidebar: Grid & Legend ----
with st.sidebar.expander("🔀 Grid & Legend"):
    show_grid = st.checkbox("Show grid", value=True)
    grid_style = st.selectbox("Grid line style", ["--", "-", ":", "-."])
    grid_width = st.slider("Grid line width", 0.3, 2.0, 0.6, 0.1)
    grid_alpha = st.slider("Grid transparency", 0.05, 1.0, 0.6, 0.05)
    legend_loc = st.selectbox("Legend location",
                              ["upper left", "upper right", "lower left", "lower right", "best"])
    legend_frame = st.checkbox("Legend frame", value=True)
    legend_fontsize = st.slider("Legend font size", 6, 20, 10)

# ---- Sidebar: Figure & Export ----
with st.sidebar.expander("🖼 Figure & Export"):
    font_size = st.slider("Font size", 8, 24, 12)
    fig_width = st.slider("Figure width (inches)", 4, 12, 8)
    fig_height = st.slider("Figure height (inches)", 3, 9, 5)
    dpi = st.selectbox("Export DPI (raster formats)", [100, 200, 300, 600], index=2)
    export_format = st.selectbox("Export format", ["png", "pdf", "svg", "eps", "tiff"],
                                 help="PDF/SVG/EPS are vector formats — ideal for journals.")
    transparent_bg = st.checkbox("Transparent background", value=False)

# Apply user's custom ordering to the dataframes
df = df_rounded if use_rounded else df_exact
df = df.set_index("Category").loc[ordered_categories].reset_index()

cfg = dict(use_rounded=use_rounded, bar_color=bar_color, line_color=line_color,
           bar_alpha=bar_alpha, bar_width=bar_width, bar_edge_color=bar_edge_color,
           bar_edge_width=bar_edge_width, line_width=line_width, line_style=line_style,
           marker_style=marker_style, marker_size=marker_size,
           marker_edge_width=marker_edge_width, marker_edge_color=marker_edge_color,
           smooth_line=smooth_line, font_family=font_family,
           mathtext_fontset=mathtext_fontset, use_usetex=use_usetex,
           ylabel_left=ylabel_left, ylabel_right=ylabel_right,
           bold_title=bold_title, label_weight=label_weight,
           tick_length=tick_length, tick_width=tick_width,
           tick_direction=tick_direction, tick_pad=tick_pad,
           minor_ticks=minor_ticks, spine_width=spine_width,
           top_spines=top_spines, x_rotation=x_rotation, show_grid=show_grid,
           grid_style=grid_style, grid_width=grid_width, grid_alpha=grid_alpha,
           legend_loc=legend_loc, legend_frame=legend_frame,
           legend_fontsize=legend_fontsize, font_size=font_size,
           fig_width=fig_width, fig_height=fig_height)

st.subheader("Data Used")
st.dataframe(df)

fig = plot_dual_axis(df, cfg)

try:
    st.pyplot(fig)
    st.markdown(get_download_link(fig, dpi, export_format, transparent_bg),
                unsafe_allow_html=True)
except Exception as e:
    st.error(f"Rendering failed — usually a LaTeX/mathtext syntax error in a label: {e}")
