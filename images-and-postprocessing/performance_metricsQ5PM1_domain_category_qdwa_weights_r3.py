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

# ------------------- QUERY INFORMATION -------------------
QUERY_ACRONYM = "Q5PM1"
QUERY_TEXT = "Innovative approaches for increasing specific energy of batteries beyond 500 W h/kg"
HIGHEST_DOMAIN = "Performance Metrics"
ALPHA = 0.25

# ------------------- DATA (Q5PM1 - QDWA Weights) -------------------
data_exact = {
    "Category": ["Cathode Materials", "Anode Materials", "Electrolyte Systems",
                 "Manufacturing", "Degradation", "Performance Metrics"],
    "raw_k": [1.5968, 3.0796, 0.0728, 0.8139, 0.1312, 5.5057],
    "w_k": [0.1454, 0.2622, 0.0254, 0.0838, 0.0300, 0.4532]
}
df_exact = pd.DataFrame(data_exact)

data_rounded = {
    "Category": ["Cathode Materials", "Anode Materials", "Electrolyte Systems",
                 "Manufacturing", "Degradation", "Performance Metrics"],
    "raw_k": [1.60, 3.08, 0.07, 0.81, 0.13, 5.51],
    "w_k": [0.145, 0.262, 0.025, 0.084, 0.030, 0.453]
}
df_rounded = pd.DataFrame(data_rounded)

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
        w_s = make_interp_spline(xs, df["w_k"].to_numpy(), k=3)(x_s)
        ax2.plot(x_s, w_s, color=cfg["line_color"],
                 linewidth=cfg["line_width"],            # <-- spline width
                 linestyle=cfg["line_style"], label="Smoothed Weight (right)")
        ax2.plot(xs, df["w_k"], linestyle="none", marker=cfg["marker_style"],
                 markersize=cfg["marker_size"], color=cfg["line_color"],
                 markeredgewidth=cfg["marker_edge_width"],
                 markeredgecolor=cfg["marker_edge_color"])
    else:
        ax2.plot(xs, df["w_k"], color=cfg["line_color"],
                 linewidth=cfg["line_width"],            # <-- spline width
                 linestyle=cfg["line_style"], marker=cfg["marker_style"],
                 markersize=cfg["marker_size"],
                 markeredgewidth=cfg["marker_edge_width"],
                 markeredgecolor=cfg["marker_edge_color"],
                 label="Smoothed Weight (right)")

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
            spine.set_linewidth(cfg["spine_width"])     # <-- spine thickness
        ax.spines["top"].set_visible(cfg["top_spines"])
    ax1.spines["right"].set_visible(False)   # hidden under ax2's colored spine
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
    title_text = (f"QDWA Weights - Query {QUERY_ACRONYM}\n"
                  f"{version} Data - Highest Domain: {HIGHEST_DOMAIN}")
    ax1.set_title(title_text,
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
            f'download="QDWA_{QUERY_ACRONYM}_chart.{fmt}">Download {fmt.upper()}</a>')
    return href

# ------------------- STREAMLIT UI -------------------
st.set_page_config(page_title=f"Dual-Axis Chart - {QUERY_ACRONYM}", layout="wide")
st.title(f"📊 QDWA Weight Analysis - Query {QUERY_ACRONYM}")

# Display query information
st.info(f"**Query:** {QUERY_TEXT}\n\n**Highest Domain:** {HIGHEST_DOMAIN}\n\n**α (prior):** {ALPHA}")

st.markdown("Compare raw evidence (bars) and smoothed weights (line) "
            "with full publication-style control.")

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

st.sidebar.header("Chart Customization")

with st.sidebar.expander("🎨 Data & Colors", expanded=True):
    use_rounded = st.checkbox("Use Rounded Data", value=False)
    bar_color = st.color_picker("Bar Color", "#1f77b4")
    line_color = st.color_picker("Line Color", "#ff7f0e")
    bar_alpha = st.slider("Bar transparency", 0.1, 1.0, 0.7, 0.05)
    bar_width = st.slider("Bar width", 0.2, 1.0, 0.8, 0.05)
    bar_edge_color = st.color_picker("Bar edge color", "#000000")
    bar_edge_width = st.slider("Bar edge width", 0.0, 2.0, 0.5, 0.1)

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

with st.sidebar.expander("🔤 Fonts & Math", expanded=True):
    font_family = st.selectbox("Font family", ["sans-serif", "serif", "monospace"])
    mathtext_fontset = st.selectbox(
        "Math font set (mathtext)",
        ["dejavusans", "dejavuserif", "cm", "stix", "stixsans"], index=2,
        help="'cm' = Computer Modern → classic LaTeX look. "
             "'stix' + serif family ≈ Times New Roman. No LaTeX install needed.")
    ylabel_left = st.text_input("Left y-label (LaTeX ok)",
                                value=r"Raw Evidence $k_{\mathrm{raw}}$")
    ylabel_right = st.text_input("Right y-label (LaTeX ok)",
                                 value=r"QDWA Weight $W_k$")
    label_weight = st.selectbox("Axis label weight", ["normal", "bold"])
    bold_title = st.checkbox("Bold title", value=True)
    if LATEX_AVAILABLE:
        use_usetex = st.checkbox("Use real LaTeX (text.usetex)", value=False)
    else:
        st.info("No LaTeX installation found → using built-in mathtext. "
                "Pick 'cm' for the LaTeX look.")
        use_usetex = False

with st.sidebar.expander(" Ticks & Spines"):
    tick_length = st.slider("Major tick length", 0, 15, 5)
    tick_width = st.slider("Major tick width", 0.1, 3.0, 1.0, 0.1)
    tick_direction = st.selectbox("Tick direction", ["out", "in", "inout"])
    tick_pad = st.slider("Tick label padding", 1, 15, 4)
    minor_ticks = st.checkbox("Show minor ticks (y axes)", value=False)
    spine_width = st.slider("Spine (frame) width", 0.5, 3.0, 1.0, 0.1)
    top_spines = st.checkbox("Show top spine", value=False)
    x_rotation = st.select_slider("X-label rotation", options=[0, 15, 30, 45, 60, 90], value=45)

with st.sidebar.expander("🔀 Grid & Legend"):
    show_grid = st.checkbox("Show grid", value=True)
    grid_style = st.selectbox("Grid line style", ["--", "-", ":", "-."])
    grid_width = st.slider("Grid line width", 0.3, 2.0, 0.6, 0.1)
    grid_alpha = st.slider("Grid transparency", 0.05, 1.0, 0.6, 0.05)
    legend_loc = st.selectbox("Legend location",
                              ["upper left", "upper right", "lower left", "lower right", "best"])
    legend_frame = st.checkbox("Legend frame", value=True)
    legend_fontsize = st.slider("Legend font size", 6, 20, 10)

with st.sidebar.expander("🖼 Figure & Export"):
    font_size = st.slider("Font size", 8, 24, 12)
    fig_width = st.slider("Figure width (inches)", 4, 12, 8)
    fig_height = st.slider("Figure height (inches)", 3, 9, 5)
    dpi = st.selectbox("Export DPI (raster formats)", [100, 200, 300, 600], index=2)
    export_format = st.selectbox("Export format", ["png", "pdf", "svg", "eps", "tiff"],
                                 help="PDF/SVG/EPS are vector formats — ideal for journals.")
    transparent_bg = st.checkbox("Transparent background", value=False)

df = df_rounded if use_rounded else df_exact

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

st.subheader("QDWA Weight Data - Query Q5PM1")
st.dataframe(df)

# Display the QDWA formula
st.latex(r"W_k = \frac{\alpha + \mathrm{raw}_k}{6\alpha + \sum_j \mathrm{raw}_j}, \quad \alpha = 0.25")

fig = plot_dual_axis(df, cfg)

try:
    st.pyplot(fig)
    st.markdown(get_download_link(fig, dpi, export_format, transparent_bg),
                unsafe_allow_html=True)
except Exception as e:
    st.error(f"Rendering failed — usually a LaTeX/mathtext syntax error in a label: {e}")
