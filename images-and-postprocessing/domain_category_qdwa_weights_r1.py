import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from io import BytesIO
import base64

# ------------------- DATA -------------------
data_exact = {
    "Category": ["Cathode Materials", "Anode Materials", "Electrolyte Systems",
                 "Manufacturing", "Degradation", "Performance Metrics"],
    "raw_k": [3.4272, 4.4412, 0.3772, 1.2086, 0.5604, 8.1854],
    "w_k": [0.1867, 0.2381, 0.0318, 0.0740, 0.0411, 0.4282]
}
df_exact = pd.DataFrame(data_exact)

data_rounded = {
    "Category": ["Cathode Materials", "Anode Materials", "Electrolyte Systems",
                 "Manufacturing", "Degradation", "Performance Metrics"],
    "raw_k": [3.5, 4.4, 0.3, 1.4, 0.4, 8.0],
    "w_k": [0.20, 0.10, 0.05, 0.05, 0.02, 0.25]
}
df_rounded = pd.DataFrame(data_rounded)

# ------------------- PLOT FUNCTION -------------------
def plot_dual_axis(df, use_rounded, bar_color, line_color, marker_style,
                   font_size, fig_width, fig_height, show_grid, dpi):
    fig, ax1 = plt.subplots(figsize=(fig_width, fig_height))

    # Bar plot (left axis)
    x = df["Category"]
    bars = ax1.bar(x, df["raw_k"], color=bar_color, alpha=0.7, label="Raw Evidence (left)")
    ax1.set_xlabel("Category", fontsize=font_size)
    ax1.set_ylabel("Raw Evidence (raw_k)", fontsize=font_size, color=bar_color)
    ax1.tick_params(axis='y', labelcolor=bar_color, labelsize=font_size)

    # ✅ FIXED: rotation via labelrotation, alignment via plt.setp
    ax1.tick_params(axis='x', labelsize=font_size, labelrotation=45)
    plt.setp(ax1.get_xticklabels(), ha='right', rotation_mode='anchor')

    # Right axis
    ax2 = ax1.twinx()
    line = ax2.plot(x, df["w_k"], color=line_color, marker=marker_style,
                    markersize=8, linewidth=2, label="Smoothed Weight (right)")
    ax2.set_ylabel("Smoothed Weight (w_k)", fontsize=font_size, color=line_color)
    ax2.tick_params(axis='y', labelcolor=line_color, labelsize=font_size)

    # Grid
    if show_grid:
        ax1.grid(True, linestyle='--', alpha=0.6, axis='y')
        ax2.grid(False)  # avoid double grid

    # Title
    version = "Rounded" if use_rounded else "Exact"
    ax1.set_title(f"Dual‑Axis Chart – {version} Data", fontsize=font_size+2)

    # Legends combined
    handles, labels = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles + handles2, labels + labels2, loc='upper left', fontsize=font_size-2)

    fig.tight_layout()
    return fig

# ------------------- DOWNLOAD FUNCTION -------------------
def get_image_download_link(fig, dpi):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    # Updated filename to reflect chart type
    href = f'<a href="data:image/png;base64,{b64}" download="dual_axis_chart.png">Download PNG</a>'
    return href

# ------------------- STREAMLIT UI -------------------
st.set_page_config(page_title="Dual‑Axis Chart", layout="wide")
st.title("📊 Publication‑Ready Dual‑Axis Chart")
st.markdown("Compare raw evidence (bars) and smoothed weights (line) for material categories.")

# Sidebar controls
st.sidebar.header("Chart Customization")
use_rounded = st.sidebar.checkbox("Use Rounded Data (from second table)", value=False)
df = df_rounded if use_rounded else df_exact

bar_color = st.sidebar.color_picker("Bar Color", "#1f77b4")
line_color = st.sidebar.color_picker("Line Color", "#ff7f0e")
marker_style = st.sidebar.selectbox("Marker Style", ["o", "s", "^", "D", "P", "*"])
font_size = st.sidebar.slider("Font Size", 8, 24, 12)
fig_width = st.sidebar.slider("Figure Width (inches)", 4, 12, 8)
fig_height = st.sidebar.slider("Figure Height (inches)", 3, 9, 5)
show_grid = st.sidebar.checkbox("Show Grid", value=True)
dpi = st.sidebar.selectbox("Export DPI", [100, 200, 300], index=1)

st.subheader("Data Used")
st.dataframe(df)

# Plot
fig = plot_dual_axis(df, use_rounded, bar_color, line_color, marker_style,
                     font_size, fig_width, fig_height, show_grid, dpi)
st.pyplot(fig)

# Download
st.markdown(get_image_download_link(fig, dpi), unsafe_allow_html=True)

st.caption("Left axis: raw evidence; Right axis: smoothed weights (w_k).")
