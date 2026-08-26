import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ------------------- DATA -------------------
data_raw = {
    "Material": ["NMC811", "Silicon", "Graphite"],
    "Time_1": [1, 275, 467],
    "Time_2": [22, 311, 405]
}
df = pd.DataFrame(data_raw)
df["Growth"] = ((df["Time_2"] - df["Time_1"]) / df["Time_1"] * 100).round(2)

# ------------------- PLOT FUNCTION -------------------
def plot_slope_chart(df, line_width, marker_size, font_size, colormap_name,
                     show_labels, log_scale, show_grid):
    fig, ax = plt.subplots(figsize=(8, 6))

    # Use plt.get_cmap (works in all recent Matplotlib versions)
    cmap = plt.get_cmap(colormap_name)
    colors = [cmap(i / len(df)) for i in range(len(df))]

    x = [1, 2]

    for idx, row in df.iterrows():
        y = [row["Time_1"], row["Time_2"]]
        color = colors[idx]

        ax.plot(x, y, color=color, lw=line_width, marker='o', ms=marker_size,
                label=row["Material"])

        if show_labels:
            ax.text(x[0]-0.05, y[0], f"{row['Material']}\n{y[0]}",
                    ha='right', va='center', fontsize=font_size, color=color)
            growth_text = f"{row['Growth']:+.2f}%"
            ax.text(x[1]+0.05, y[1], f"{y[1]}\n{growth_text}",
                    ha='left', va='center', fontsize=font_size, color=color)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Time 1", "Time 2"], fontsize=font_size+2)
    ax.set_ylabel("Count", fontsize=font_size+2)
    ax.set_title("Slope Chart of Material Occurrences", fontsize=font_size+4)

    if log_scale:
        ax.set_yscale('log')
        ax.set_ylabel("Count (log scale)", fontsize=font_size+2)

    ax.grid(show_grid, linestyle='--', alpha=0.6)

    if not show_labels:
        ax.legend(fontsize=font_size)

    ax.tick_params(axis='both', labelsize=font_size)

    st.pyplot(fig)

# ------------------- STREAMLIT UI -------------------
st.set_page_config(page_title="Slope Chart - Material Growth", layout="wide")
st.title("📈 Slope Chart for Material Occurrences")
st.markdown("Customize the chart using the controls on the left.")

# Sidebar controls
st.sidebar.header("Chart Customization")
line_width = st.sidebar.slider("Line Width", 1, 10, 3)
marker_size = st.sidebar.slider("Marker Size", 4, 20, 10)
font_size = st.sidebar.slider("Font Size", 8, 24, 12)
colormap_name = st.sidebar.selectbox(
    "Colormap",
    ["tab10", "Set1", "Set2", "Dark2", "Paired", "viridis", "plasma", "inferno", "magma"]
)
show_labels = st.sidebar.checkbox("Show Value Labels", value=True)
log_scale = st.sidebar.checkbox("Log Scale for Y-axis", value=False)
show_grid = st.sidebar.checkbox("Show Grid", value=True)

st.subheader("Raw Data")
st.dataframe(df)

plot_slope_chart(df, line_width, marker_size, font_size, colormap_name,
                 show_labels, log_scale, show_grid)

st.markdown("---")
st.caption("Growth rates are calculated as ((Time2 - Time1) / Time1) × 100")
