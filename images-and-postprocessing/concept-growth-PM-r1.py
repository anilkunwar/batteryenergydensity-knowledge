import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

# ------------------- DATA -------------------
data_raw = {
    "Material": ["NMC811", "Silicon", "Graphite"],
    "Time_1": [1, 275, 467],
    "Time_2": [22, 311, 405],
    "Symbol": ["●", "■", "◆"],
    "Highlight": [True, False, False]  # NMC811 is the star
}
df = pd.DataFrame(data_raw)
df["Growth"] = ((df["Time_2"] - df["Time_1"]) / df["Time_1"] * 100).round(2)
df["Growth_Str"] = df["Growth"].apply(lambda g: f"+{g:.2f}%" if g >= 0 else f"{g:.2f}%")

# Color palette — curated, high-contrast
PALETTE = {
    "NMC811":    "#E63946",  # vivid red
    "Silicon":   "#457B9D",  # steel blue
    "Graphite":  "#2A9D8F",  # teal
}
MARKER_STYLE = {
    "NMC811":    "o",
    "Silicon":   "s",
    "Graphite":  "D",
}

# ------------------- PLOT FUNCTION -------------------
def plot_slope_chart(
    df_active,
    line_width, marker_size, font_size,
    show_labels, log_scale, show_grid,
    highlight_star, show_annotation, bg_style,
):
    n = len(df_active)
    if n == 0:
        st.info("No materials selected. Toggle at least one in the sidebar.")
        return

    # ---------- figure setup ----------
    fig, ax = plt.subplots(figsize=(10, 6.5))
    fig.patch.set_facecolor("#FAFAFA" if bg_style == "Light" else "#1E1E2F")
    ax.set_facecolor("#FFFFFF" if bg_style == "Light" else "#2B2B3D")

    text_color = "#222222" if bg_style == "Light" else "#E0E0E0"
    grid_color = "#CCCCCC" if bg_style == "Light" else "#444466"
    spine_color = "#AAAAAA" if bg_style == "Light" else "#555577"

    x = [1, 2]

    for idx, row in df_active.iterrows():
        mat = row["Material"]
        y = [row["Time_1"], row["Time_2"]]
        color = PALETTE[mat]
        marker = MARKER_STYLE[mat]
        is_star = row["Highlight"] and highlight_star

        lw = line_width * (1.8 if is_star else 1.0)
        ms = marker_size * (1.4 if is_star else 1.0)
        alpha = 1.0 if is_star else 0.85
        zorder = 10 if is_star else 5

        # shadow / glow for highlighted line
        if is_star:
            ax.plot(x, y, color=color, lw=lw + 4, alpha=0.15, zorder=zorder - 1)
            ax.plot(x, y, color=color, lw=lw + 2, alpha=0.25, zorder=zorder - 1)

        # main line
        ax.plot(
            x, y, color=color, lw=lw, marker=marker, ms=ms,
            alpha=alpha, zorder=zorder, label=mat,
            markeredgecolor="white" if bg_style == "Light" else "#1E1E2F",
            markeredgewidth=1.5,
            solid_capstyle="round",
        )

        # ---- labels ----
        if show_labels:
            stroke = [pe.withStroke(linewidth=2.5, foreground="white")] if bg_style == "Light" else [pe.withStroke(linewidth=2.5, foreground="#1E1E2F")]
            fs_label = font_size - 1

            # Left label
            ax.text(
                x[0] - 0.08, y[0],
                f"{row['Symbol']} {mat}\n{y[0]:,}",
                ha="right", va="center", fontsize=fs_label,
                color=color, fontweight="bold" if is_star else "normal",
                path_effects=stroke,
            )
            # Right label
            ax.text(
                x[1] + 0.08, y[1],
                f"{y[1]:,}  ({row['Growth_Str']})",
                ha="left", va="center", fontsize=fs_label,
                color=color, fontweight="bold" if is_star else "normal",
                path_effects=stroke,
            )

    # ---- annotation arrow for biggest growth ----
    if show_annotation and not df_active.empty:
        star_row = df_active[df_active["Highlight"] == True]
        if not star_row.empty:
            sr = star_row.iloc[0]
            mid_x = 1.5
            mid_y = (sr["Time_1"] + sr["Time_2"]) / 2
            offset_y = mid_y * 0.35 if log_scale else 80
            ax.annotate(
                f"🔥 {sr['Growth_Str']}",
                xy=(mid_x, mid_y), xytext=(mid_x, mid_y + offset_y),
                fontsize=font_size + 2, fontweight="bold",
                color=PALETTE[sr["Material"]],
                ha="center", va="bottom",
                arrowprops=dict(
                    arrowstyle="->", color=PALETTE[sr["Material"]],
                    lw=1.8, connectionstyle="arc3,rad=-0.2",
                ),
                path_effects=[pe.withStroke(linewidth=2, foreground="white")] if bg_style == "Light" else [pe.withStroke(linewidth=2, foreground="#1E1E2F")],
            )

    # ---- axes ----
    ax.set_xticks([1, 2])
    ax.set_xticklabels(
        ["Before 2021", "After 2021"],
        fontsize=font_size + 2, fontweight="bold", color=text_color,
    )
    ax.set_ylabel("Occurrences", fontsize=font_size + 2, color=text_color, labelpad=10)
    title = "Slope Chart — Material Occurrences Over Time"
    ax.set_title(title, fontsize=font_size + 5, fontweight="bold", color=text_color, pad=15)

    if log_scale:
        ax.set_yscale("log")
        ax.set_ylabel("Occurrences (log scale)", fontsize=font_size + 2, color=text_color, labelpad=10)

    ax.grid(show_grid, linestyle="--", alpha=0.4, color=grid_color)

    # clean spines
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color(spine_color)

    ax.tick_params(axis="both", labelsize=font_size, colors=text_color)
    ax.set_xlim(0.5, 2.5)

    # ---- LEGEND with symbols ----
    handles, labels = ax.get_legend_handles_labels()
    new_labels = []
    for lab in labels:
        sym = df[df["Material"] == lab]["Symbol"].values[0]
        new_labels.append(f"  {sym}  {lab}")
    legend = ax.legend(
        handles, new_labels,
        loc="lower right",
        fontsize=font_size + 1,
        frameon=True,
        fancybox=True,
        shadow=True,
        edgecolor=spine_color,
        facecolor="#FFFFFF" if bg_style == "Light" else "#2B2B3D",
        labelcolor=text_color,
        borderpad=0.8,
        handletextpad=0.6,
    )
    legend.get_frame().set_linewidth(1.2)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)


# ------------------- STREAMLIT UI -------------------
st.set_page_config(page_title="Slope Chart — Material Growth", layout="wide")

# ---- top header ----
st.html("""
<div style="display:flex;align-items:center;gap:12px;margin-bottom:4px">
    <span style="font-size:2.2rem">📈</span>
    <span style="font-size:1.7rem;font-weight:700;background:linear-gradient(90deg,#E63946,#457B9D);-webkit-background-clip:text;-webkit-text-fill-color:transparent">
        Slope Chart — Material Occurrences
    </span>
</div>
<p style="color:#888;margin-top:-4px;margin-bottom:16px">
    Visualizing the growth of battery material mentions from before 2021 to after 2021.
</p>
""")

# ---- sidebar ----
with st.sidebar:
    st.header("🎛️  Controls")

    st.subheader("Material Toggles")
    toggle_states = {}
    cols = st.columns(3)
    for i, mat in enumerate(df["Material"]):
        color_hex = PALETTE[mat]
        sym = df[df["Material"] == mat]["Symbol"].values[0]
        with cols[i]:
            toggle_states[mat] = st.toggle(
                f"{sym} {mat}",
                value=True,
                key=f"toggle_{mat}",
            )

    st.divider()

    st.subheader("Style")
    line_width = st.slider("Line Width", 1, 10, 3)
    marker_size = st.slider("Marker Size", 4, 20, 10)
    font_size = st.slider("Font Size", 8, 24, 12)
    bg_style = st.radio("Background", ["Light", "Dark"], horizontal=True)

    st.divider()

    st.subheader("Options")
    show_labels = st.checkbox("Show Value Labels", value=True)
    log_scale = st.checkbox("Log Scale (Y-axis)", value=False)
    show_grid = st.checkbox("Show Grid", value=True)
    highlight_star = st.checkbox("Highlight Fastest Growth", value=True)
    show_annotation = st.checkbox("Show Growth Annotation", value=True)

# ---- active dataframe ----
df_active = df[[toggle_states[m] for m in df["Material"]]].copy()

# ---- data table ----
with st.expander("📊  View Raw Data", expanded=False):
    st.dataframe(
        df[["Material", "Time_1", "Time_2", "Growth_Str"]].rename(
            columns={"Growth_Str": "Growth"}
        ),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Material": st.column_config.TextColumn("Material"),
            "Time_1": st.column_config.NumberColumn("Before 2021", format="%d"),
            "Time_2": st.column_config.NumberColumn("After 2021", format="%d"),
            "Growth": st.column_config.TextColumn("Growth"),
        },
    )

# ---- plot ----
plot_slope_chart(
    df_active,
    line_width, marker_size, font_size,
    show_labels, log_scale, show_grid,
    highlight_star, show_annotation, bg_style,
)

# ---- footer ----
st.markdown("---")
st.caption("Growth = ((After 2021 − Before 2021) / Before 2021) × 100  ·  Built with Streamlit & Matplotlib")
