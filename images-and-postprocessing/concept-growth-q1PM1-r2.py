import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import io

# Optional: interactive hover (install with `pip install mplcursors`)
try:
    import mplcursors
    HAVE_MPLCURSORS = True
except ImportError:
    HAVE_MPLCURSORS = False

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

# Default palette (will be overridden by user colors)
DEFAULT_PALETTE = {
    "NMC811":    "#E63946",
    "Silicon":   "#457B9D",
    "Graphite":  "#2A9D8F",
}
MARKER_STYLE = {
    "NMC811":    "o",
    "Silicon":   "s",
    "Graphite":  "D",
}
LINE_STYLE_DEFAULT = {
    "NMC811":    "-",
    "Silicon":   "-",
    "Graphite":  "-",
}

# ------------------- PLOT FUNCTION -------------------
def plot_slope_chart(
    df_active,
    # Core styling
    line_width, marker_size, font_size,
    show_labels, log_scale, show_grid,
    highlight_star, bg_style,
    # New customisations
    custom_colors,
    line_styles,           # dict material -> linestyle string
    marker_overrides,      # dict material -> marker symbol
    shadow_alpha,
    annotate_material,     # material name or None
    y_min, y_max,
    legend_loc,
    title_text,
    xlabel_text,
    ylabel_text,
    show_hover,
):
    n = len(df_active)
    if n == 0:
        st.info("No materials selected. Toggle at least one in the sidebar.")
        return

    # ---------- figure setup ----------
    fig, ax = plt.subplots(figsize=(10, 6.5))
    bg_face = "#FAFAFA" if bg_style == "Light" else "#1E1E2F"
    ax_face = "#FFFFFF" if bg_style == "Light" else "#2B2B3D"
    fig.patch.set_facecolor(bg_face)
    ax.set_facecolor(ax_face)

    text_color = "#222222" if bg_style == "Light" else "#E0E0E0"
    grid_color = "#CCCCCC" if bg_style == "Light" else "#444466"
    spine_color = "#AAAAAA" if bg_style == "Light" else "#555577"
    edge_color = "white" if bg_style == "Light" else "#1E1E2F"

    x = [1, 2]

    for idx, row in df_active.iterrows():
        mat = row["Material"]
        y = [row["Time_1"], row["Time_2"]]
        color = custom_colors.get(mat, DEFAULT_PALETTE[mat])
        marker = marker_overrides.get(mat, MARKER_STYLE[mat])
        linestyle = line_styles.get(mat, "-")
        is_star = row["Highlight"] and highlight_star

        lw = line_width * (1.8 if is_star else 1.0)
        ms = marker_size * (1.4 if is_star else 1.0)
        alpha = 1.0 if is_star else 0.85
        zorder = 10 if is_star else 5

        # shadow / glow for highlighted line (if alpha > 0)
        if is_star and shadow_alpha > 0:
            ax.plot(x, y, color=color, lw=lw + 4, alpha=shadow_alpha*0.5, zorder=zorder - 1)
            ax.plot(x, y, color=color, lw=lw + 2, alpha=shadow_alpha, zorder=zorder - 1)

        # main line
        line, = ax.plot(
            x, y, color=color, lw=lw, marker=marker, ms=ms,
            alpha=alpha, zorder=zorder, label=mat,
            markeredgecolor=edge_color,
            markeredgewidth=1.5,
            solid_capstyle="round",
            linestyle=linestyle,
        )

        # ---- labels ----
        if show_labels:
            stroke = [pe.withStroke(linewidth=2.5, foreground=edge_color)]
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

    # ---- annotation for chosen material ----
    if annotate_material and annotate_material in df_active["Material"].values:
        sr = df_active[df_active["Material"] == annotate_material].iloc[0]
        mid_x = 1.5
        mid_y = (sr["Time_1"] + sr["Time_2"]) / 2
        offset_y = mid_y * 0.35 if log_scale else 80
        ax.annotate(
            f"🔥 {sr['Growth_Str']}",
            xy=(mid_x, mid_y), xytext=(mid_x, mid_y + offset_y),
            fontsize=font_size + 2, fontweight="bold",
            color=custom_colors.get(annotate_material, DEFAULT_PALETTE[annotate_material]),
            ha="center", va="bottom",
            arrowprops=dict(
                arrowstyle="->",
                color=custom_colors.get(annotate_material, DEFAULT_PALETTE[annotate_material]),
                lw=1.8, connectionstyle="arc3,rad=-0.2",
            ),
            path_effects=[pe.withStroke(linewidth=2, foreground=edge_color)],
        )

    # ---- axes ----
    ax.set_xticks([1, 2])
    ax.set_xticklabels(
        ["Before 2021", "After 2021"],
        fontsize=font_size + 2, fontweight="bold", color=text_color,
    )
    ax.set_ylabel(ylabel_text, fontsize=font_size + 2, color=text_color, labelpad=10)
    ax.set_title(title_text, fontsize=font_size + 5, fontweight="bold", color=text_color, pad=15)

    if log_scale:
        ax.set_yscale("log")
        ax.set_ylabel(ylabel_text + " (log scale)", fontsize=font_size + 2, color=text_color, labelpad=10)
    else:
        # manual Y limits if provided
        if y_min is not None and y_max is not None and y_max > y_min:
            ax.set_ylim(y_min, y_max)

    ax.grid(show_grid, linestyle="--", alpha=0.4, color=grid_color)

    # clean spines
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color(spine_color)

    ax.tick_params(axis="both", labelsize=font_size, colors=text_color)
    ax.set_xlim(0.5, 2.5)

    # ---- LEGEND ----
    handles, labels = ax.get_legend_handles_labels()
    new_labels = []
    for lab in labels:
        sym = df[df["Material"] == lab]["Symbol"].values[0]
        new_labels.append(f"  {sym}  {lab}")
    legend = ax.legend(
        handles, new_labels,
        loc=legend_loc,
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

    # ---- interactive hover (optional) ----
    if show_hover and HAVE_MPLCURSORS:
        cursor = mplcursors.cursor(ax.lines, hover=True)
        cursor.connect("add", lambda sel: sel.annotation.set_text(
            f"{sel.artist.get_label()}: {sel.target[1]:.0f}"
        ))

    st.pyplot(fig, use_container_width=True)
    return fig  # return for export


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

    # Material selection
    with st.expander("📌 Material Toggles", expanded=True):
        toggle_states = {}
        cols = st.columns(3)
        for i, mat in enumerate(df["Material"]):
            sym = df[df["Material"] == mat]["Symbol"].values[0]
            with cols[i]:
                toggle_states[mat] = st.toggle(
                    f"{sym} {mat}",
                    value=True,
                    key=f"toggle_{mat}",
                )

    # Per-material styling
    with st.expander("🎨 Per‑Material Styling", expanded=False):
        st.markdown("**Colors**")
        custom_colors = {}
        cols = st.columns(3)
        for i, mat in enumerate(df["Material"]):
            with cols[i]:
                custom_colors[mat] = st.color_picker(
                    mat, DEFAULT_PALETTE[mat], key=f"color_{mat}"
                )

        st.markdown("**Line styles**")
        line_styles = {}
        cols = st.columns(3)
        style_opts = ["-", "--", "-.", ":"]
        for i, mat in enumerate(df["Material"]):
            with cols[i]:
                line_styles[mat] = st.selectbox(
                    f"{mat}", style_opts, index=0, key=f"linestyle_{mat}"
                )

        st.markdown("**Marker symbols**")
        marker_overrides = {}
        cols = st.columns(3)
        marker_opts = ["o", "s", "D", "^", "v", "*", "p", "X"]
        for i, mat in enumerate(df["Material"]):
            default_idx = marker_opts.index(MARKER_STYLE[mat])
            with cols[i]:
                marker_overrides[mat] = st.selectbox(
                    f"{mat}", marker_opts, index=default_idx, key=f"marker_{mat}"
                )

    with st.expander("✨ Glow / Shadow", expanded=False):
        shadow_alpha = st.slider("Shadow intensity", 0.0, 1.0, 0.25, 0.05)

    with st.expander("📝 Annotation", expanded=False):
        annot_opts = [None] + list(df["Material"])
        annotate_material = st.selectbox(
            "Material to annotate (growth arrow)",
            annot_opts,
            format_func=lambda x: "None" if x is None else x,
            index=1,  # NMC811 by default
        )

    with st.expander("⚙️ Axis & Layout", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            y_min = st.number_input("Y‑axis min (if not log)", value=None, step=10)
        with col2:
            y_max = st.number_input("Y‑axis max (if not log)", value=None, step=10)
        legend_loc = st.selectbox(
            "Legend position",
            ["best", "upper right", "upper left", "lower left", "lower right", "center"],
            index=4,
        )

    with st.expander("✏️ Titles & Labels", expanded=False):
        title_text = st.text_input("Chart Title", "Slope Chart — Material Occurrences Over Time")
        xlabel_text = st.text_input("X‑axis label", "Time Period")
        ylabel_text = st.text_input("Y‑axis label", "Occurrences")

    # Core style controls (always visible)
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
    show_hover = st.checkbox("Hover Tooltips (requires mplcursors)", value=True)

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
fig = plot_slope_chart(
    df_active,
    line_width, marker_size, font_size,
    show_labels, log_scale, show_grid,
    highlight_star, bg_style,
    custom_colors,
    line_styles,
    marker_overrides,
    shadow_alpha,
    annotate_material,
    y_min if y_min != 0 else None,
    y_max if y_max != 0 else None,
    legend_loc,
    title_text,
    xlabel_text,
    ylabel_text,
    show_hover,
)

# ---- export button ----
if fig is not None:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    st.download_button(
        label="📥 Download Chart as PNG",
        data=buf,
        file_name="slope_chart.png",
        mime="image/png",
        use_container_width=True,
    )

# ---- footer ----
st.markdown("---")
st.caption(
    "Growth = ((After 2021 − Before 2021) / Before 2021) × 100  ·  "
    "Built with Streamlit & Matplotlib  ·  "
    "Hover tooltips require `mplcursors` (install with `pip install mplcursors`)"
)
