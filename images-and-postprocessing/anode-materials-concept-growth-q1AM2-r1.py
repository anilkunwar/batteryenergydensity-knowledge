import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.patches import FancyBboxPatch
import numpy as np
import io

try:
    import mplcursors
    HAVE_MPLCURSORS = True
except ImportError:
    HAVE_MPLCURSORS = False

# ═══════════════════════════════════════════════════════════════
#  SAFE COLORMAP GETTER
# ═══════════════════════════════════════════════════════════════
def safe_get_cmap(name):
    try:
        return plt.colormaps.get_cmap(name) if hasattr(plt.colormaps, "get_cmap") else plt.colormaps[name]
    except Exception:
        try:
            return cm.get_cmap(name)
        except Exception:
            return plt.cm.viridis

def get_all_colormaps():
    try:
        return sorted(list(plt.colormaps()))
    except AttributeError:
        return sorted(list(cm._colormaps.keys()))

ALL_CMAPS = get_all_colormaps()

# ═══════════════════════════════════════════════════════════════
#  ANNOTATION SYMBOL OPTIONS
# ═══════════════════════════════════════════════════════════════
ANN_SYMBOLS = {
    "★  Star":       "★",
    "▲  Triangle":   "▲",
    "●  Circle":     "●",
    "◆  Diamond":    "◆",
    "▶  Arrow":      "▶",
    "✦  Star Open":  "✦",
    "■  Square":     "■",
    "None":          "",
}
ANN_ARROW_STYLES = {
    "→  Standard":   "->",
    "▷  Open":       "-|>",
    "⟶  Fancy":      "fancy",
    "—  Simple":     "simple",
}
ANN_BOX_STYLES = {
    "Rounded":       "round,pad=0.4",
    "Square":        "square,pad=0.4",
    "Sawtooth":      "sawtooth,pad=0.4",
    "None":          None,
}

# ═══════════════════════════════════════════════════════════════
#  BACKGROUND PRESETS
# ═══════════════════════════════════════════════════════════════
BG_PRESETS = {
    "None":           ("#FFFFFF", "#FFFFFF", "#FFFFFF"),
    "Sunset":         ("#FFE5B4", "#FF7F50", "#CD5C5C"),
    "Ocean":          ("#E0F7FA", "#4FC3F7", "#01579B"),
    "Forest":         ("#E8F5E9", "#66BB6A", "#1B5E20"),
    "Lavender":       ("#F3E5F5", "#CE93D8", "#4A148C"),
    "Twilight":       ("#FFF9C4", "#FFB74D", "#4E342E"),
    "Arctic":         ("#E3F2FD", "#90CAF9", "#1A237E"),
    "Rose Garden":    ("#FCE4EC", "#F48FB1", "#880E4F"),
    "Mint":           ("#E0F2F1", "#80CBC4", "#004D40"),
    "Midnight":       ("#1A1A2E", "#16213E", "#0F3460"),
    "Ember":          ("#FFF3E0", "#FF8A65", "#BF360C"),
    "Peach":          ("#FFF8E1", "#FFCC80", "#E65100"),
    "Skyline":        ("#E1F5FE", "#4FC3F7", "#0277BD"),
    "Neon Night":     ("#0D0D0D", "#1A0033", "#0D0D0D"),
    "Cherry Blossom": ("#FFF0F5", "#FFB6C1", "#C71585"),
    "Custom":         ("#FF6B6B", "#4ECDC4", "#45B7D1"),
}

# ═══════════════════════════════════════════════════════════════
#  DATA — Q1AM2: Overcoming silicon anode volume expansion
#  *** 0-0 attributes (sei_formation, cycle_life, etc.) excluded ***
# ═══════════════════════════════════════════════════════════════
data_raw = {
    "Material":  ["Silicon", "FEC", "Graphite", "NMC811"],
    "Time_1":    [275, 18, 467, 1],
    "Time_2":    [311, 49, 405, 22],
    "Symbol":    ["■", "●", "◆", "▲"],
    "Highlight": [True, False, False, False],
}
df = pd.DataFrame(data_raw)
df["Growth"]     = ((df["Time_2"] - df["Time_1"]) / df["Time_1"] * 100).round(2)
df["Growth_Str"] = df["Growth"].apply(lambda g: f"+{g:.2f}%" if g >= 0 else f"{g:.2f}%")

# Q1AM2-themed palette: silicon anode volume expansion context
DEFAULT_PALETTE = {
    "Silicon":  "#E63946",   # Vivid red  — primary anode under study
    "FEC":      "#F4A261",   # Warm orange — electrolyte additive for SEI stabilisation
    "Graphite": "#457B9D",   # Steel blue  — conventional anode (declining reference)
    "NMC811":   "#2A9D8F",   # Teal        — high-Ni cathode counterpart
}
MARKER_STYLE = {
    "Silicon":  "s",
    "FEC":      "o",
    "Graphite": "D",
    "NMC811":   "^",
}

# ═══════════════════════════════════════════════════════════════
#  HELPER — Quadratic Bézier curved line
# ═══════════════════════════════════════════════════════════════
def make_curved_line(x1, y1, x2, y2, curvature=0.0, n_pts=80):
    t  = np.linspace(0, 1, n_pts)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2 + curvature * max(abs(y2 - y1), 1)
    x  = (1 - t)**2 * x1 + 2 * (1 - t) * t * cx + t**2 * x2
    y  = (1 - t)**2 * y1 + 2 * (1 - t) * t * cy + t**2 * y2
    return x, y

# ═══════════════════════════════════════════════════════════════
#  MAIN PLOT FUNCTION
# ═══════════════════════════════════════════════════════════════
def plot_slope_chart(df_active, **kw):
    # --- labels ---
    show_left   = kw.get("show_left_labels",  True)
    show_right  = kw.get("show_right_labels", True)
    show_sym    = kw.get("show_symbols",      True)
    show_gpct   = kw.get("show_growth_pct",   True)
    label_oy    = kw.get("label_offset_y",    0)
    label_bg    = kw.get("label_bg",          False)
    label_rot   = kw.get("label_rotation",    0)
    conn_lines  = kw.get("connector_lines",   False)

    # --- line ---
    line_w      = kw.get("line_width",   3.0)
    curv        = kw.get("curvature",    0.0)
    line_alpha  = kw.get("line_alpha",   0.85)
    show_arrow  = kw.get("show_arrow",   False)

    # --- colormap ---
    use_cmap     = kw.get("use_cmap",      False)
    cmap_name    = kw.get("cmap_name",     "viridis")
    show_cbar    = kw.get("show_colorbar", True)
    cmap_reverse = kw.get("cmap_reverse",  False)

    # --- per-material ---
    cust_col = kw.get("custom_colors",    DEFAULT_PALETTE)
    ln_styles= kw.get("line_styles",      {})
    mk_over  = kw.get("marker_overrides", MARKER_STYLE)

    # --- gradient bg ---
    tri_bg  = kw.get("three_color_bg",        False)
    bg1     = kw.get("bg_color1",             "#FFE5B4")
    bg2     = kw.get("bg_color2",             "#FF7F50")
    bg3     = kw.get("bg_color3",             "#CD5C5C")
    bg_alpha= kw.get("bg_gradient_alpha",     0.15)
    bg_dir  = kw.get("bg_gradient_direction", "Vertical (Top→Bottom)")

    # --- axes box ---
    box_on      = kw.get("box_visible",       True)
    box_col     = kw.get("box_color",         "#888888")
    box_w       = kw.get("box_width",         2.0)
    box_ls      = kw.get("box_linestyle",     "solid")
    box_rad     = kw.get("box_corner_radius", 0.02)
    box_shad    = kw.get("box_shadow",        True)
    box_fill    = kw.get("box_fill",          False)
    box_fill_col= kw.get("box_fill_color",    "#FFFFFF")
    box_fill_al = kw.get("box_fill_alpha",    0.05)

    # --- highlight ---
    hi_star   = kw.get("highlight_star",    True)
    shad_alpha= kw.get("shadow_alpha",      0.25)

    # --- annotation ---
    ann_mat       = kw.get("annotate_material",  None)
    ann_symbol    = kw.get("ann_symbol",         "★")
    ann_box_style = kw.get("ann_box_style",      "round,pad=0.4")
    ann_arrow_sty = kw.get("ann_arrow_style",    "->")
    ann_arrow_lw  = kw.get("ann_arrow_lw",       2.5)
    ann_offset    = kw.get("ann_offset",         0.35)
    ann_curve_rad = kw.get("ann_curve_rad",      -0.2)
    ann_font_extra= kw.get("ann_font_extra",     2)

    # --- axes ---
    log_sc    = kw.get("log_scale",     False)
    show_grid = kw.get("show_grid",     True)
    grid_style= kw.get("grid_style",    "--")
    y_min     = kw.get("y_min",         None)
    y_max     = kw.get("y_max",         None)
    leg_loc   = kw.get("legend_loc",    "None")
    sp_w      = kw.get("spine_width",   1.0)
    tk_len    = kw.get("tick_length",   6)
    tk_w      = kw.get("tick_width",    1.0)

    # --- text ---
    title    = kw.get("title_text",     "Q1AM2 — Silicon Anode Volume Expansion")
    subtitle = kw.get("subtitle_text",  "Overcoming volume expansion to maintain cycle life")
    xl_text  = kw.get("xlabel_text",    "Time Period")
    yl_text  = kw.get("ylabel_text",    "Publication Occurrences")
    watermark= kw.get("watermark_text", "")

    # --- theme / layout ---
    bg_st     = kw.get("bg_style",      "Light")
    mk_sz     = kw.get("marker_size",   10)
    fs        = kw.get("font_size",     12)
    fw_val    = kw.get("fig_width",     10)
    fh_val    = kw.get("fig_height",    6.5)
    show_hover= kw.get("show_hover",    True)

    n = len(df_active)
    if n == 0:
        st.info("No concepts selected — toggle at least one in the sidebar.")
        return None

    # ─── figure ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(fw_val, fh_val))
    bg_face = "#FAFAFA" if bg_st == "Light" else "#1E1E2F"
    ax_face = "#FFFFFF" if bg_st == "Light" else "#2B2B3D"
    fig.patch.set_facecolor(bg_face)
    ax.set_facecolor(ax_face)
    txt_c  = "#222222" if bg_st == "Light" else "#E0E0E0"
    grd_c  = "#CCCCCC" if bg_st == "Light" else "#444466"
    sp_c   = "#AAAAAA" if bg_st == "Light" else "#555577"
    edge_c = "white"  if bg_st == "Light" else "#1E1E2F"

    xp = [1, 2]

    # ─── colormap ─────────────────────────────────────────────
    cmap_obj = norm_obj = None
    if use_cmap and n > 0:
        cname    = cmap_name + "_r" if cmap_reverse else cmap_name
        cmap_obj = safe_get_cmap(cname)
        gv = df_active["Growth"].values
        vmin, vmax = gv.min(), gv.max()
        if vmin == vmax:
            vmax = vmin + 1
        norm_obj = mcolors.Normalize(vmin=vmin, vmax=vmax)

    def col_for(mat, growth):
        if use_cmap and cmap_obj and norm_obj:
            return cmap_obj(norm_obj(growth))
        return cust_col.get(mat, DEFAULT_PALETTE.get(mat, "#333333"))

    # ─── draw each slope ──────────────────────────────────────
    for idx, row in df_active.iterrows():
        mat   = row["Material"]
        yv    = [row["Time_1"], row["Time_2"]]
        color = col_for(mat, row["Growth"])
        marker= mk_over.get(mat, MARKER_STYLE.get(mat, "o"))
        ls    = ln_styles.get(mat, "-")
        star  = row["Highlight"] and hi_star

        lw = line_w * (1.8 if star else 1.0)
        ms = mk_sz  * (1.4 if star else 1.0)
        al = min(line_alpha, 1.0) if star else line_alpha * 0.85
        zo = 10 if star else 5

        use_curve = abs(curv) > 0.001
        if use_curve:
            xc, yc = make_curved_line(xp[0], yv[0], xp[1], yv[1], curv)
        else:
            xc, yc = xp, yv

        # glow
        if star and shad_alpha > 0:
            ax.plot(xc, yc, color=color, lw=lw + 4,
                    alpha=shad_alpha * 0.5, zorder=zo - 1)
            ax.plot(xc, yc, color=color, lw=lw + 2,
                    alpha=shad_alpha,       zorder=zo - 1)

        # main line
        ax.plot(xc, yc, color=color, lw=lw, alpha=al, zorder=zo,
                linestyle=ls, solid_capstyle="round",
                dash_capstyle="round", label=mat)

        # endpoint markers
        ax.plot(xc[0],  yc[0],  marker=marker, ms=ms, color=color,
                zorder=zo + 1, markeredgecolor=edge_c, markeredgewidth=1.5)
        ax.plot(xc[-1], yc[-1], marker=marker, ms=ms, color=color,
                zorder=zo + 1, markeredgecolor=edge_c, markeredgewidth=1.5)

        # arrow
        if show_arrow:
            ax.annotate("", xy=(xp[1] + 0.06, yv[1]),
                        xytext=(xp[1] - 0.08, yv[1]),
                        arrowprops=dict(arrowstyle="->", color=color,
                                        lw=lw * 0.7), zorder=zo + 2)

        # ─── labels ───────────────────────────────────────────
        stroke = [pe.withStroke(linewidth=2.5, foreground=edge_c)]
        fl     = fs - 1
        sym    = row["Symbol"] if show_sym else ""
        oy     = label_oy

        bbox_p = (dict(boxstyle="round,pad=0.3", facecolor=ax_face,
                       edgecolor=color, alpha=0.75, linewidth=0.8)
                  if label_bg else None)

        if conn_lines:
            ax.plot([xp[0] - 0.04, xp[0]], [yv[0] + oy, yv[0]],
                    color=color, lw=0.6, alpha=0.5, zorder=zo - 1,
                    linestyle=":")
            ax.plot([xp[1], xp[1] + 0.04], [yv[1], yv[1] + oy],
                    color=color, lw=0.6, alpha=0.5, zorder=zo - 1,
                    linestyle=":")

        if show_left:
            ltxt = f"{sym} {mat}\n{yv[0]:,}".strip()
            ax.text(xp[0] - 0.08, yv[0] + oy, ltxt,
                    ha="right", va="center", fontsize=fl,
                    rotation=label_rot, color=color,
                    fontweight="bold" if star else "normal",
                    path_effects=stroke, bbox=bbox_p)

        if show_right:
            gp   = f"  ({row['Growth_Str']})" if show_gpct else ""
            rtxt = f"{yv[1]:,}{gp}"
            ax.text(xp[1] + 0.08, yv[1] + oy, rtxt,
                    ha="left", va="center", fontsize=fl,
                    rotation=label_rot, color=color,
                    fontweight="bold" if star else "normal",
                    path_effects=stroke, bbox=bbox_p)

    # ─── ANNOTATION ──────────────────────────────────────────
    if ann_mat and ann_mat in df_active["Material"].values:
        sr  = df_active[df_active["Material"] == ann_mat].iloc[0]
        mx  = 1.5
        my  = (sr["Time_1"] + sr["Time_2"]) / 2
        oy2 = my * ann_offset if log_sc else 80
        ac  = col_for(ann_mat, sr["Growth"])

        prefix  = ann_symbol if ann_symbol else ""
        ann_txt = f"{prefix}  {sr['Growth_Str']}" if prefix else sr['Growth_Str']

        bbox_ann = None
        if ann_box_style:
            bbox_ann = dict(
                boxstyle=ann_box_style,
                facecolor=ax_face,
                edgecolor=ac,
                alpha=0.92,
                linewidth=1.8,
            )

        ax.annotate(
            ann_txt,
            xy=(mx, my),
            xytext=(mx, my + oy2),
            fontsize=fs + ann_font_extra,
            fontweight="bold",
            color=ac,
            ha="center",
            va="bottom",
            bbox=bbox_ann,
            arrowprops=dict(
                arrowstyle=ann_arrow_sty,
                color=ac,
                lw=ann_arrow_lw,
                connectionstyle=f"arc3,rad={ann_curve_rad}",
                shrinkA=5,
                shrinkB=8,
                mutation_scale=20,
            ),
            path_effects=[pe.withStroke(linewidth=2, foreground=edge_c)],
            zorder=25,
        )

    # ─── axes setup ─────────────────────────────────────────
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Before 2021", "After 2021"],
                       fontsize=fs + 2, fontweight="bold", color=txt_c)
    ax.set_ylabel(yl_text, fontsize=fs + 2, color=txt_c, labelpad=10)

    full_title = title + (f"\n{subtitle}" if subtitle else "")
    ax.set_title(full_title, fontsize=fs + 5, fontweight="bold",
                 color=txt_c, pad=15, linespacing=1.4)

    if log_sc:
        ax.set_yscale("log")
        ax.set_ylabel(yl_text + "  (log scale)", fontsize=fs + 2,
                      color=txt_c, labelpad=10)
    elif y_min is not None and y_max is not None and y_max > y_min:
        ax.set_ylim(y_min, y_max)

    ax.grid(show_grid, linestyle=grid_style, alpha=0.4, color=grd_c)
    ax.tick_params(axis="both", labelsize=fs, colors=txt_c,
                   length=tk_len, width=tk_w)
    ax.set_xlim(0.5, 2.5)

    # ─── three-color gradient background ──────────────────────
    if tri_bg:
        clist = [mcolors.to_rgba(bg1), mcolors.to_rgba(bg2),
                 mcolors.to_rgba(bg3)]
        xl, xr = ax.get_xlim()
        yb, yt = ax.get_ylim()
        if "Vertical" in bg_dir:
            grad = np.linspace(1, 0, 256).reshape(-1, 1)
            grad = np.hstack([grad] * 2)
        else:
            grad = np.linspace(0, 1, 256).reshape(1, -1)
            grad = np.vstack([grad] * 2)
        cm_bg = mcolors.LinearSegmentedColormap.from_list("tbg", clist,
                                                          N=256)
        ax.imshow(grad, aspect="auto", cmap=cm_bg, alpha=bg_alpha,
                  extent=[xl, xr, yb, yt], origin="lower", zorder=0)

    # ─── colorbar ─────────────────────────────────────────────
    if use_cmap and show_cbar and cmap_obj and norm_obj:
        sm = cm.ScalarMappable(cmap=cmap_obj, norm=norm_obj)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02, shrink=0.8)
        cbar.set_label("Growth (%)", fontsize=fs, color=txt_c)
        cbar.ax.tick_params(colors=txt_c, labelsize=fs - 1)
        cbar.outline.set_edgecolor(sp_c)
        cbar.outline.set_linewidth(0.8)

    # ─── legend ───────────────────────────────────────────────
    handles, labels = ax.get_legend_handles_labels()
    new_lab = []
    for lab in labels:
        sym = df[df["Material"] == lab]["Symbol"].values[0]
        if show_gpct:
            g = df[df["Material"] == lab]["Growth_Str"].values[0]
            new_lab.append(f"  {sym}  {lab}  ({g})")
        else:
            new_lab.append(f"  {sym}  {lab}")
    if handles and leg_loc != "None":
        leg = ax.legend(handles, new_lab, loc=leg_loc, fontsize=fs + 1,
                        frameon=True, fancybox=True, shadow=True,
                        edgecolor=sp_c,
                        facecolor=("#FFFFFF" if bg_st == "Light"
                                   else "#2B2B3D"),
                        labelcolor=txt_c, borderpad=0.8,
                        handletextpad=0.6)
        leg.get_frame().set_linewidth(1.2)

    # ─── watermark ────────────────────────────────────────────
    if watermark:
        fig.text(0.99, 0.01, watermark, fontsize=8, color=txt_c,
                 alpha=0.3, ha="right", va="bottom", style="italic")

    # ─── AXES BOX ─────────────────────────────────────────────
    ls_map = {"solid": "-", "dashed": "--",
              "dotted": ":", "dashdot": "-."}
    bls = ls_map.get(box_ls, "-")

    if box_on:
        for sp_name in ax.spines.values():
            sp_name.set_visible(False)
        if box_shad:
            ax.add_patch(FancyBboxPatch(
                (0.004, -0.004), 0.996, 1.004,
                boxstyle=f"round,pad=0,rounding_size={box_rad}",
                facecolor="none", edgecolor=(0, 0, 0, 0.12),
                linewidth=box_w + 2, linestyle=bls,
                transform=ax.transAxes, zorder=19, clip_on=False))
        if box_fill:
            ax.add_patch(FancyBboxPatch(
                (0, 0), 1, 1,
                boxstyle=f"round,pad=0,rounding_size={box_rad}",
                facecolor=(*mcolors.to_rgb(box_fill_col), box_fill_al),
                edgecolor="none",
                transform=ax.transAxes, zorder=0, clip_on=False))
        ax.add_patch(FancyBboxPatch(
            (0, 0), 1, 1,
            boxstyle=f"round,pad=0,rounding_size={box_rad}",
            facecolor="none", edgecolor=box_col,
            linewidth=box_w, linestyle=bls,
            transform=ax.transAxes, zorder=20, clip_on=False))
    else:
        for sp_name in ax.spines.values():
            sp_name.set_linewidth(sp_w)
            sp_name.set_color(sp_c)
        for sp_name in ("top", "right"):
            ax.spines[sp_name].set_visible(False)

    fig.tight_layout()

    # ─── hover ────────────────────────────────────────────────
    if show_hover and HAVE_MPLCURSORS:
        cursor = mplcursors.cursor(ax.lines, hover=True)
        cursor.connect("add", lambda sel: sel.annotation.set_text(
            f"{sel.artist.get_label()}: {sel.target[1]:.0f}"))

    st.pyplot(fig, use_container_width=True)
    return fig


# ═══════════════════════════════════════════════════════════════
#  STREAMLIT PAGE
# ═══════════════════════════════════════════════════════════════
st.set_page_config(page_title="Q1AM2 — Silicon Anode Volume Expansion", layout="wide")

st.html("""<div style="display:flex;align-items:center;gap:12px;margin-bottom:4px">
<span style="font-size:2.2rem">🔋</span>
<span style="font-size:1.7rem;font-weight:700;
background:linear-gradient(90deg,#E63946,#F4A261,#457B9D);
-webkit-background-clip:text;-webkit-text-fill-color:transparent">
Q1AM2 — Silicon Anode Volume Expansion</span></div>
<p style="color:#888;margin-top:-4px;margin-bottom:16px">
Concept Growth Analysis &middot; Pre-2021 vs Post-2021 &middot;
0-0 Attributes Excluded &middot; Cycle Life Preservation Focus</p>""")

# ─── sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.header("🎛️  Controls")

    # ── 1. Concept toggles ──
    with st.expander("📌 Concept Toggles", expanded=True):
        toggle_states = {}
        cols = st.columns(2)
        for i, mat in enumerate(df["Material"]):
            sym = df[df["Material"] == mat]["Symbol"].values[0]
            with cols[i % 2]:
                toggle_states[mat] = st.toggle(
                    f"{sym} {mat}", True, key=f"tog_{mat}")

    st.caption("ℹ️  Concepts with 0 occurrences in both periods "
               "(e.g. sei_formation, cycle_life) are excluded.")

    # ── 2. Label controls ──
    with st.expander("🏷️  Label Controls", expanded=True):
        show_left  = st.checkbox("Left Labels  (name + value)", True)
        show_right = st.checkbox("Right Labels (value + growth)", True)
        c1, c2 = st.columns(2)
        with c1:
            show_sym  = st.checkbox("Symbols  ● ■ ◆ ▲", True)
        with c2:
            show_gpct = st.checkbox("Growth %", True)
        label_oy   = st.slider("Label Vertical Offset", -150, 150, 0, 5)
        label_rot  = st.slider("Label Rotation (°)",   -45, 45, 0, 1)
        label_bg   = st.checkbox("Label Background Boxes", False)
        conn_lines = st.checkbox("Connector Dots → Labels", False)

    # ── 3. Line / spline style ──
    with st.expander("✏️  Line & Spline Style", expanded=True):
        line_w    = st.slider("Spline Thickness (line width)",
                              0.5, 14.0, 3.0, 0.5)
        curv      = st.slider("Curvature / Spline Bend",
                              -1.0, 1.0, 0.0, 0.05,
                              help="0 = straight · + = bulge up · "
                                   "− = bulge down")
        line_alph = st.slider("Line Opacity", 0.1, 1.0, 0.85, 0.05)
        show_arrow= st.checkbox("Arrow at Line End", False)

    # ── 4. Colormap mode ──
    with st.expander("🌈  Colormap Mode  (50+ maps)", expanded=False):
        use_cmap    = st.checkbox("Color Lines by Growth Rate", False)
        cmap_search = st.text_input("Filter colormaps…", "", key="cms")
        filtered = ([c for c in ALL_CMAPS
                     if cmap_search.lower() in c.lower()]
                    if cmap_search else ALL_CMAPS)
        cmap_name = st.selectbox(
            "Colormap", filtered,
            index=(filtered.index("viridis")
                   if "viridis" in filtered else 0))
        cmap_reverse = st.checkbox("Reverse Colormap", False)

        if use_cmap:
            pc = safe_get_cmap(
                cmap_name + ("_r" if cmap_reverse else ""))
            st.image(pc(np.linspace(0, 1, 512).reshape(1, -1)),
                     use_container_width=True)
            st.caption(
                f"Showing: **{cmap_name}**  ·  "
                f"{len(ALL_CMAPS)} total maps")

        show_cbar = st.checkbox("Show Colorbar", True)

    # ── 5. Per-concept styling ──
    custom_colors  = DEFAULT_PALETTE.copy()
    ln_styles_dict = {m: "-" for m in df["Material"]}
    mk_over_dict   = MARKER_STYLE.copy()

    with st.expander("🎨  Per-Concept Styling", expanded=False):
        st.markdown("**Colors**")
        cc = {}; cols = st.columns(2)
        for i, mat in enumerate(df["Material"]):
            with cols[i % 2]:
                cc[mat] = st.color_picker(
                    mat, DEFAULT_PALETTE[mat], key=f"clr_{mat}")
        if not use_cmap:
            custom_colors = cc

        st.markdown("**Line Styles**")
        ls_d = {}; cols = st.columns(2)
        for i, mat in enumerate(df["Material"]):
            with cols[i % 2]:
                ls_d[mat] = st.selectbox(
                    mat, ["-", "--", "-.", ":"], key=f"ls_{mat}")
        ln_styles_dict = ls_d

        st.markdown("**Markers**")
        mo = {}; cols = st.columns(2)
        mk_opts = ["o", "s", "D", "^", "v", "*", "p", "X", "h", "P", "8"]
        for i, mat in enumerate(df["Material"]):
            di = mk_opts.index(MARKER_STYLE[mat])
            with cols[i % 2]:
                mo[mat] = st.selectbox(
                    mat, mk_opts, index=di, key=f"mk_{mat}")
        mk_over_dict = mo

    # ── 6. Three-color gradient background ──
    with st.expander("🌅  Three-Color Gradient / Shade",
                     expanded=False):
        tri_bg = st.checkbox("Enable Gradient Background", False)
        bg_pre = st.selectbox("Preset",
                              list(BG_PRESETS.keys()), index=0)
        p1, p2, p3 = BG_PRESETS[bg_pre]
        cols = st.columns(3)
        with cols[0]:
            bg1 = st.color_picker("Top / Left",    p1, key="bg1")
        with cols[1]:
            bg2 = st.color_picker("Middle",         p2, key="bg2")
        with cols[2]:
            bg3 = st.color_picker("Bottom / Right", p3, key="bg3")
        bg_alpha = st.slider("Gradient Opacity", 0.0, 0.8, 0.15, 0.05)
        bg_dir   = st.radio("Direction",
                            ["Vertical (Top→Bottom)",
                             "Horizontal (Left→Right)"],
                            horizontal=True)

    # ── 7. Axes box / border ──
    with st.expander("📦  Axes Box / Border", expanded=False):
        box_on  = st.checkbox("Show Axes Box", True)
        box_col = st.color_picker("Border Color", "#888888", key="bxcol")
        box_w   = st.slider("Border Width",   0.5, 8.0, 2.0, 0.5)
        box_ls  = st.selectbox("Border Style",
                               ["solid", "dashed", "dotted", "dashdot"])
        box_rad = st.slider("Corner Roundness", 0.0, 0.1, 0.02, 0.005)
        box_shad= st.checkbox("Drop Shadow", True)
        box_fill= st.checkbox("Box Fill Tint", False)
        box_fill_col = st.color_picker("Fill Tint Color",
                                       "#FFFFFF", key="bxfill")
        box_fill_al  = st.slider("Fill Tint Opacity",
                                 0.0, 0.3, 0.05, 0.01)

    # ── 8. Annotation callout ──
    with st.expander("📌  Annotation Callout", expanded=False):
        a_opts  = [None] + list(df["Material"])
        ann_mat = st.selectbox(
            "Annotate Concept", a_opts,
            format_func=lambda x: "None" if x is None else x,
            index=2)  # default: FEC (fastest meaningful growth)

        if ann_mat:
            st.markdown("**Prefix Symbol**  *(no emoji — renders "
                        "in all backends)*")
            ann_sym_key = st.selectbox(
                "Symbol",
                list(ANN_SYMBOLS.keys()), index=0, key="ann_sym")
            ann_symbol = ANN_SYMBOLS[ann_sym_key]

            st.markdown("**Text Box**")
            ann_box_key = st.selectbox(
                "Box Style",
                list(ANN_BOX_STYLES.keys()), index=0, key="ann_box")
            ann_box_style = ANN_BOX_STYLES[ann_box_key]

            st.markdown("**Arrow**")
            ann_arr_key = st.selectbox(
                "Arrow Head",
                list(ANN_ARROW_STYLES.keys()), index=0, key="ann_arr")
            ann_arrow_sty = ANN_ARROW_STYLES[ann_arr_key]

            ann_arrow_lw  = st.slider("Arrow Thickness",
                                      1.0, 6.0, 2.5, 0.5)
            ann_curve_rad = st.slider("Arrow Curve",
                                      -0.5, 0.5, -0.2, 0.05)
            ann_offset    = st.slider("Callout Distance",
                                      0.1, 1.0, 0.35, 0.05)
            ann_font_extra= st.slider("Extra Font Size",
                                      0, 6, 2, 1)
        else:
            ann_symbol     = "★"
            ann_box_style  = "round,pad=0.4"
            ann_arrow_sty  = "->"
            ann_arrow_lw   = 2.5
            ann_curve_rad  = -0.2
            ann_offset     = 0.35
            ann_font_extra = 2

    # ── 9. Glow / highlight ──
    with st.expander("✨  Glow / Highlight", expanded=False):
        hi_star    = st.checkbox("Highlight Silicon (Primary Material)", True)
        shad_alpha = st.slider("Glow Intensity", 0.0, 1.0, 0.25, 0.05)

    # ── 10. Titles & text ──
    with st.expander("📝  Titles & Text", expanded=False):
        title_t = st.text_input(
            "Title",
            "Q1AM2 — Concept Growth: Silicon Anode Volume Expansion")
        sub_t   = st.text_input("Subtitle",
                                "Overcoming volume expansion to maintain cycle life · "
                                "0-0 attributes excluded")
        xl_t    = st.text_input("X-Axis Label", "Time Period")
        yl_t    = st.text_input("Y-Axis Label", "Publication Occurrences")
        wm_t    = st.text_input("Watermark", "")

    # ── 11. Axes & grid ──
    with st.expander("⚙️  Axes & Grid", expanded=False):
        log_sc    = st.checkbox("Log Scale (Y)", False)
        show_grid = st.checkbox("Show Grid",    True)
        grid_sty  = st.selectbox("Grid Style",
                                 ["--", ":", "-.", "-"])
        cust_yl   = st.checkbox("Custom Y-Limits", False)
        y_min = y_max = None
        if cust_yl:
            c1, c2 = st.columns(2)
            with c1:
                y_min = st.number_input("Y-min", value=0,
                                        step=10, key="ymin")
            with c2:
                y_max = st.number_input("Y-max", value=500,
                                        step=10, key="ymax")
        leg_loc = st.selectbox(
            "Legend Position",
            ["None", "best", "upper right", "upper left",
             "lower left", "lower right", "center"],
            index=0)
        st.markdown("**Spines & Ticks**")
        sp_w   = st.slider("Spine Width",  0.5, 5.0, 1.0, 0.1)
        tk_len = st.slider("Tick Length",   2, 20, 6, 1)
        tk_w   = st.slider("Tick Width",    0.5, 5.0, 1.0, 0.1)

    # ── 12. Theme & layout ──
    st.divider()
    st.subheader("🎨  Theme & Layout")
    bg_st  = st.radio("Theme", ["Light", "Dark"], horizontal=True)
    mk_sz  = st.slider("Marker Size", 4, 28, 10)
    fs_val = st.slider("Font Size",   8, 26, 12)
    asp_map = {"4:3": (10, 7.5), "16:9": (12, 6.75),
               "3:2": (10.5, 7), "1:1": (8, 8), "Wide": (14, 6)}
    asp = st.selectbox("Aspect Ratio", list(asp_map.keys()), index=0)
    fw_val, fh_val = asp_map[asp]

    st.divider()
    show_hover = st.checkbox("Hover Tooltips", True,
                              disabled=not HAVE_MPLCURSORS)

# ─── active data ─────────────────────────────────────────────
df_active = df[[toggle_states[m] for m in df["Material"]]].copy()

# ─── data table ──────────────────────────────────────────────
with st.expander("📊  View Raw Data  (0-0 attributes excluded)", expanded=False):
    st.dataframe(
        df[["Material", "Time_1", "Time_2", "Growth_Str"]].rename(
            columns={"Time_1": "Before 2021",
                     "Time_2": "After 2021",
                     "Growth_Str": "Growth"}),
        use_container_width=True, hide_index=True,
        column_config={
            "Material": st.column_config.TextColumn("Concept"),
            "Before 2021": st.column_config.NumberColumn(
                "Before 2021", format="%d"),
            "After 2021": st.column_config.NumberColumn(
                "After 2021", format="%d"),
            "Growth":   st.column_config.TextColumn("Growth"),
        })
    st.caption("Excluded (0 in both periods): sei_formation, cycle_life, "
               "and other zero-count concepts from the Q1AM2 attribute set.")

# ─── plot ────────────────────────────────────────────────────
fig = plot_slope_chart(
    df_active,
    show_left_labels=show_left,   show_right_labels=show_right,
    show_symbols=show_sym,        show_growth_pct=show_gpct,
    label_offset_y=label_oy,      label_bg=label_bg,
    label_rotation=label_rot,     connector_lines=conn_lines,
    line_width=line_w,            curvature=curv,
    line_alpha=line_alph,         show_arrow=show_arrow,
    use_cmap=use_cmap,            cmap_name=cmap_name,
    show_colorbar=show_cbar,      cmap_reverse=cmap_reverse,
    custom_colors=custom_colors,  line_styles=ln_styles_dict,
    marker_overrides=mk_over_dict,
    three_color_bg=tri_bg,        bg_color1=bg1,
    bg_color2=bg2,                bg_color3=bg3,
    bg_gradient_alpha=bg_alpha,   bg_gradient_direction=bg_dir,
    box_visible=box_on,           box_color=box_col,
    box_width=box_w,              box_linestyle=box_ls,
    box_corner_radius=box_rad,    box_shadow=box_shad,
    box_fill=box_fill,            box_fill_color=box_fill_col,
    box_fill_alpha=box_fill_al,
    highlight_star=hi_star,       shadow_alpha=shad_alpha,
    # annotation params
    annotate_material=ann_mat,    ann_symbol=ann_symbol,
    ann_box_style=ann_box_style,  ann_arrow_style=ann_arrow_sty,
    ann_arrow_lw=ann_arrow_lw,    ann_offset=ann_offset,
    ann_curve_rad=ann_curve_rad,  ann_font_extra=ann_font_extra,
    # axes
    log_scale=log_sc,             show_grid=show_grid,
    grid_style=grid_sty,
    y_min=y_min,                  y_max=y_max,
    legend_loc=leg_loc,           spine_width=sp_w,
    tick_length=tk_len,           tick_width=tk_w,
    title_text=title_t,           subtitle_text=sub_t,
    xlabel_text=xl_t,             ylabel_text=yl_t,
    watermark_text=wm_t,
    bg_style=bg_st,               marker_size=mk_sz,
    font_size=fs_val,             fig_width=fw_val,
    fig_height=fh_val,            show_hover=show_hover,
)

# ─── export ──────────────────────────────────────────────────
if fig is not None:
    c1, c2, c3 = st.columns(3)
    for col, fmt, ext, mime in [
        (c1, "png", "png",  "image/png"),
        (c2, "svg", "svg",  "image/svg+xml"),
        (c3, "pdf", "pdf",  "application/pdf"),
    ]:
        buf = io.BytesIO()
        fig.savefig(buf, format=fmt, dpi=300, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        buf.seek(0)
        with col:
            st.download_button(
                f"📥 {ext.upper()}", data=buf,
                file_name=f"Q1AM2_slope_chart.{ext}",
                mime=mime, use_container_width=True)

# ─── footer ──────────────────────────────────────────────────
st.markdown("---")
st.caption(
    f"Q1AM2: Overcoming silicon anode volume expansion to maintain cycle life  ·  "
    f"Growth = ((After − Before) / Before) × 100  ·  "
    f"0-0 attributes filtered out  ·  "
    f"Available colormaps: **{len(ALL_CMAPS)}**  ·  "
    "Built with Streamlit & Matplotlib")
