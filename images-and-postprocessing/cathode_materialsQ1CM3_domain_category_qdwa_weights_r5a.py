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
#  DATA — Q1CM3: Cathode Materials Comparison
#  Layered oxides (NMC811, NMC622, NMC532, NMC333, LCO, NCA)
#  vs Olivine (LFP)
#  NMC811 (Ni-rich, 80% Ni) shows explosive growth (+2100%)
# ═══════════════════════════════════════════════════════════════
data_raw = {
    "Material":  ["nmc811", "lfp",   "nmc622", "lco",   "nca",   "nmc532", "nmc333"],
    "Time_1":    [1,        55,      15,       28,      26,      10,       2],
    "Time_2":    [22,       114,     13,       24,      21,      4,        0],
    "Symbol":    ["▲",      "■",     "◆",      "●",     "★",     "▼",      "✕"],
    "Highlight": [True,     False,   False,    False,   False,   False,    False],
}
df = pd.DataFrame(data_raw)
df["Growth"]     = ((df["Time_2"] - df["Time_1"]) / df["Time_1"] * 100).round(2)
df["Growth_Str"] = df["Growth"].apply(lambda g: f"+{g:.2f}%" if g >= 0 else f"{g:.2f}%")

# Q1CM3-themed palette: cathode material context
# Layered oxides in warm/cool tones; LFP (olivine) in distinct green
DEFAULT_PALETTE = {
    "nmc811":  "#E63946",   # Vivid red   — Ni-rich layered oxide, primary focus
    "lfp":     "#2A9D8F",   # Teal-green  — olivine structure, only non-layered
    "nmc622":  "#457B9D",   # Steel blue  — layered oxide (mid-Ni)
    "lco":     "#7B2D8E",   # Purple      — traditional layered oxide (Co-rich)
    "nca":     "#F4A261",   # Warm orange — Ni-rich layered oxide (Al-stabilised)
    "nmc532":  "#6C757D",   # Slate gray  — layered oxide (declining)
    "nmc333":  "#ADB5BD",   # Light gray  — layered oxide (extinct)
}
MARKER_STYLE = {
    "nmc811":  "^",   # triangle up   — explosive growth
    "lfp":     "s",   # square        — stable olivine
    "nmc622":  "D",   # diamond
    "lco":     "o",   # circle
    "nca":     "p",   # pentagon
    "nmc532":  "v",   # triangle down — declining
    "nmc333":  "X",   # X             — extinct
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
    
    # --- x-axis order ---
    x_axis_order = kw.get("x_axis_order", ["Early Period", "Recent Period"])
    if len(x_axis_order) != 2:
        x_axis_order = ["Early Period", "Recent Period"]
    swapped = (x_axis_order[0] == "Recent Period")

    # --- text ---
    title    = kw.get("title_text",     "Q1CM3 — Cathode Materials: Layered Oxide vs Olivine")
    subtitle = kw.get("subtitle_text",  "Ni-rich NMC811 surge vs stable LFP olivine · 7 cathodes compared")
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
        # Dynamic y-values based on x_axis_order
        if swapped:
            yv = [row["Time_2"], row["Time_1"]]
        else:
            yv = [row["Time_1"], row["Time_2"]]
            
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
            gp   = f"  ({row['
