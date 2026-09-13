import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.lines as mlines  # Added for better legend handles
from matplotlib.patches import FancyBboxPatch
import numpy as np
import io
import os
import glob

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
        return sorted(list(plt.colormaps))
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
#  DATA LOADING — read all CSVs from concept-growth-datasets/
# ═══════════════════════════════════════════════════════════════
def get_data_dir():
    dir_name = "concept-growth-datasets"
    if "__file__" in globals():
        base = os.path.dirname(os.path.abspath(__file__))
        p = os.path.join(base, dir_name)
        if os.path.isdir(p):
            return p
    p = os.path.join(os.getcwd(), dir_name)
    if os.path.isdir(p):
        return p
    return dir_name

CSV_DIR = get_data_dir()

SYMBOLS_POOL = ["●", "■", "◆", "▲", "▶", "✦", "▼", "◉", "◇", "◈", "▣", "◐"]
MARKERS_POOL = ["o", "s", "D", "^", ">", "*", "v", "p", "X", "h", "P", "8"]
PALETTE_POOL = [
    "#D62828", "#6C757D", "#457B9D", "#2A9D8F", "#E9C46A",
    "#F4A261", "#E76F51", "#264653", "#8338EC", "#3A86FF",
    "#FF006E", "#FB5607", "#06D6A0", "#118AB2", "#073B4C",
    "#EF476F", "#FFD166", "#7209B7", "#4361EE", "#4CC9F0",
]

def _find_col(cols, keyword):
    for c in cols:
        if keyword.lower() in str(c).lower():
            return c
    return None

def _file_signature(csv_dir):
    if not os.path.isdir(csv_dir):
        return ()
    sig = []
    for fp in sorted(glob.glob(os.path.join(csv_dir, "*.csv"))):
        try:
            st_ = os.stat(fp)
            sig.append((os.path.basename(fp), int(st_.st_mtime), st_.st_size))
        except OSError:
            sig.append((os.path.basename(fp), 0, 0))
    return tuple(sig)

def load_all_concepts(csv_dir):
    if not os.path.isdir(csv_dir):
        return None, (f"Directory '{csv_dir}' not found. "
                      f"Make sure the folder sits next to your app script.")

    csv_files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if not csv_files:
        return None, f"No CSV files found in '{csv_dir}'."

    frames = []
    errors = []
    for csv_path in csv_files:
        try:
            d = pd.read_csv(csv_path, encoding="utf-8-sig")
        except Exception as e:
            errors.append(f"{os.path.basename(csv_path)}: {e}")
            continue

        d = d.loc[:, ~d.columns.astype(str).str.startswith("Unnamed")]
        d.columns = [str(c).strip() for c in d.columns]

        c_concept = _find_col(d.columns, "concept")
        c_early   = _find_col(d.columns, "early")
        c_recent  = _find_col(d.columns, "recent")
        c_growth  = _find_col(d.columns, "growth")

        if not all([c_concept, c_early, c_recent, c_growth]):
            errors.append(f"{os.path.basename(csv_path)}: missing required columns")
            continue

        sub = d[[c_concept, c_early, c_recent, c_growth]].copy()
        sub.columns = ["Concept", "Early Count", "Recent Count", "Growth Rate (%)"]

        sub["Early Count"]     = pd.to_numeric(sub["Early Count"],     errors="coerce").fillna(0)
        sub["Recent Count"]    = pd.to_numeric(sub["Recent Count"],    errors="coerce").fillna(0)
        sub["Growth Rate (%)"] = pd.to_numeric(sub["Growth Rate (%)"], errors="coerce").fillna(0)
        sub["Concept"]         = sub["Concept"].astype(str).str.strip()

        sub["Domain"] = os.path.splitext(os.path.basename(csv_path))[0]
        sub["_src_order"] = len(frames)
        frames.append(sub)

    if not frames:
        return None, "No valid CSV files could be parsed.\n" + "\n".join(errors)

    combined = pd.concat(frames, ignore_index=True)
    total_before = len(combined)

    combined = (combined
                .sort_values("_src_order")
                .drop_duplicates(subset="Concept", keep="first")
                .drop(columns="_src_order")
                .reset_index(drop=True))
    n_dupes = total_before - len(combined)

    mask_zero = (
        (combined["Early Count"] == 0)
        & (combined["Recent Count"] == 0)
        & (combined["Growth Rate (%)"] == 0)
    )
    n_excluded = int(mask_zero.sum())
    combined = combined[~mask_zero].reset_index(drop=True)

    combined = combined.drop_duplicates(
        subset="Concept", keep="first"
    ).reset_index(drop=True)

    msg = (f"Loaded {len(csv_files)} CSV file(s) · "
           f"{len(combined)} unique concept(s) · "
           f"{n_dupes} duplicate row(s) merged · "
           f"{n_excluded} all-zero concept(s) excluded.")
    if errors:
        msg += "\n⚠️  " + " | ".join(errors)
    return combined, msg


@st.cache_data(show_spinner=False)
def _cached_load(csv_dir, _signature):
    return load_all_concepts(csv_dir)

_sig = _file_signature(CSV_DIR)
_combined_df, _load_msg = _cached_load(CSV_DIR, _sig)

if _combined_df is None or len(_combined_df) == 0:
    st.error(f"⚠️  Could not load concept data.\n\n{_load_msg}")
    st.stop()

_combined_df = _combined_df.sort_values(
    "Growth Rate (%)", ascending=False
).reset_index(drop=True)

df = pd.DataFrame({
    "Material":  _combined_df["Concept"].astype(str).values,
    "Time_1":    _combined_df["Early Count"].astype(float).values,
    "Time_2":    _combined_df["Recent Count"].astype(float).values,
    "Symbol":    [SYMBOLS_POOL[i % len(SYMBOLS_POOL)] for i in range(len(_combined_df))],
    "Highlight": [i == 0 for i in range(len(_combined_df))],
    "Domain":    _combined_df["Domain"].astype(str).values,
})
df["Growth"]     = _combined_df["Growth Rate (%)"].astype(float).round(2).values
df["Growth_Str"] = df["Growth"].apply(
    lambda g: f"+{g:.2f}%" if g >= 0 else f"{g:.2f}%"
)

df = df.reset_index(drop=True)
df["RowId"]  = df.index.astype(int)
df["RowKey"] = df["RowId"].astype(str) + "::" + df["Material"]

DEFAULT_PALETTE = {m: PALETTE_POOL[i % len(PALETTE_POOL)]
                   for i, m in enumerate(df["Material"])}
MARKER_STYLE    = {m: MARKERS_POOL[i % len(MARKERS_POOL)]
                   for i, m in enumerate(df["Material"])}

HIGHLIGHT_CONCEPT = df.iloc[0]["Material"] if len(df) else None
HIGHLIGHT_ROWKEY  = df.iloc[0]["RowKey"]   if len(df) else None

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
    show_left   = kw.get("show_left_labels",  True)
    show_right  = kw.get("show_right_labels", True)
    show_sym    = kw.get("show_symbols",      True)
    show_gpct   = kw.get("show_growth_pct",   True)
    label_oy    = kw.get("label_offset_y",    0)
    label_bg    = kw.get("label_bg",          False)
    label_rot   = kw.get("label_rotation",    0)
    conn_lines  = kw.get("connector_lines",   False)

    line_w      = kw.get("line_width",   3.0)
    curv        = kw.get("curvature",    0.0)
    line_alpha  = kw.get("line_alpha",   0.85)
    show_arrow  = kw.get("show_arrow",   False)

    use_cmap     = kw.get("use_cmap",      False)
    cmap_name    = kw.get("cmap_name",     "viridis")
    show_cbar    = kw.get("show_colorbar", True)
    cmap_reverse = kw.get("cmap_reverse",  False)

    cust_col = kw.get("custom_colors",    DEFAULT_PALETTE)
    ln_styles= kw.get("line_styles",      {})
    mk_over  = kw.get("marker_overrides", MARKER_STYLE)

    tri_bg  = kw.get("three_color_bg",        False)
    bg1     = kw.get("bg_color1",             "#FFE5B4")
    bg2     = kw.get("bg_color2",             "#FF7F50")
    bg3     = kw.get("bg_color3",             "#CD5C5C")
    bg_alpha= kw.get("bg_gradient_alpha",     0.15)
    bg_dir  = kw.get("bg_gradient_direction", "Vertical (Top→Bottom)")

    box_on      = kw.get("box_visible",       True)
    box_col     = kw.get("box_color",         "#888888")
    box_w       = kw.get("box_width",         2.0)
    box_ls      = kw.get("box_linestyle",     "solid")
    box_rad     = kw.get("box_corner_radius", 0.02)
    box_shad    = kw.get("box_shadow",        True)
    box_fill    = kw.get("box_fill",          False)
    box_fill_col= kw.get("box_fill_color",    "#FFFFFF")
    box_fill_al = kw.get("box_fill_alpha",     0.05)

    hi_star   = kw.get("highlight_star",    True)
    shad_alpha= kw.get("shadow_alpha",      0.25)

    ann_rowkey    = kw.get("annotate_rowkey",    None)
    ann_symbol    = kw.get("ann_symbol",         "★")
    ann_box_style = kw.get("ann_box_style",      "round,pad=0.4")
    ann_arrow_sty = kw.get("ann_arrow_style",    "->")
    ann_arrow_lw  = kw.get("ann_arrow_lw",       2.5)
    ann_offset    = kw.get("ann_offset",         0.35)
    ann_curve_rad = kw.get("ann_curve_rad",      -0.2)
    ann_font_extra= kw.get("ann_font_extra",     2)

    log_sc    = kw.get("log_scale",     False)
    show_grid = kw.get("show_grid",     True)
    grid_style= kw.get("grid_style",    "--")
    y_min     = kw.get("y_min",         None)
    y_max     = kw.get("y_max",         None)
    leg_loc   = kw.get("legend_loc",    "None")
    sp_w      = kw.get("spine_width",   1.0)
    tk_len    = kw.get("tick_length",   6)
    tk_w      = kw.get("tick_width",    1.0)

    title    = kw.get("title_text",     "Concept Growth — Slope Chart")
    subtitle = kw.get("subtitle_text",  "")
    xl_text  = kw.get("xlabel_text",    "Time Period")
    yl_text  = kw.get("ylabel_text",    "Publication Occurrences")
    watermark= kw.get("watermark_text", "")

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

    for idx, row in df_active.iterrows():
        mat   = row["Material"]
        yv    = [row["Time_1"], row["Time_2"]]
        color = col_for(mat, row["Growth"])
        marker= mk_over.get(mat, MARKER_STYLE.get(mat, "o"))
        ls    = ln_styles.get(mat, "-")
        star  = bool(row["Highlight"]) and hi_star

        lw = line_w * (1.8 if star else 1.0)
        ms = mk_sz  * (1.4 if star else 1.0)
        al = min(line_alpha, 1.0) if star else line_alpha * 0.85
        zo = 10 if star else 5

        use_curve = abs(curv) > 0.001
        if use_curve:
            xc, yc = make_curved_line(xp[0], yv[0], xp[1], yv[1], curv)
        else:
            xc, yc = xp, yv

        if star and shad_alpha > 0:
            ax.plot(xc, yc, color=color, lw=lw + 4,
                    alpha=shad_alpha * 0.5, zorder=zo - 1)
            ax.plot(xc, yc, color=color, lw=lw + 2,
                    alpha=shad_alpha,       zorder=zo - 1)

        line_label = row["RowKey"]
        ax.plot(xc, yc, color=color, lw=lw, alpha=al, zorder=zo,
                linestyle=ls, solid_capstyle="round",
                dash_capstyle="round", label=line_label)

        ax.plot(xc[0],  yc[0],  marker=marker, ms=ms, color=color,
                zorder=zo + 1, markeredgecolor=edge_c, markeredgewidth=1.5)
        ax.plot(xc[-1], yc[-1], marker=marker, ms=ms, color=color,
                zorder=zo + 1, markeredgecolor=edge_c, markeredgewidth=1.5)

        if show_arrow:
            ax.annotate("", xy=(xp[1] + 0.06, yv[1]),
                        xytext=(xp[1] - 0.08, yv[1]),
                        arrowprops=dict(arrowstyle="->", color=color,
                                        lw=lw * 0.7), zorder=zo + 2)

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
            ltxt = f"{sym} {mat}\n{yv[0]:,.0f}".strip()
            ax.text(xp[0] - 0.08, yv[0] + oy, ltxt,
                    ha="right", va="center", fontsize=fl,
                    rotation=label_rot, color=color,
                    fontweight="bold" if star else "normal",
                    path_effects=stroke, bbox=bbox_p)

        if show_right:
            gp   = f"  ({row['Growth_Str']})" if show_gpct else ""
            rtxt = f"{yv[1]:,.0f}{gp}"
            ax.text(xp[1] + 0.08, yv[1] + oy, rtxt,
                    ha="left", va="center", fontsize=fl,
                    rotation=label_rot, color=color,
                    fontweight="bold" if star else "normal",
                    path_effects=stroke, bbox=bbox_p)

    # ─── ANNOTATION (Skipped if ann_rowkey is None) ────────────
    if ann_rowkey and ann_rowkey in df_active["RowKey"].values:
        sr  = df_active[df_active["RowKey"] == ann_rowkey].iloc[0]
        mx  = 1.5
        my  = (sr["Time_1"] + sr["Time_2"]) / 2
        oy2 = my * ann_offset if log_sc else 80
        ac  = col_for(sr["Material"], sr["Growth"])

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

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Early Period", "Recent Period"],
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
        cm_bg = mcolors.LinearSegmentedColormap.from_list("tbg", clist, N=256)
        ax.imshow(grad, aspect="auto", cmap=cm_bg, alpha=bg_alpha,
                  extent=[xl, xr, yb, yt], origin="lower", zorder=0)

    if use_cmap and show_cbar and cmap_obj and norm_obj:
        sm = cm.ScalarMappable(cmap=cmap_obj, norm=norm_obj)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.02, shrink=0.8)
        cbar.set_label("Growth (%)", fontsize=fs, color=txt_c)
        cbar.ax.tick_params(colors=txt_c, labelsize=fs - 1)
        cbar.outline.set_edgecolor(sp_c)
        cbar.outline.set_linewidth(0.8)

    # ─── LEGEND GENERATION (Using Line2D for perfect accuracy) ───
    if leg_loc != "None" and n > 0:
        legend_handles = []
        for idx, row in df_active.iterrows():
            mat    = row["Material"]
            color  = col_for(mat, row["Growth"])
            marker = mk_over.get(mat, MARKER_STYLE.get(mat, "o"))
            ls     = ln_styles.get(mat, "-")
            
            # Format label: Symbol + Concept + Growth %
            lbl = f"{row['Symbol']}  {mat}  ({row['Growth_Str']})"
            
            handle = mlines.Line2D([], [], color=color, marker=marker,
                                   linestyle=ls, markersize=8,
                                   markeredgecolor=edge_c, markeredgewidth=1.5,
                                   label=lbl)
            legend_handles.append(handle)

        leg = ax.legend(handles=legend_handles, loc=leg_loc, fontsize=fs + 1,
                        frameon=True, fancybox=True, shadow=True,
                        edgecolor=sp_c,
                        facecolor=("#FFFFFF" if bg_st == "Light" else "#2B2B3D"),
                        labelcolor=txt_c, borderpad=0.8,
                        handletextpad=0.6)
        leg.get_frame().set_linewidth(1.2)

    if watermark:
        fig.text(0.99, 0.01, watermark, fontsize=8, color=txt_c,
                 alpha=0.3, ha="right", va="bottom", style="italic")

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

    if show_hover and HAVE_MPLCURSORS:
        cursor = mplcursors.cursor(ax.lines, hover=True)
        cursor.connect("add", lambda sel: sel.annotation.set_text(
            f"{sel.artist.get_label()}: {sel.target[1]:.0f}"))

    st.pyplot(fig, use_container_width=True)
    return fig


# ═══════════════════════════════════════════════════════════════
#  STREAMLIT PAGE
# ═══════════════════════════════════════════════════════════════
st.set_page_config(page_title="Concept Growth Slope Chart", layout="wide")

_domains    = sorted(df["Domain"].unique())
_n_concepts = len(df)
_top_growth = df.iloc[0]["Growth"] if _n_concepts else 0

st.html(f"""<div style="display:flex;align-items:center;gap:12px;margin-bottom:4px">
<span style="font-size:2.2rem">📈</span>
<span style="font-size:1.7rem;font-weight:700;
background:linear-gradient(90deg,#D62828,#6C757D,#457B9D);
-webkit-background-clip:text;-webkit-text-fill-color:transparent">
Concept Growth — Slope Chart</span></div>
<p style="color:#888;margin-top:-4px;margin-bottom:16px">
Loaded <b>{_n_concepts}</b> unique concept(s) from
<b>{len(_domains)}</b> domain file(s) in
<code>concept-growth-datasets/</code> &middot;
Highest growth: <b>{HIGHLIGHT_CONCEPT}</b>
({_top_growth:+.2f}%)</p>""")

st.caption("ℹ️  " + _load_msg)

# ─── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.header("🎛️  Controls")

    # ── 1. Domain filter + concept toggles ──
    with st.expander("📌 Concept Toggles", expanded=True):
        sel_domains = st.multiselect(
            "Filter by Domain file", _domains, default=_domains,
            key="domain_filter")

        vis_df = df[df["Domain"].isin(sel_domains)].reset_index(drop=True)
        visible_keys  = vis_df["RowKey"].tolist()
        _n_visible    = len(vis_df)

        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            if st.button("✅ Select All", use_container_width=True):
                for rk in visible_keys:
                    st.session_state[f"tog_{rk}"] = True
        with cc2:
            if st.button("❌ Clear All", use_container_width=True):
                for rk in visible_keys:
                    st.session_state[f"tog_{rk}"] = False
        with cc3:
            if st.button("⭐ Top 5 Growth", use_container_width=True):
                top5 = set(df.head(5)["RowKey"].tolist())
                for rk in visible_keys:
                    st.session_state[f"tog_{rk}"] = rk in top5

        toggle_states = {}
        n_cols = 3
        cols = st.columns(n_cols)
        for i, row in vis_df.iterrows():
            rk   = row["RowKey"]
            mat  = row["Material"]
            sym  = row["Symbol"]
            default = (i < 5)  # <--- CHANGE THIS TO True IF YOU WANT ALL ON BY DEFAULT
            with cols[i % n_cols]:
                cur = st.session_state.get(f"tog_{rk}", default)
                toggle_states[rk] = st.toggle(
                    f"{sym} {mat}", cur, key=f"tog_{rk}")

        st.caption(f"ℹ️  {_n_concepts} unique concept(s) available "
                   f"({_n_visible} shown after domain filter).  "
                   f"Top growth: {HIGHLIGHT_CONCEPT} "
                   f"({df.iloc[0]['Growth_Str']}).")

    # ── 2. Label controls ──
    with st.expander("🏷️  Label Controls", expanded=True):
        show_left  = st.checkbox("Left Labels  (name + value)", True)
        show_right = st.checkbox("Right Labels (value + growth)", True)
        c1, c2 = st.columns(2)
        with c1:
            show_sym  = st.checkbox("Symbols", True)
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
                              -1.0, 1.0, 0.0, 0.05)
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
    ln_styles_dict = {m: "-" for m in df["Material"].unique()}
    mk_over_dict   = MARKER_STYLE.copy()

    with st.expander("🎨  Per-Concept Styling", expanded=False):
        st.markdown(f"**Colors** ({_n_visible} visible)")
        cc = {}
        for i, row in vis_df.iterrows():
            rk  = row["RowKey"]
            mat = row["Material"]
            if i % 3 == 0:
                cols = st.columns(3)
            with cols[i % 3]:
                cc[mat] = st.color_picker(
                    mat[:20], DEFAULT_PALETTE.get(mat, "#888888"),
                    key=f"clr_{rk}")
        if not use_cmap:
            custom_colors.update(cc)

        st.markdown("**Line Styles**")
        ls_d = {}
        for i, row in vis_df.iterrows():
            rk  = row["RowKey"]
            mat = row["Material"]
            if i % 3 == 0:
                cols = st.columns(3)
            with cols[i % 3]:
                ls_d[mat] = st.selectbox(
                    mat[:20], ["-", "--", "-.", ":"], key=f"ls_{rk}")
        ln_styles_dict.update(ls_d)

        st.markdown("**Markers**")
        mo = {}
        mk_opts = ["o", "s", "D", "^", "v", "*", "p", "X", "h", "P", "8"]
        for i, row in vis_df.iterrows():
            rk  = row["RowKey"]
            mat = row["Material"]
            if i % 3 == 0:
                cols = st.columns(3)
            _mk = MARKER_STYLE.get(mat, "o")
            di = mk_opts.index(_mk) if _mk in mk_opts else 0
            with cols[i % 3]:
                mo[mat] = st.selectbox(
                    mat[:20], mk_opts, index=di, key=f"mk_{rk}")
        mk_over_dict.update(mo)

    # ── 6. Three-color gradient background ──
    with st.expander("🌅  Three-Color Gradient / Shade", expanded=False):
        tri_bg = st.checkbox("Enable Gradient Background", False)
        bg_pre = st.selectbox("Preset", list(BG_PRESETS.keys()), index=0)
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
        a_opts  = [None] + list(df["RowKey"])
        # CHANGED: Default to None so it doesn't show the 2100% box by default
        default_idx = 0 
        ann_rowkey = st.selectbox(
            "Annotate Concept", a_opts,
            format_func=lambda x: (
                "None" if x is None
                else (x.split("::", 1)[1]
                      if "::" in x else str(x))),
            index=default_idx)

        if ann_rowkey:
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
        # CHANGED: Default to False so the highest growth isn't automatically highlighted
        hi_star = st.checkbox(
            f"Highlight {HIGHLIGHT_CONCEPT} (Highest Growth)",
            False)
        shad_alpha = st.slider("Glow Intensity", 0.0, 1.0, 0.25, 0.05)

    # ── 10. Titles & text ──
    with st.expander("📝  Titles & Text", expanded=False):
        title_t = st.text_input(
            "Title",
            "Concept Growth — Early vs Recent Period")
        sub_t   = st.text_input(
            "Subtitle",
            f"{_n_concepts} unique concepts from {len(_domains)} domain file(s)")
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
        # CHANGED: Default index from 0 to 1 ("best" instead of "None") to show legend by default
        leg_loc = st.selectbox(
            "Legend Position",
            ["None", "best", "upper right", "upper left",
             "lower left", "lower right", "center"],
            index=1)
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

# ─── Active data (RowKey-based, always unique) ───────────────
active_keys = [rk for rk, on in toggle_states.items() if on]
df_active = df[df["RowKey"].isin(active_keys)].copy()

# ─── Data table ──────────────────────────────────────────────
with st.expander(f"📊  View Raw Data  ({len(df)} unique concepts)",
                 expanded=False):
    view_df = df[["Material", "Domain", "Time_1", "Time_2", "Growth_Str"]].copy()
    st.dataframe(
        view_df.rename(columns={
            "Material":    "Concept",
            "Time_1":      "Early Count",
            "Time_2":      "Recent Count",
            "Growth_Str":  "Growth"}),
        use_container_width=True, hide_index=True,
        column_config={
            "Concept":      st.column_config.TextColumn("Concept"),
            "Domain":       st.column_config.TextColumn("First Seen In (Domain File)"),
            "Early Count":  st.column_config.NumberColumn(
                "Early Count", format="%d"),
            "Recent Count": st.column_config.NumberColumn(
                "Recent Count", format="%d"),
            "Growth":       st.column_config.TextColumn("Growth"),
        })
    st.caption(
        f"Source: 'concept-growth-datasets/' directory.  "
        f"Duplicate concepts across files are merged once "
        f"(first file wins — counts are NOT summed).  "
        f"Concepts where Early Count, Recent Count AND Growth Rate "
        f"are all 0 have been excluded.  "
        f"Highest growth: {HIGHLIGHT_CONCEPT} "
        f"({df.iloc[0]['Growth_Str']}).")

# ─── Plot ────────────────────────────────────────────────────
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
    annotate_rowkey=ann_rowkey,   ann_symbol=ann_symbol,
    ann_box_style=ann_box_style,  ann_arrow_style=ann_arrow_sty,
    ann_arrow_lw=ann_arrow_lw,    ann_offset=ann_offset,
    ann_curve_rad=ann_curve_rad,  ann_font_extra=ann_font_extra,
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

# ─── Export ──────────────────────────────────────────────────
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
                file_name=f"concept_growth_slope_chart.{ext}",
                mime=mime, use_container_width=True)

# ─── Footer ──────────────────────────────────────────────────
st.markdown("---")
st.caption(
    f"Concept Growth Slope Chart  ·  "
    f"Source directory: concept-growth-datasets/  ·  "
    f"Duplicate concepts across files merged once (no summing)  ·  "
    f"Growth Rate (%) taken directly from CSVs  ·  "
    f"All-zero concepts excluded  ·  "
    f"Available colormaps: **{len(ALL_CMAPS)}**  ·  "
    "Built with Streamlit & Matplotlib")
