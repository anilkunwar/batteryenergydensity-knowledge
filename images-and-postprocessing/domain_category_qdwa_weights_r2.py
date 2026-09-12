import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
import matplotlib.cm as cm
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
#  OPTION DICTS  (unchanged)
# ═══════════════════════════════════════════════════════════════
ANN_SYMBOLS = {
    "★  Star": "★", "▲  Triangle": "▲", "●  Circle": "●",
    "◆  Diamond": "◆", "▶  Arrow": "▶", "✦  Star Open": "✦",
    "■  Square": "■", "None": "",
}
ANN_ARROW_STYLES = {
    "→  Standard": "->", "▷  Open": "-|>",
    "⟶  Fancy": "fancy", "—  Simple": "simple",
}
ANN_BOX_STYLES = {
    "Rounded": "round,pad=0.4", "Square": "square,pad=0.4",
    "Sawtooth": "sawtooth,pad=0.4", "None": None,
}
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
#  DATA LOADING
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

# Friendly short names for the domain files (used in badges/tables)
DOMAIN_ALIAS = {
    "battery-degradation-Q1D5-concept-growth-rate":     "degradation",
    "concept-growth-q1am2-anode-materials":             "anode",
    "concept-growth-q1pm1-performance-metrics":         "performance",
    "concept-growth-rate-q1m4-manufacturing":           "manufacturing",
    "electrolyte-systems-q1es6-concept-growth":         "electrolyte",
}
def _alias(domain: str) -> str:
    return DOMAIN_ALIAS.get(domain, domain[:14])


def _find_col(cols, keyword):
    for c in cols:
        if keyword.lower() in str(c).lower():
            return c
    return None


def load_all_concepts(csv_dir):
    """Read every CSV and return a *per-file row* DataFrame (not yet
    deduplicated)."""
    if not os.path.isdir(csv_dir):
        return None, (f"Directory '{csv_dir}' not found.")
    files = sorted(glob.glob(os.path.join(csv_dir, "*.csv")))
    if not files:
        return None, f"No CSV files found in '{csv_dir}'."

    frames, errors = [], []
    for fp in files:
        try:
            d = pd.read_csv(fp)
        except Exception as e:
            errors.append(f"{os.path.basename(fp)}: {e}")
            continue
        d = d.loc[:, ~d.columns.astype(str).str.startswith("Unnamed")]
        d.columns = [str(c).strip() for c in d.columns]

        c_concept = _find_col(d.columns, "concept")
        c_early   = _find_col(d.columns, "early")
        c_recent  = _find_col(d.columns, "recent")
        c_growth  = _find_col(d.columns, "growth")
        if not all([c_concept, c_early, c_recent, c_growth]):
            errors.append(f"{os.path.basename(fp)}: missing required columns")
            continue

        sub = d[[c_concept, c_early, c_recent, c_growth]].copy()
        sub.columns = ["Concept", "Early Count", "Recent Count", "Growth Rate (%)"]
        sub["Early Count"]     = pd.to_numeric(sub["Early Count"],     errors="coerce").fillna(0)
        sub["Recent Count"]    = pd.to_numeric(sub["Recent Count"],    errors="coerce").fillna(0)
        sub["Growth Rate (%)"] = pd.to_numeric(sub["Growth Rate (%)"], errors="coerce").fillna(0)
        sub["Domain"] = os.path.splitext(os.path.basename(fp))[0]
        frames.append(sub)

    if not frames:
        return None, "No valid CSV files could be parsed.\n" + "\n".join(errors)

    combined = pd.concat(frames, ignore_index=True)

    # Drop all-zero rows
    mask_zero = ((combined["Early Count"] == 0)
                 & (combined["Recent Count"] == 0)
                 & (combined["Growth Rate (%)"] == 0))
    n_excluded = int(mask_zero.sum())
    combined = combined[~mask_zero].reset_index(drop=True)

    return combined, (f"Loaded {len(files)} CSV file(s) · "
                      f"{len(combined)} raw concept-row(s) · "
                      f"{n_excluded} all-zero row(s) excluded.")


@st.cache_data(show_spinner=False)
def _cached_load(csv_dir):
    return load_all_concepts(csv_dir)


# ═══════════════════════════════════════════════════════════════
#  ▓▓ REWRITTEN ▓▓  DEDUPLICATE — same concept name = same concept.
#  Counts are expected to be identical across files; we pick them
#  once and record domain membership + integrity warnings.
# ═══════════════════════════════════════════════════════════════
def consolidate_concepts(per_file_df: pd.DataFrame):
    """Return (unique_concepts_df, anomalies_df)."""
    rows, anomalies = [], []
    for concept, grp in per_file_df.groupby("Concept", sort=False):
        e_vals = sorted(set(grp["Early Count"].tolist()))
        r_vals = sorted(set(grp["Recent Count"].tolist()))
        domains = sorted(grp["Domain"].unique().tolist())

        consistent = (len(e_vals) == 1 and len(r_vals) == 1)

        if consistent:
            early  = float(e_vals[0])
            recent = float(r_vals[0])
        else:
            # Same concept with conflicting counts across files — flag it,
            # and fall back to the maximum observed (safest default).
            early  = float(grp["Early Count"].max())
            recent = float(grp["Recent Count"].max())
            anomalies.append({
                "Concept":  concept,
                "Domains":  ", ".join(_alias(d) for d in domains),
                "Early values":  e_vals,
                "Recent values": r_vals,
            })

        # Recompute growth from the (deduplicated) counts when defined.
        if early > 0:
            growth = round((recent - early) / early * 100.0, 2)
        else:
            # early == 0 → growth is mathematically undefined
            growth = float(grp["Growth Rate (%)"].max())

        rows.append({
            "Concept":         concept,
            "Early Count":     early,
            "Recent Count":    recent,
            "Growth Rate (%)": growth,
            "Domains":         domains,
            "Domain_Count":    len(domains),
            "Domains_Short":   ", ".join(_alias(d) for d in domains),
            "Inconsistent":    not consistent,
        })

    unique_df = pd.DataFrame(rows).sort_values(
        "Growth Rate (%)", ascending=False
    ).reset_index(drop=True)

    anomalies_df = pd.DataFrame(anomalies)
    return unique_df, anomalies_df


_combined_raw, _load_msg = _cached_load(CSV_DIR)
if _combined_raw is None or len(_combined_raw) == 0:
    st.error(f"⚠️  Could not load concept data.\n\n{_load_msg}")
    st.stop()

# ▓▓ Deduplicate concepts ▓▓
UNIQUE_DF, ANOMALIES_DF = consolidate_concepts(_combined_raw)


# ═══════════════════════════════════════════════════════════════
#  ▓▓ REWRITTEN ▓▓  INSIGHTS — now based on unique concepts
# ═══════════════════════════════════════════════════════════════
def compute_insights(unique_df: pd.DataFrame, per_file_df: pd.DataFrame) -> dict:
    if unique_df is None or len(unique_df) == 0:
        return {}
    ins = {
        "early_min":  float(unique_df["Early Count"].min()),
        "early_max":  float(unique_df["Early Count"].max()),
        "recent_min": float(unique_df["Recent Count"].min()),
        "recent_max": float(unique_df["Recent Count"].max()),
        "growth_min": float(unique_df["Growth Rate (%)"].min()),
        "growth_max": float(unique_df["Growth Rate (%)"].max()),
        "n_concepts": int(len(unique_df)),
        "n_domains":  int(per_file_df["Domain"].nunique()),
        "n_raw_rows": int(len(per_file_df)),
    }

    # Concept × domain presence matrix
    presence = (per_file_df.assign(v=1)
                .pivot_table(index="Concept", columns="Domain",
                             values="v", aggfunc="max", fill_value=0))
    ins["presence_matrix"] = presence

    # Concepts that span multiple domains
    multi = unique_df[unique_df["Domain_Count"] > 1].copy()
    ins["multi_domain_concepts"] = multi.sort_values(
        ["Domain_Count", "Recent Count"], ascending=False
    )
    ins["n_multi_domain"] = int(len(multi))

    # Concepts unique to one domain
    single = unique_df[unique_df["Domain_Count"] == 1].copy()
    ins["n_single_domain"] = int(len(single))

    # Per-domain unique-concept counts
    dom_counts = (per_file_df.drop_duplicates(["Domain", "Concept"])
                  .groupby("Domain")["Concept"].count()
                  .sort_values(ascending=False))
    ins["concepts_per_domain"] = dom_counts

    return ins

INSIGHTS = compute_insights(UNIQUE_DF, _combined_raw)


# ═══════════════════════════════════════════════════════════════
#  ▓▓ REWRITTEN ▓▓  Working DataFrame — ONE row per unique concept
# ═══════════════════════════════════════════════════════════════
df = pd.DataFrame({
    "Material":     UNIQUE_DF["Concept"].astype(str).values,
    "Time_1":       UNIQUE_DF["Early Count"].astype(float).values,
    "Time_2":       UNIQUE_DF["Recent Count"].astype(float).values,
    "Symbol":       [SYMBOLS_POOL[i % len(SYMBOLS_POOL)] for i in range(len(UNIQUE_DF))],
    "Highlight":    [i == 0 for i in range(len(UNIQUE_DF))],
    "Domains":      UNIQUE_DF["Domains"].values,
    "Domain_Count": UNIQUE_DF["Domain_Count"].values,
    "Domains_Short": UNIQUE_DF["Domains_Short"].values,
    "Inconsistent": UNIQUE_DF["Inconsistent"].values,
})
df["Growth"]     = UNIQUE_DF["Growth Rate (%)"].astype(float).round(2).values
df["Growth_Str"] = df["Growth"].apply(lambda g: f"+{g:.2f}%" if g >= 0 else f"{g:.2f}%")

# RowKey is now unique by concept name (no domain prefix — concept IS the key)
df["RowKey"] = df["Material"]

# Compact badge: "[×2]" when in multiple domains, "" otherwise.
df["Domain_Badge"] = df["Domain_Count"].apply(
    lambda n: "" if n <= 1 else f" [×{n}]"
)

DEFAULT_PALETTE = {m: PALETTE_POOL[i % len(PALETTE_POOL)]
                   for i, m in enumerate(df["Material"].unique())}
MARKER_STYLE    = {m: MARKERS_POOL[i % len(MARKERS_POOL)]
                   for i, m in enumerate(df["Material"].unique())}

HIGHLIGHT_CONCEPT = df.iloc[0]["Material"] if len(df) else None
HIGHLIGHT_ROWKEY  = df.iloc[0]["RowKey"]   if len(df) else None


# ═══════════════════════════════════════════════════════════════
#  Helper — Bézier curve (unchanged)
# ═══════════════════════════════════════════════════════════════
def make_curved_line(x1, y1, x2, y2, curvature=0.0, n_pts=80):
    t  = np.linspace(0, 1, n_pts)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2 + curvature * max(abs(y2 - y1), 1)
    x  = (1 - t)**2 * x1 + 2 * (1 - t) * t * cx + t**2 * x2
    y  = (1 - t)**2 * y1 + 2 * (1 - t) * t * cy + t**2 * y2
    return x, y


# ═══════════════════════════════════════════════════════════════
#  MAIN PLOT FUNCTION — unchanged (same signature, same behaviour)
# ═══════════════════════════════════════════════════════════════
def plot_slope_chart(df_active, **kw):
    # ... [IDENTICAL BODY TO YOUR ORIGINAL plot_slope_chart] ...
    # Copy verbatim from your file — no changes needed here.
    pass  # <-- keep your original implementation


# NOTE: Paste your original plot_slope_chart() body here.
# Nothing inside it needs to change — the columns it reads
# (Material, Time_1, Time_2, Growth, Growth_Str, Symbol, Highlight,
#  RowKey, Domain_Badge) are all still present in `df`.


# ═══════════════════════════════════════════════════════════════
#  STREAMLIT PAGE HEADER
# ═══════════════════════════════════════════════════════════════
st.set_page_config(page_title="Concept Growth Slope Chart", layout="wide")

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
<b>{INSIGHTS['n_domains']}</b> domain file(s) &middot;
Highest growth: <b>{HIGHLIGHT_CONCEPT}</b> ({_top_growth:+.2f}%)</p>""")

st.caption("ℹ️  " + _load_msg +
           f"  ·  Deduplicated to {_n_concepts} unique concept(s).")


# ═══════════════════════════════════════════════════════════════
#  ▓▓ REWRITTEN ▓▓  INSIGHTS CARD
# ═══════════════════════════════════════════════════════════════
if INSIGHTS:
    with st.expander("🔎  Dataset Insights  (auto-learned)",
                     expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Unique concepts",       INSIGHTS["n_concepts"])
        c2.metric("Domain files",          INSIGHTS["n_domains"])
        c3.metric("Cross-domain concepts", INSIGHTS["n_multi_domain"])
        c4.metric("Single-domain concepts",INSIGHTS["n_single_domain"])

        c1, c2, c3 = st.columns(3)
        c1.metric("Early count range",
                  f"{INSIGHTS['early_min']:.0f} → {INSIGHTS['early_max']:.0f}")
        c2.metric("Recent count range",
                  f"{INSIGHTS['recent_min']:.0f} → {INSIGHTS['recent_max']:.0f}")
        c3.metric("Growth range",
                  f"{INSIGHTS['growth_min']:+.2f}% → {INSIGHTS['growth_max']:+.2f}%")

        # ─── Integrity warnings ─────────────────────────────────
        if len(ANOMALIES_DF) > 0:
            st.warning(
                f"⚠️  {len(ANOMALIES_DF)} concept(s) have *different* "
                f"counts across domain files. Values below use the maximum "
                f"observed; please check the source CSVs.")
            st.dataframe(ANOMALIES_DF, use_container_width=True,
                         hide_index=True)
        else:
            st.success(
                "✅  All duplicated concepts have identical counts across "
                "domain files — safe to deduplicate. No data integrity "
                "issues detected.")

        # ─── Concept × Domain presence matrix ───────────────────
        st.markdown("**Concept presence matrix** "
                    "(● = concept appears in this domain file)")
        pm = INSIGHTS["presence_matrix"].copy()
        pm.index.name = "Concept"
        pm.columns = [_alias(c) for c in pm.columns]
        st.dataframe(
            pm.replace({0: "·", 1: "●"}),
            use_container_width=True)

        # ─── Multi-domain concept table ─────────────────────────
        st.markdown("**Concepts spanning multiple domain files**")
        mdc = INSIGHTS["multi_domain_concepts"]
        if len(mdc):
            st.dataframe(
                mdc[["Concept", "Domain_Count", "Early Count",
                     "Recent Count", "Growth Rate (%)", "Domains_Short"]]
                  .rename(columns={
                      "Domain_Count":     "Files",
                      "Early Count":      "Early",
                      "Recent Count":     "Recent",
                      "Growth Rate (%)":  "Growth %",
                      "Domains_Short":    "Appears in"}),
                use_container_width=True, hide_index=True,
                column_config={
                    "Growth %": st.column_config.NumberColumn(format="%.2f")})
        else:
            st.info("No concept appears in more than one domain file.")

        st.markdown("**Concepts per domain file** (raw, before dedup)")
        st.dataframe(
            INSIGHTS["concepts_per_domain"]
                .rename("Unique concepts").reset_index()
                .rename(columns={"Domain": "Domain file"}),
            use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("🎛️  Controls")

    # ── Search & range filters (unchanged behaviour) ────────
    with st.expander("🔍  Search & Range Filters", expanded=False):
        concept_search = st.text_input(
            "Search concept name",
            "", key="concept_search",
            placeholder="e.g. nmc, fec, silicon")

        min_domains = st.slider(
            "Must appear in ≥ N domain files",
            min_value=1,
            max_value=max(1, int(df["Domain_Count"].max())),
            value=1, step=1, key="min_domains_filter",
            help="2 = only concepts that span at least 2 domain files.")

        early_rng = st.slider(
            "Early Count range",
            min_value=int(INSIGHTS["early_min"]),
            max_value=int(INSIGHTS["early_max"]),
            value=(int(INSIGHTS["early_min"]), int(INSIGHTS["early_max"])),
            step=max(1, int((INSIGHTS["early_max"] - INSIGHTS["early_min"]) / 100) or 1),
            key="early_rng")

        recent_rng = st.slider(
            "Recent Count range",
            min_value=int(INSIGHTS["recent_min"]),
            max_value=int(INSIGHTS["recent_max"]),
            value=(int(INSIGHTS["recent_min"]), int(INSIGHTS["recent_max"])),
            step=max(1, int((INSIGHTS["recent_max"] - INSIGHTS["recent_min"]) / 100) or 1),
            key="recent_rng")

        growth_rng = st.slider(
            "Growth (%) range",
            min_value=float(np.floor(INSIGHTS["growth_min"])),
            max_value=float(np.ceil(INSIGHTS["growth_max"])),
            value=(float(np.floor(INSIGHTS["growth_min"])),
                   float(np.ceil(INSIGHTS["growth_max"]))),
            step=1.0, key="growth_rng")

        sort_by = st.selectbox(
            "Sort concepts by",
            ["Growth Rate", "Recent Count", "Early Count",
             "Total Count", "Concept Name", "Domain Count"],
            index=0, key="sort_by")

    # ── ▓▓ REWRITTEN ▓▓  Domain membership filter ───────────
    with st.expander("🗂️  Domain Membership", expanded=False):
        st.caption("A concept is shown when it appears in **at least one** "
                   "of the selected domain files.")

        dom_options = sorted(
            set(d for doms in df["Domains"] for d in doms))
        dom_labels = {d: _alias(d) for d in dom_options}

        sel_domains = st.multiselect(
            "Concepts that appear in:",
            dom_options,
            default=dom_options,
            format_func=lambda d: dom_labels.get(d, d),
            key="domain_filter")

        only_multi = st.checkbox(
            "Only concepts spanning ≥ 2 domain files",
            value=False, key="only_multi")

        only_inconsistent = st.checkbox(
            "Only concepts flagged as inconsistent",
            value=False, key="only_inconsistent")

    # ── Concept toggles ─────────────────────────────────────
    with st.expander("📌 Concept Toggles", expanded=True):
        # Apply filters
        dom_set = set(sel_domains)
        def _matches_doms(doms):
            return any(d in dom_set for d in doms)

        mask = (df["Domains"].apply(_matches_doms)
                & df["Domain_Count"].ge(min_domains)
                & df["Time_1"].between(*early_rng)
                & df["Time_2"].between(*recent_rng)
                & df["Growth"].between(*growth_rng))
        if concept_search.strip():
            mask &= df["Material"].str.contains(
                concept_search.strip(), case=False, na=False)
        if only_multi:
            mask &= df["Domain_Count"].ge(2)
        if only_inconsistent:
            mask &= df["Inconsistent"]

        vis_df = df[mask].copy()

        # Sort
        if sort_by == "Total Count":
            vis_df["_total"] = vis_df["Time_1"] + vis_df["Time_2"]
            vis_df = vis_df.sort_values("_total", ascending=False)
        else:
            col_asc = {
                "Growth Rate":  ("Growth",       False),
                "Recent Count": ("Time_2",       False),
                "Early Count":  ("Time_1",       False),
                "Concept Name": ("Material",     True),
                "Domain Count": ("Domain_Count", False),
            }[sort_by]
            vis_df = vis_df.sort_values(col_asc[0], ascending=col_asc[1])
        vis_df = vis_df.reset_index(drop=True)

        visible_keys = vis_df["RowKey"].tolist()
        _n_visible   = len(vis_df)

        cc1, cc2, cc3, cc4 = st.columns(4)
        with cc1:
            if st.button("✅ All", use_container_width=True):
                for rk in visible_keys:
                    st.session_state[f"tog_{rk}"] = True
        with cc2:
            if st.button("❌ None", use_container_width=True):
                for rk in visible_keys:
                    st.session_state[f"tog_{rk}"] = False
        with cc3:
            if st.button("⭐ Top 5", use_container_width=True):
                top5 = set(vis_df.head(5)["RowKey"].tolist())
                for rk in visible_keys:
                    st.session_state[f"tog_{rk}"] = rk in top5
        with cc4:
            if st.button("🔁 Inv", use_container_width=True):
                for rk in visible_keys:
                    st.session_state[f"tog_{rk}"] = not st.session_state.get(
                        f"tog_{rk}", False)

        toggle_states = {}
        n_cols = 3
        cols = st.columns(n_cols)
        for i, row in vis_df.iterrows():
            rk, mat, sym = row["RowKey"], row["Material"], row["Symbol"]
            badge = row.get("Domain_Badge", "")
            label = f"{sym} {mat}{badge}"
            # Full domain list as toggle tooltip via help=
            help_txt = "Appears in: " + ", ".join(
                _alias(d) for d in row["Domains"])
            if row.get("Inconsistent"):
                help_txt += "   ⚠️  counts differ across files"
            with cols[i % n_cols]:
                cur = st.session_state.get(f"tog_{rk}", i < 5)
                toggle_states[rk] = st.toggle(
                    label, cur, key=f"tog_{rk}", help=help_txt)

        st.caption(
            f"ℹ️  {_n_concepts} unique concept(s) · "
            f"{_n_visible} match current filters · "
            f"{sum(toggle_states.values())} selected.")

    # ── All remaining sidebar sections are IDENTICAL to your original ──
    with st.expander("🏷️  Label Controls", expanded=True):
        show_left  = st.checkbox("Left Labels",  True)
        show_right = st.checkbox("Right Labels", True)
        c1, c2 = st.columns(2)
        with c1: show_sym  = st.checkbox("Symbols", True)
        with c2: show_gpct = st.checkbox("Growth %", True)
        label_oy   = st.slider("Label Vertical Offset", -150, 150, 0, 5)
        label_rot  = st.slider("Label Rotation (°)",   -45, 45, 0, 1)
        label_bg   = st.checkbox("Label Background Boxes", False)
        conn_lines = st.checkbox("Connector Dots → Labels", False)

    with st.expander("✏️  Line & Spline Style", expanded=True):
        line_w    = st.slider("Spline Thickness", 0.5, 14.0, 3.0, 0.5)
        curv      = st.slider("Curvature", -1.0, 1.0, 0.0, 0.05)
        line_alph = st.slider("Line Opacity", 0.1, 1.0, 0.85, 0.05)
        show_arrow= st.checkbox("Arrow at Line End", False)

    with st.expander("🌈  Colormap Mode", expanded=False):
        use_cmap    = st.checkbox("Color Lines by Growth Rate", False)
        cmap_search = st.text_input("Filter colormaps…", "", key="cms")
        filtered = ([c for c in ALL_CMAPS if cmap_search.lower() in c.lower()]
                    if cmap_search else ALL_CMAPS)
        cmap_name = st.selectbox(
            "Colormap", filtered,
            index=(filtered.index("viridis") if "viridis" in filtered else 0))
        cmap_reverse = st.checkbox("Reverse Colormap", False)
        if use_cmap and filtered:
            pc = safe_get_cmap(cmap_name + ("_r" if cmap_reverse else ""))
            st.image(pc(np.linspace(0, 1, 512).reshape(1, -1)),
                     use_container_width=True)
        show_cbar = st.checkbox("Show Colorbar", True)

    custom_colors  = DEFAULT_PALETTE.copy()
    ln_styles_dict = {m: "-" for m in df["Material"].unique()}
    mk_over_dict   = MARKER_STYLE.copy()

    with st.expander("🎨  Per-Concept Styling", expanded=False):
        st.markdown(f"**Colors** ({_n_visible} visible)")
        cc = {}
        for i, row in vis_df.iterrows():
            rk, mat = row["RowKey"], row["Material"]
            if i % 3 == 0: cols = st.columns(3)
            with cols[i % 3]:
                cc[mat] = st.color_picker(
                    mat[:20], DEFAULT_PALETTE.get(mat, "#888888"),
                    key=f"clr_{rk}")
        if not use_cmap:
            custom_colors.update(cc)

        st.markdown("**Line Styles**")
        ls_d = {}
        for i, row in vis_df.iterrows():
            rk, mat = row["RowKey"], row["Material"]
            if i % 3 == 0: cols = st.columns(3)
            with cols[i % 3]:
                ls_d[mat] = st.selectbox(
                    mat[:20], ["-", "--", "-.", ":"], key=f"ls_{rk}")
        ln_styles_dict.update(ls_d)

        st.markdown("**Markers**")
        mo = {}
        mk_opts = ["o", "s", "D", "^", "v", "*", "p", "X", "h", "P", "8"]
        for i, row in vis_df.iterrows():
            rk, mat = row["RowKey"], row["Material"]
            if i % 3 == 0: cols = st.columns(3)
            _mk = MARKER_STYLE.get(mat, "o")
            di = mk_opts.index(_mk) if _mk in mk_opts else 0
            with cols[i % 3]:
                mo[mat] = st.selectbox(mat[:20], mk_opts, index=di, key=f"mk_{rk}")
        mk_over_dict.update(mo)

    with st.expander("🌅  Three-Color Gradient", expanded=False):
        tri_bg = st.checkbox("Enable Gradient Background", False)
        bg_pre = st.selectbox("Preset", list(BG_PRESETS.keys()), index=0)
        p1, p2, p3 = BG_PRESETS[bg_pre]
        cols = st.columns(3)
        with cols[0]: bg1 = st.color_picker("Top / Left",    p1, key="bg1")
        with cols[1]: bg2 = st.color_picker("Middle",        p2, key="bg2")
        with cols[2]: bg3 = st.color_picker("Bottom / Right",p3, key="bg3")
        bg_alpha = st.slider("Gradient Opacity", 0.0, 0.8, 0.15, 0.05)
        bg_dir   = st.radio("Direction",
                            ["Vertical (Top→Bottom)", "Horizontal (Left→Right)"],
                            horizontal=True)

    with st.expander("📦  Axes Box / Border", expanded=False):
        box_on  = st.checkbox("Show Axes Box", True)
        box_col = st.color_picker("Border Color", "#888888", key="bxcol")
        box_w   = st.slider("Border Width",   0.5, 8.0, 2.0, 0.5)
        box_ls  = st.selectbox("Border Style",
                               ["solid", "dashed", "dotted", "dashdot"])
        box_rad = st.slider("Corner Roundness", 0.0, 0.1, 0.02, 0.005)
        box_shad= st.checkbox("Drop Shadow", True)
        box_fill= st.checkbox("Box Fill Tint", False)
        box_fill_col = st.color_picker("Fill Tint Color", "#FFFFFF", key="bxfill")
        box_fill_al  = st.slider("Fill Tint Opacity", 0.0, 0.3, 0.05, 0.01)

    with st.expander("📌  Annotation Callout", expanded=False):
        a_opts  = [None] + list(df["RowKey"])
        default_idx = 1 if HIGHLIGHT_ROWKEY else 0
        ann_rowkey = st.selectbox(
            "Annotate Concept", a_opts,
            format_func=lambda x: (
                "None" if x is None
                else f"{x}  ({df.loc[df['RowKey']==x, 'Domains_Short'].iloc[0]})"),
            index=default_idx)
        if ann_rowkey:
            ann_symbol  = ANN_SYMBOLS[st.selectbox(
                "Symbol", list(ANN_SYMBOLS.keys()), index=0, key="ann_sym")]
            ann_box_style = ANN_BOX_STYLES[st.selectbox(
                "Box Style", list(ANN_BOX_STYLES.keys()), index=0, key="ann_box")]
            ann_arrow_sty = ANN_ARROW_STYLES[st.selectbox(
                "Arrow Head", list(ANN_ARROW_STYLES.keys()), index=0, key="ann_arr")]
            ann_arrow_lw  = st.slider("Arrow Thickness", 1.0, 6.0, 2.5, 0.5)
            ann_curve_rad = st.slider("Arrow Curve", -0.5, 0.5, -0.2, 0.05)
            ann_offset    = st.slider("Callout Distance", 0.1, 1.0, 0.35, 0.05)
            ann_font_extra= st.slider("Extra Font Size", 0, 6, 2, 1)
        else:
            ann_symbol, ann_box_style, ann_arrow_sty = "★", "round,pad=0.4", "->"
            ann_arrow_lw, ann_curve_rad, ann_offset, ann_font_extra = 2.5, -0.2, 0.35, 2

    with st.expander("✨  Glow / Highlight", expanded=False):
        hi_star = st.checkbox(f"Highlight {HIGHLIGHT_CONCEPT} (Highest Growth)", True)
        shad_alpha = st.slider("Glow Intensity", 0.0, 1.0, 0.25, 0.05)

    with st.expander("📝  Titles & Text", expanded=False):
        title_t = st.text_input("Title", "Concept Growth — Early vs Recent Period")
        sub_t   = st.text_input(
            "Subtitle",
            f"{_n_concepts} unique concepts from {INSIGHTS['n_domains']} domain files")
        xl_t    = st.text_input("X-Axis Label", "Time Period")
        yl_t    = st.text_input("Y-Axis Label", "Publication Occurrences")
        wm_t    = st.text_input("Watermark", "")

    with st.expander("⚙️  Axes & Grid", expanded=False):
        log_sc    = st.checkbox("Log Scale (Y)", False)
        show_grid = st.checkbox("Show Grid",    True)
        grid_sty  = st.selectbox("Grid Style", ["--", ":", "-.", "-"])
        cust_yl   = st.checkbox("Custom Y-Limits", False)
        y_min = y_max = None
        if cust_yl:
            c1, c2 = st.columns(2)
            with c1: y_min = st.number_input("Y-min", value=0, step=10, key="ymin")
            with c2: y_max = st.number_input("Y-max", value=500, step=10, key="ymax")
        leg_loc = st.selectbox(
            "Legend Position",
            ["None", "best", "upper right", "upper left",
             "lower left", "lower right", "center"], index=0)
        sp_w   = st.slider("Spine Width",  0.5, 5.0, 1.0, 0.1)
        tk_len = st.slider("Tick Length",   2, 20, 6, 1)
        tk_w   = st.slider("Tick Width",    0.5, 5.0, 1.0, 0.1)

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


# ─── Active data ─────────────────────────────────────────────
active_keys = [rk for rk, on in toggle_states.items() if on]
df_active = df[df["RowKey"].isin(active_keys)].copy()


# ─── Data table ──────────────────────────────────────────────
with st.expander(f"📊  View Data  ({len(df)} unique concepts)", expanded=False):
    view_df = df[["Material", "Domains_Short", "Time_1", "Time_2",
                  "Growth_Str", "Domain_Count"]].copy()
    st.dataframe(
        view_df.rename(columns={
            "Material":     "Concept",
            "Domains_Short":"Appears in domain files",
            "Time_1":       "Early Count",
            "Time_2":       "Recent Count",
            "Growth_Str":   "Growth",
            "Domain_Count": "N domain files"}),
        use_container_width=True, hide_index=True,
        column_config={
            "Concept":              st.column_config.TextColumn("Concept"),
            "Appears in domain files": st.column_config.TextColumn(width="medium"),
            "Early Count":          st.column_config.NumberColumn(format="%d"),
            "Recent Count":         st.column_config.NumberColumn(format="%d"),
            "Growth":               st.column_config.TextColumn("Growth"),
            "N domain files":       st.column_config.NumberColumn(format="%d"),
        })
    st.caption(
        f"Counts are per-concept (deduplicated). The same concept appearing in "
        f"multiple domain files has its counts shown ONCE. "
        f"'{INSIGHTS['n_multi_domain']}' concept(s) span multiple domain files.")


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


st.markdown("---")
st.caption(
    f"Concept Growth Slope Chart · Source: concept-growth-datasets/ · "
    f"Concepts deduplicated across domain files · "
    f"Available colormaps: **{len(ALL_CMAPS)}** · "
    "Built with Streamlit & Matplotlib")
