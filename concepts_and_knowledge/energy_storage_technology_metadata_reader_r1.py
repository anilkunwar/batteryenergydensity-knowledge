import streamlit as st
import os
import json
import pandas as pd
from pathlib import Path
import math

# ─── Page Config ───
st.set_page_config(
    page_title="JSON Metadata Explorer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Paths ───
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_METADATA_DIR = os.path.join(SCRIPT_DIR, "json_metadatabase")
os.makedirs(JSON_METADATA_DIR, exist_ok=True)

# ─── Caching ───
@st.cache_data(show_spinner=False)
def load_all_json_files(directory):
    """Load every .json file in *directory* and return a list of (filepath, records)."""
    files = sorted(Path(directory).glob("*.json"))
    if not files:
        return []
    loaded = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)          # Python allows NaN by default
            if isinstance(data, list):
                loaded.append((str(fp.name), data))
            elif isinstance(data, dict):
                loaded.append((str(fp.name), [data]))
            else:
                loaded.append((str(fp.name), []))
        except Exception as e:
            st.error(f"Error loading {fp.name}: {e}")
    return loaded

@st.cache_data(show_spinner=False)
def build_master_dataframe(file_records):
    """Flatten all records into one DataFrame."""
    rows = []
    for fname, records in file_records:
        for rec in records:
            if not isinstance(rec, dict):
                continue
            rec = dict(rec)          # shallow copy
            rec["_source_file"] = fname
            rows.append(rec)
    if not rows:
        return pd.DataFrame()
    df = pd.json_normalize(rows)
    # Replace literal NaN / null with pandas NA for cleaner display
    df = df.replace({float("nan"): pd.NA, None: pd.NA, "NaN": pd.NA, "": pd.NA})
    # Ensure Year is numeric where possible
    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    return df

# ─── Load Data ───
with st.spinner("Scanning json_metadatabase …"):
    file_records = load_all_json_files(JSON_METADATA_DIR)
    df = build_master_dataframe(file_records)

# ─── Sidebar ───
st.sidebar.title("📂 Metadata Explorer")
st.sidebar.caption(f"Directory: `{JSON_METADATA_DIR}`")

if not file_records:
    st.sidebar.warning("No .json files found.")
    st.info("Place JSON metadata files in the `json_metadatabase/` folder next to this script.")
    st.stop()

st.sidebar.success(f"Loaded {len(file_records)} file(s)  •  {len(df)} record(s)")

# File selector
file_names = [f[0] for f in file_records]
selected_files = st.sidebar.multiselect("Filter by source file", file_names, default=file_names)

# ─── Main Tabs ───
tab_dashboard, tab_table, tab_detail, tab_export = st.tabs([
    "📊 Dashboard", "📋 Table View", "📄 Paper Details", "💾 Export"
])

# ─── Pre-filtered DataFrame ───
if selected_files:
    df_filtered = df[df["_source_file"].isin(selected_files)].copy()
else:
    df_filtered = df.copy()

# ─── Dashboard Tab ───
with tab_dashboard:
    st.header("Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", len(df_filtered))
    if "Year" in df_filtered.columns:
        yrs = df_filtered["Year"].dropna()
        c2.metric("Year Range", f"{int(yrs.min())}–{int(yrs.max())}" if len(yrs) else "N/A")
    if "Authors" in df_filtered.columns:
        c3.metric("Unique Authors", df_filtered["Authors"].nunique())
    if "Cited by" in df_filtered.columns:
        cites = pd.to_numeric(df_filtered["Cited by"], errors="coerce").sum()
        c4.metric("Total Citations", int(cites) if not math.isnan(cites) else 0)

    st.divider()
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Records by Year")
        if "Year" in df_filtered.columns:
            yr_counts = df_filtered["Year"].value_counts().sort_index().reset_index()
            yr_counts.columns = ["Year", "Count"]
            st.bar_chart(yr_counts.set_index("Year"))
        else:
            st.info("No 'Year' column found.")

        st.subheader("Document Types")
        if "Document Type" in df_filtered.columns:
            dt = df_filtered["Document Type"].value_counts().reset_index()
            dt.columns = ["Type", "Count"]
            st.dataframe(dt, use_container_width=True, hide_index=True)
        else:
            st.info("No 'Document Type' column found.")

    with col_right:
        st.subheader("Top Sources")
        if "Source title" in df_filtered.columns:
            src = df_filtered["Source title"].value_counts().head(10).reset_index()
            src.columns = ["Source", "Count"]
            st.dataframe(src, use_container_width=True, hide_index=True)
        else:
            st.info("No 'Source title' column found.")

        st.subheader("Top Authors")
        if "Authors" in df_filtered.columns:
            auth_counts = df_filtered["Authors"].value_counts().head(10).reset_index()
            auth_counts.columns = ["Authors", "Count"]
            st.dataframe(auth_counts, use_container_width=True, hide_index=True)
        else:
            st.info("No 'Authors' column found.")

# ─── Table View Tab ───
with tab_table:
    st.header("Search & Filter")
    search_text = st.text_input("Keyword search (searches Title, Abstract, Authors, Keywords)", "")

    # Column filters
    filter_cols = st.columns(3)
    with filter_cols[0]:
        if "Year" in df_filtered.columns:
            yrs = sorted(df_filtered["Year"].dropna().unique())
            if yrs:
                sel_years = st.multiselect("Year", yrs)
            else:
                sel_years = []
        else:
            sel_years = []
    with filter_cols[1]:
        if "Document Type" in df_filtered.columns:
            dtypes = sorted(df_filtered["Document Type"].dropna().unique())
            sel_dtypes = st.multiselect("Document Type", dtypes)
        else:
            sel_dtypes = []
    with filter_cols[2]:
        if "Source title" in df_filtered.columns:
            sources = sorted(df_filtered["Source title"].dropna().unique())
            sel_sources = st.multiselect("Source", sources)
        else:
            sel_sources = []

    # Apply filters
    dff = df_filtered.copy()
    if search_text:
        cols_to_search = [c for c in ["Title", "Abstract", "Authors", "Author Keywords", "Index Keywords"] if c in dff.columns]
        if cols_to_search:
            mask = dff[cols_to_search].fillna("").apply(lambda row: row.str.contains(search_text, case=False, na=False)).any(axis=1)
            dff = dff[mask]
    if sel_years:
        dff = dff[dff["Year"].isin(sel_years)]
    if sel_dtypes:
        dff = dff[dff["Document Type"].isin(sel_dtypes)]
    if sel_sources:
        dff = dff[dff["Source title"].isin(sel_sources)]

    st.write(f"Showing **{len(dff)}** record(s)")
    # Choose display columns
    preferred = ["Title", "Authors", "Year", "Source title", "Document Type", "DOI", "Cited by"]
    display_cols = [c for c in preferred if c in dff.columns]
    if not display_cols:
        display_cols = list(dff.columns)
    st.dataframe(dff[display_cols], use_container_width=True, hide_index=True)

# ─── Detail View Tab ───
with tab_detail:
    st.header("Paper Details")
    if len(dff) == 0:
        st.warning("No records match the current filters.")
    else:
        # Build a selector based on title + authors
        label_col = []
        for _, row in dff.iterrows():
            title = row.get("Title", "Untitled")
            authors = row.get("Authors", "")
            year = row.get("Year", "")
            label = f"[{year}] {title[:80]}{'…' if len(str(title))>80 else ''}  —  {authors[:60]}{'…' if len(str(authors))>60 else ''}"
            label_col.append(label)
        dff = dff.copy()
        dff["__label"] = label_col
        selected_label = st.selectbox("Select a paper", dff["__label"])
        selected_row = dff[dff["__label"] == selected_label].iloc[0]

        st.divider()
        st.subheader(selected_row.get("Title", "Untitled"))
        meta_cols = st.columns([2, 1, 1, 1])
        meta_cols[0].markdown(f"**Authors:** {selected_row.get('Authors', 'N/A')}")
        meta_cols[1].markdown(f"**Year:** {selected_row.get('Year', 'N/A')}")
        meta_cols[2].markdown(f"**Type:** {selected_row.get('Document Type', 'N/A')}")
        meta_cols[3].markdown(f"**Cited by:** {selected_row.get('Cited by', 'N/A')}")

        if "DOI" in selected_row and pd.notna(selected_row["DOI"]):
            st.markdown(f"**DOI:** [{selected_row['DOI']}](https://doi.org/{selected_row['DOI']})")
        if "Link" in selected_row and pd.notna(selected_row["Link"]):
            st.markdown(f"**Link:** [{selected_row['Link'][:60]}…]({selected_row['Link']})")
        if "Source title" in selected_row and pd.notna(selected_row["Source title"]):
            st.markdown(f"**Source:** {selected_row['Source title']}")

        with st.expander("Abstract", expanded=True):
            st.write(selected_row.get("Abstract", "No abstract available."))

        with st.expander("Keywords & Index Terms"):
            if "Author Keywords" in selected_row and pd.notna(selected_row["Author Keywords"]):
                st.markdown(f"**Author Keywords:** {selected_row['Author Keywords']}")
            if "Index Keywords" in selected_row and pd.notna(selected_row["Index Keywords"]):
                st.markdown(f"**Index Keywords:** {selected_row['Index Keywords']}")

        with st.expander("Full Metadata Record"):
            # Drop internal helper columns
            full = selected_row.drop(labels=["__label", "_source_file"], errors="ignore")
            st.json(full.to_dict())

# ─── Export Tab ───
with tab_export:
    st.header("Export Filtered Data")
    if len(dff) == 0:
        st.warning("Nothing to export — adjust filters first.")
    else:
        export_cols = [c for c in dff.columns if c not in {"__label"}]
        export_df = dff[export_cols].copy()
        # Replace pd.NA with None for clean JSON
        export_json = export_df.to_json(orient="records", indent=2, force_ascii=False)
        export_csv = export_df.to_csv(index=False)
        c1, c2 = st.columns(2)
        c1.download_button("⬇ Download JSON", export_json, file_name="filtered_metadata.json", mime="application/json")
        c2.download_button("⬇ Download CSV", export_csv, file_name="filtered_metadata.csv", mime="text/csv")
        st.write(f"Ready to export **{len(export_df)}** record(s).")
