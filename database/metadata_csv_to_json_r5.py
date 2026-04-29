"""
Scopus CSV to JSON Converter - Multi-file, Multi-section
- Upload M CSV files, each may contain any number of metadata sections (blank-line separated)
- Each article receives a unique ID (UUID v4)
- Output: single combined JSON + optional per‑file JSONs
"""

import streamlit as st
import pandas as pd
import json
import uuid
import re
from datetime import datetime
from io import StringIO


def parse_single_csv_section(section_content: str, source_filename: str, section_index: int) -> list:
    """Parse one CSV section, return list of dicts with unique IDs."""
    try:
        df = pd.read_csv(StringIO(section_content), encoding='utf-8')
        df = df.where(pd.notnull(df), None)
        records = df.to_dict(orient='records')
        
        for record in records:
            record['unique_id'] = str(uuid.uuid4())
            record['source_file'] = source_filename
            record['section_index'] = section_index
            record['import_timestamp'] = datetime.now().isoformat()
        
        return records
    except Exception as e:
        st.warning(f"Failed to parse section {section_index} in {source_filename}: {e}")
        return []


def parse_multi_section_csv(file_content: str, source_filename: str) -> tuple[list, dict]:
    """
    Split file by blank lines, parse each chunk as CSV.
    Returns (all_articles, summary) where summary = {sections_parsed, total_articles}
    """
    sections = re.split(r'\n\s*\n', file_content.strip())
    all_articles = []
    sections_parsed = 0
    for i, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue
        if ',' in section and '\n' in section:
            articles = parse_single_csv_section(section, source_filename, i)
            all_articles.extend(articles)
            sections_parsed += 1
        else:
            st.warning(f"Skipping section {i+1} in `{source_filename}` – not valid CSV")
    return all_articles, {"sections_parsed": sections_parsed, "total_articles": len(all_articles)}


def parse_single_csv_file(file_content: str, source_filename: str) -> tuple[list, dict]:
    """Parse a CSV file (may be multi-section or single). Returns (articles, summary)."""
    if re.search(r'\n\s*\n', file_content.strip()):
        st.info(f"📄 `{source_filename}`: multi-section detected")
        return parse_multi_section_csv(file_content, source_filename)
    else:
        articles = parse_single_csv_section(file_content, source_filename, 0)
        summary = {"sections_parsed": 1 if articles else 0, "total_articles": len(articles)}
        return articles, summary


def process_uploaded_files(uploaded_files, progress_bar=None) -> tuple[list, dict]:
    """
    Process all uploaded files.
    Returns (all_articles, file_summaries) where file_summaries = {filename: summary}
    """
    all_articles = []
    file_summaries = {}
    total_files = len(uploaded_files)
    
    for idx, uploaded_file in enumerate(uploaded_files):
        if progress_bar:
            progress_bar.progress((idx + 1) / total_files, text=f"Processing {uploaded_file.name}...")
        
        try:
            content = uploaded_file.read().decode('utf-8')
            articles, summary = parse_single_csv_file(content, uploaded_file.name)
            all_articles.extend(articles)
            file_summaries[uploaded_file.name] = summary
            if articles:
                st.success(f"✅ `{uploaded_file.name}` → {summary['total_articles']} articles (from {summary['sections_parsed']} sections)")
            else:
                st.warning(f"⚠️ No articles found in `{uploaded_file.name}`")
        except Exception as e:
            st.error(f"❌ Error processing `{uploaded_file.name}`: {e}")
            file_summaries[uploaded_file.name] = {"error": str(e)}
    
    return all_articles, file_summaries


def save_per_file_json(articles_by_file: dict, timestamp: str) -> dict:
    """Save separate JSON files for each original CSV file. Returns dict of filenames."""
    per_file_paths = {}
    for filename, articles in articles_by_file.items():
        if articles:
            out_filename = f"{filename.replace('.csv', '')}_{timestamp}.json"
            json_out = json.dumps(articles, indent=2, ensure_ascii=False)
            per_file_paths[filename] = out_filename
    return per_file_paths


def main():
    st.set_page_config(page_title="Scopus Multi-CSV to JSON", page_icon="📚", layout="wide")
    st.title("📚 Scopus Multi‑File / Multi‑Section CSV to JSON")
    st.markdown(
        "Upload **M CSV files** – each may contain **any number of metadata sections** (blank‑line separated). "
        "Every article receives a **unique ID** (UUID v4). You can download a **combined JSON** (all articles) "
        "or **separate JSONs per original file**."
    )
    
    with st.sidebar:
        st.header("How to use")
        st.markdown(
            "1. Export Scopus articles as CSV (one or multiple exports).\n"
            "2. (Optional) Concatenate several CSV exports into one file, with **blank lines** between them.\n"
            "3. Upload all CSV files here.\n"
            "4. Click **Convert**.\n"
            "5. Download the combined JSON or individual JSONs.\n\n"
            "**Example of a multi‑section CSV file:**\n\n"
            '    "Authors","Title","Year"\n'
            '    "Smith J.","Study A","2023"\n\n'
            '    "Authors","Title","Year"\n'
            '    "Jones M.","Study B","2024"'
        )
    
    uploaded_files = st.file_uploader(
        "Upload Scopus CSV files (multiple allowed)",
        type=['csv'],
        accept_multiple_files=True,
        help="Each file may contain one or more CSV sections separated by blank lines."
    )
    
    if uploaded_files:
        st.subheader(f"📂 {len(uploaded_files)} file(s) selected")
        
        if st.button("🚀 Convert to JSON", type="primary"):
            progress_bar = st.progress(0, text="Starting...")
            with st.spinner("Processing files..."):
                all_articles, file_summaries = process_uploaded_files(uploaded_files, progress_bar)
            progress_bar.empty()
            
            if not all_articles:
                st.error("No valid articles found in any uploaded file.")
                return
            
            # Global summary
            total_articles = len(all_articles)
            total_files_with_data = sum(1 for s in file_summaries.values() if s.get("total_articles", 0) > 0)
            st.success(f"✅ **Total: {total_articles} articles** from {total_files_with_data} file(s)")
            
            # Display per‑file summary table
            summary_df = pd.DataFrame([
                {
                    "File": fname,
                    "Sections": summary.get("sections_parsed", 0),
                    "Articles": summary.get("total_articles", 0),
                    "Status": "Error" if "error" in summary else "OK"
                }
                for fname, summary in file_summaries.items()
            ])
            st.dataframe(summary_df, use_container_width=True)
            
            # Preview first IDs
            with st.expander("🔍 Preview first 5 unique IDs"):
                for i, article in enumerate(all_articles[:5]):
                    st.code(f"ID: {article['unique_id']}\nTitle: {article.get('Title', 'N/A')[:80]}...")
            
            # Prepare combined JSON
            combined_json = json.dumps(all_articles, indent=2, ensure_ascii=False)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            combined_filename = f"scopus_all_articles_{timestamp}.json"
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="⬇️ Download Combined JSON (all articles)",
                    data=combined_json,
                    file_name=combined_filename,
                    mime="application/json"
                )
            
            # Optional per‑file JSONs (if user wants)
            # Group articles by source_file
            articles_by_file = {}
            for art in all_articles:
                fname = art["source_file"]
                articles_by_file.setdefault(fname, []).append(art)
            
            with col2:
                if len(articles_by_file) > 1:
                    if st.button("📁 Generate per‑file JSONs"):
                        per_file_paths = save_per_file_json(articles_by_file, timestamp)
                        for fname, out_name in per_file_paths.items():
                            json_data = json.dumps(articles_by_file[fname], indent=2, ensure_ascii=False)
                            st.download_button(
                                label=f"⬇️ {out_name}",
                                data=json_data,
                                file_name=out_name,
                                mime="application/json",
                                key=f"perfile_{fname}"
                            )
                else:
                    st.info("Only one source file – use the combined download.")
            
            with st.expander("📄 Full JSON preview (first article)"):
                if all_articles:
                    st.json(all_articles[0], expanded=True)
    else:
        st.info("👆 Please upload at least one CSV file.")


if __name__ == "__main__":
    main()
