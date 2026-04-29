"""
Scopus CSV to JSON Converter - Multi-file, Multi-section
- Read CSV files from local 'metadatabase' folder (if present)
- Also accept user uploads
- All articles get unique IDs and are combined into a single JSON
"""

import streamlit as st
import pandas as pd
import json
import uuid
import re
import os
from pathlib import Path
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


def get_csv_files_from_folder(folder_name: str = "metadatabase") -> list:
    """
    Scan a local folder (relative to app root) for all .csv files.
    Returns list of (file_path, file_content_bytes) but we read content later.
    We'll return list of file paths (absolute or relative).
    """
    folder_path = Path(folder_name)
    if not folder_path.exists() or not folder_path.is_dir():
        st.info(f"📁 Folder '{folder_name}' not found – skipping folder scanning.")
        return []
    
    csv_files = list(folder_path.glob("*.csv"))
    if not csv_files:
        st.info(f"📁 Folder '{folder_name}' contains no CSV files.")
    else:
        st.info(f"📁 Found {len(csv_files)} CSV file(s) in '{folder_name}'.")
    return csv_files


def process_file_from_path(file_path: Path, progress_callback=None) -> tuple[list, dict]:
    """Read a CSV file from disk and parse it. Returns (articles, summary)."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        filename = file_path.name
        articles, summary = parse_single_csv_file(content, filename)
        if progress_callback:
            progress_callback(filename, summary)
        return articles, summary
    except Exception as e:
        st.error(f"❌ Error reading `{file_path}`: {e}")
        return [], {"error": str(e)}


def process_uploaded_files(uploaded_files, progress_bar=None) -> tuple[list, dict]:
    """
    Process user-uploaded files.
    Returns (all_articles, file_summaries)
    """
    all_articles = []
    file_summaries = {}
    total_files = len(uploaded_files)
    
    for idx, uploaded_file in enumerate(uploaded_files):
        if progress_bar:
            progress_bar.progress((idx + 1) / total_files, text=f"Processing upload {uploaded_file.name}...")
        
        try:
            content = uploaded_file.read().decode('utf-8')
            articles, summary = parse_single_csv_file(content, uploaded_file.name)
            all_articles.extend(articles)
            file_summaries[uploaded_file.name] = summary
            if articles:
                st.success(f"✅ Upload `{uploaded_file.name}` → {summary['total_articles']} articles (from {summary['sections_parsed']} sections)")
            else:
                st.warning(f"⚠️ No articles found in upload `{uploaded_file.name}`")
        except Exception as e:
            st.error(f"❌ Error processing upload `{uploaded_file.name}`: {e}")
            file_summaries[uploaded_file.name] = {"error": str(e)}
    
    return all_articles, file_summaries


def process_folder_files(folder_csv_paths, progress_bar=None) -> tuple[list, dict]:
    """
    Process all CSV files from the local folder.
    Returns (all_articles, file_summaries)
    """
    all_articles = []
    file_summaries = {}
    total_files = len(folder_csv_paths)
    
    for idx, file_path in enumerate(folder_csv_paths):
        if progress_bar:
            progress_bar.progress((idx + 1) / total_files, text=f"Processing folder file {file_path.name}...")
        
        articles, summary = process_file_from_path(file_path)
        all_articles.extend(articles)
        file_summaries[file_path.name] = summary
        if articles:
            st.success(f"✅ Folder `{file_path.name}` → {summary['total_articles']} articles (from {summary['sections_parsed']} sections)")
        else:
            if "error" not in summary:
                st.warning(f"⚠️ No articles found in folder file `{file_path.name}`")
            else:
                st.error(f"❌ Folder file `{file_path.name}` error: {summary['error']}")
    
    return all_articles, file_summaries


def main():
    st.set_page_config(page_title="Scopus Multi-CSV to JSON (Local Folder + Uploads)", page_icon="📚", layout="wide")
    st.title("📚 Scopus CSV to JSON – Local Folder + Uploads")
    st.markdown(
        "**Sources:**\n"
        "1. CSV files in `metadatabase` folder (automatically detected)\n"
        "2. Uploaded CSV files (any number)\n\n"
        "All articles receive a **unique ID** (UUID v4). Output is a single combined JSON."
    )
    
    with st.sidebar:
        st.header("How to use")
        st.markdown(
            "1. Place your Scopus CSV files inside a folder named **`metadatabase`** (same directory as this app).\n"
            "2. Or upload CSV files directly using the button below.\n"
            "3. Click **Convert** – files from both sources are processed.\n"
            "4. Download the combined JSON.\n\n"
            "**Note:** Each CSV file can contain multiple metadata sections separated by blank lines."
        )
    
    # ---- Folder scanning (always) ----
    folder_csv_paths = get_csv_files_from_folder("metadatabase")
    
    # ---- Upload widget ----
    uploaded_files = st.file_uploader(
        "Upload additional CSV files (multiple allowed)",
        type=['csv'],
        accept_multiple_files=True,
        help="These will be merged with files from metadatabase folder."
    )
    
    if st.button("🚀 Convert to JSON", type="primary"):
        all_articles = []
        file_summaries = {}
        
        # Process folder files
        if folder_csv_paths:
            with st.spinner("Scanning metadatabase folder..."):
                folder_articles, folder_summaries = process_folder_files(folder_csv_paths)
                all_articles.extend(folder_articles)
                file_summaries.update(folder_summaries)
        
        # Process uploaded files
        if uploaded_files:
            with st.spinner("Processing uploaded files..."):
                upload_articles, upload_summaries = process_uploaded_files(uploaded_files)
                all_articles.extend(upload_articles)
                file_summaries.update(upload_summaries)
        
        if not all_articles:
            st.error("No articles found in metadatabase folder or uploaded files.")
            return
        
        # Global summary
        total_articles = len(all_articles)
        total_files_with_data = sum(1 for s in file_summaries.values() if s.get("total_articles", 0) > 0)
        st.success(f"✅ **Total: {total_articles} articles** from {total_files_with_data} file(s)")
        
        # Per-file summary table
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
                title = article.get('Title', article.get('title', 'N/A'))
                st.code(f"ID: {article['unique_id']}\nTitle: {title[:80]}...")
        
        # Combined JSON download
        combined_json = json.dumps(all_articles, indent=2, ensure_ascii=False)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_filename = f"scopus_all_articles_{timestamp}.json"
        
        st.download_button(
            label="⬇️ Download Combined JSON (all articles, all sources)",
            data=combined_json,
            file_name=combined_filename,
            mime="application/json"
        )
        
        with st.expander("📄 Full JSON preview (first article)"):
            if all_articles:
                st.json(all_articles[0], expanded=True)


if __name__ == "__main__":
    main()
