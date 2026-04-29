"""
Scopus CSV to JSON Converter (Multi-section CSV support)
- Handles CSV files with multiple sections separated by blank lines
- Each section is parsed independently
- Each article gets a unique ID
- Supports multiple file uploads
"""

import streamlit as st
import pandas as pd
import json
import uuid
import re
from datetime import datetime
from io import StringIO


def parse_single_csv_section(section_content: str, source_filename: str, section_index: int) -> list:
    """
    Parse a single CSV section (must have header row).
    Returns list of article dicts with unique IDs.
    """
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


def parse_multi_section_csv(file_content: str, source_filename: str) -> list:
    """
    Split file content by blank lines (two or more newlines).
    Each chunk is treated as a separate CSV (with header).
    """
    sections = re.split(r'\n\s*\n', file_content.strip())
    all_articles = []
    
    for i, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue
        # Check if section looks like CSV (has at least one comma and newline)
        if ',' in section and '\n' in section:
            articles = parse_single_csv_section(section, source_filename, i)
            all_articles.extend(articles)
            if articles:
                st.info(f"  Section {i+1} of `{source_filename}`: {len(articles)} articles")
        else:
            st.warning(f"Skipping section {i+1} in `{source_filename}` – does not look like CSV data")
    
    return all_articles


def parse_single_csv_file(file_content: str, source_filename: str) -> list:
    """
    Parse a single CSV file. If it contains blank lines separating multiple sections,
    handle accordingly. Otherwise, treat as normal CSV.
    """
    if re.search(r'\n\s*\n', file_content.strip()):
        st.info(f"📄 `{source_filename}` appears to contain multiple CSV sections (blank lines detected). Parsing each separately...")
        return parse_multi_section_csv(file_content, source_filename)
    else:
        return parse_single_csv_section(file_content, source_filename, 0)


def process_uploaded_files(uploaded_files) -> list:
    """
    Process multiple uploaded files. For each file, detect if multi-section.
    Returns combined list of all articles with unique IDs.
    """
    all_articles = []
    for uploaded_file in uploaded_files:
        try:
            content = uploaded_file.read().decode('utf-8')
            articles = parse_single_csv_file(content, uploaded_file.name)
            all_articles.extend(articles)
            if articles:
                st.success(f"✅ Loaded {len(articles)} articles from `{uploaded_file.name}`")
            else:
                st.warning(f"⚠️ No articles found in `{uploaded_file.name}`")
        except Exception as e:
            st.error(f"❌ Error processing `{uploaded_file.name}`: {e}")
    return all_articles


def main():
    st.set_page_config(
        page_title="Scopus Multi-Section CSV to JSON",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Scopus Multi-Section CSV to JSON Converter")
    st.markdown(
        "**Upload CSV files** – each file may contain a single standard Scopus CSV "
        "or multiple metadata sections separated by blank lines (each section has its own header row). "
        "Every article receives a unique ID (UUID v4). All articles from all files/sections are combined into one JSON file."
    )
    
    with st.sidebar:
        st.header("How to use")
        st.markdown(
            "1. Export from Scopus as CSV (one or multiple exports).\n"
            "2. If you have multiple exports, you can concatenate them into one file with blank lines between each CSV.\n"
            "3. Upload the file(s) here.\n"
            "4. Click Convert.\n"
            "5. Download the JSON with unique IDs.\n\n"
            "**Example of multi-section CSV:**\n\n"
            '    "Authors","Title","Year"\n'
            '    "Smith J.","Study A","2023"\n\n'
            '    "Authors","Title","Year"\n'
            '    "Jones M.","Study B","2024"'
        )
    
    uploaded_files = st.file_uploader(
        "Upload Scopus CSV files",
        type=['csv'],
        accept_multiple_files=True,
        help="Supports multiple files, and files with blank-line separated sections."
    )
    
    if uploaded_files:
        st.subheader(f"📂 {len(uploaded_files)} file(s) selected")
        
        if st.button("🚀 Convert to JSON", type="primary"):
            with st.spinner("Processing files..."):
                all_articles = process_uploaded_files(uploaded_files)
            
            if not all_articles:
                st.error("No valid articles found in the uploaded files.")
                return
            
            st.success(f"✅ **{len(all_articles)} articles** processed successfully.")
            
            # Preview first few IDs
            with st.expander("🔍 Preview first 5 unique IDs"):
                for i, article in enumerate(all_articles[:5]):
                    st.code(f"ID: {article['unique_id']}\nTitle: {article.get('Title', 'N/A')[:80]}...")
            
            # JSON output
            json_output = json.dumps(all_articles, indent=2, ensure_ascii=False)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"scopus_all_articles_{timestamp}.json"
            
            st.download_button(
                label="⬇️ Download JSON (all articles with IDs)",
                data=json_output,
                file_name=filename,
                mime="application/json"
            )
            
            with st.expander("📄 Full JSON preview (first article)"):
                if all_articles:
                    st.json(all_articles[0], expanded=True)
    else:
        st.info("👆 Please upload at least one CSV file.")


if __name__ == "__main__":
    main()
