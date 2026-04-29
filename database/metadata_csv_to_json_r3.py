"""
Scopus to JSON Converter (CSV-only, multi-file, with unique IDs)
- Upload one or more Scopus CSV exports
- Each article gets a unique ID (UUID)
- Combined JSON output with all articles
"""

import streamlit as st
import pandas as pd
import json
import uuid
from datetime import datetime
from io import StringIO


def parse_scopus_csv(csv_content: str, source_filename: str = None) -> list:
    """
    Convert Scopus CSV (UTF-8) to a list of dictionaries.
    Adds metadata: source_filename and a unique ID (if not already present).
    """
    df = pd.read_csv(StringIO(csv_content), encoding='utf-8')
    # Replace NaN with None for clean JSON
    df = df.where(pd.notnull(df), None)
    records = df.to_dict(orient='records')
    
    # Add unique ID and source file info to each record
    for record in records:
        # Generate a unique ID (UUID4) for each article
        record['unique_id'] = str(uuid.uuid4())
        if source_filename:
            record['source_file'] = source_filename
        # Optionally, add timestamp
        record['import_timestamp'] = datetime.now().isoformat()
    
    return records


def process_uploaded_files(uploaded_files) -> list:
    """
    Process multiple uploaded CSV files.
    Returns a combined list of all articles with unique IDs.
    """
    all_articles = []
    for uploaded_file in uploaded_files:
        try:
            content = uploaded_file.read().decode('utf-8')
            articles = parse_scopus_csv(content, uploaded_file.name)
            all_articles.extend(articles)
            st.success(f"✅ Loaded {len(articles)} articles from `{uploaded_file.name}`")
        except Exception as e:
            st.error(f"❌ Error processing `{uploaded_file.name}`: {e}")
    return all_articles


def main():
    st.set_page_config(
        page_title="Scopus CSV to JSON Converter",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Scopus CSV to JSON Converter")
    st.markdown("""
    **Upload one or more Scopus CSV exports** – each article will receive a **unique ID**.
    The output is a single JSON array containing all articles, ready for retrieval by ID.
    """)
    
    # Sidebar: instructions
    with st.sidebar:
        st.header("How to export from Scopus")
        st.markdown("""
        1. In Scopus, select your articles.
        2. Click **Export** → **CSV**.
        3. Choose **All available fields** (recommended).
        4. Download the CSV file(s).
        5. Upload them here.
        """)
        st.divider()
        st.caption("The unique ID is a UUID v4. Use it to retrieve individual abstracts later.")
    
    # Main area
    uploaded_files = st.file_uploader(
        "Upload Scopus CSV files",
        type=['csv'],
        accept_multiple_files=True,
        help="You can select multiple CSV files at once."
    )
    
    if uploaded_files:
        st.subheader(f"📂 {len(uploaded_files)} file(s) selected")
        
        if st.button("🚀 Convert to JSON", type="primary"):
            with st.spinner("Processing CSV files..."):
                all_articles = process_uploaded_files(uploaded_files)
            
            if not all_articles:
                st.error("No valid articles found in the uploaded files.")
                return
            
            # Display stats
            st.success(f"✅ Total articles processed: **{len(all_articles)}**")
            
            # Show first few IDs as preview
            with st.expander("🔍 Preview first 5 unique IDs"):
                for i, article in enumerate(all_articles[:5]):
                    st.code(f"ID: {article['unique_id']}\nTitle: {article.get('Title', 'N/A')[:80]}...")
            
            # Prepare JSON output
            json_output = json.dumps(all_articles, indent=2, ensure_ascii=False)
            
            # Provide download button
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"scopus_all_articles_{timestamp}.json"
            
            st.download_button(
                label="⬇️ Download JSON (all articles with IDs)",
                data=json_output,
                file_name=filename,
                mime="application/json"
            )
            
            # Optional: show full JSON preview (collapsed)
            with st.expander("📄 Full JSON preview (first article)"):
                if all_articles:
                    st.json(all_articles[0], expanded=True)
    
    else:
        st.info("👆 Please upload at least one CSV file exported from Scopus.")


if __name__ == "__main__":
    main()
