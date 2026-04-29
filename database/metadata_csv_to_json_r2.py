"""
Scopus to JSON Converter
Supports:
1. Scopus text export (unstructured) – using regex parsing
2. Scopus CSV export (structured) – using pandas
"""

import streamlit as st
import json
import re
import pandas as pd
from datetime import datetime
from io import StringIO

# ----------------------------------------------------------------------
# 1. Parser for unstructured Scopus TEXT export (with Unicode support)
# ----------------------------------------------------------------------
UNICODE_LETTER = r'[A-Za-z\u00C0-\u024F\u1E00-\u1EFF]'
AUTHOR_NAME_PATTERN = rf'{UNICODE_LETTER}+(?:[-\s]{UNICODE_LETTER}+)*'

def parse_scopus_text(text):
    articles = []
    raw_records = re.split(r'(?=EXPORT DATE:)', text.strip())
    
    for raw_record in raw_records:
        raw_record = raw_record.strip()
        if not raw_record or len(raw_record) < 50:
            continue
        article = {}

        # Export date
        match = re.search(r'EXPORT DATE:\s*(.+?)(?:\n|$)', raw_record)
        if match:
            article['export_date'] = match.group(1).strip()

        # Authors (short format)
        match = re.search(
            r'^(' + AUTHOR_NAME_PATTERN + r'(?:,\s*[A-Z]\.)+(?:;\s*' + AUTHOR_NAME_PATTERN + r'(?:,\s*[A-Z]\.)+)*)',
            raw_record, re.MULTILINE
        )
        if match:
            article['authors'] = [a.strip() for a in match.group(1).split(';')]

        # Author full names
        match = re.search(r'AUTHOR FULL NAMES:\s*(.+?)(?:\n\s*\d+|\n\s*[^\d]|$)', raw_record, re.DOTALL)
        if match:
            full_names_text = match.group(1).strip()
            author_entries = re.findall(r'([^;]+?\(\d+\))', full_names_text)
            authors_full = []
            for entry in author_entries:
                name_match = re.match(r'(.+?),\s*(.+?)\s*\((\d+)\)', entry.strip())
                if name_match:
                    authors_full.append({
                        'last_name': name_match.group(1).strip(),
                        'first_name': name_match.group(2).strip(),
                        'scopus_id': name_match.group(3)
                    })
            article['authors_full'] = authors_full

        # Author IDs
        match = re.search(r'\n(\d+(?:;\s*\d+)+)\s*\n', raw_record)
        if match:
            article['author_ids'] = [id.strip() for id in match.group(1).split(';')]

        # Title
        match = re.search(r'\d+(?:;\s*\d+)+\s*\n\s*\n(.+?)\n\s*\(\d{4}\)', raw_record, re.DOTALL)
        if match:
            article['title'] = ' '.join(match.group(1).split())

        # Year, journal, volume, article number
        match = re.search(r'\((\d{4})\)\s+(.+?),\s+(\d+)(?:,\s+art\.\s+no\.\s+(\d+))?', raw_record)
        if match:
            article['year'] = int(match.group(1))
            article['journal'] = match.group(2).strip()
            article['volume'] = match.group(3)
            if match.group(4):
                article['article_number'] = match.group(4)

        # Cited count
        match = re.search(r'Cited\s+(\d+)\s+time', raw_record)
        if match:
            article['cited_count'] = int(match.group(1))

        # DOI, URL, affiliations, abstract, keywords, etc.
        match = re.search(r'DOI:\s*(10\.\S+)', raw_record)
        if match:
            article['doi'] = match.group(1)

        match = re.search(r'(https://www\.scopus\.com/inward/record\.uri\?[^\s]+)', raw_record)
        if match:
            article['scopus_url'] = match.group(1)

        match = re.search(r'AFFILIATIONS:\s*(.+?)(?:\n\s*ABSTRACT:|$)', raw_record, re.DOTALL)
        if match:
            article['affiliations'] = [a.strip() for a in match.group(1).split(';')]

        match = re.search(r'ABSTRACT:\s*(.+?)(?:\n\s*AUTHOR KEYWORDS:|$)', raw_record, re.DOTALL)
        if match:
            article['abstract'] = ' '.join(match.group(1).split())

        match = re.search(r'AUTHOR KEYWORDS:\s*(.+?)(?:\n\s*INDEX KEYWORDS:|$)', raw_record, re.DOTALL)
        if match:
            article['author_keywords'] = [k.strip() for k in match.group(1).split(';')]

        match = re.search(r'INDEX KEYWORDS:\s*(.+?)(?:\n\s*CORRESPONDENCE ADDRESS:|$)', raw_record, re.DOTALL)
        if match:
            article['index_keywords'] = [k.strip() for k in match.group(1).split(';')]

        match = re.search(r'CORRESPONDENCE ADDRESS:\s*(.+?)(?:\n\s*PUBLISHER:|$)', raw_record, re.DOTALL)
        if match:
            article['correspondence_address'] = ' '.join(match.group(1).split())
            email_match = re.search(r'email:\s*([^\s;]+)', article['correspondence_address'], re.IGNORECASE)
            if email_match:
                article['correspondence_email'] = email_match.group(1)

        match = re.search(r'PUBLISHER:\s*(.+?)(?:\n|$)', raw_record)
        if match:
            article['publisher'] = match.group(1).strip()

        match = re.search(r'ISSN:\s*(\d+)', raw_record)
        if match:
            article['issn'] = match.group(1)

        match = re.search(r'CODEN:\s*([A-Z]+)', raw_record)
        if match:
            article['coden'] = match.group(1)

        match = re.search(r'LANGUAGE OF ORIGINAL DOCUMENT:\s*(.+?)(?:\n|$)', raw_record)
        if match:
            article['language'] = match.group(1).strip()

        match = re.search(r'ABBREVIATED SOURCE TITLE:\s*(.+?)(?:\n|$)', raw_record)
        if match:
            article['abbreviated_source_title'] = match.group(1).strip()

        match = re.search(r'DOCUMENT TYPE:\s*(.+?)(?:\n|$)', raw_record)
        if match:
            article['document_type'] = match.group(1).strip()

        match = re.search(r'PUBLICATION STAGE:\s*(.+?)(?:\n|$)', raw_record)
        if match:
            article['publication_stage'] = match.group(1).strip()

        match = re.search(r'SOURCE:\s*(.+?)(?:\n|$)', raw_record)
        if match:
            article['source'] = match.group(1).strip()

        if 'title' in article or 'authors' in article:
            articles.append(article)

    return articles


# ----------------------------------------------------------------------
# 2. Converter for structured Scopus CSV export
# ----------------------------------------------------------------------
def parse_scopus_csv(csv_content):
    """
    Convert Scopus CSV (UTF-8) to a list of dictionaries.
    Handles the exact column names shown in the user's example.
    """
    df = pd.read_csv(StringIO(csv_content), encoding='utf-8')
    # Replace NaN with None for clean JSON
    df = df.where(pd.notnull(df), None)
    records = df.to_dict(orient='records')
    return records


# ----------------------------------------------------------------------
# Streamlit App
# ----------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Scopus to JSON Converter", page_icon="📚", layout="wide")
    st.title("📚 Scopus to JSON Converter")
    st.markdown("""
    **Two input formats supported:**
    1. **CSV** (recommended) – export from Scopus as CSV, then upload.  
    2. **Text** – the raw Scopus text export (unstructured, but also works).
    """)

    # Choose input type
    input_type = st.radio("Choose export format", ("CSV (structured)", "Text (unstructured)"))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input")
        uploaded_file = st.file_uploader(
            f"Upload Scopus {input_type.split()[0]} file",
            type=['csv'] if input_type.startswith("CSV") else ['txt']
        )

        if input_type.startswith("CSV"):
            if uploaded_file is not None:
                csv_content = uploaded_file.read().decode('utf-8')
                st.text_area("CSV Preview (first 500 chars)", csv_content[:500], height=200)
            else:
                csv_content = None
                st.info("Please upload a CSV file exported from Scopus.")
        else:
            if uploaded_file is not None:
                text_content = uploaded_file.read().decode('utf-8')
                st.text_area("Scopus Text", value=text_content, height=400)
            else:
                text_content = st.text_area("Scopus Text", height=400, placeholder="Paste your Scopus text export here...")

    # Process button
    if st.button("Convert to JSON", type="primary"):
        if input_type.startswith("CSV"):
            if not uploaded_file:
                st.error("Please upload a CSV file.")
                return
            with st.spinner("Parsing CSV..."):
                try:
                    articles = parse_scopus_csv(csv_content)
                except Exception as e:
                    st.error(f"Error parsing CSV: {e}")
                    return
        else:
            if not text_content or text_content.strip() == "":
                st.error("Please provide Scopus text.")
                return
            with st.spinner("Parsing text..."):
                articles = parse_scopus_text(text_content)

        if not articles:
            st.error("No articles found. Please check the file format.")
            return

        with col2:
            st.subheader("Output")
            st.metric("Articles Parsed", len(articles))

            json_output = json.dumps(articles, indent=2, ensure_ascii=False)
            st.text_area("JSON Preview", value=json_output, height=300)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"scopus_{timestamp}.json"
            st.download_button(
                label="⬇️ Download JSON",
                data=json_output,
                file_name=filename,
                mime="application/json"
            )

            with st.expander("View first article details"):
                if articles:
                    st.json(articles[0], expanded=False)

    with st.expander("📖 How to export CSV from Scopus"):
        st.markdown("""
        1. In Scopus, select your articles.
        2. Click **Export** → **CSV**.
        3. Choose **All available fields** (or at least the columns you need).
        4. Download the CSV file.
        5. Upload it here – the converter automatically transforms it to JSON.
        """)

if __name__ == "__main__":
    main()
