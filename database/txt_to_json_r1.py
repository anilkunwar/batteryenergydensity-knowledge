"""
Scopus to JSON Converter
A Streamlit app that parses Scopus text exports and converts them to structured JSON.
"""

import streamlit as st
import json
import re
from datetime import datetime
from pathlib import Path


def parse_scopus_text(text):
    """
    Parse Scopus text export into a list of article dictionaries.
    """
    articles = []
    
    # Split by "EXPORT DATE:" to separate individual records
    # The first split item is usually empty or header text
    raw_records = re.split(r'(?=EXPORT DATE:)', text.strip())
    
    for raw_record in raw_records:
        raw_record = raw_record.strip()
        if not raw_record or len(raw_record) < 50:
            continue
            
        article = {}
        
        # EXPORT DATE
        export_date_match = re.search(r'EXPORT DATE:\s*(.+?)(?:\n|$)', raw_record)
        if export_date_match:
            article['export_date'] = export_date_match.group(1).strip()
        
        # AUTHORS (short format)
        authors_short = re.search(r'^([A-Z][a-zA-Z\-]+(?:,\s*[A-Z]\.)+(?:;\s*[A-Z][a-zA-Z\-]+(?:,\s*[A-Z]\.)+)*)', raw_record, re.MULTILINE)
        if authors_short:
            authors_text = authors_short.group(1)
            article['authors'] = [a.strip() for a in authors_text.split(';')]
        
        # AUTHOR FULL NAMES with IDs
        author_full_match = re.search(r'AUTHOR FULL NAMES:\s*(.+?)(?:\n\s*\d+|\n\s*[^\d]|$)', raw_record, re.DOTALL)
        if author_full_match:
            full_names_text = author_full_match.group(1).strip()
            # Parse "Name, Firstname (ID); Name2, Firstname2 (ID2)"
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
        
        # AUTHOR IDs line (the line with just numbers)
        author_ids_match = re.search(r'\n(\d+(?:;\s*\d+)+)\s*\n', raw_record)
        if author_ids_match:
            ids_text = author_ids_match.group(1)
            article['author_ids'] = [id.strip() for id in ids_text.split(';')]
        
        # TITLE
        # Title is usually the line after author IDs, before the year/journal line
        title_match = re.search(r'\d+(?:;\s*\d+)+\s*\n\s*\n(.+?)\n\s*\(\d{4}\)', raw_record, re.DOTALL)
        if title_match:
            article['title'] = ' '.join(title_match.group(1).split())
        
        # JOURNAL INFO
        journal_match = re.search(r'\((\d{4})\)\s+(.+?),\s+(\d+)(?:,\s+art\.\s+no\.\s+(\d+))?', raw_record)
        if journal_match:
            article['year'] = int(journal_match.group(1))
            article['journal'] = journal_match.group(2).strip()
            article['volume'] = journal_match.group(3)
            if journal_match.group(4):
                article['article_number'] = journal_match.group(4)
        
        # CITED COUNT
        cited_match = re.search(r'Cited\s+(\d+)\s+time', raw_record)
        if cited_match:
            article['cited_count'] = int(cited_match.group(1))
        
        # DOI
        doi_match = re.search(r'DOI:\s*(10\.\S+)', raw_record)
        if doi_match:
            article['doi'] = doi_match.group(1)
        
        # Scopus URL
        url_match = re.search(r'(https://www\.scopus\.com/inward/record\.uri\?[^\s]+)', raw_record)
        if url_match:
            article['scopus_url'] = url_match.group(1)
        
        # AFFILIATIONS
        affil_match = re.search(r'AFFILIATIONS:\s*(.+?)(?:\n\s*ABSTRACT:|$)', raw_record, re.DOTALL)
        if affil_match:
            affils_text = affil_match.group(1).strip()
            article['affiliations'] = [a.strip() for a in affils_text.split(';')]
        
        # ABSTRACT
        abstract_match = re.search(r'ABSTRACT:\s*(.+?)(?:\n\s*AUTHOR KEYWORDS:|$)', raw_record, re.DOTALL)
        if abstract_match:
            article['abstract'] = ' '.join(abstract_match.group(1).split())
        
        # AUTHOR KEYWORDS
        auth_kw_match = re.search(r'AUTHOR KEYWORDS:\s*(.+?)(?:\n\s*INDEX KEYWORDS:|$)', raw_record, re.DOTALL)
        if auth_kw_match:
            kw_text = auth_kw_match.group(1).strip()
            article['author_keywords'] = [k.strip() for k in kw_text.split(';')]
        
        # INDEX KEYWORDS
        idx_kw_match = re.search(r'INDEX KEYWORDS:\s*(.+?)(?:\n\s*CORRESPONDENCE ADDRESS:|$)', raw_record, re.DOTALL)
        if idx_kw_match:
            kw_text = idx_kw_match.group(1).strip()
            article['index_keywords'] = [k.strip() for k in kw_text.split(';')]
        
        # CORRESPONDENCE ADDRESS
        corr_match = re.search(r'CORRESPONDENCE ADDRESS:\s*(.+?)(?:\n\s*PUBLISHER:|$)', raw_record, re.DOTALL)
        if corr_match:
            article['correspondence_address'] = ' '.join(corr_match.group(1).split())
            # Extract email
            email_match = re.search(r'email:\s*([^\s;]+)', article['correspondence_address'])
            if email_match:
                article['correspondence_email'] = email_match.group(1)
        
        # PUBLISHER
        pub_match = re.search(r'PUBLISHER:\s*(.+?)(?:\n|$)', raw_record)
        if pub_match:
            article['publisher'] = pub_match.group(1).strip()
        
        # ISSN
        issn_match = re.search(r'ISSN:\s*(\d+)', raw_record)
        if issn_match:
            article['issn'] = issn_match.group(1)
        
        # CODEN
        coden_match = re.search(r'CODEN:\s*([A-Z]+)', raw_record)
        if coden_match:
            article['coden'] = coden_match.group(1)
        
        # LANGUAGE
        lang_match = re.search(r'LANGUAGE OF ORIGINAL DOCUMENT:\s*(.+?)(?:\n|$)', raw_record)
        if lang_match:
            article['language'] = lang_match.group(1).strip()
        
        # ABBREVIATED SOURCE TITLE
        abbrev_match = re.search(r'ABBREVIATED SOURCE TITLE:\s*(.+?)(?:\n|$)', raw_record)
        if abbrev_match:
            article['abbreviated_source_title'] = abbrev_match.group(1).strip()
        
        # DOCUMENT TYPE
        doc_type_match = re.search(r'DOCUMENT TYPE:\s*(.+?)(?:\n|$)', raw_record)
        if doc_type_match:
            article['document_type'] = doc_type_match.group(1).strip()
        
        # PUBLICATION STAGE
        stage_match = re.search(r'PUBLICATION STAGE:\s*(.+?)(?:\n|$)', raw_record)
        if stage_match:
            article['publication_stage'] = stage_match.group(1).strip()
        
        # SOURCE
        source_match = re.search(r'SOURCE:\s*(.+?)(?:\n|$)', raw_record)
        if source_match:
            article['source'] = source_match.group(1).strip()
        
        # Only add if we found at least a title
        if 'title' in article and article['title']:
            articles.append(article)
    
    return articles


def main():
    st.set_page_config(
        page_title="Scopus to JSON Converter",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Scopus to JSON Converter")
    st.markdown("""
    Convert Scopus text exports into structured JSON format.
    Paste your Scopus text below or upload a `.txt` file.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input")
        
        # File upload
        uploaded_file = st.file_uploader("Upload Scopus text file", type=['txt'])
        
        # Text area
        if uploaded_file is not None:
            text_content = uploaded_file.read().decode('utf-8')
            st.text_area("Scopus Text", value=text_content, height=400, key="input_text")
        else:
            text_content = st.text_area(
                "Scopus Text", 
                height=400, 
                placeholder="Paste your Scopus export text here...",
                key="input_text"
            )
    
    # Process button
    if st.button("Convert to JSON", type="primary"):
        if not text_content or text_content.strip() == "":
            st.error("Please provide Scopus text to convert.")
            return
        
        with st.spinner("Parsing Scopus data..."):
            articles = parse_scopus_text(text_content)
        
        if not articles:
            st.error("No articles found. Please check your Scopus text format.")
            return
        
        with col2:
            st.subheader("Output")
            
            # Convert to JSON
            json_output = json.dumps(articles, indent=2, ensure_ascii=False)
            
            # Display stats
            st.metric("Articles Parsed", len(articles))
            
            # Display JSON preview
            st.text_area("JSON Preview", value=json_output, height=300)
            
            # Download button
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"scopus_export_{timestamp}.json"
            
            st.download_button(
                label="⬇️ Download JSON",
                data=json_output,
                file_name=filename,
                mime="application/json"
            )
            
            # Show expandable details for each article
            with st.expander("View Parsed Articles Details"):
                for i, article in enumerate(articles, 1):
                    st.markdown(f"**Article {i}: {article.get('title', 'Untitled')}**")
                    st.json(article, expanded=False)
                    st.divider()
    
    # Instructions
    with st.expander("📖 How to use"):
        st.markdown("""
        1. **Export from Scopus**: In Scopus, select your articles and choose "Export" → "Text" format
        2. **Copy the text**: Copy the exported text (including all fields like EXPORT DATE, AUTHORS, etc.)
        3. **Paste or Upload**: Paste the text in the left panel or save it as a `.txt` file and upload it
        4. **Convert**: Click "Convert to JSON" to parse the data
        5. **Download**: Save the resulting JSON file for further analysis
        
        ### Supported Fields
        The parser extracts: Export Date, Authors (short & full names with Scopus IDs), Title, 
        Year, Journal, Volume, Article Number, Cited Count, DOI, Scopus URL, Affiliations, 
        Abstract, Author Keywords, Index Keywords, Correspondence Address, Publisher, ISSN, 
        CODEN, Language, Abbreviated Source Title, Document Type, Publication Stage, and Source.
        """)


if __name__ == "__main__":
    main()
