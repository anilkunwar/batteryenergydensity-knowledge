"""
Scopus CSV to JSON Converter - Multi-file, Multi-section
- Read CSV files from local 'metadatabase' folder (if present)
- Also accept user uploads
- All articles get unique IDs and are combined into a single JSON
- Supports deduplication by DOI/Scopus ID
- Custom folder path input via UI
- Debug info for troubleshooting path issues
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
from typing import Optional


def parse_single_csv_section(section_content: str, source_filename: str, section_index: int) -> list:
    """
    Parse one CSV section, return list of dicts with unique IDs.
    
    Args:
        section_content: Raw CSV text for one section
        source_filename: Name of the source file for tracking
        section_index: Index of this section within the file
        
    Returns:
        List of article dictionaries with metadata fields added
    """
    try:
        # Read CSV with UTF-8 encoding, handle missing values
        df = pd.read_csv(StringIO(section_content), encoding='utf-8', dtype=str)
        df = df.where(pd.notnull(df), None)
        records = df.to_dict(orient='records')
        
        # Add tracking metadata to each record
        for record in records:
            record['unique_id'] = str(uuid.uuid4())
            record['source_file'] = source_filename
            record['section_index'] = section_index
            record['import_timestamp'] = datetime.now().isoformat()
        
        return records
    except pd.errors.EmptyDataError:
        st.warning(f"⚠️ Empty CSV section {section_index} in {source_filename}")
        return []
    except pd.errors.ParserError as e:
        st.warning(f"⚠️ CSV parse error in section {section_index} of {source_filename}: {e}")
        return []
    except UnicodeDecodeError as e:
        st.warning(f"⚠️ Encoding error in {source_filename}: {e}. Try saving as UTF-8.")
        return []
    except Exception as e:
        st.warning(f"⚠️ Failed to parse section {section_index} in {source_filename}: {type(e).__name__}: {e}")
        return []


def parse_multi_section_csv(file_content: str, source_filename: str) -> tuple[list, dict]:
    """
    Split file by blank lines, parse each chunk as CSV.
    
    Args:
        file_content: Full text content of the CSV file
        source_filename: Name of the source file
        
    Returns:
        Tuple of (all_articles list, summary dict with sections_parsed and total_articles)
    """
    # Split by one or more blank lines (handles \n\n, \n \n, etc.)
    sections = re.split(r'\n\s*\n', file_content.strip())
    all_articles = []
    sections_parsed = 0
    
    for i, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue
        # Basic validation: must have comma and newline to be CSV
        if ',' in section and '\n' in section:
            articles = parse_single_csv_section(section, source_filename, i)
            if articles:
                all_articles.extend(articles)
                sections_parsed += 1
            else:
                st.warning(f"⚠️ Section {i+1} in `{source_filename}` parsed but yielded no articles")
        else:
            st.warning(f"⚠️ Skipping section {i+1} in `{source_filename}` – not valid CSV format (missing comma or newline)")
    
    summary = {
        "sections_parsed": sections_parsed,
        "total_articles": len(all_articles),
        "total_sections_found": len([s for s in sections if s.strip()])
    }
    return all_articles, summary


def parse_single_csv_file(file_content: str, source_filename: str) -> tuple[list, dict]:
    """
    Parse a CSV file (may be multi-section or single).
    
    Args:
        file_content: Full text content of the CSV file
        source_filename: Name of the source file
        
    Returns:
        Tuple of (articles list, summary dict)
    """
    file_content_stripped = file_content.strip()
    
    # Detect multi-section by looking for blank line separators
    if re.search(r'\n\s*\n', file_content_stripped):
        st.info(f"📄 `{source_filename}`: multi-section format detected")
        return parse_multi_section_csv(file_content, source_filename)
    else:
        # Single-section CSV
        articles = parse_single_csv_section(file_content_stripped, source_filename, 0)
        summary = {
            "sections_parsed": 1 if articles else 0,
            "total_articles": len(articles),
            "total_sections_found": 1
        }
        return articles, summary


def get_csv_files_from_folder(folder_name: str) -> list:
    """
    Scan a folder for all .csv files.
    
    Args:
        folder_name: Path to folder (relative or absolute)
        
    Returns:
        List of Path objects for CSV files found
    """
    folder_path = Path(folder_name)
    
    if not folder_path.exists():
        return []
    if not folder_path.is_dir():
        st.warning(f"⚠️ Path '{folder_name}' exists but is not a directory")
        return []
    
    csv_files = sorted(folder_path.glob("*.csv"))
    
    if not csv_files:
        st.info(f"📁 Folder '{folder_name}' contains no .csv files")
    else:
        st.success(f"📁 Found {len(csv_files)} CSV file(s) in '{folder_name}'")
    
    return csv_files


def process_file_from_path(file_path: Path) -> tuple[list, dict]:
    """
    Read a CSV file from disk and parse it.
    
    Args:
        file_path: Path object pointing to the CSV file
        
    Returns:
        Tuple of (articles list, summary dict)
    """
    filename = file_path.name
    
    try:
        # Try UTF-8 first, fall back to latin-1 for older Scopus exports
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("Could not decode file with any supported encoding")
        
        articles, summary = parse_single_csv_file(content, filename)
        return articles, summary
        
    except FileNotFoundError:
        st.error(f"❌ File not found: `{file_path}`")
        return [], {"error": f"File not found: {file_path}"}
    except PermissionError:
        st.error(f"❌ Permission denied reading: `{file_path}`")
        return [], {"error": f"Permission denied: {file_path}"}
    except Exception as e:
        st.error(f"❌ Error reading `{file_path}`: {type(e).__name__}: {e}")
        return [], {"error": f"{type(e).__name__}: {str(e)}"}


def process_uploaded_files(uploaded_files: list) -> tuple[list, dict]:
    """
    Process user-uploaded files.
    
    Args:
        uploaded_files: List of Streamlit UploadedFile objects
        
    Returns:
        Tuple of (all_articles list, file_summaries dict)
    """
    all_articles = []
    file_summaries = {}
    total_files = len(uploaded_files)
    
    for idx, uploaded_file in enumerate(uploaded_files):
        try:
            # Read and decode file content
            file_bytes = uploaded_file.read()
            content = file_bytes.decode('utf-8')
            
            articles, summary = parse_single_csv_file(content, uploaded_file.name)
            all_articles.extend(articles)
            file_summaries[uploaded_file.name] = summary
            
            if articles:
                st.success(f"✅ Upload `{uploaded_file.name}` → {summary['total_articles']} articles (from {summary['sections_parsed']} sections)")
            else:
                st.warning(f"⚠️ No articles found in upload `{uploaded_file.name}`")
                
        except UnicodeDecodeError:
            # Try fallback encoding
            try:
                content = file_bytes.decode('latin-1')
                articles, summary = parse_single_csv_file(content, uploaded_file.name)
                all_articles.extend(articles)
                file_summaries[uploaded_file.name] = summary
                st.success(f"✅ Upload `{uploaded_file.name}` (latin-1) → {summary['total_articles']} articles")
            except Exception as e2:
                st.error(f"❌ Encoding error in `{uploaded_file.name}`: {e2}")
                file_summaries[uploaded_file.name] = {"error": f"Encoding: {e2}"}
        except Exception as e:
            st.error(f"❌ Error processing upload `{uploaded_file.name}`: {type(e).__name__}: {e}")
            file_summaries[uploaded_file.name] = {"error": f"{type(e).__name__}: {str(e)}"}
    
    return all_articles, file_summaries


def process_folder_files(folder_csv_paths: list) -> tuple[list, dict]:
    """
    Process all CSV files from the local folder.
    
    Args:
        folder_csv_paths: List of Path objects for CSV files
        
    Returns:
        Tuple of (all_articles list, file_summaries dict)
    """
    all_articles = []
    file_summaries = {}
    
    for file_path in folder_csv_paths:
        articles, summary = process_file_from_path(file_path)
        all_articles.extend(articles)
        file_summaries[file_path.name] = summary
        
        if "error" in summary:
            st.error(f"❌ Folder file `{file_path.name}`: {summary['error']}")
        elif articles:
            st.success(f"✅ Folder `{file_path.name}` → {summary['total_articles']} articles (from {summary['sections_parsed']} sections)")
        else:
            st.warning(f"⚠️ No articles found in folder file `{file_path.name}`")
    
    return all_articles, file_summaries


def deduplicate_articles(articles: list, key_field: str = 'DOI') -> tuple[list, int]:
    """
    Remove duplicate articles based on a key field.
    
    Args:
        articles: List of article dictionaries
        key_field: Field name to use for deduplication (default: 'DOI')
        
    Returns:
        Tuple of (deduplicated list, number of duplicates removed)
    """
    if not articles:
        return [], 0
    
    seen = set()
    unique = []
    duplicates_removed = 0
    
    for article in articles:
        # Try multiple potential ID fields in order of preference
        key = None
        for field in [key_field, 'Scopus ID', 'EID', 'DOI', 'unique_id']:
            if field in article and article[field]:
                key = f"{field}:{article[field]}"
                break
        
        if key is None:
            # Fallback to unique_id if no other identifier found
            key = article.get('unique_id')
        
        if key and key not in seen:
            seen.add(key)
            unique.append(article)
        else:
            duplicates_removed += 1
    
    return unique, duplicates_removed


def main():
    # Page configuration
    st.set_page_config(
        page_title="Scopus Multi-CSV to JSON Converter",
        page_icon="📚",
        layout="wide",
        menu_items={
            'Get Help': 'https://github.com/yourusername/scopus-converter',
            'Report a bug': 'https://github.com/yourusername/scopus-converter/issues',
            'About': "### Scopus CSV to JSON Converter\n\nConvert Scopus export CSV files (including multi-section formats) to a unified JSON with unique IDs."
        }
    )
    
    # Header
    st.title("📚 Scopus CSV to JSON Converter")
    st.markdown(
        "**Convert Scopus export CSV files to unified JSON format**\n\n"
        "**Features:**\n"
        "- ✅ Auto-detect CSV files in `metadatabase` folder\n"
        "- ✅ Upload additional CSV files (multiple allowed)\n"
        "- ✅ Handle multi-section CSV exports (separated by blank lines)\n"
        "- ✅ Assign unique UUID v4 to every article\n"
        "- ✅ Optional deduplication by DOI or Scopus ID\n"
        "- ✅ Single combined JSON output with metadata"
    )
    
    # Sidebar: Instructions and settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Folder path configuration
        st.subheader("📁 Source Folder")
        folder_path_input = st.text_input(
            "CSV folder path (relative or absolute)",
            value="metadatabase",
            help="Enter path to folder containing Scopus CSV exports. Use '.' for current directory."
        )
        
        # Deduplication option
        st.subheader("🔁 Deduplication")
        enable_dedup = st.checkbox(
            "Remove duplicate articles",
            value=False,
            help="Check to remove articles with matching DOI, Scopus ID, or EID"
        )
        dedup_field = st.selectbox(
            "Primary key for deduplication",
            options=['DOI', 'Scopus ID', 'EID', 'Title'],
            index=0,
            disabled=not enable_dedup,
            help="Field to use as primary identifier when removing duplicates"
        )
        
        # Debug info toggle
        st.subheader("🔍 Debug")
        show_debug = st.checkbox("Show working directory info", value=False)
        
        if show_debug:
            st.code(f"📍 Working directory: {os.getcwd()}")
            st.code(f"📁 Folder exists: {Path(folder_path_input).exists()}")
            st.code(f"📁 Is directory: {Path(folder_path_input).is_dir() if Path(folder_path_input).exists() else 'N/A'}")
            if Path(folder_path_input).exists() and Path(folder_path_input).is_dir():
                csv_count = len(list(Path(folder_path_input).glob("*.csv")))
                st.code(f"📄 CSV files found: {csv_count}")
        
        # Usage instructions
        st.header("📖 How to Use")
        st.markdown(
            "1. **Folder mode**: Place Scopus CSV exports in the folder specified above\n"
            "2. **Upload mode**: Use the file uploader below for additional files\n"
            "3. **Configure**: Enable deduplication if needed\n"
            "4. **Convert**: Click the button to process all sources\n"
            "5. **Download**: Get your combined JSON file\n\n"
            "**Note**: Multi-section CSVs (with blank line separators) are auto-detected."
        )
    
    # Main content area
    st.divider()
    
    # Folder scanning
    st.subheader("📁 Folder Source")
    folder_csv_paths = get_csv_files_from_folder(folder_path_input)
    
    if folder_csv_paths:
        with st.expander(f"View {len(folder_csv_paths)} CSV file(s) found", expanded=False):
            for fp in folder_csv_paths:
                st.text(f"• {fp.name} ({fp.stat().st_size:,} bytes)")
    
    # File upload section
    st.subheader("📤 Upload Additional Files")
    uploaded_files = st.file_uploader(
        "Select CSV files to upload (multiple allowed)",
        type=['csv'],
        accept_multiple_files=True,
        help="These files will be merged with folder sources. Each file can contain multiple sections separated by blank lines."
    )
    
    if uploaded_files:
        with st.expander(f"View {len(uploaded_files)} uploaded file(s)", expanded=False):
            for uf in uploaded_files:
                st.text(f"• {uf.name} ({uf.size:,} bytes)")
    
    # Convert button
    st.divider()
    col1, col2 = st.columns([1, 3])
    with col1:
        convert_btn = st.button("🚀 Convert to JSON", type="primary", use_container_width=True)
    
    # Processing logic
    if convert_btn:
        all_articles = []
        file_summaries = {}
        processing_errors = []
        
        # Progress indicator
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        
        total_sources = len(folder_csv_paths) + (len(uploaded_files) if uploaded_files else 0)
        if total_sources == 0:
            st.warning("⚠️ No source files found. Add files to the folder or upload CSV files to continue.")
            return
        
        progress_bar = st.progress(0)
        
        # Process folder files
        if folder_csv_paths:
            status_placeholder.info(f"🔄 Processing {len(folder_csv_paths)} file(s) from folder...")
            folder_articles, folder_summaries = process_folder_files(folder_csv_paths)
            all_articles.extend(folder_articles)
            file_summaries.update(folder_summaries)
            progress_bar.progress(len(folder_csv_paths) / max(total_sources, 1))
        
        # Process uploaded files
        if uploaded_files:
            status_placeholder.info(f"🔄 Processing {len(uploaded_files)} uploaded file(s)...")
            upload_articles, upload_summaries = process_uploaded_files(uploaded_files)
            all_articles.extend(upload_articles)
            file_summaries.update(upload_summaries)
            progress_bar.progress(1.0)
        
        status_placeholder.empty()
        progress_bar.empty()
        
        # Deduplication step
        if enable_dedup and all_articles:
            original_count = len(all_articles)
            all_articles, removed_count = deduplicate_articles(all_articles, key_field=dedup_field)
            if removed_count > 0:
                st.info(f"🔁 Deduplication: Removed {removed_count} duplicate(s), {len(all_articles)} unique article(s) remaining")
            else:
                st.success(f"🔁 No duplicates found based on '{dedup_field}'")
        
        # Final results
        if not all_articles:
            st.error("❌ No articles found in any source files. Please check your CSV format and encoding.")
            return
        
        # Success summary
        total_articles = len(all_articles)
        total_files_with_data = sum(1 for s in file_summaries.values() if s.get("total_articles", 0) > 0)
        st.success(f"✅ **Success!** Processed **{total_articles:,} article(s)** from **{total_files_with_data} file(s)**")
        
        # Per-file summary table
        summary_data = []
        for fname, summary in file_summaries.items():
            summary_data.append({
                "File": fname,
                "Sections Parsed": summary.get("sections_parsed", 0),
                "Total Sections": summary.get("total_sections_found", "N/A"),
                "Articles": summary.get("total_articles", 0),
                "Status": "❌ Error" if "error" in summary else "✅ OK"
            })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            st.subheader("📊 Processing Summary")
            st.dataframe(
                summary_df.style.applymap(
                    lambda x: "color: red; font-weight: bold" if "Error" in str(x) else "",
                    subset=["Status"]
                ),
                use_container_width=True,
                hide_index=True
            )
        
        # Preview section
        with st.expander("🔍 Preview: First 5 Articles (Unique IDs)", expanded=False):
            for i, article in enumerate(all_articles[:5], 1):
                title = article.get('Title') or article.get('title') or article.get('Document title') or 'N/A'
                authors = article.get('Authors') or article.get('authors') or 'N/A'
                year = article.get('Year') or article.get('publicationYear') or 'N/A'
                source = article.get('source_file', 'unknown')
                uid = article.get('unique_id', 'N/A')
                
                st.markdown(f"**{i}.** `{uid[:8]}...` | **{title[:100]}{'...' if len(str(title)) > 100 else ''}**")
                st.caption(f"Authors: {authors} | Year: {year} | Source: {source}")
                st.divider()
        
        # JSON output
        st.subheader("💾 Download Combined JSON")
        combined_json = json.dumps(all_articles, indent=2, ensure_ascii=False)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_filename = f"scopus_combined_{timestamp}.json"
        
        # Download button
        st.download_button(
            label=f"⬇️ Download JSON ({len(all_articles):,} articles, {len(combined_json)/1024:.1f} KB)",
            data=combined_json,
            file_name=combined_filename,
            mime="application/json",
            use_container_width=True
        )
        
        # Optional: Show full JSON preview (collapsible)
        with st.expander("📄 Full JSON Preview (First Article)", expanded=False):
            if all_articles:
                st.json(all_articles[0], expanded=False)
        
        # Copy JSON to clipboard option (Streamlit >= 0.87)
        st.caption("💡 Tip: Use the download button for large outputs. For small datasets, you can also copy from the preview above.")
    
    # Footer
    st.divider()
    st.caption(
        "Scopus CSV to JSON Converter • "
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} • "
        "All articles assigned UUID v4 unique identifiers"
    )


if __name__ == "__main__":
    main()
