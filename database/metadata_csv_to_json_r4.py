""")

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


# Required import for re (add at top if not already present)
import re

if __name__ == "__main__":
main()
